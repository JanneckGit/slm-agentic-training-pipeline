"""
evaluation/rescore.py
=====================
Offline re-scoring of saved Text-to-SQL predictions — WITHOUT re-running
inference. Reads the per-example *_predictions.jsonl files written by
evaluate.py, re-applies extract_sql + execution_match (loose/strict) to the
stored raw_output, OVERWRITES the result files in place (keeping a one-time
.orig backup), and re-logs the corrected metrics to the matching MLflow
baseline runs.

All scoring logic lives in evaluate.py — this module imports it (single source
of truth) and only re-runs it over saved outputs. evaluate.py has no top-level
torch/transformers imports, so importing it here is cheap.

Why: an earlier eval run used a buggy extract_sql (left Markdown fences /
reasoning in the SQL), so several models scored a false 0%. Re-scoring fixes
the metrics without paying for inference again.

The rescore is idempotent: it re-extracts from the preserved raw_output, so it
can be run again (e.g. after another extract_sql fix) and the .orig backup
always keeps the very first original.

Usage:
    python evaluation/rescore.py --config config/pipeline_config.yaml
    python evaluation/rescore.py --eval-dir data/final/eval/Qwen3.6-27B
    python evaluation/rescore.py --no-mlflow          # files only, skip MLflow
    python evaluation/rescore.py --tracking-uri file:///abs/path/to/mlruns
"""

import argparse
import json
import logging
import os
from pathlib import Path

import yaml

from evaluation.evaluate import (
    MLFLOW_EXPERIMENT,
    baseline_run_name,
    execution_match,
    extract_sql,
    normalize_sql,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_PRED_SUFFIX = "_predictions.jsonl"


def load_test_map(test_path: Path) -> dict:
    """Maps question -> {schema, gold_sql} from the eval test set, so stored
    predictions can be re-executed without re-running inference. Same source +
    fields evaluate.py reads (ex['schema'], ex['sql'])."""
    qmap = {}
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            qmap[ex.get("question", "")] = {
                "schema": ex.get("schema", ""),
                "gold_sql": ex.get("sql", ""),
            }
    return qmap


def rescore_file(pred_path: Path, qmap: dict) -> dict:
    """Re-scores one *_predictions.jsonl file. Returns the new aggregate, the
    re-scored per-example rows (new schema, raw_output preserved), the
    gold_failed count, and the pre-rescore execution accuracy (for reporting)."""
    rows = []
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n = n_em = n_ex = n_ex_strict = n_gold_failed = n_old_ex = 0
    by_complexity: dict[str, dict] = {}
    rescored_rows = []
    for r in rows:
        question = r.get("question", "")
        complexity = r.get("complexity", "unknown")
        raw_output = r.get("raw_output", "")
        # Test set is authoritative for schema + gold (evaluate.py loads the same).
        meta = qmap.get(question, {})
        schema = meta.get("schema", "")
        gold_sql = meta.get("gold_sql") or r.get("gold_sql", "")

        pred_sql = extract_sql(raw_output)
        em = normalize_sql(pred_sql) == normalize_sql(gold_sql)
        ex_loose, ex_strict, exec_error, gold_failed = execution_match(
            pred_sql, gold_sql, schema)

        n += 1
        n_em += int(em)
        n_ex += int(ex_loose)
        n_ex_strict += int(ex_strict)
        n_gold_failed += int(gold_failed)
        n_old_ex += int(bool(r.get("execution_match")))
        bc = by_complexity.setdefault(
            complexity, {"em": 0, "ex": 0, "ex_strict": 0, "total": 0})
        bc["total"] += 1
        bc["em"] += int(em)
        bc["ex"] += int(ex_loose)
        bc["ex_strict"] += int(ex_strict)

        rescored_rows.append({
            **r,
            "gold_sql": gold_sql,
            "extracted_sql": pred_sql,
            "exact_match": em,
            "execution_match": ex_loose,
            "execution_match_strict": ex_strict,
            "gold_failed": gold_failed,
            "error": exec_error,
        })

    overall = {
        "n": n,
        "exact_match": round(n_em / n, 4) if n else 0,
        "execution_accuracy": round(n_ex / n, 4) if n else 0,
        "execution_accuracy_strict": round(n_ex_strict / n, 4) if n else 0,
    }
    agg = {
        "overall": overall,
        "by_complexity": {
            k: {
                "n": v["total"],
                "exact_match": round(v["em"] / v["total"], 4) if v["total"] else 0,
                "execution_accuracy": round(v["ex"] / v["total"], 4) if v["total"] else 0,
                "execution_accuracy_strict": round(v["ex_strict"] / v["total"], 4) if v["total"] else 0,
            }
            for k, v in by_complexity.items()
        },
    }
    return {
        "agg": agg,
        "rescored_rows": rescored_rows,
        "gold_failed": n_gold_failed,
        "old_execution_accuracy": round(n_old_ex / n, 4) if n else 0,
    }


def write_back(pred_path: Path, result: dict) -> str:
    """Overwrites <stem>_predictions.jsonl + <stem>.json in place. Creates a
    one-time <stem>_predictions.jsonl.orig backup (never overwritten, so the
    very first original survives re-rescoring). Returns the stem."""
    stem = pred_path.name[: -len(_PRED_SUFFIX)]
    backup = pred_path.with_name(pred_path.name + ".orig")
    if not backup.exists():
        backup.write_bytes(pred_path.read_bytes())

    with open(pred_path, "w") as f:
        for r in result["rescored_rows"]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    agg_path = pred_path.with_name(f"{stem}.json")
    with open(agg_path, "w") as f:
        json.dump(result["agg"], f, indent=2)
    return stem


def run_name_for(pred_path: Path) -> str:
    """MLflow run name for a prediction file, via evaluate.py's convention:
    the eval subdir name == Path(model_path).name, run == baseline_<name.lower()>."""
    return baseline_run_name(pred_path.parent.name)


def resolve_tracking_uri(arg: str | None) -> str | None:
    """--tracking-uri > $MLFLOW_TRACKING_URI > local ./mlruns (where runs live
    when running on the host; evaluate.py uses file:///app/mlruns in Docker)."""
    if arg:
        return arg
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    local = Path("mlruns")
    if local.is_dir():
        return local.resolve().as_uri()
    return None


def update_mlflow(tracking_uri: str, results_by_path: dict) -> list:
    """Re-logs corrected metrics to the matching baseline runs.

    For each prediction file: find the run named baseline_<model> (the LATEST
    started one if a model was evaluated multiple times), resume it, and re-log
    eval_ex (= loose), eval_ex_strict, eval_em + per-complexity eval_ex_<klasse>
    (loose). MLflow keeps the latest value as current. Tags the run rescored=true.

    Returns [(model, run_id, old_eval_ex, new_eval_ex), ...]. Degrades
    gracefully (warns, returns what it did) if MLflow is unreachable.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except Exception as e:  # mlflow not installed
        logger.warning(f"  [mlflow] not importable ({e}); skipping run updates.")
        return []
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
        if exp is None:
            logger.warning(f"  [mlflow] experiment '{MLFLOW_EXPERIMENT}' not found "
                           f"at {tracking_uri}; skipping run updates.")
            return []
        runs = client.search_runs([exp.experiment_id],
                                  order_by=["attributes.start_time DESC"],
                                  max_results=2000)
    except Exception as e:
        logger.warning(f"  [mlflow] cannot reach store at {tracking_uri} ({e}); "
                       f"skipping run updates.")
        return []

    # runs come newest-first → first occurrence of a name is the latest run.
    latest_by_name: dict[str, object] = {}
    for run in runs:
        name = run.data.tags.get("mlflow.runName", "")
        latest_by_name.setdefault(name, run)

    report = []
    for pred_path, result in sorted(results_by_path.items()):
        run_name = run_name_for(pred_path)
        run = latest_by_name.get(run_name)
        if run is None:
            logger.warning(f"  [mlflow] no run named '{run_name}' "
                           f"({pred_path.parent.name}); skipped.")
            continue
        run_id = run.info.run_id
        old_ex = run.data.metrics.get("eval_ex")
        overall = result["agg"]["overall"]
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("eval_ex", overall["execution_accuracy"])
                mlflow.log_metric("eval_ex_strict", overall["execution_accuracy_strict"])
                mlflow.log_metric("eval_em", overall["exact_match"])
                for klasse, v in result["agg"]["by_complexity"].items():
                    mlflow.log_metric(f"eval_ex_{klasse.replace(' ', '_')}",
                                      v["execution_accuracy"])
                mlflow.set_tag("rescored", "true")
            old_str = f"{old_ex:.4f}" if isinstance(old_ex, (int, float)) else str(old_ex)
            logger.info(f"  [mlflow] {run_name} ({run_id}): "
                        f"eval_ex {old_str} -> {overall['execution_accuracy']:.4f}")
            report.append((pred_path.parent.name, run_id, old_ex,
                           overall["execution_accuracy"]))
        except Exception as e:
            logger.warning(f"  [mlflow] failed to update {run_name} ({e}).")
    return report


def print_table(results_by_path: dict) -> None:
    """Comparison table over all rescored models: strict / loose / EM, the
    loose-vs-strict delta, and the gold_failed count."""
    rows = []
    for pred_path, result in sorted(results_by_path.items()):
        o = result["agg"]["overall"]
        rows.append((
            pred_path.parent.name,
            o["execution_accuracy_strict"],
            o["execution_accuracy"],
            o["exact_match"],
            round(o["execution_accuracy"] - o["execution_accuracy_strict"], 4),
            result["gold_failed"],
        ))
    if not rows:
        logger.info("No files rescored.")
        return
    w = max(len("model"), max(len(r[0]) for r in rows))
    print("\n=== Re-scored comparison (execution_accuracy = loose) ===")
    print(f"{'model':<{w}}  {'strict':>7}  {'loose':>7}  {'EM':>7}  "
          f"{'Δ(L-S)':>7}  {'gold_fail':>9}")
    print("-" * (w + 46))
    for name, strict, loose, em, delta, gf in rows:
        print(f"{name:<{w}}  {strict:>7.3f}  {loose:>7.3f}  {em:>7.3f}  "
              f"{delta:>+7.3f}  {gf:>9d}")


def main():
    ap = argparse.ArgumentParser(
        description="Offline re-score saved predictions (overwrites in place).")
    ap.add_argument("--config", default="config/pipeline_config.yaml")
    ap.add_argument("--eval-dir", default=None,
                    help="Re-score only this eval subdir (default: all under final_dir/eval)")
    ap.add_argument("--test-file", default=None, help="Override eval test path")
    ap.add_argument("--tracking-uri", default=None,
                    help="MLflow tracking URI (default: ./mlruns or $MLFLOW_TRACKING_URI)")
    ap.add_argument("--no-mlflow", action="store_true",
                    help="Re-score files only; do not update MLflow runs")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    test_path = Path(args.test_file) if args.test_file else \
        Path(config["data"]["final_dir"]) / config["data"].get("eval_test_file", "test_clean.jsonl")
    qmap = load_test_map(test_path)
    logger.info(f"Loaded test map: {len(qmap)} questions from {test_path}")

    if args.eval_dir:
        pred_files = sorted(Path(args.eval_dir).glob(f"*{_PRED_SUFFIX}"))
    else:
        eval_root = Path(config["data"]["final_dir"]) / "eval"
        pred_files = sorted(eval_root.glob(f"*/*{_PRED_SUFFIX}"))
    # Never treat a previous rescore sidecar as input (legacy naming).
    pred_files = [p for p in pred_files
                  if not p.name.endswith(f"_rescored{_PRED_SUFFIX}")]
    logger.info(f"Found {len(pred_files)} prediction files")

    results_by_path = {}
    for pred_path in pred_files:
        result = rescore_file(pred_path, qmap)
        stem = write_back(pred_path, result)
        results_by_path[pred_path] = result
        o = result["agg"]["overall"]
        logger.info(f"  {pred_path.parent.name}: "
                    f"loose={o['execution_accuracy']:.3f} "
                    f"strict={o['execution_accuracy_strict']:.3f} "
                    f"EM={o['exact_match']:.3f} gold_fail={result['gold_failed']} "
                    f"(n={o['n']}) [overwrote {pred_path.name} + {stem}.json]")

    if not args.no_mlflow:
        tracking_uri = resolve_tracking_uri(args.tracking_uri)
        if tracking_uri:
            logger.info(f"Updating MLflow runs at {tracking_uri} ...")
            update_mlflow(tracking_uri, results_by_path)
        else:
            logger.info("No MLflow store found; skipping (pass --tracking-uri to set one).")

    print_table(results_by_path)


if __name__ == "__main__":
    main()

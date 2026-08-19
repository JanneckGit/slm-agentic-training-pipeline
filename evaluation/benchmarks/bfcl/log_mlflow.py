#!/usr/bin/env python3
"""Log one BFCL run to MLflow (experiment `bfcl_eval`).

Runs under .venv-tau2 — the bfcl venv carries no mlflow. It only reads what is on disk:
run_manifest.json (params) and score/*/*/BFCL_v4_*_score.json (metrics). No bfcl imports.

    MLFLOW_TRACKING_URI=file://$PWD/mlruns MLFLOW_ALLOW_FILE_STORE=true \
      .venv-tau2/bin/python evaluation/benchmarks/bfcl/log_mlflow.py \
        --run data/generated/eval/bfcl/qwen3-4b_base --experiment bfcl_eval

Run name = the label. Running the same label again creates a NEW run (e.g. after a resume covering
more categories) — so the history stays traceable instead of being overwritten.
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--experiment", default="bfcl_eval")
    a = ap.parse_args()

    manifest_path = a.run / "run_manifest.json"
    if not manifest_path.is_file():
        print(f"no run_manifest.json under {a.run} — MLflow skipped", file=sys.stderr)
        return 0
    m = json.loads(manifest_path.read_text())

    metrics, correct, total = {}, 0, 0
    # Recursive on purpose: the memory categories nest THREE levels deep
    # (score/<model>/agentic/memory/<backend>/...) — a two-level glob drops them silently.
    for f in sorted(a.run.glob("score/**/BFCL_v4_*_score.json")):
        cat = f.name[len("BFCL_v4_"):-len("_score.json")]
        with f.open() as fh:
            head = json.loads(fh.readline())
        if head.get("accuracy") is None:
            continue
        metrics[f"acc_{cat}"] = float(head["accuracy"])
        correct += int(head.get("correct_count") or 0)
        total += int(head.get("total_count") or 0)
    if total:
        metrics["acc_overall"] = correct / total
        metrics["items_scored"] = total

    if not metrics:
        print("no scores found — MLflow skipped", file=sys.stderr)
        return 0

    import mlflow
    mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name=m.get("label")):
        mlflow.log_params({
            "model": m.get("model"),
            "model_key": m.get("model_key"),
            "model_alias": m.get("model_alias"),
            "model_fingerprint": m.get("model_fingerprint"),
            "checkpoint": m.get("model_dir_resolved"),
            "temperature": (m.get("sampling") or {}).get("temperature"),
            "generation_config": json.dumps(m.get("generation_config")),
            "max_model_len": (m.get("serving") or {}).get("max_model_len"),
            # fast/slow seit dem Concurrency-Split; Fallback auf den Alt-Key fuer alte Manifeste
            "num_threads_fast": (m.get("serving") or {}).get("num_threads_fast",
                                                             (m.get("serving") or {}).get("num_threads")),
            "num_threads_slow": (m.get("serving") or {}).get("num_threads_slow",
                                                             (m.get("serving") or {}).get("num_threads")),
            "bfcl_version": m.get("bfcl_version"),
            "repo_commit": m.get("repo_commit"),
            "categories": ",".join(m.get("categories") or []),
            "categories_skipped": ",".join(m.get("categories_skipped") or []),
            "items_planned": m.get("items_total"),
            "registry_injected": (m.get("registry") or {}).get("injected"),
            "display_name": (m.get("registry") or {}).get("display_name"),
            "run_ids_file": m.get("run_ids_file"),
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(manifest_path))
    print(f"-- MLflow: {a.experiment}/{m.get('label')} — {len(metrics)} metrics, "
          f"overall {100 * metrics.get('acc_overall', 0):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

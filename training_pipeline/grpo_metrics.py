#!/usr/bin/env python3
"""Post-hoc metrics for a Stage-2 GRPO run: ledger → informative-group share and coverage per epoch,
validation dumps → EXACT pass^k and mean per validation step, stop-rule recommendation, checkpoint
selection. Plain python3 (+ mlflow-skinny optional for --mlflow).

    python3 training_pipeline/grpo_metrics.py r1                 # table
    python3 training_pipeline/grpo_metrics.py r1 --mlflow        # + log to the verl run (exp db_bahn_grpo)
    python3 training_pipeline/grpo_metrics.py r1 --select        # write selection.json + `selected` symlink

Why exact pass^k here and not from verl: verl's `worst@N` is a bootstrap estimate (metric_utils.py),
i.e. a smoothed pass^k; binary rewards allow the exact value = mean over tasks of min(score over n).

Data sources (all written by the run itself):
  <run_dir>/ledger/*.jsonl            one line per scored train rollout {task_id, score, ts, pid}
  <run_dir>/sampler/epoch_XXX.json    plan + weights + diag per epoch (CorridorDataset)
  <run_dir>/val_dump/<step>.jsonl     verl validation generations: input, output, gts (= task id), score, step
  <ckpt_dir>/global_step_N/           verl checkpoints (LoRA adapter inside)
Groups in the ledger are reconstructed per task in arrival order in chunks of the profile's `n`
(a task drawn twice in one step yields two groups) — good enough for the informative share.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_pipeline.grpo_config import load_profile, run_paths  # noqa: E402
from training_pipeline.grpo_sampling import informative_share  # noqa: E402

HOST_PREFIX = "/app/"


def host(p: str) -> Path:
    """container path -> host path (repo root == /app)."""
    repo = Path(__file__).resolve().parents[1]
    return repo / p[len(HOST_PREFIX):] if p.startswith(HOST_PREFIX) else Path(p)


def read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def ledger_epochs(run_dir: Path, n: int) -> list[dict]:
    """Per sampler epoch: informative-group share, draws, coverage. Uses the epoch plans' timestamps
    to slice the ledger (entries between plan[e].ts and plan[e+1].ts belong to epoch e)."""
    plans = sorted(run_dir.glob("sampler/epoch_*.json"))
    entries = sorted((e for f in glob.glob(str(run_dir / "ledger/*.jsonl")) for e in read_jsonl(Path(f))),
                     key=lambda e: e.get("ts", 0))
    if not plans:
        return []
    bounds = [json.loads(p.read_text()) for p in plans]
    out = []
    for i, pl in enumerate(bounds):
        t0 = pl["ts"]
        t1 = bounds[i + 1]["ts"] if i + 1 < len(bounds) else float("inf")
        ep = [e for e in entries if t0 <= e.get("ts", 0) < t1]
        by_task: dict[str, list[float]] = defaultdict(list)
        for e in ep:
            by_task[e["task_id"]].append(float(e["score"]))
        groups = [scores[j:j + n] for scores in by_task.values() for j in range(0, len(scores), n)]
        planned = set(pl["plan_task_ids"])
        out.append({
            "epoch": pl["epoch"], "mode": pl.get("mode"), "rollouts": len(ep), "groups": len(groups),
            "informative_share": round(informative_share(groups), 3) if groups else None,
            "mean_reward": round(sum(e["score"] for e in ep) / len(ep), 3) if ep else None,
            "planned_tasks": len(planned), "sibling_draw_share": pl.get("share_sibling_draws"),
            "n_forced": pl["diag"].get("n_forced"), "weight_total": pl["diag"].get("weight_total"),
        })
    return out


def coverage(run_dir: Path) -> dict:
    plans = [json.loads(p.read_text()) for p in sorted(run_dir.glob("sampler/epoch_*.json"))]
    if not plans:
        return {}
    all_ids = set(plans[0]["weights"].keys())
    last_seen: dict[str, int] = {}
    for pl in plans:
        for tid in set(pl["plan_task_ids"]):
            last_seen[tid] = pl["epoch"]
    e = plans[-1]["epoch"]
    never = [t for t in all_ids if t not in last_seen]
    waits = [e - last_seen[t] for t in last_seen]
    return {"epochs": len(plans), "tasks": len(all_ids), "never_drawn": len(never),
            "max_wait_epochs": max(waits) if waits else None}


def validation_metrics(val_dump: Path, n: int) -> list[dict]:
    """Exact pass^n and mean per validation step from verl's validation dumps."""
    out = []
    for f in sorted(val_dump.glob("*.jsonl"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1):
        rows = read_jsonl(f)
        if not rows:
            continue
        by_task: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            tid = r.get("gts") or r.get("ground_truth") or r.get("task_id")
            sc = r.get("score")
            if isinstance(sc, dict):
                sc = sc.get("score")
            if tid is not None and sc is not None:
                by_task[str(tid)].append(float(sc))
        complete = {t: v for t, v in by_task.items() if len(v) >= n}
        if not complete:
            continue
        pass_k = sum(1 for v in complete.values() if min(v[:n]) >= 1) / len(complete)
        mean = sum(sum(v[:n]) for v in complete.values()) / (n * len(complete))
        out.append({"step": int(f.stem) if f.stem.isdigit() else f.stem, "tasks": len(complete),
                    "pass_k": round(pass_k, 4), "mean": round(mean, 4), "rollouts": len(rows)})
    return out


def stop_recommendation(ep_rows: list[dict], vals: list[dict], min_share: float = 0.20, patience: int = 2) -> str:
    reasons = []
    if ep_rows and ep_rows[-1]["informative_share"] is not None and ep_rows[-1]["informative_share"] < min_share:
        reasons.append(f"informative share {ep_rows[-1]['informative_share']} < {min_share} in last epoch")
    if len(vals) >= patience + 1:
        best = max(v["pass_k"] for v in vals)
        recent = [v["pass_k"] for v in vals[-patience:]]
        if all(x < best for x in recent) and vals[-patience - 1]["pass_k"] >= best:
            reasons.append(f"heldout pass^k not improved for {patience} validations (best {best})")
    return "STOP: " + "; ".join(reasons) if reasons else "continue"


def select(vals: list[dict], ckpt_dir: Path, adapter_root: Path, run: str, dry: bool) -> dict:
    """Best exact pass^k among validations that have a checkpoint (step > 0); tie → higher mean → earlier
    step. Step 0 (val_before_train) is the baseline: if nothing beats it, no selection."""
    baseline = next((v for v in vals if v["step"] == 0), None)
    cands = [v for v in vals if isinstance(v["step"], int) and v["step"] > 0
             and (ckpt_dir / f"global_step_{v['step']}").exists()]
    decision = {"run": run, "baseline": baseline, "candidates": cands, "selected": None, "reason": None}
    if not cands:
        decision["reason"] = "no validated checkpoint"
        return decision
    best = sorted(cands, key=lambda v: (-v["pass_k"], -v["mean"], v["step"]))[0]
    if baseline and (best["pass_k"], best["mean"]) <= (baseline["pass_k"], baseline["mean"]):
        decision["reason"] = f"no checkpoint beats the step-0 baseline (pass^k {baseline['pass_k']}, mean {baseline['mean']}) — RL brought no gain, ep3 stays"
        return decision
    decision["selected"] = best
    decision["reason"] = "best exact heldout pass^k (tie: mean, then earlier step)"
    if not dry:
        adapter_root.mkdir(parents=True, exist_ok=True)
        link = adapter_root / "selected"
        target = ckpt_dir / f"global_step_{best['step']}"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    return decision


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--run", default=None)
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--dry", action="store_true", help="with --select: decide but write nothing")
    a = ap.parse_args()
    r = load_profile(a.profile, run=a.run)
    rp = run_paths(r)
    run_dir, val_dump, ckpt_dir = host(rp["run_dir"]), host(rp["val_dump"]), host(rp["ckpt_dir"])
    n_train, n_val = r["train"]["n"], r["val"]["n"]

    ep_rows = ledger_epochs(run_dir, n_train)
    cov = coverage(run_dir)
    vals = validation_metrics(val_dump, n_val) if val_dump.exists() else []

    print(f"\nrun {r['run']} ({r['profile']}) — {run_dir}")
    print(f"{'epoch':>5s} {'mode':>8s} {'rollouts':>8s} {'groups':>6s} {'informative':>11s} {'reward':>7s} {'forced':>6s} {'siblings':>8s}")
    for e in ep_rows:
        print(f"{e['epoch']:5d} {str(e['mode']):>8s} {e['rollouts']:8d} {e['groups']:6d} "
              f"{str(e['informative_share']):>11s} {str(e['mean_reward']):>7s} {str(e['n_forced']):>6s} {str(e['sibling_draw_share']):>8s}")
    if cov:
        print(f"coverage: {cov}")
    print(f"\n{'val step':>8s} {'tasks':>5s} {f'pass^{n_val}':>8s} {'mean':>7s} {'rollouts':>8s}")
    for v in vals:
        print(f"{str(v['step']):>8s} {v['tasks']:5d} {v['pass_k']:8.4f} {v['mean']:7.4f} {v['rollouts']:8d}")
    print("\nrecommendation:", stop_recommendation(ep_rows, vals))

    if a.select:
        adapter_root = host(f"/app/data/final/checkpoints/db_bahn_grpo_lora_qwen3-4b/{r['run']}")
        dec = select(vals, ckpt_dir, adapter_root, r["run"], a.dry)
        print("\nselection:", json.dumps({k: dec[k] for k in ("selected", "reason", "baseline")}, indent=1))
        if not a.dry:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "selection.json").write_text(json.dumps(dec, indent=1))
            print(f"wrote {run_dir / 'selection.json'}"
                  + (f" and symlink {adapter_root / 'selected'} -> global_step_{dec['selected']['step']}" if dec["selected"] else ""))

    if a.mlflow:
        try:
            import mlflow
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            mlflow.set_tracking_uri((Path(__file__).resolve().parents[1] / "mlruns").as_uri())
            exp = mlflow.get_experiment_by_name(r["mlflow"]["project"])
            runs = mlflow.search_runs([exp.experiment_id], filter_string=f"tags.mlflow.runName = '{r['run']}'") if exp else None
            if runs is None or runs.empty:
                print("[mlflow] no run named", r["run"])
                return
            run_id = runs.iloc[0]["run_id"]
            with mlflow.start_run(run_id=run_id):
                for e in ep_rows:
                    if e["informative_share"] is not None:
                        mlflow.log_metric("corridor/informative_share", e["informative_share"], step=e["epoch"])
                        mlflow.log_metric("corridor/sibling_draw_share", e["sibling_draw_share"] or 0, step=e["epoch"])
                for v in vals:
                    if isinstance(v["step"], int):
                        mlflow.log_metric(f"val/pass{n_val}_exact", v["pass_k"], step=v["step"])
                        mlflow.log_metric(f"val/mean{n_val}_exact", v["mean"], step=v["step"])
                if a.select and dec.get("selected"):
                    mlflow.set_tag("selected_step", str(dec["selected"]["step"]))
            print(f"[mlflow] logged to run {run_id}")
        except Exception as e:  # tracking must never fail the analysis
            print(f"[mlflow] failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

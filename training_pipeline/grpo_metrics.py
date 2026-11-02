#!/usr/bin/env python3
"""Post-hoc metrics for a Stage-2 GRPO run: ledger → informative-group share and coverage per epoch,
validation dumps → EXACT pass^k and mean per validation step, divergence alarm, failures by template
class, reward error flags, stop-rule recommendation, checkpoint selection. Plain python3 + numpy (+
mlflow-skinny optional for --mlflow).

    python3 training_pipeline/grpo_metrics.py <run>              # table
    python3 training_pipeline/grpo_metrics.py <run> --mlflow     # + log to the verl run (run it inside the grpo container)
    python3 training_pipeline/grpo_metrics.py <run> --select     # write selection.json + `selected` symlink
    python3 training_pipeline/grpo_metrics.py <run> --pick-step N --note "why" [--by NAME]   # manual selection
    python3 training_pipeline/grpo_metrics.py <run> --divergence-json out.json               # alarm state for scripts

Divergence (config stop.entropy_factor / clip_ratio_max / consecutive): actor/entropy above factor x its
epoch-0 mean (checked from epoch 1 on) or response_length/clip_ratio above the cap (from step 1 on), in
`consecutive` consecutive steps. It is an ALARM, not a stop — the run continues, and --select excludes the
checkpoints from the first of those steps on. Per-step values come from the MLflow file store (read as
text), else from the console log.

Selection is protected: once selection.json carries a manual `override` (--pick-step), --select refuses
to overwrite it (and the `selected` symlink) without --force.

The run's knobs (n, validation n, stop rule, paths) come from runs/<run>/resolved_config.json, written by
ops/grpo_run.sh at start — never from the current YAML defaults.

Why exact pass^k here and not from verl: verl's `worst@N` is a bootstrap estimate (metric_utils.py),
i.e. a smoothed pass^k; binary rewards allow the exact value = mean over tasks of min(score over n).

Data sources (all written by the run itself):
  <run_dir>/ledger/*.jsonl            one line per scored train rollout {task_id, score, ts, pid}
  <run_dir>/sampler/epoch_XXX.json    plan + weights + diag per epoch (CorridorDataset)
  <run_dir>/val_dump/<step>.jsonl     verl validation generations: input, output, gts (= task id), score, step
  <ckpt_dir>/global_step_N/           verl checkpoints (LoRA adapter inside)
  <run_dir>/rollout_dump/<step>.jsonl verl training generations (gts, score, reward extras) → classes, error flags
  mlruns/<exp>/<run_id>/metrics/<key>  per-step metrics '<ts_ms> <value> <step>' → divergence, lr check
Groups in the ledger are reconstructed per task in arrival order in chunks of the profile's `n`
(a task drawn twice in one step yields two groups) — good enough for the informative share.

Template classes (a priori, from the pool manifest = the init model's corridor measurement): a template
is `echt` when more of its tasks failed on a real verifier component than only on the wording
(`communicate`), `wortlaut` in the opposite case, `gemischt` on a tie, `keiner` when none failed and
`nicht_im_pool` when the pool does not hold it. The training side uses the per-task signal.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_pipeline.grpo_config import load_profile, load_run, lr_factor, run_paths, steps_per_epoch  # noqa: E402
from training_pipeline.grpo_sampling import informative_share  # noqa: E402

HOST_PREFIX = "/app/"
REPO = Path(__file__).resolve().parents[1]
CLASSES = ("wortlaut", "echt", "gemischt", "keiner", "nicht_im_pool")
STEP_KEYS = ("actor/entropy", "response_length/clip_ratio", "actor/lr", "actor/kl_loss",
             "actor/conf_downweighted_frac")


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


# --- per-step training metrics ----------------------------------------------------------------------
def _experiment_dir(mlruns: Path, experiment: str) -> Path | None:
    for meta in sorted(mlruns.glob("*/meta.yaml")):
        for line in meta.read_text(errors="replace").splitlines():
            if line.startswith("name:") and line.split(":", 1)[1].strip().strip("'\"") == experiment:
                return meta.parent
    return None


def mlflow_step_metrics(run: str, experiment: str, keys=STEP_KEYS, mlruns: Path | None = None) -> dict[str, dict[int, float]]:
    """Per-step metrics of `run` straight from the MLflow FILE store (no mlflow import): every ACTIVE MLflow
    run named `run` in the experiment (a resume may open a second one) — metrics/<key> holds lines
    '<ts_ms> <value> <step>'; per step the latest timestamp wins."""
    exp = _experiment_dir(mlruns or REPO / "mlruns", experiment)
    out: dict[str, dict[int, float]] = {k: {} for k in keys}
    if exp is None:
        return out
    stamp: dict[tuple[str, int], int] = {}
    for rd in sorted(p for p in exp.iterdir() if p.is_dir()):
        name_f, meta_f = rd / "tags" / "mlflow.runName", rd / "meta.yaml"
        if not name_f.exists() or name_f.read_text().strip() != run:
            continue
        if meta_f.exists() and "lifecycle_stage: deleted" in meta_f.read_text(errors="replace"):
            continue
        for k in keys:
            f = rd / "metrics" / k
            if not f.exists():
                continue
            for line in f.read_text(errors="replace").splitlines():
                parts = line.split()
                if len(parts) < 3:
                    continue
                ts, val, step = int(parts[0]), float(parts[1]), int(parts[2])
                if ts >= stamp.get((k, step), -1):
                    stamp[(k, step)] = ts
                    out[k][step] = val
    return out


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_STEP_LINE = re.compile(r"(?:^|\s)step:(\d+) - ")


def log_step_metrics(log: Path, keys=STEP_KEYS) -> dict[str, dict[int, float]]:
    """Fallback: verl's console step lines 'step:N - key:value - …' (values may be wrapped as
    np.float64(…); ANSI colours stripped). A resumed run repeats steps — the last line wins."""
    out: dict[str, dict[int, float]] = {k: {} for k in keys}
    if not log.exists():
        return out
    pats = {k: re.compile(r"(?:^|\s)" + re.escape(k) + r":(?:np\.[a-z0-9]+\()?(-?[0-9.]+(?:e[-+]?[0-9]+)?)") for k in keys}
    with open(log, errors="replace") as f:
        for line in f:
            line = _ANSI.sub("", line)
            m = _STEP_LINE.search(line)
            if not m:
                continue
            step = int(m.group(1))
            for k, pat in pats.items():
                mm = pat.search(line)
                if mm:
                    out[k][step] = float(mm.group(1))
    return out


def _first_run(steps: list[int], ok, consecutive: int) -> tuple[int | None, int | None]:
    """First window of `consecutive` consecutive integer steps that all satisfy ok -> (onset, fired_at)."""
    streak: list[int] = []
    for s in sorted(steps):
        if ok(s) and (not streak or s == streak[-1] + 1):
            streak.append(s)
        elif ok(s):
            streak = [s]
        else:
            streak = []
        if len(streak) >= consecutive:
            return streak[-consecutive], s
    return None, None


def divergence(steps: dict[str, dict[int, float]], spe: int, entropy_factor: float = 2.0,
               clip_ratio_max: float = 0.05, consecutive: int = 2) -> dict:
    """Divergence alarm. Baseline = mean actor/entropy over epoch 0 (steps 1..spe; valid once all of
    them are logged); the entropy rule watches the steps after epoch 0, the clip rule every step. The
    earlier rule wins; `onset` = first step of the firing window (checkpoints >= onset are excluded)."""
    ent = steps.get("actor/entropy", {})
    clip = steps.get("response_length/clip_ratio", {})
    out = {"baseline": None, "threshold": None, "entropy_rule": None, "clip_rule": None, "fired": False,
           "rule": None, "onset": None, "fired_at": None, "low_entropy_steps": [],
           "status": "no_baseline", "steps_seen": max([*ent, *clip], default=0)}
    epoch0 = range(1, spe + 1)
    rules = []
    if spe > 0 and all(s in ent for s in epoch0):
        base = sum(ent[s] for s in epoch0) / spe
        out["baseline"], out["threshold"] = round(base, 4), round(entropy_factor * base, 4)
        later = [s for s in ent if s > spe]
        onset, at = _first_run(later, lambda s: ent[s] > entropy_factor * base, consecutive)
        out["entropy_rule"] = {"onset": onset, "fired_at": at}
        if at is not None:
            rules.append(("entropy", onset, at))
        out["low_entropy_steps"] = sorted(s for s in later if ent[s] < 0.5 * base)
        out["status"] = "ok"
    onset, at = _first_run(list(clip), lambda s: clip[s] > clip_ratio_max, consecutive)
    out["clip_rule"] = {"onset": onset, "fired_at": at}
    if at is not None:
        rules.append(("clip_ratio", onset, at))
    if rules:
        rule, onset, at = min(rules, key=lambda x: (x[2], x[1]))
        out.update(fired=True, rule=rule, onset=onset, fired_at=at, status="fired")
    return out


def lr_check(steps: dict[str, dict[int, float]], train: dict) -> dict | None:
    """Logged actor/lr (read after the scheduler stepped: lr x lr_factor(k + 1)) vs the closed form."""
    lrs = steps.get("actor/lr", {})
    if not lrs:
        return None
    sched = train.get("lr_schedule") or "constant"
    expected = {k: train["lr"] * lr_factor(k + 1, int(train.get("warmup_steps", -1)), train.get("total_training_steps"),
                                           float(train.get("min_lr_ratio", 0.0)), sched) for k in lrs}
    dev = {k: lrs[k] / e - 1 for k, e in expected.items() if e > 0}
    worst = max(dev, key=lambda k: abs(dev[k]), default=None)
    return {"steps": len(lrs), "schedule": sched, "max_rel_dev": round(abs(dev[worst]), 6) if worst else 0.0,
            "worst_step": worst, "last": (max(lrs), lrs[max(lrs)])}


# --- template classes ---------------------------------------------------------------------------
def pool_manifest(r: dict) -> dict:
    return json.loads((host(r["paths"]["pool_dir"]) / "pool_manifest.json").read_text())


def template_class(manifest: dict, template: str) -> str:
    t = manifest.get("templates", {}).get(template)
    if t is None:
        return "nicht_im_pool"
    e, w = t.get("signal", {}).get("echt", 0), t.get("signal", {}).get("wortlaut", 0)
    return "echt" if e > w else "wortlaut" if w > e else ("keiner" if e == 0 else "gemischt")


def task_signals(r: dict, manifest: dict) -> dict[str, str]:
    """task id -> per-task signal (wortlaut / echt / keiner) of the pool's train rows. The parquet needs
    pyarrow; without it the same signal is re-derived from the manifest's corridor traces."""
    try:
        import pandas as pd
        df = pd.read_parquet(host(r["paths"]["pool_dir"]) / "train.parquet")
        return {rm["ground_truth"]: ei["signal"] for rm, ei in zip(df["reward_model"], df["extra_info"])}
    except ImportError:
        from training_pipeline.build_grpo_pool import signal_by_task
        tmpl = set(manifest.get("templates", {}))
        templates = {tid: k["template"] for tid, k in answer_keys().items()}
        return {tid: s for tid, s in signal_by_task(host(manifest["traces_file"])).items() if templates.get(tid) in tmpl}


def answer_keys() -> dict:
    return json.loads((REPO / "data/raw/db_sandbox/answer_keys.json").read_text())


def _dump_files(d: Path) -> list[Path]:
    return sorted((f for f in d.glob("*.jsonl") if f.stem.isdigit()), key=lambda f: int(f.stem))


def train_classes(rollout_dump: Path, signals: dict[str, str], spe: int) -> list[dict]:
    """Per epoch and per-task signal: groups / informative groups / failed rollouts / rollouts."""
    by_epoch: dict[int, dict] = {}
    for f in _dump_files(rollout_dump):
        ep = (int(f.stem) - 1) // spe
        groups: dict[str, list[float]] = defaultdict(list)
        for row in read_jsonl(f):
            groups[str(row.get("gts"))].append(float(row["score"]))
        c = by_epoch.setdefault(ep, {"epoch": ep, "steps": set(), **{k: {"groups": 0, "informative": 0, "fail": 0,
                                                                            "rollouts": 0} for k in ("wortlaut", "echt", "keiner", "?")}})
        c["steps"].add(int(f.stem))
        for tid, sc in groups.items():
            k = c[signals.get(tid, "?")]
            k["groups"] += 1
            k["rollouts"] += len(sc)
            k["fail"] += sum(x < 1 for x in sc)
            k["informative"] += 0 < sum(sc) / len(sc) < 1
    return [{**c, "steps": sorted(c["steps"])} for _, c in sorted(by_epoch.items())]


def val_classes(val_dump: Path, manifest: dict, n: int) -> list[dict]:
    """Per validation step and template class: failed rollouts / rollouts / tasks failed at pass^n.
    Uses verl's reward (score < 1), not eval_report.accepted (no truncation/degeneration flags in dumps)."""
    templates = {tid: k["template"] for tid, k in answer_keys().items()}
    out = []
    for f in _dump_files(val_dump):
        by_task: dict[str, list[float]] = defaultdict(list)
        for row in read_jsonl(f):
            by_task[str(row.get("gts"))].append(float(row["score"]))
        c = {k: {"fail": 0, "rollouts": 0, "tasks_failed": 0, "tasks": 0} for k in CLASSES}
        for tid, sc in by_task.items():
            k = c[template_class(manifest, templates.get(tid, ""))]
            sc = sc[:n]
            k["fail"] += sum(x < 1 for x in sc)
            k["rollouts"] += len(sc)
            k["tasks"] += 1
            k["tasks_failed"] += min(sc) < 1
        out.append({"step": int(f.stem), **c})
    return out


def error_flags(dump: Path) -> dict | None:
    """Sum of the reward error flags over all dump files; None when the run predates them."""
    tot = {"rows": 0, "adapter_error": 0.0, "verifier_error": 0.0, "files": 0}
    seen = False
    for f in _dump_files(dump):
        tot["files"] += 1
        for row in read_jsonl(f):
            tot["rows"] += 1
            if "adapter_error" in row:
                seen = True
                tot["adapter_error"] += float(row["adapter_error"])
                tot["verifier_error"] += float(row.get("verifier_error", 0.0))
    return tot if seen else None


def paired_bootstrap(diffs: list[int], n_boot: int = 10000, seed: int = 0) -> tuple[int, int, int]:
    """Sum of per-task paired differences and its 95 % interval (resampling tasks; order statistics
    at 2.5 / 97.5 % of the sorted bootstrap sums). Pinned: numpy default_rng(seed)."""
    import numpy as np
    d = np.asarray(diffs, dtype=np.int64)
    if d.size == 0:
        return 0, 0, 0
    rng = np.random.default_rng(seed)
    sums = np.sort(d[rng.integers(0, d.size, size=(n_boot, d.size))].sum(axis=1))
    return int(d.sum()), int(sums[int(0.025 * n_boot)]), int(sums[int(0.975 * n_boot)])


def stop_recommendation(ep_rows: list[dict], vals: list[dict], min_share: float = 0.20, patience: int = 3,
                        div: dict | None = None) -> str:
    """STOP when (a) the last epoch's informative-group share fell below min_share, or (b) the best
    heldout pass^k lies at least `patience` validations back (i.e. `patience` consecutive validations
    without a new best). A fired divergence rule adds an ALARM (advisory: the run continues)."""
    alarm = (f" | ALARM: divergence ({div['rule']}, steps {div['onset']}-{div['fired_at']}) — checkpoints from step "
             f"{div['onset']} on are excluded from --select" if div and div.get("fired") else "")
    reasons = []
    if ep_rows and ep_rows[-1]["informative_share"] is not None and ep_rows[-1]["informative_share"] < min_share:
        reasons.append(f"informative share {ep_rows[-1]['informative_share']} < {min_share} in last epoch")
    if len(vals) >= patience + 1:
        best_i = max(range(len(vals)), key=lambda i: (vals[i]["pass_k"], vals[i]["mean"], -i))
        since_best = len(vals) - 1 - best_i
        if since_best >= patience:
            reasons.append(f"heldout pass^k not improved for {since_best} validations "
                           f"(best {vals[best_i]['pass_k']} at step {vals[best_i]['step']})")
    return ("STOP: " + "; ".join(reasons) if reasons else "continue") + alarm


def select(vals: list[dict], ckpt_dir: Path, adapter_root: Path, run: str, dry: bool,
           exclude_from: int | None = None) -> dict:
    """Best exact pass^k among validations that have a checkpoint (step > 0, step < exclude_from = the
    divergence onset); tie → higher mean → earlier step. Step 0 (val_before_train) is the baseline: if
    nothing beats it, no selection."""
    baseline = next((v for v in vals if v["step"] == 0), None)
    with_ckpt = [v for v in vals if isinstance(v["step"], int) and v["step"] > 0
                 and (ckpt_dir / f"global_step_{v['step']}").exists()]
    cands = [v for v in with_ckpt if exclude_from is None or v["step"] < exclude_from]
    decision = {"run": run, "baseline": baseline, "candidates": cands, "selected": None, "reason": None,
                "excluded_by_divergence": [v["step"] for v in with_ckpt if v not in cands]}
    if not cands:
        decision["reason"] = "no validated checkpoint" + (f" before the divergence onset (step {exclude_from})"
                                                          if with_ckpt else "")
        return decision
    best = sorted(cands, key=lambda v: (-v["pass_k"], -v["mean"], v["step"]))[0]
    if baseline and (best["pass_k"], best["mean"]) <= (baseline["pass_k"], baseline["mean"]):
        decision["reason"] = f"no checkpoint beats the step-0 baseline (pass^k {baseline['pass_k']}, mean {baseline['mean']}) — RL brought no gain, ep3 stays"
        return decision
    decision["selected"] = best
    decision["reason"] = "best exact heldout pass^k (tie: mean, then earlier step)" + (
        f"; checkpoints from step {exclude_from} on excluded (divergence)" if exclude_from is not None else "")
    if not dry:
        link_selected(adapter_root, ckpt_dir, best["step"])
    return decision


def link_selected(adapter_root: Path, ckpt_dir: Path, step: int) -> None:
    adapter_root.mkdir(parents=True, exist_ok=True)
    link = adapter_root / "selected"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(ckpt_dir / f"global_step_{step}")


def existing_override(run_dir: Path) -> dict | None:
    f = run_dir / "selection.json"
    return json.loads(f.read_text()).get("override") if f.exists() else None


def pick_step(vals: list[dict], ckpt_dir: Path, adapter_root: Path, run_dir: Path, run: str, step: int, note: str,
              by: str, onset: int | None, force: bool, dry: bool) -> dict:
    """Manual selection (e.g. after the external K=8 eval): selection.json gets the chosen step with an
    `override` block (date, by, auto_selected_step, note; an earlier override is kept as `previous`) and
    the `selected` symlink points to it. A step at/after the divergence onset needs --force."""
    if not (ckpt_dir / f"global_step_{step}").exists():
        raise SystemExit(f"no checkpoint {ckpt_dir / f'global_step_{step}'}")
    if onset is not None and step >= onset and not force:
        raise SystemExit(f"step {step} lies at/after the divergence onset (step {onset}) — add --force to pick it anyway")
    auto = select(vals, ckpt_dir, adapter_root, run, dry=True, exclude_from=onset)
    chosen = next((v for v in vals if v["step"] == step), {"step": step})
    override = {"date": datetime.date.today().isoformat(), "by": by,
                "auto_selected_step": (auto["selected"] or {}).get("step"), "note": "set with grpo_metrics.py --pick-step"}
    prev = existing_override(run_dir)
    if prev:
        override["previous"] = prev
    dec = {**auto, "selected": chosen, "reason": note, "override": override}
    if not dry:
        link_selected(adapter_root, ckpt_dir, step)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "selection.json").write_text(json.dumps(dec, indent=1))
    return dec


def main() -> None:
    ap = argparse.ArgumentParser(description="metrics, divergence alarm, classes and checkpoint selection for a GRPO run")
    ap.add_argument("run", help="run name (runs/<run>/resolved_config.json)")
    ap.add_argument("--profile", default=None, help="fallback: resolve the knobs from this YAML profile instead")
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--dry", action="store_true", help="with --select/--pick-step: decide but write nothing")
    ap.add_argument("--pick-step", type=int, default=None, metavar="N", help="manual selection of global_step_N")
    ap.add_argument("--note", default=None, help="with --pick-step: why (lands in selection.json reason)")
    ap.add_argument("--by", default="Janneck", help="with --pick-step: who decided")
    ap.add_argument("--force", action="store_true", help="overwrite a manual override / pick a step after the divergence onset")
    ap.add_argument("--divergence-json", type=Path, default=None, help="write the divergence state to this file")
    ap.add_argument("--no-classes", action="store_true", help="skip the template-class tables")
    a = ap.parse_args()
    if a.pick_step is not None and not a.note:
        ap.error("--pick-step needs --note")
    r = load_profile(a.profile, run=a.run) if a.profile else load_run(a.run)
    rp = run_paths(r)
    run_dir, val_dump, ckpt_dir = host(rp["run_dir"]), host(rp["val_dump"]), host(rp["ckpt_dir"])
    n_train, n_val = r["train"]["n"], r["val"]["n"]
    spe = steps_per_epoch(r)
    stop_cfg = r.get("stop", {})

    ep_rows = ledger_epochs(run_dir, n_train)
    cov = coverage(run_dir)
    vals = validation_metrics(val_dump, n_val) if val_dump.exists() else []
    steps = mlflow_step_metrics(r["run"], r["mlflow"]["project"])
    src = "mlflow"
    if not steps["actor/entropy"]:
        steps, src = log_step_metrics(REPO / "logs" / f"{r['run']}.log"), "console log"
    div = divergence(steps, spe, float(stop_cfg.get("entropy_factor", 2.0)), float(stop_cfg.get("clip_ratio_max", 0.05)),
                     int(stop_cfg.get("consecutive", 2)))
    onset = div["onset"] if div["fired"] else None

    print(f"\nrun {r['run']} (profile {r.get('profile')}) — {run_dir}")
    print(f"{'epoch':>5s} {'mode':>8s} {'rollouts':>8s} {'groups':>6s} {'informative':>11s} {'reward':>7s} {'forced':>6s} {'siblings':>8s}")
    for e in ep_rows:
        print(f"{e['epoch']:5d} {str(e['mode']):>8s} {e['rollouts']:8d} {e['groups']:6d} "
              f"{str(e['informative_share']):>11s} {str(e['mean_reward']):>7s} {str(e['n_forced']):>6s} {str(e['sibling_draw_share']):>8s}")
    if cov:
        print(f"coverage: {cov}")
    print(f"\n{'val step':>8s} {'tasks':>5s} {f'pass^{n_val}':>8s} {'mean':>7s} {'rollouts':>8s}")
    for v in vals:
        print(f"{str(v['step']):>8s} {v['tasks']:5d} {v['pass_k']:8.4f} {v['mean']:7.4f} {v['rollouts']:8d}")

    print(f"\ndivergence ({src}, {div['steps_seen']} steps): status {div['status']}"
          + (f", entropy baseline {div['baseline']} -> threshold {div['threshold']}" if div["baseline"] is not None else
             f" (entropy rule needs all {spe} epoch-0 steps)")
          + (f", FIRED by {div['rule']} at step {div['fired_at']} (onset {div['onset']})" if div["fired"] else ""))
    if div["clip_rule"] and div["clip_rule"]["fired_at"] and div["rule"] != "clip_ratio":
        print(f"  clip-ratio rule also fired at step {div['clip_rule']['fired_at']} (onset {div['clip_rule']['onset']})")
    if div["low_entropy_steps"]:
        print(f"  INFO entropy below half the baseline (collapse side) at steps {div['low_entropy_steps']}")
    lrc = lr_check(steps, r["train"])
    if lrc:
        print(f"lr check: {lrc['steps']} logged steps, {lrc['schedule']}, max rel. deviation from the closed form "
              f"{lrc['max_rel_dev']} (step {lrc['worst_step']}); last step {lrc['last'][0]} lr {lrc['last'][1]:.3e}")
    for name, d in (("train", host(rp["rollout_dump"])), ("val", val_dump)):
        ef = error_flags(d) if d.exists() else None
        print(f"reward error flags ({name} dumps): " + (f"adapter_error {ef['adapter_error']:.0f}, verifier_error "
                                                        f"{ef['verifier_error']:.0f} of {ef['rows']} rollouts in {ef['files']} files"
                                                        if ef else "n/a (no dumps or run predates the flags)"))

    if not a.no_classes:
        try:
            man = pool_manifest(r)
            tc = train_classes(host(rp["rollout_dump"]), task_signals(r, man), spe)
            if tc:
                print("\ntrain rollouts by per-task signal — failed/rollouts (informative/groups)")
                for c in tc:
                    print(f"  epoch {c['epoch']} steps {c['steps'][0]}-{c['steps'][-1]}: " + " | ".join(
                        f"{k} {c[k]['fail']}/{c[k]['rollouts']} ({c[k]['informative']}/{c[k]['groups']})"
                        for k in ("wortlaut", "echt", "keiner", "?") if c[k]["groups"]))
            vc = val_classes(val_dump, man, n_val) if val_dump.exists() else []
            if vc:
                print(f"\nvalidation by template class — failed rollouts/rollouts (tasks failed at pass^{n_val}/tasks)")
                for c in vc:
                    print(f"  step {c['step']:>3d}: " + " | ".join(
                        f"{k} {c[k]['fail']}/{c[k]['rollouts']} ({c[k]['tasks_failed']}/{c[k]['tasks']})"
                        for k in CLASSES if c[k]["tasks"]))
        except Exception as e:   # the class view must never take the core report down
            print(f"\n[classes] skipped: {type(e).__name__}: {e}")

    print("\nrecommendation:", stop_recommendation(ep_rows, vals, float(stop_cfg.get("min_informative_share", 0.20)),
                                                   int(stop_cfg.get("patience_validations", 3)), div))
    if a.divergence_json:
        a.divergence_json.write_text(json.dumps(div, indent=1))

    adapter_root = host(f"/app/data/final/checkpoints/db_bahn_grpo_lora_qwen3-4b/{r['run']}")
    dec = None
    if a.pick_step is not None:
        dec = pick_step(vals, ckpt_dir, adapter_root, run_dir, r["run"], a.pick_step, a.note, a.by, onset, a.force, a.dry)
        print("\nmanual selection:", json.dumps({k: dec[k] for k in ("selected", "reason", "override")}, indent=1))
        if not a.dry:
            print(f"wrote {run_dir / 'selection.json'} and symlink {adapter_root / 'selected'} -> global_step_{a.pick_step}")
    elif a.select:
        prev = existing_override(run_dir)
        if prev and not a.dry and not a.force:
            raise SystemExit(f"{run_dir / 'selection.json'} carries a manual override (step chosen {prev.get('date')} by "
                             f"{prev.get('by')}) — --select would overwrite it and the `selected` symlink. "
                             f"Use --dry to compare, --force to overwrite.")
        dec = select(vals, ckpt_dir, adapter_root, r["run"], a.dry, exclude_from=onset)
        print("\nselection:", json.dumps({k: dec[k] for k in ("selected", "reason", "baseline", "excluded_by_divergence")}, indent=1))
        if prev and a.dry:
            manual = (json.loads((run_dir / "selection.json").read_text()).get("selected") or {}).get("step")
            print(f"(dry run — selection.json keeps its manual override: step {manual}, set {prev.get('date')} by {prev.get('by')})")
        if not a.dry:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "selection.json").write_text(json.dumps({**dec, "divergence": div}, indent=1))
            print(f"wrote {run_dir / 'selection.json'}"
                  + (f" and symlink {adapter_root / 'selected'} -> global_step_{dec['selected']['step']}" if dec["selected"] else ""))

    if a.mlflow:
        try:
            import mlflow
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            mlflow.set_tracking_uri((REPO / "mlruns").as_uri())
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
                if dec and dec.get("selected") and not a.dry:
                    mlflow.set_tag("selected_step", str(dec["selected"]["step"]))
                if div["fired"]:
                    mlflow.set_tag("divergence_onset", str(div["onset"]))
            print(f"[mlflow] logged to run {run_id}")
        except Exception as e:  # tracking must never fail the analysis
            print(f"[mlflow] failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired comparison of two K-rollout heldout trace files by template class.

    python3 evaluation/k8_class_compare.py \
        --base data/generated/eval/db_traces_heldout_eval_after_grpo_db_bahn_r2_step24_k8.jsonl \
        --cand data/generated/eval/db_traces_heldout_eval_after_grpo_db_bahn_r3_stepN_k8.jsonl

Per task: failed rollouts = K minus accepted ones (evaluation/eval_report.accepted — the heldout eval's
own acceptance incl. truncation/degeneration). Per class: sum of the per-task differences cand − base and
a paired bootstrap interval over tasks. The classes are the a-priori template classes of the GRPO pool
(training_pipeline/grpo_metrics.template_class: echt / wortlaut / gemischt / keiner / nicht_im_pool) —
the view that showed r2 trading real errors for wording. Both files must hold the same tasks with the
same K. Plain python3 + numpy (+ PyYAML for the default pool dir).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.eval_report import accepted  # noqa: E402
from training_pipeline.grpo_metrics import CLASSES, host, paired_bootstrap, template_class  # noqa: E402


def fails_per_task(path: Path) -> tuple[dict[str, int], dict[str, int], dict[str, str]]:
    """-> (failed rollouts per task, rollouts per task, template per task)."""
    fails: dict[str, int] = defaultdict(int)
    n: dict[str, int] = defaultdict(int)
    tmpl: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n[r["task_id"]] += 1
            fails[r["task_id"]] += 0 if accepted(r) else 1
            tmpl[r["task_id"]] = r["template"]
    return dict(fails), dict(n), tmpl


def compare(base: Path, cand: Path, manifest: dict, n_boot: int = 10000, seed: int = 0) -> dict:
    fb, nb, tb = fails_per_task(base)
    fc, nc, _ = fails_per_task(cand)
    if set(nb) != set(nc):
        raise SystemExit(f"task sets differ: {len(set(nb) ^ set(nc))} tasks in only one file")
    ks = set(nb.values()) | set(nc.values())
    if len(ks) != 1:
        raise SystemExit(f"rollouts per task differ: K values {sorted(ks)}")
    by: dict[str, list[str]] = defaultdict(list)
    for t in sorted(nb):
        by[template_class(manifest, tb[t])].append(t)
    out = {"base": str(base), "cand": str(cand), "k": ks.pop(), "tasks": len(nb), "classes": {}}
    for k in (*CLASSES, "gesamt"):
        ts = sorted(nb) if k == "gesamt" else by.get(k, [])
        if not ts:
            continue
        d, lo, hi = paired_bootstrap([fc[t] - fb[t] for t in ts], n_boot, seed)
        out["classes"][k] = {"tasks": len(ts), "templates": len({tb[t] for t in ts}),
                             "fail_base": sum(fb[t] for t in ts), "fail_cand": sum(fc[t] for t in ts),
                             "diff": d, "ci95": [lo, hi]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="paired K-rollout comparison by template class")
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--pool-dir", type=Path, default=None, help="GRPO pool (pool_manifest.json); default: config paths.pool_dir")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path, default=None, help="also write the result as JSON")
    a = ap.parse_args()
    pool = a.pool_dir
    if pool is None:
        import yaml
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/grpo_db_bahn.yaml").read_text())
        pool = host(cfg["paths"]["pool_dir"])
    manifest = json.loads((pool / "pool_manifest.json").read_text())
    res = compare(a.base, a.cand, manifest, a.boot, a.seed)
    print(f"base {a.base.name}\ncand {a.cand.name}\n{res['tasks']} tasks x K={res['k']}; failed rollouts per class, "
          f"diff = cand - base, paired bootstrap 95 % over tasks ({a.boot}, seed {a.seed})")
    print(f"  {'class':14s} {'tasks':>5s} {'tmpl':>4s} {'base':>5s} {'cand':>5s} {'diff':>5s}  ci95")
    for k, c in res["classes"].items():
        print(f"  {k:14s} {c['tasks']:5d} {c['templates']:4d} {c['fail_base']:5d} {c['fail_cand']:5d} "
              f"{c['diff']:+5d}  [{c['ci95'][0]:+d}, {c['ci95'][1]:+d}]")
    if a.json:
        a.json.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

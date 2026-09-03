#!/usr/bin/env python3
"""Build the Stage-2 GRPO task pool (verl parquet) from the corridor measurement.

Usage (tau2 venv, repo root):
    PYTHONPATH=. .venv-tau2/bin/python training_pipeline/build_grpo_pool.py \
        --corridor data/generated/eval/rl_corridor_ep3_corridor_k8.json \
        --traces   data/generated/eval/db_traces_rl_train_ep3_corridor_k8.jsonl \
        --pool-id  corridor_ep3_k8_20260902

Pool design (2026-09-03, docs/stage2-RL.md):
  * train.parquet — ALL 20 tasks of every template that has ≥1 corridor task (1–7 of 8 successes
    in the measurement). The corridor tasks are the core; their 8/8 "siblings" hedge the ±1/8 noise of
    an 8-rollout measurement and add ticket variety. Templates with 160/160 stay out. Each row carries
    `rate0` (own rate), `template_rate0` (template mean), `ever_failed`, `signal` (echt / wortlaut /
    keiner — which verifier component failed) for training_pipeline/grpo_dataset.py, which turns them
    into draw weights (see grpo_sampling.py). Compute per step is fixed (128 draws/epoch), so the pool
    size costs nothing; it only spreads the draws.
  * val.parquet — the full heldout_eval split (480 tasks): verl validates on it every epoch with n=4
    → exact pass^4 from the validation dumps (training_pipeline/grpo_metrics.py).
  * smoke_val.parquet — 24 heldout tasks (12 that ep3 failed at pass^4, 12 it passed) for the GPU smoke.
  * pool_manifest.json — provenance (corridor sha, trace sha) and counts per template.

verl schema (verified against the frozen image): the reward manager reads reward_model.ground_truth
and hands extra_info through untouched; RLDataset promotes extra_info.tools_kwargs into per-sample
tool create_kwargs. Nested dicts go in as JSON strings so pyarrow can't trip over heterogeneous
task schemas. The ledger path is NOT in the parquet — the dataset injects the run's ledger dir at
__getitem__ time (one pool, many runs).

System prompt: policy + the "Arbeitsweise" block of sdg_pipeline/db_bahn/rollout.py:SYSTEM_TEMPLATE,
WITHOUT the tools block — verl's ToolAgentLoop passes the schemas to the chat template, which renders
its own "# Tools" section. (Known divergence from the eval prompt; the step-0 validation is the gate.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from training_pipeline.grpo_sampling import PoolTask, compute_weights  # noqa: E402

DATA_DIR = REPO / "data/raw/db_sandbox"
POOL_ROOT = REPO / "data/final/grpo/db_bahn_pool"
HELDOUT_K4_DEFAULT = REPO / "data/generated/eval/db_traces_heldout_eval_after_qwen3-4b_ep3_k4.jsonl"
SEED = 42


def sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def arbeitsweise_block() -> str:
    """The eval prompt's working-rules block, without the tools block (the chat template adds one)."""
    from sdg_pipeline.db_bahn.rollout import SYSTEM_TEMPLATE
    tail = SYSTEM_TEMPLATE.rsplit("</tools>", 1)[1]   # LAST occurrence: the sentence "within <tools></tools> XML tags" precedes the block
    return tail.format()  # SYSTEM_TEMPLATE is a format string ({{ }} → { })


def signal_by_task(traces: Path) -> dict[str, str]:
    """echt = at least one failed rollout with a non-communicate component; wortlaut = failures only in
    `communicate`; keiner = no failed rollout."""
    failed: dict[str, set] = defaultdict(set)
    seen: set[str] = set()
    with open(traces) as f:
        for line in f:
            r = json.loads(line)
            seen.add(r["task_id"])
            sc = r["score"]
            if sc["score"] < 1:
                comps = {c for c, v in sc["components"].items() if not v} or {"other"}
                failed[r["task_id"]] |= comps
    out = {}
    for tid in seen:
        comps = failed.get(tid)
        out[tid] = "keiner" if not comps else ("wortlaut" if comps == {"communicate"} else "echt")
    return out


def _row(task: dict, key: dict, idx: int, sys_prompt: str, tool_names: list[str], extra: dict) -> dict:
    init_json = json.dumps(((task.get("initial_state") or {}).get("initialization_actions")) or [],
                           ensure_ascii=False)
    return {
        "data_source": "db_bahn",
        "prompt": [{"role": "system", "content": sys_prompt},
                   {"role": "user", "content": task.get("ticket") or ""}],
        "reward_model": {"style": "rule", "ground_truth": task["id"]},
        "extra_info": {
            "index": idx,
            "need_tools_kwargs": True,
            "task": json.dumps(task, ensure_ascii=False),
            "answer_key": json.dumps(key, ensure_ascii=False),
            # every tool gets the same create_kwargs: whichever fires first builds the episode env
            "tools_kwargs": {n: {"create_kwargs": {"initialization_actions_json": init_json}}
                             for n in tool_names},
            **extra,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="corridor JSON -> verl GRPO pool parquets")
    ap.add_argument("--corridor", type=Path, required=True, help="rl_corridor_*.json (corridor_from_traces.py)")
    ap.add_argument("--traces", type=Path, required=True, help="the k=8 rollout JSONL behind the corridor")
    ap.add_argument("--pool-id", required=True, help="folder name under data/final/grpo/db_bahn_pool/")
    ap.add_argument("--heldout-k4", type=Path, default=HELDOUT_K4_DEFAULT,
                    help="ep3 heldout k=4 traces (picks the smoke-val tasks)")
    ap.add_argument("--smoke-val-n", type=int, default=24)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or (POOL_ROOT / args.pool_id)
    tasks = {t["id"]: t for t in json.loads((DATA_DIR / "tasks.json").read_text())}
    keys = json.loads((DATA_DIR / "answer_keys.json").read_text())
    splits = json.loads((DATA_DIR / "split_tasks.json").read_text())
    corr = json.loads(args.corridor.read_text())
    ctasks: dict[str, dict] = corr["tasks"]
    k = corr["summary"]["k"]

    from sdg_pipeline.db_bahn.tau2_domain import get_environment
    env = get_environment(solo_mode=True)
    tool_names = [t.name for t in env.get_tools()]
    sys_prompt = env.get_policy() + arbeitsweise_block()

    # --- train pool: all tasks of every template with >=1 corridor task ---------------------------
    by_tmpl: dict[str, list[str]] = defaultdict(list)
    for tid, v in ctasks.items():
        by_tmpl[v["template"]].append(tid)
    selected_tmpls = sorted(t for t, ids in by_tmpl.items() if any(ctasks[i]["corridor"] for i in ids))
    tmpl_rate = {t: sum(ctasks[i]["ok"] for i in ids) / (k * len(ids)) for t, ids in by_tmpl.items()}
    signal = signal_by_task(args.traces)
    rl_set = set(splits["rl_train"])

    train_ids = [i for t in selected_tmpls for i in sorted(by_tmpl[t])]
    assert all(i in rl_set for i in train_ids), "corridor file must be measured on the rl_train split"
    pool_tasks, rows = [], []
    for n, tid in enumerate(train_ids):
        v = ctasks[tid]
        extra = {"pool_role": "train", "split": "rl_train", "pool_id": args.pool_id,
                 "template": v["template"], "rate0": v["ok"] / k, "template_rate0": tmpl_rate[v["template"]],
                 "ever_failed": bool(v["ok"] < k), "corridor": bool(v["corridor"]),
                 "signal": signal.get(tid, "keiner")}
        rows.append(_row(tasks[tid], keys[tid], n, sys_prompt, tool_names, extra))
        pool_tasks.append(PoolTask(tid, v["template"], extra["rate0"], extra["template_rate0"], extra["ever_failed"]))

    # --- val: full heldout; smoke_val: 12 ep3-non-pass^4 + 12 passed ------------------------------
    held_ids = sorted(splits["heldout_eval"])
    val_rows = [_row(tasks[tid], keys[tid], n, sys_prompt, tool_names,
                     {"pool_role": "val", "split": "heldout_eval", "pool_id": args.pool_id,
                      "template": keys[tid]["template"]}) for n, tid in enumerate(held_ids)]
    ok_by_task: dict[str, list[float]] = defaultdict(list)
    if args.heldout_k4.exists():
        with open(args.heldout_k4) as f:
            for line in f:
                r = json.loads(line)
                ok_by_task[r["task_id"]].append(r["score"]["score"])
    failed_p4 = sorted(t for t, v in ok_by_task.items() if min(v) < 1)
    passed_p4 = sorted(t for t, v in ok_by_task.items() if v and min(v) >= 1)
    rng = random.Random(SEED)
    half = args.smoke_val_n // 2
    smoke_ids = sorted(rng.sample(failed_p4, min(half, len(failed_p4)))
                       + rng.sample(passed_p4, args.smoke_val_n - min(half, len(failed_p4))))
    smoke_rows = [r for r in val_rows if r["reward_model"]["ground_truth"] in set(smoke_ids)]
    for n, r in enumerate(smoke_rows):
        r = dict(r)
        r["extra_info"] = {**r["extra_info"], "index": n, "pool_role": "smoke_val"}
        smoke_rows[n] = r

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "train.parquet")
    pd.DataFrame(val_rows).to_parquet(out / "val.parquet")
    pd.DataFrame(smoke_rows).to_parquet(out / "smoke_val.parquet")

    # --- manifest ----------------------------------------------------------------------------------
    w = compute_weights(pool_tasks)
    tot = sum(w)
    per_tmpl = {t: {"n": len(by_tmpl[t]), "corridor": sum(1 for i in by_tmpl[t] if ctasks[i]["corridor"]),
                    "template_rate0": round(tmpl_rate[t], 4),
                    "signal": dict(Counter(signal.get(i, "keiner") for i in by_tmpl[t]))} for t in selected_tmpls}
    manifest = {
        "pool_id": args.pool_id,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corridor_file": str(args.corridor), "corridor_sha256_16": sha256_16(args.corridor),
        "traces_file": str(args.traces), "traces_sha256_16": sha256_16(args.traces),
        "corridor_provenance": corr.get("provenance"),
        "k": k, "n_templates": len(selected_tmpls), "n_train": len(rows),
        "n_corridor": sum(1 for i in train_ids if ctasks[i]["corridor"]),
        "n_ever_failed": sum(1 for i in train_ids if ctasks[i]["ok"] < k),
        "n_val": len(val_rows), "n_smoke_val": len(smoke_rows), "smoke_val_ids": smoke_ids,
        "signal_counts": dict(Counter(signal.get(i, "keiner") for i in train_ids)),
        "start_weights": {"eps": 0.005, "mix": 0.5, "total": round(tot, 4),
                          "share_corridor": round(sum(x for x, i in zip(w, train_ids) if ctasks[i]["corridor"]) / tot, 4)},
        "templates": per_tmpl,
        "system_prompt_sha256_16": hashlib.sha256(sys_prompt.encode()).hexdigest()[:16],
        "tool_names": tool_names,
    }
    (out / "pool_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    print(f"pool {args.pool_id}: templates={len(selected_tmpls)} train={len(rows)} "
          f"(corridor {manifest['n_corridor']}, ever_failed {manifest['n_ever_failed']}) "
          f"val={len(val_rows)} smoke_val={len(smoke_rows)}")
    print(f"signal: {manifest['signal_counts']} | start weight share corridor "
          f"{100 * manifest['start_weights']['share_corridor']:.0f} %")
    print("out:", out)


if __name__ == "__main__":
    main()

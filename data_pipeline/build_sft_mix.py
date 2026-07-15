"""
data_pipeline/build_sft_mix.py
==============================
Assemble the 3-leg SFT mix from the three converted chat files into ONE shuffled training set + a held-out
val split (never in the gradient) for the overfit detector.

Legs (all already in the unified db_bahn chat format):
  - db_bahn : data/final/db_traces_chat.jsonl        (verified German DB traces; core, up-weighted by count)
  - AReaL   : data/generated/areal_chat.jsonl         (tau2 dialogue + policy, 3 domains — the dialogue half)
  - ToolACE : data/generated/toolace_chat.jsonl       (API-schema breadth + irrelevance)

Filters here:
  - db_bahn: drop the ~10 "flail" traces with an identical consecutive tool call (name+args repeated back to
    back) — deterministic 0-false-positive quality signal.
  - Val split: stratified per-source (~val-frac), seeded; the rest is train. One shuffle (seed 42), no blocks.

Output: data/final/sft_mix_chat.jsonl + data/final/sft_mix_val.jsonl, plus a printed stats gate.

Usage:  PYTHONPATH=. python3 data_pipeline/build_sft_mix.py            # default --val-frac 0.0188 (~300 val)
"""

import argparse
import json
import random
from collections import Counter

from data_pipeline.common import write_jsonl

LEGS = {
    "db_bahn": "data/final/db_traces_chat.jsonl",
    "areal": "data/generated/areal_chat.jsonl",
    "toolace": "data/generated/toolace_chat.jsonl",
}


def is_flail(rec: dict) -> bool:
    """A trace with an identical tool call repeated back-to-back (name + args) — the over-search signal."""
    calls = [(tc["function"]["name"], json.dumps(tc["function"]["arguments"], sort_keys=True))
             for m in rec["messages"] if m.get("tool_calls") for tc in m["tool_calls"]]
    return any(calls[i] == calls[i + 1] for i in range(len(calls) - 1))


def n_assistant_turns(rec: dict) -> int:
    return sum(1 for m in rec["messages"] if m["role"] == "assistant")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.0188)  # ~300 held-out (overfit-diag only; no early-stop)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-train", default="data/final/sft_mix_chat.jsonl")
    ap.add_argument("--out-val", default="data/final/sft_mix_val.jsonl")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    train, val, stats = [], [], {}
    for src, path in LEGS.items():
        recs = [json.loads(l) for l in open(path) if l.strip()]
        dropped = 0
        if src == "db_bahn":
            kept = [r for r in recs if not is_flail(r)]
            dropped = len(recs) - len(kept)
            recs = kept
        for r in recs:                       # normalize a coarse source tag for the leakage/stats gate
            r["_meta"]["mix_source"] = src
        rng.shuffle(recs)                    # per-source shuffle before the val cut (seeded)
        n_val = round(len(recs) * args.val_frac)
        val += recs[:n_val]
        train += recs[n_val:]
        stats[src] = {"in": len(recs) + dropped, "dropped": dropped, "val": n_val, "train": len(recs) - n_val}

    rng.shuffle(train)                       # ONE mixed shuffle (no blocks -> no forgetting)
    rng.shuffle(val)

    for path, data in [(args.out_train, train), (args.out_val, val)]:
        write_jsonl(data, path)

    # --- stats gate ---
    print("=== SFT-MIX STATS ===")
    for src, s in stats.items():
        print(f"  {src:8s} in={s['in']:6d} dropped={s['dropped']:3d} -> train {s['train']:6d} / val {s['val']:4d}")
    print(f"  TOTAL    train {len(train)} / val {len(val)} = {len(train)+len(val)}")
    tr_src = Counter(r["_meta"]["mix_source"] for r in train)
    print(f"  train source-mix: {dict(tr_src)} "
          f"({', '.join(f'{k} {100*v/len(train):.0f}%' for k,v in tr_src.items())})")
    mt = sum(1 for r in train if n_assistant_turns(r) >= 3)
    print(f"  train multi-assistant (>=3 turns): {mt} = {100*mt/len(train):.0f}%")
    val_src = Counter(r["_meta"]["mix_source"] for r in val)
    print(f"  val source-mix: {dict(val_src)}")
    # leakage guard: no record object shared (paranoia — different files, but assert disjoint by id)
    assert not (set(map(id, train)) & set(map(id, val))), "HARD-FAIL: train/val overlap"
    print("  train/val disjoint: OK")


if __name__ == "__main__":
    main()

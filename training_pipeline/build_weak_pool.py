"""
training_pipeline/build_weak_pool.py
==========================
Stage 1 of the RLVR-GRPO pilot: build the DISJOINT weak-category prompt sets.

From data/raw/seed_full.jsonl, for the weak categories {window functions, set operations}:
  1. EXECUTABLE GATE  keep a record iff its schema sets up in SQLite AND its gold `sql` executes
     against it (reuses evaluation.evaluate.execute_sql_on_schema + the build_train_clean
     schema_sets_up check). Drops CTE / PostgreSQL-DML golds that don't run in SQLite.
  2. LEAKAGE GUARD    drop any record whose normalized question appears in the SFT training data
     (the 14B-thinking adapter's data_snapshot train+eval chat) OR in test_clean.jsonl -> true OOD.
  3. DEDUP            drop duplicate normalized questions within the pool.
  4. CARVE            hold out a disjoint weak EVAL set (~HELD_OUT_PER_CAT/category) ->
     data/final/weak_test_clean.jsonl; the remainder (capped) is the RL candidate pool ->
     data/final/grpo/weak_candidates.jsonl.
  5. REPORT           per-category surviving counts at every stage. Set operations is the thin
     category — accept fewer, never force/fill.

Run inside the training container (root-owned data dirs + yaml/sqlite available):
    python3 training_pipeline/build_weak_pool.py
"""
import argparse
import json
import random
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import yaml

from evaluation.evaluate import execute_sql_on_schema  # single source of truth

WEAK = {"window functions", "set operations"}
EXTRA = {"subqueries"}          # Pre-Pilot [3]: 3rd RL category (frequent pattern, deep pool ~6.7k raw)
ALL_CATS = WEAK | EXTRA
SEED = 42
HELD_OUT_PER_CAT = 80          # disjoint held-out weak eval per category (plan: ~50-80)
CANDIDATE_CAP_PER_CAT = 600    # canonical RL candidate pool cap (kept stable, documented artifact)
# Pre-Pilot [4]: EXPANDED probe pool. Low per-category hit-rate (~23-40%) means 600 candidates
# yield <300 reachable for window/subqueries -> pull more from the deep pool. Per-cat caps; a
# huge cap = "take all available" (set operations is supply-capped, ~606 after held-out).
PROBE_CAP_PER_CAT = {"set operations": 10**9, "window functions": 1800, "subqueries": 1200}

SRC = Path("data/raw/seed_full.jsonl")
TEST = Path("data/final/test_clean.jsonl")
# Default SFT data_snapshot for the leakage guard. OVERRIDE via grpo.sft_data_snapshot in the
# config (or --adapter-snap); main() HARD-FAILS if the resolved path is missing, so a renamed or
# retrained SFT checkpoint can never silently disable the SFT-leakage filtering.
DEFAULT_ADAPTER_SNAP = Path(
    "data/final/checkpoints/"
    "t-qwen3635ba3b-bf16_s-qwen314b_sdg750_2ep_seed42_20260618_0053_thinking/data_snapshot"
)
OUT_DIR = Path("data/final/grpo")
HELD_OUT = Path("data/final/weak_test_clean.jsonl")
CANDIDATES = OUT_DIR / "weak_candidates.jsonl"            # canonical 600/cat
CANDIDATES_PROBE = OUT_DIR / "weak_candidates_probe.jsonl"  # expanded pool for the reachability probe


def norm_q(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def schema_sets_up(ddl: str) -> bool:
    try:
        c = sqlite3.connect(":memory:")
        c.executescript(ddl)
        c.close()
        return True
    except Exception:
        return False


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def sft_questions(adapter_snap: Path) -> set:
    """Normalized questions the SFT model already saw (+ the held-out test set)."""
    qs = set()
    for r in load_jsonl(TEST):
        qs.add(norm_q(r.get("question", "")))
    for fn in ("train_chat.jsonl", "eval_chat.jsonl"):
        for r in load_jsonl(adapter_snap / fn):
            for m in r.get("messages", []):
                if m.get("role") == "user" and "Question:" in m.get("content", ""):
                    qs.add(norm_q(m["content"].split("Question:")[-1]))
    qs.discard("")
    return qs


def slim(r: dict) -> dict:
    return {
        "question": r.get("question", ""),
        "schema": r.get("schema", ""),
        "gold_sql": r.get("sql", ""),
        "complexity": r.get("complexity", ""),
        "domain": r.get("domain", ""),
        "source": "rl_weak_pool",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline_config.yaml",
                    help="reads grpo.sft_data_snapshot for the leakage guard")
    ap.add_argument("--adapter-snap", default=None,
                    help="SFT data_snapshot dir for the leakage guard (overrides grpo.sft_data_snapshot)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)) if Path(args.config).exists() else {}
    adapter_snap = Path(args.adapter_snap or (cfg.get("grpo") or {}).get("sft_data_snapshot") or DEFAULT_ADAPTER_SNAP)

    # The leakage guard MUST have real inputs, or it silently lets SFT/test questions into the RL
    # pool. Hard-fail (not warn) so a renamed/retrained SFT checkpoint can never disable filtering.
    if not TEST.exists():
        raise FileNotFoundError(
            f"Leakage guard: test set {TEST} not found — refusing to build the weak pool "
            f"without it (leakage risk).")
    snap_files = [adapter_snap / fn for fn in ("train_chat.jsonl", "eval_chat.jsonl")]
    if not adapter_snap.exists() or not any(p.exists() for p in snap_files):
        raise FileNotFoundError(
            f"Leakage guard: SFT data_snapshot {adapter_snap} missing or has no "
            f"train_chat.jsonl/eval_chat.jsonl. Set grpo.sft_data_snapshot (or --adapter-snap) to "
            f"the CURRENT SFT checkpoint's data_snapshot before building the weak pool.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    recs = [r for r in load_jsonl(SRC) if r.get("complexity") in ALL_CATS]
    print(f"[0] seed_full weak records: {dict(Counter(r['complexity'] for r in recs))} total={len(recs)}")
    print(f"[0] leakage-guard snapshot : {adapter_snap}")

    # 1. executable gate
    kept = []
    for r in recs:
        schema, sql = r.get("schema", ""), r.get("sql", "")
        if schema_sets_up(schema):
            ok, _ = execute_sql_on_schema(sql, schema)
            if ok:
                kept.append(r)
    print(f"[1] after executable gate : {dict(Counter(r['complexity'] for r in kept))} total={len(kept)}")

    # 2. leakage guard
    leak = sft_questions(adapter_snap)
    if not leak:
        raise RuntimeError(
            f"Leakage guard produced 0 known questions from {adapter_snap} + {TEST} — the "
            f"snapshot is empty/wrong; refusing to build (would skip leakage filtering).")
    kept = [r for r in kept if norm_q(r.get("question", "")) not in leak]
    print(f"[2] after leakage guard   : {dict(Counter(r['complexity'] for r in kept))} total={len(kept)} "
          f"(vs {len(leak)} SFT/test questions)")

    # 3. dedup within pool
    seen, dedup = set(), []
    for r in kept:
        k = norm_q(r.get("question", ""))
        if k and k not in seen:
            seen.add(k)
            dedup.append(r)
    print(f"[3] after dedup           : {dict(Counter(r['complexity'] for r in dedup))} total={len(dedup)}")

    # 4. carve held-out + candidate pool, per category, disjoint.
    #    WEAK carved first with the original shared rng -> byte-identical to the 2-category build;
    #    EXTRA (subqueries) carved with its OWN rng so it does NOT perturb the WEAK shuffle/split
    #    (a single shared rng would shift window's shuffle and silently rewrite its existing split).
    def carve(cat, r):
        pool = [x for x in dedup if x["complexity"] == cat]
        r.shuffle(pool)
        h = pool[:HELD_OUT_PER_CAT]
        rest = pool[HELD_OUT_PER_CAT:]
        c = rest[:CANDIDATE_CAP_PER_CAT]                                    # canonical 600/cat (stable)
        cp = rest[:PROBE_CAP_PER_CAT.get(cat, CANDIDATE_CAP_PER_CAT)]       # expanded probe pool (superset)
        print(f"    {cat:18s}: held_out={len(h):4d}  candidates={len(c):4d}  probe_pool={len(cp):4d}  (of {len(pool)} unique)")
        return h, c, cp

    held, cand, cand_probe = [], [], []
    for cat in sorted(WEAK):                       # shared rng -> reproduces existing window/set-ops
        h, c, cp = carve(cat, rng)
        held += h
        cand += c
        cand_probe += cp
    for cat in sorted(EXTRA):                       # independent rng -> appended, weak split untouched
        h, c, cp = carve(cat, random.Random(SEED))
        held += h
        cand += c
        cand_probe += cp

    with open(HELD_OUT, "w") as f:
        for r in held:
            f.write(json.dumps(slim(r), ensure_ascii=False) + "\n")
    with open(CANDIDATES, "w") as f:
        for r in cand:
            f.write(json.dumps(slim(r), ensure_ascii=False) + "\n")
    with open(CANDIDATES_PROBE, "w") as f:                                  # expanded reachability input
        for r in cand_probe:
            f.write(json.dumps(slim(r), ensure_ascii=False) + "\n")

    hq = {norm_q(r["question"]) for r in held}
    cq = {norm_q(r["question"]) for r in cand}
    assert not (hq & cq), "held-out and candidate pool overlap!"
    print(f"[4] wrote held_out={len(held)} -> {HELD_OUT}")
    print(f"    wrote candidates={len(cand)} -> {CANDIDATES}")
    print(f"    wrote probe_pool={len(cand_probe)} -> {CANDIDATES_PROBE} "
          f"({dict(Counter(r['complexity'] for r in cand_probe))})")
    print(f"    held n cand overlap        : {len(hq & cq)} (expect 0)")
    print(f"    held n SFT/test overlap    : {len(hq & leak)} (expect 0)")
    print(f"    candidate n SFT/test overlap: {len(cq & leak)} (expect 0)")


if __name__ == "__main__":
    main()

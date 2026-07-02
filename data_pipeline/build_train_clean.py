"""
data_pipeline/build_train_clean.py
============================
Turns the raw teacher trace-distillation dump into training-ready splits.

Pipeline (all inside the training container so the root-owned data dirs are
writable):

  1. FILTER  trace_distill.jsonl -> trace_clean.jsonl
       DROP a record if EITHER
         (a) its schema does not set up in SQLite (executescript raises), OR
         (b) the TEACHER `sql` does not execute against that schema.
       Reuses evaluation.evaluate.execute_sql_on_schema for (b) — the same
       harness eval uses — so "executable" means the same thing everywhere.
       We do NOT filter on gold_sql match.

  2. LEAKAGE GUARD  confirm 0 overlap between the kept questions and
       data/final/test_clean.jsonl on normalized question text. If any test
       question leaked in, it is removed (and reported).

  3. SPLIT  trace_clean.jsonl -> train_split.jsonl (~90%) + val_split.jsonl (~10%)
       random, fixed seed, disjoint. test_clean.jsonl is never touched.

Usage (inside container):
    python data_pipeline/build_train_clean.py
"""

import json
import random
import re
import sqlite3
import sys
from pathlib import Path

from evaluation.evaluate import execute_sql_on_schema  # single source of truth

GEN_DIR = Path("data/generated")
# SRC overridable via argv[1] -> point at the cleaned trace file (Phase 1 output)
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else GEN_DIR / "trace_distill.jsonl"
CLEAN = GEN_DIR / "trace_clean.jsonl"
TRAIN = GEN_DIR / "train_split.jsonl"
VAL = GEN_DIR / "val_split.jsonl"
TEST = Path("data/final/test_clean.jsonl")

SEED = 42
VAL_FRACTION = 0.10


def norm_q(s: str) -> str:
    """Same normalization the SDG leakage guard uses: collapse ws, strip, lower."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def schema_sets_up(schema_ddl: str) -> bool:
    """True iff the full schema DDL applies cleanly via executescript()."""
    try:
        conn = sqlite3.connect(":memory:")
        conn.executescript(schema_ddl)
        conn.close()
        return True
    except Exception:
        return False


def load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    records = load_jsonl(SRC)
    print(f"[1] FILTER  loaded {len(records)} records from {SRC}")

    kept, drop_schema, drop_sql = [], [], []
    for r in records:
        schema = r.get("schema", "")
        sql = r.get("sql", "")
        if not schema_sets_up(schema):
            drop_schema.append(r)
            continue
        ok, _ = execute_sql_on_schema(sql, schema)
        if not ok:
            drop_sql.append(r)
            continue
        kept.append(r)

    print(f"    dropped (schema does not set up) : {len(drop_schema)}")
    print(f"    dropped (teacher sql non-exec)   : {len(drop_sql)}")
    print(f"    kept                             : {len(kept)}")

    with open(CLEAN, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"    wrote {len(kept)} -> {CLEAN}")

    # ---- 2. LEAKAGE GUARD ----------------------------------------------------
    test = load_jsonl(TEST)
    test_qs = {norm_q(ex.get("question", "")) for ex in test}
    overlap = [r for r in kept if norm_q(r.get("question", "")) in test_qs]
    print(f"\n[2] LEAKAGE  test set: {len(test)} questions ({len(test_qs)} unique normalized)")
    print(f"    overlap (kept ∩ test, normalized question): {len(overlap)}")
    if overlap:
        print("    >0 overlap — removing leaked questions from trace_clean")
        for r in overlap[:5]:
            print(f"      - {r.get('question', '')[:80]}")
        kept = [r for r in kept if norm_q(r.get("question", "")) not in test_qs]
        with open(CLEAN, "w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"    rewrote {len(kept)} -> {CLEAN}")
    else:
        print("    ✅ 0 overlap — test set is disjoint, no removal needed")

    # ---- 3. SPLIT ------------------------------------------------------------
    idx = list(range(len(kept)))
    random.Random(SEED).shuffle(idx)
    n_val = max(1, round(len(kept) * VAL_FRACTION))
    val_idx = set(idx[:n_val])
    train_recs = [kept[i] for i in idx[n_val:]]
    val_recs = [kept[i] for i in idx[:n_val]]

    with open(TRAIN, "w") as f:
        for r in train_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(VAL, "w") as f:
        for r in val_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # disjointness sanity check on question text
    tq = {norm_q(r["question"]) for r in train_recs}
    vq = {norm_q(r["question"]) for r in val_recs}
    print(f"\n[3] SPLIT  seed={SEED}  val_fraction={VAL_FRACTION}")
    print(f"    train: {len(train_recs)}  -> {TRAIN}")
    print(f"    val  : {len(val_recs)}  -> {VAL}")
    print(f"    train+val = {len(train_recs) + len(val_recs)} (== clean {len(kept)})")
    print(f"    train∩val question overlap: {len(tq & vq)} (expect 0)")


if __name__ == "__main__":
    main()

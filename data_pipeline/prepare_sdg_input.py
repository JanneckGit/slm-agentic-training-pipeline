"""
data_pipeline/prepare_sdg_input.py
============================
Builds the leakage-free, SQLite-executable SDG input for the trace-distillation
run. Two filters, both reported stage by stage:

  1. LEAKAGE GUARD (critical): drop any seed example whose normalized question
     also appears in the eval test set (data/final/test_clean.jsonl). We must
     never generate teacher traces for test questions. Exclusion is on the
     normalized question (broad/safe); the question+SQL overlap is also reported.

  2. SQLite PREFILTER (efficiency): drop seed examples whose GOLD SQL does not
     execute in SQLite (PostgreSQL-only CTEs/DML etc.) — those traces would be
     dropped downstream anyway. Reuses evaluation.evaluate.execute_sql_on_schema.

Writes the surviving examples to --output and reports the final count (and
whether >= --target remain).

Usage (training container, data/ is root-owned):
    python data_pipeline/prepare_sdg_input.py \
        --config config/pipeline_config.local.yaml \
        --output data/raw/seed_sdg_input.jsonl --target 750
"""

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from evaluation.evaluate import execute_sql_on_schema, normalize_sql


def norm_q(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline_config.yaml")
    ap.add_argument("--seed-file", default=None)
    ap.add_argument("--test-file", default=None)
    ap.add_argument("--output", default="data/raw/seed_sdg_input.jsonl")
    ap.add_argument("--target", type=int, default=750)
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]

    seed_path = Path(args.seed_file) if args.seed_file else \
        Path(data_cfg["raw_dir"]) / "seed_sample.jsonl"
    test_path = Path(args.test_file) if args.test_file else \
        Path(data_cfg["final_dir"]) / data_cfg.get("eval_test_file", "test_clean.jsonl")

    seed = load_jsonl(seed_path)
    test = load_jsonl(test_path)
    print(f"SDG input  : {len(seed)}  ({seed_path})")
    print(f"Test set   : {len(test)}  ({test_path})")

    # -- 1. Leakage guard --
    test_q = {norm_q(ex.get("question", "")) for ex in test}
    test_qs = {(norm_q(ex.get("question", "")), normalize_sql(ex.get("sql", "")))
               for ex in test}

    overlap_q = [ex for ex in seed if norm_q(ex.get("question", "")) in test_q]
    overlap_qs = [ex for ex in seed
                  if (norm_q(ex.get("question", "")), normalize_sql(ex.get("sql", ""))) in test_qs]
    print("\n[1] LEAKAGE GUARD")
    print(f"    overlap by question only      : {len(overlap_q)}")
    print(f"    overlap by question+SQL       : {len(overlap_qs)}")

    seed_noleak = [ex for ex in seed if norm_q(ex.get("question", "")) not in test_q]
    print(f"    excluded (question match)     : {len(seed) - len(seed_noleak)}")
    print(f"    remaining after leakage guard : {len(seed_noleak)}")

    # -- 2. SQLite prefilter --
    kept, sql_fail = [], 0
    for ex in seed_noleak:
        ok, _ = execute_sql_on_schema(ex.get("sql", ""), ex.get("schema", ""))
        if ok:
            kept.append(ex)
        else:
            sql_fail += 1
    print("\n[2] SQLite PREFILTER (gold SQL must execute)")
    print(f"    dropped (non-executable gold) : {sql_fail}")
    print(f"    remaining after SQLite filter : {len(kept)}")

    # -- Write filtered input --
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("\n=== FINAL ===")
    print(f"    leakage-free, SQLite-OK input : {len(kept)}  → {out_path}")
    ok_target = len(kept) >= args.target
    print(f"    target {args.target}: {'OK (enough)' if ok_target else 'SHORT — run will have fewer'} "
          f"({len(kept)} available)")


if __name__ == "__main__":
    main()

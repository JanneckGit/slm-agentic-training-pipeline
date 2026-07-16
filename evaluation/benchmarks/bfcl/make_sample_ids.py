#!/usr/bin/env python3
"""Deterministic BFCL sample-ID lists (seed 42) — the versioned subset for quick-run AND later full-run reuse.

Usage:  .venv-bfcl/bin/python evaluation/benchmarks/bfcl/make_sample_ids.py [--outdir <dir>]

Reads the category datasets from the *installed* bfcl_eval package (no network), samples per-category
IDs with a fixed seed and writes into --outdir (default: this file's directory):
  sample_ids_v1.json   all 100 IDs (12 categories)          -> committed reference
  mt.json              the 20 multi_turn IDs (phase 1)
  st_live.json         the 80 single-turn + live IDs (phase 2)
  smoke.json           4 smoke IDs, disjoint from the sample (throwaway)
Each file is in the `test_case_ids_to_generate.json` format bfcl expects: {"<category>": ["<id>", ...]}.
"""
import argparse
import json
import random
import re
from pathlib import Path

import bfcl_eval

# category -> n  (quick-run v1; category-atomic, multi_turn has priority and is never cut)
COUNTS = {
    "multi_turn_base": 5,
    "multi_turn_miss_func": 5,
    "multi_turn_miss_param": 5,
    "multi_turn_long_context": 5,
    "simple_python": 10,
    "multiple": 10,
    "parallel": 10,
    "parallel_multiple": 10,
    "live_irrelevance": 20,
    "live_simple": 8,
    "live_multiple": 8,
    "live_relevance": 4,
}
SMOKE_COUNTS = {"simple_python": 2, "live_simple": 1, "multi_turn_base": 1}
SEED = 42


def category_ids(data_dir: Path, cat: str) -> list[str]:
    pat = re.compile(rf"BFCL_v\d+_{re.escape(cat)}\.json")
    matches = [p for p in data_dir.iterdir() if pat.fullmatch(p.name)]
    if len(matches) != 1:
        avail = sorted(p.name for p in data_dir.glob("BFCL_*.json"))
        raise SystemExit(f"category '{cat}': {len(matches)} dataset files matched; available: {avail}")
    ids = []
    with matches[0].open() as f:
        first = f.read(1)
        f.seek(0)
        entries = json.load(f) if first == "[" else [json.loads(l) for l in f if l.strip()]
    for e in entries:
        ids.append(e["id"])
    return sorted(ids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path(__file__).parent)
    out = ap.parse_args().outdir
    data_dir = Path(bfcl_eval.__file__).parent / "data"

    rng = random.Random(SEED)
    sample, smoke = {}, {}
    for cat, n in COUNTS.items():
        ids = category_ids(data_dir, cat)
        if n > len(ids):
            raise SystemExit(f"category '{cat}': want {n}, dataset has only {len(ids)}")
        sample[cat] = sorted(rng.sample(ids, n))
        if cat in SMOKE_COUNTS:  # smoke IDs: first eligible ones NOT in the sample (deterministic)
            rest = [i for i in ids if i not in set(sample[cat])]
            smoke[cat] = rest[: SMOKE_COUNTS[cat]]

    files = {
        "sample_ids_v1.json": sample,
        "mt.json": {c: v for c, v in sample.items() if c.startswith("multi_turn")},
        "st_live.json": {c: v for c, v in sample.items() if not c.startswith("multi_turn")},
        "smoke.json": smoke,
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (out / name).write_text(json.dumps(content, indent=1) + "\n")
        print(f"{name}: {sum(len(v) for v in content.values())} ids / {len(content)} categories")


if __name__ == "__main__":
    main()

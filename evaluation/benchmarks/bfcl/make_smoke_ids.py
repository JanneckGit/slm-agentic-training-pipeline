#!/usr/bin/env python3
"""Deterministic BFCL id lists for smokes and spot probes (seed 42).

The full run needs NO id list — `bfcl generate --test-category <group>` runs the whole category.
Id lists are only needed for short verifications, and that is what this script produces.
(Its predecessor was `make_sample_ids.py` for the 100-id quick run; that quick run has been
superseded and its id list archived alongside the run.)

Usage:
    # standard smoke (4 ids across 3 categories) -> smoke.json
    .venv-bfcl/bin/python evaluation/benchmarks/bfcl/make_smoke_ids.py

    # ad-hoc probe for a single category
    .venv-bfcl/bin/python evaluation/benchmarks/bfcl/make_smoke_ids.py \
        --category multi_turn_long_context --n 2 --out /tmp/probe_longctx.json

    # memory: --n counts REAL entries, the prereqs come along automatically (1 -> 11 ids)
    .venv-bfcl/bin/python evaluation/benchmarks/bfcl/make_smoke_ids.py \
        --category memory_vector --n 1 --out /tmp/probe_memvec.json

    # EXACT ids (validated against the loader; memory ids pull their prereqs in automatically)
    .venv-bfcl/bin/python evaluation/benchmarks/bfcl/make_smoke_ids.py \
        --category multi_turn_long_context \
        --ids multi_turn_long_context_66,multi_turn_long_context_123 --out /tmp/ctxprobe.json

    # several categories in one file: category:count pairs
    .venv-bfcl/bin/python evaluation/benchmarks/bfcl/make_smoke_ids.py \
        --spec simple_python:2,live_simple:1,memory_kv:1 --out /tmp/crossgroup.json

Ids are drawn from the datasets of the *installed* bfcl_eval package (no network). The output
format is what bfcl expects as `test_case_ids_to_generate.json`: {"<category>": ["<id>", ...]}.
"""
import argparse
import json
import random
from pathlib import Path

# Standard smoke: one cheap category from each of the three families (non-live / live /
# multi-turn), so a smoke touches all three evaluation paths. 4 ids = ~2 min including serving.
SMOKE_COUNTS = {"simple_python": 2, "live_simple": 1, "multi_turn_base": 1}
SEED = 42


def category_entries(cat: str) -> list[dict]:
    """All test entries of a category — via bfcl's OWN loader, in dataset order.

    Do NOT read the raw file: `BFCL_v4_memory.json` carries ids like `memory_103-student-23`, but
    the actual category ids are `memory_kv_103-student-23` (backend prefix, assigned per category).
    An id taken from the raw file matches nothing, and bfcl then generates ZERO entries without
    comment — the smoke would look green having checked nothing.
    """
    from bfcl_eval.utils import load_dataset_entry
    return load_dataset_entry(cat)


def memory_ids(cat: str, n: int) -> list[str]:
    """Ids for a memory category: n real entries PLUS their prereqs.

    Memory categories are sequential — every real entry has a `depends_on` on prereq entries that
    populate the memory first. In id mode bfcl does NOT pull those in automatically
    (`load_test_entries_from_id_file` filters by id only). Both mistakes have been made here:
      * real entries only   -> they run against an empty memory
      * prereq entries only -> `clean_up_memory_prereq_entries` drops them as soon as the matching
        real entry is absent, and bfcl reports "previously generated" for an EMPTY directory — a
        smoke that looks like it ran.
    Hence: take real entries in dataset order and pass their prereqs along.
    """
    entries = category_entries(cat)
    real = [e for e in entries if "_prereq_" not in e["id"]][:n]
    ids = []
    for e in real:
        ids.extend(e.get("depends_on") or [])
        ids.append(e["id"])
    return list(dict.fromkeys(ids))   # dedupe, keep order


def exact_ids(cat: str, wanted: list[str]) -> list[str]:
    """Hand-picked ids, validated against the loader — never trust ids from the raw files.

    For memory categories the named REAL entries additionally pull in their `depends_on` prereqs
    (the same trap as in memory_ids: without them the entry runs against an empty memory).
    """
    entries = {e["id"]: e for e in category_entries(cat)}
    missing = [i for i in wanted if i not in entries]
    if missing:
        raise SystemExit(f"category '{cat}': unknown ids {missing} — ids must come from "
                         "load_dataset_entry(), not from the raw files (backend prefixes!)")
    ids = []
    for i in wanted:
        ids.extend(entries[i].get("depends_on") or [])
        ids.append(i)
    return list(dict.fromkeys(ids))


def parse_spec(spec: str) -> dict[str, int]:
    """'simple_python:2,live_simple:1' -> {'simple_python': 2, 'live_simple': 1}"""
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        cat, _, n = part.partition(":")
        if not n.isdigit():
            raise SystemExit(f"--spec entry '{part}' is not category:count")
        out[cat] = int(n)
    return out


def pick(counts: dict[str, int]) -> dict[str, list[str]]:
    """Seed-42 selection per category. Fresh RNG per call -> category order does not matter."""
    out = {}
    for cat, n in counts.items():
        if cat.startswith("memory"):
            out[cat] = memory_ids(cat, n)
            continue
        ids = [e["id"] for e in category_entries(cat)]
        if n > len(ids):
            raise SystemExit(f"category '{cat}': {n} requested, dataset only has {len(ids)}")
        # Draw from the SORTED set: random.sample depends on the order of the input list, and
        # dataset order is not a stable contract.
        out[cat] = sorted(random.Random(SEED).sample(sorted(ids), n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", help="a single category instead of the standard smoke")
    ap.add_argument("--n", type=int, default=2, help="number of ids with --category (default 2)")
    ap.add_argument("--ids", help="comma list of EXACT ids (requires --category); validated")
    ap.add_argument("--spec", help="category:count[,category:count...] — several categories at once")
    ap.add_argument("--out", type=Path, help="target file (default: smoke.json next to this script)")
    a = ap.parse_args()

    if a.ids and not a.category:
        raise SystemExit("--ids requires --category")
    if a.spec and (a.ids or a.category):
        raise SystemExit("--spec replaces --category/--ids — pick one form")

    out = a.out or (Path(__file__).parent / "smoke.json")
    if a.ids:
        content = {a.category: exact_ids(a.category, [i.strip() for i in a.ids.split(",") if i.strip()])}
    elif a.spec:
        content = pick(parse_spec(a.spec))
    else:
        content = pick({a.category: a.n} if a.category else SMOKE_COUNTS)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(content, indent=1) + "\n")
    total = sum(len(v) for v in content.values())
    print(f"{out}: {total} ids / {len(content)} categories")
    for cat, ids in content.items():
        print(f"  {cat}: {', '.join(ids)}")


if __name__ == "__main__":
    main()

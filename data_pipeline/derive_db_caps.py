"""
data_pipeline/derive_db_caps.py
===============================
Derive per-template caps for the db_bahn SFT leg from a heldout base-model eval — the measurement
half of the model-aware downweighting (the apply half lives in build_sft_mix.py --db-caps).

Three bands on the single-shot heldout yield y (accept gate = evaluation/eval_report.accepted):
  beherrscht  y >= 0.85  -> cap 60   (replay floor: the only evidenced cliff is exactly zero)
  korridor    0.45..0.85 -> cap 250  (per-task saturation; ~median template size)
  lernkern    y <  0.45  -> uncapped (the actual learning material)
Thresholds sit BETWEEN the discrete 10-pp steps of a 10-rollout measurement (+-18 pp noise per
template) — finer bands would be pseudo-precision. Rationale + literature: docs/dataset-edits-db.md.

Hard gates: dups in the eval (resumed into a stale file) and records without a template abort.
A leg template without (full) eval coverage is WARNED and left uncapped (cap null) — the safe
default for future wave-2 templates that exist in the leg before a new eval ran. A 0%-yield
template gets a spot-check warning (its teacher traces must be vetted once, see the doc).

Output JSON carries provenance (model/label/eval + sha256_16 of eval AND leg file, same truncation
convention as seed_worldstate.py's world manifest); build_sft_mix.py hard-fails on a leg-hash
mismatch, so stale caps can never silently mis-cap a changed leg. No timestamp in the output —
a re-run on the same inputs is byte-identical (determinism check relies on it).

Usage:  PYTHONPATH=. python3 data_pipeline/derive_db_caps.py          # defaults = current base eval,
        # schreibt die LABEL-Datei db_bahn_caps_<label>.json (nie durch den kanonischen Symlink)
        (fuer andere Studenten: ops/build_sft_data.sh MODEL LABEL — leitet --eval/--model/--label ab
         und zeigt danach die kanonischen Symlinks um)
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter

from evaluation.eval_report import load

TH_BEHERRSCHT = 0.85
TH_KORRIDOR = 0.45
BAND_CAPS = {"beherrscht": 60, "korridor": 250, "lernkern": None}
MIN_EVAL_N = 10          # a template needs at least this many eval rollouts to be banded


def sha256_16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def band_of(y: float) -> str:
    return "beherrscht" if y >= TH_BEHERRSCHT else "korridor" if y >= TH_KORRIDOR else "lernkern"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="data/generated/eval/db_traces_heldout_eval_base_qwen3-4b.jsonl")
    ap.add_argument("--leg", default="data/generated/legs/db_traces_chat.jsonl")
    ap.add_argument("--out", default=None,
                    help="default: data/generated/eval/db_bahn_caps_<label>.json (Label-Datei; den "
                         "kanonischen Symlink db_bahn_caps.json zeigt ops/build_sft_data.sh um)")
    ap.add_argument("--model", default="Qwen/Qwen3-4B", help="provenance only")
    ap.add_argument("--label", default="base_qwen3-4b", help="provenance only")
    args = ap.parse_args()
    if args.out is None:
        args.out = f"data/generated/eval/db_bahn_caps_{args.label}.json"
    if os.path.islink(args.out):
        sys.exit(f"HARD-FAIL: {args.out} ist ein Symlink (kanonischer Name) — nicht durch den Link "
                 f"schreiben. ops/build_sft_data.sh nutzen oder einen expliziten --out Pfad angeben.")

    d = load(args.eval)
    per, meta = dict(d["per"]), d["meta"]      # dict(): load() returns a defaultdict — no phantom keys
    if meta["n"] == 0:
        sys.exit(f"HARD-FAIL: {args.eval} is empty")
    if meta["dups"] > 0:
        sys.exit(f"HARD-FAIL: {meta['dups']} duplicate (task_id, sample_idx) in {args.eval} — "
                 f"resumed into a stale file; caps from a mixed eval would be junk")
    if "?" in per:
        sys.exit(f"HARD-FAIL: {per['?']['n']} records without a template field in {args.eval}")

    teachers = Counter(json.loads(l).get("teacher") for l in open(args.eval) if l.strip())
    if len(teachers) > 1:
        print(f"WARN: eval mischt mehrere teacher-Labels: {dict(teachers)} — Datei pruefen!")
    elif args.label not in teachers:
        print(f"HINWEIS: eval teacher-Label {list(teachers)} != --label {args.label!r} "
              f"(ok bei umbenannter Datei, z. B. base_think -> base_qwen3-4b)")

    leg_counts = Counter()
    for line in open(args.leg):
        if line.strip():
            leg_counts[json.loads(line)["_meta"]["template"]] += 1

    templates = {}
    for t in sorted(leg_counts):
        e = per.get(t)
        if e is None or e["n"] < MIN_EVAL_N:
            print(f"WARN: {t} ohne (volle) Eval-Abdeckung (n={e['n'] if e else 0}) -> UNGEDECKELT "
                  f"(Safe-Default fuer neue Templates; Eval neu laufen lassen, um es zu banden)")
            templates[t] = {"n_eval": e["n"] if e else 0, "ok": e["ok"] if e else 0,
                            "yield": None, "band": None, "cap": None, "n_leg": leg_counts[t]}
            continue
        y = e["ok"] / e["n"]
        band = band_of(y)
        templates[t] = {"n_eval": e["n"], "ok": e["ok"], "yield": y,
                        "band": band, "cap": BAND_CAPS[band], "n_leg": leg_counts[t]}
        if y == 0.0:
            print(f"WARN: {t} bei 0% Yield — Teacher-Traces einmal stichprobenartig pruefen "
                  f"(lernbare Demonstration vs. Task-Defekt; Befund fuer die aktuellen zwei: "
                  f"docs/dataset-edits-db.md)")
    for t in sorted(set(per) - set(leg_counts)):
        print(f"WARN: {t} in der Eval, aber nicht im Leg — ignoriert")

    out = {
        "provenance": {
            "model": args.model, "label": args.label,
            "eval_file": args.eval, "eval_sha256_16": sha256_16(args.eval),
            "leg_file": args.leg, "leg_sha256_16": sha256_16(args.leg),
            "thresholds": {"beherrscht_min_yield": TH_BEHERRSCHT, "korridor_min_yield": TH_KORRIDOR},
            "band_caps": BAND_CAPS, "min_eval_n": MIN_EVAL_N,
        },
        "templates": templates,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")

    # --- report ---
    print(f"\n=== DB-CAPS ({args.model}, eval {args.eval}) ===")
    print(f"{'Template':32s} {'yield':>10s} {'band':>10s} {'cap':>6s} {'n_leg':>6s} {'behalten':>8s}")
    for t, e in sorted(templates.items(), key=lambda kv: (-(kv[1]["yield"] or 0), kv[0])):
        y = "  n/a" if e["yield"] is None else f"{e['ok']}/{e['n_eval']}"
        cap = "voll" if e["cap"] is None else str(e["cap"])
        kept = e["n_leg"] if e["cap"] is None else min(e["n_leg"], e["cap"])
        print(f"{t:32s} {y:>10s} {e['band'] or 'ungedeckelt':>10s} {cap:>6s} {e['n_leg']:6d} {kept:8d}")
    bands = {}
    for e in templates.values():
        b = bands.setdefault(e["band"] or "ungedeckelt", {"tmpl": 0, "vor": 0, "nach": 0})
        b["tmpl"] += 1
        b["vor"] += e["n_leg"]
        b["nach"] += e["n_leg"] if e["cap"] is None else min(e["n_leg"], e["cap"])
    print("-" * 76)
    for band, b in sorted(bands.items()):
        print(f"{band:32s} {b['tmpl']:2d} Templates {b['vor']:6d} -> {b['nach']:6d}")
    tot_v = sum(b["vor"] for b in bands.values())
    tot_n = sum(b["nach"] for b in bands.values())
    print(f"{'LEG GESAMT (vor Flail-Filter)':32s} {len(templates):2d} Templates {tot_v:6d} -> {tot_n:6d}")
    print(f"caps -> {args.out}")


if __name__ == "__main__":
    main()

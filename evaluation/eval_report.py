"""
evaluation/eval_report.py
=========================
Per-template report over a rollout JSONL (sdg_pipeline/db_bahn/rollout.py), optionally as a delta
against a baseline run. Yield = the same accept gate rollout.py prints and logs to MLflow:
score==1.0 AND not truncated AND not degenerate.

Counts are shown next to the percentage on purpose: heldout_eval has 10 rollouts per template, so a
single flip moves the number by 10 pp — a bare percentage would fake a precision that is not there.

Plain python3 (no tau2 needed, no repo imports).

Usage:
    python3 evaluation/eval_report.py --input data/generated/db_traces_heldout_base_think.jsonl
    python3 evaluation/eval_report.py --input  data/generated/db_traces_heldout_after_ep2.jsonl \
                                      --baseline data/generated/db_traces_heldout_before.jsonl
"""

import argparse
import json
from collections import Counter, defaultdict

# same thresholds as sdg_pipeline/db_bahn/rollout.py (duplicated so this stays import-free)
DEGEN_MAX_THINK_CHARS = 12_000
DEGEN_MAX_DUP8_RATIO = 0.5


def accepted(rec: dict) -> bool:
    d = rec.get("degen") or {}
    return ((rec.get("score") or {}).get("score") == 1.0
            and not rec.get("truncated")
            and d.get("think_ngram_dup_ratio", 0.0) <= DEGEN_MAX_DUP8_RATIO
            and d.get("max_think_chars", 0) <= DEGEN_MAX_THINK_CHARS)


def load(path: str) -> dict:
    """-> {template: {n, ok, turns, calls}} plus file-level counters."""
    per = defaultdict(lambda: {"n": 0, "ok": 0, "turns": 0, "calls": 0})
    meta = {"n": 0, "truncated": 0, "degen": 0, "finish": Counter(), "seen": set(), "dups": 0}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        key = (r.get("task_id"), r.get("sample_idx"))
        if key in meta["seen"]:
            meta["dups"] += 1
        meta["seen"].add(key)
        e = per[r.get("template", "?")]
        sc = r.get("score") or {}
        d = r.get("degen") or {}
        e["n"] += 1
        e["ok"] += accepted(r)
        e["turns"] += sc.get("turns_used", 0)
        e["calls"] += sc.get("n_tool_calls", 0)
        meta["n"] += 1
        meta["truncated"] += bool(r.get("truncated"))
        meta["degen"] += (d.get("think_ngram_dup_ratio", 0.0) > DEGEN_MAX_DUP8_RATIO
                          or d.get("max_think_chars", 0) > DEGEN_MAX_THINK_CHARS)
        meta["finish"][str(r.get("finish_reason", "?")).split(":")[0]] += 1
    return {"per": per, "meta": meta}


def pct(ok: int, n: int) -> str:
    return f"{100 * ok / n:5.1f}% ({ok}/{n})" if n else "–"


def main():
    ap = argparse.ArgumentParser(description="per-template report over a rollout JSONL")
    ap.add_argument("--input", required=True, help="rollout JSONL to report on")
    ap.add_argument("--baseline", default=None,
                    help="second rollout JSONL; adds a per-template delta column (pp)")
    args = ap.parse_args()

    cur = load(args.input)
    base = load(args.baseline) if args.baseline else None
    per, meta = cur["per"], cur["meta"]

    tot_ok = sum(e["ok"] for e in per.values())
    print(f"\n{args.input}")
    if base:
        b = base["meta"]
        b_ok = sum(e["ok"] for e in base["per"].values())
        delta = (100 * tot_ok / max(1, meta["n"])) - (100 * b_ok / max(1, b["n"]))
        print(f"vs. {args.baseline}")
        print(f"\nTOTAL  {pct(tot_ok, meta['n'])}   baseline {pct(b_ok, b['n'])}   Δ {delta:+.1f} pp")
    else:
        print(f"\nTOTAL  {pct(tot_ok, meta['n'])}")
    print(f"       truncated {meta['truncated']}   degenerate {meta['degen']}")

    off = {k: v for k, v in meta["finish"].items() if k != "final_answer"}
    if off:  # infra failures look exactly like model failures in the yield — surface them
        print(f"       finish_reason != final_answer: {dict(off)}")
    if meta["dups"]:
        print(f"       WARNING {meta['dups']} duplicate (task_id, sample_idx) — resumed into a stale file?")

    hdr = f"\n{'Template':34s} {'yield':>16s}"
    if base:
        hdr += f" {'baseline':>16s} {'Δ pp':>7s}"
    hdr += f" {'turns':>6s} {'calls':>6s}"
    print(hdr)
    print("-" * len(hdr.strip()))

    def sort_key(item):
        t, e = item
        if not base:
            return e["ok"] / max(1, e["n"])
        b = base["per"].get(t)
        by = b["ok"] / max(1, b["n"]) if b else 0.0
        return e["ok"] / max(1, e["n"]) - by          # worst delta first

    for t, e in sorted(per.items(), key=sort_key):
        row = f"{t[:34]:34s} {pct(e['ok'], e['n']):>16s}"
        if base:
            b = base["per"].get(t)
            if b and b["n"]:
                d = 100 * e["ok"] / max(1, e["n"]) - 100 * b["ok"] / b["n"]
                row += f" {pct(b['ok'], b['n']):>16s} {d:>+7.1f}"
            else:
                row += f" {'–':>16s} {'–':>7s}"
        row += f" {e['turns'] / max(1, e['n']):>6.1f} {e['calls'] / max(1, e['n']):>6.1f}"
        print(row)
    print()


if __name__ == "__main__":
    main()

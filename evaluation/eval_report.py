"""
evaluation/eval_report.py
=========================
Per-template report over a rollout JSONL (sdg_pipeline/db_bahn/rollout.py), optionally as a delta
against a baseline run. Yield = the same accept gate rollout.py prints and logs to MLflow:
score==1.0 AND not truncated AND not degenerate.

Counts are shown next to the percentage on purpose: heldout_eval has 10 rollouts per template, so a
single flip moves the number by 10 pp — a bare percentage would fake a precision that is not there.

k > 1 (ops/eval_heldout.sh default K=8): `yield` is then the MEAN over all
task×sample rollouts (= a lower-noise pass@1, comparable to the old k=1 files), and an extra
`pass^k` column counts a task only if ALL k of its samples are accepted (consistency measure, the
same pass^k τ²-bench reports). Tasks with fewer than k samples (interrupted run) are excluded from
pass^k and reported as a WARNING. With k=1 the output is byte-identical to the pre-k version.

Plain python3 (no tau2 needed, no repo imports).

Usage:
    python3 evaluation/eval_report.py --input data/generated/eval/db_traces_heldout_base_think.jsonl
    python3 evaluation/eval_report.py --input  data/generated/eval/db_traces_heldout_after_ep2.jsonl \
                                      --baseline data/generated/eval/db_traces_heldout_before.jsonl
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


def load(path: str, max_sample: int | None = None) -> dict:
    """-> {template: {n, ok, turns, calls, tasks}} plus file-level counters.
    `tasks` maps task_id -> list of accepted-flags (one per sample) for the pass^k view.
    max_sample: keep only records with sample_idx < max_sample (an aborted round-order run -> clean pass^r)."""
    per = defaultdict(lambda: {"n": 0, "ok": 0, "turns": 0, "calls": 0, "tasks": defaultdict(list)})
    meta = {"n": 0, "truncated": 0, "degen": 0, "finish": Counter(), "seen": set(), "dups": 0}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if max_sample is not None and (r.get("sample_idx") or 0) >= max_sample:
            continue
        key = (r.get("task_id"), r.get("sample_idx"))
        if key in meta["seen"]:
            meta["dups"] += 1
        meta["seen"].add(key)
        e = per[r.get("template", "?")]
        sc = r.get("score") or {}
        d = r.get("degen") or {}
        acc = accepted(r)
        e["n"] += 1
        e["ok"] += acc
        e["turns"] += sc.get("turns_used", 0)
        e["calls"] += sc.get("n_tool_calls", 0)
        e["tasks"][r.get("task_id")].append(acc)
        meta["n"] += 1
        meta["truncated"] += bool(r.get("truncated"))
        meta["degen"] += (d.get("think_ngram_dup_ratio", 0.0) > DEGEN_MAX_DUP8_RATIO
                          or d.get("max_think_chars", 0) > DEGEN_MAX_THINK_CHARS)
        meta["finish"][str(r.get("finish_reason", "?")).split(":")[0]] += 1
    _add_passk(per, meta)
    return {"per": per, "meta": meta}


def _add_passk(per: dict, meta: dict) -> None:
    """k = the largest sample count any task has. Per template: n_tasks (complete ones), passk_ok
    (all k samples accepted), incomplete (fewer than k samples -> excluded)."""
    k = max((len(v) for e in per.values() for v in e["tasks"].values()), default=1)
    meta["k"] = k
    meta["incomplete"] = 0
    for e in per.values():
        e["n_tasks"] = e["passk_ok"] = e["incomplete"] = 0
        for flags in e["tasks"].values():
            if len(flags) < k:
                e["incomplete"] += 1
                continue
            e["n_tasks"] += 1
            e["passk_ok"] += all(flags)
        meta["incomplete"] += e["incomplete"]


def passk_at(per: dict, k: int, n: int) -> tuple[float, int] | None:
    """Unbiased pass^n from k >= n samples per task: mean over complete tasks of C(c, n) / C(k, n), c = accepted
    samples (the estimator τ²-bench uses). Lets a K=8 run be compared with a K=2 run's direct pass^2.
    -> (estimate, n_tasks), or None if k < n."""
    from math import comb
    if k < n:
        return None
    vals = [comb(sum(flags), n) / comb(k, n)
            for e in per.values() for flags in e["tasks"].values() if len(flags) == k]
    return (sum(vals) / len(vals) if vals else 0.0), len(vals)


def pct(ok: int, n: int) -> str:
    return f"{100 * ok / n:5.1f}% ({ok}/{n})" if n else "–"


def main():
    ap = argparse.ArgumentParser(description="per-template report over a rollout JSONL")
    ap.add_argument("--input", required=True, help="rollout JSONL to report on")
    ap.add_argument("--baseline", default=None,
                    help="second rollout JSONL; adds a per-template delta column (pp)")
    ap.add_argument("--passk-at", type=int, default=None, metavar="N",
                    help="also print the unbiased pass^N estimate from k >= N samples per task (e.g. 2: compare "
                         "a K=8 run with a K=2 API run); off by default -> output unchanged")
    ap.add_argument("--max-sample", type=int, default=None, metavar="R",
                    help="only records with sample_idx < R (input AND baseline): an aborted --order round run "
                         "has R complete rounds over all tasks -> clean pass^R; off by default -> output unchanged")
    args = ap.parse_args()

    cur = load(args.input, args.max_sample)
    base = load(args.baseline, args.max_sample) if args.baseline else None
    per, meta = cur["per"], cur["meta"]
    k = meta["k"]
    base_k = base["meta"]["k"] if base else 1
    show_passk = k > 1
    show_base_passk = show_passk and base_k > 1

    tot_ok = sum(e["ok"] for e in per.values())
    print(f"\n{args.input}" + (f"   [nur sample_idx < {args.max_sample}]" if args.max_sample is not None else ""))
    if base:
        b = base["meta"]
        b_ok = sum(e["ok"] for e in base["per"].values())
        delta = (100 * tot_ok / max(1, meta["n"])) - (100 * b_ok / max(1, b["n"]))
        print(f"vs. {args.baseline}")
        print(f"\nTOTAL  {pct(tot_ok, meta['n'])}   baseline {pct(b_ok, b['n'])}   Δ {delta:+.1f} pp")
    else:
        print(f"\nTOTAL  {pct(tot_ok, meta['n'])}")
    if show_passk:
        pk_ok = sum(e["passk_ok"] for e in per.values())
        pk_n = sum(e["n_tasks"] for e in per.values())
        line = f"pass^{k} {pct(pk_ok, pk_n)}   (yield above = mean over {k} samples/task)"
        if show_base_passk:
            bk_ok = sum(e["passk_ok"] for e in base["per"].values())
            bk_n = sum(e["n_tasks"] for e in base["per"].values())
            d = 100 * pk_ok / max(1, pk_n) - 100 * bk_ok / max(1, bk_n)
            line = f"pass^{k} {pct(pk_ok, pk_n)}   baseline {pct(bk_ok, bk_n)}   Δ {d:+.1f} pp"
        elif base:
            line += f"  [baseline is k={base_k}: no pass^k delta]"
        print(line)
    if args.passk_at:
        n_at = args.passk_at
        for name, run in (("", cur), ("baseline ", base)):
            if run is None:
                continue
            est = passk_at(run["per"], run["meta"]["k"], n_at)
            print(f"{name}pass^{n_at} (Schaetzer aus k={run['meta']['k']}) " +
                  (f"{100 * est[0]:5.1f}% ueber {est[1]} Tasks" if est else f"– (k < {n_at})"))
    print(f"       truncated {meta['truncated']}   degenerate {meta['degen']}")

    off = {k_: v for k_, v in meta["finish"].items() if k_ != "final_answer"}
    if off:  # infra failures look exactly like model failures in the yield — surface them
        print(f"       finish_reason != final_answer: {dict(off)}")
    if meta["dups"]:
        print(f"       WARNING {meta['dups']} duplicate (task_id, sample_idx) — resumed into a stale file?")
    if meta["incomplete"]:
        print(f"       WARNING {meta['incomplete']} task(s) with fewer than k={k} samples — "
              f"excluded from pass^{k} (interrupted run?)")
        r = min(len(f) for e in per.values() for f in e["tasks"].values())
        if r >= 1:   # an --order round run aborted in round r+1: every task has its first r samples
            print(f"       hint: every task has >= {r} samples -> --max-sample {r} gives a clean pass^{r} "
                  f"(if the run used --order round)")

    hdr = f"\n{'Template':34s} {'yield':>16s}"
    if base:
        hdr += f" {'baseline':>16s} {'Δ pp':>7s}"
    if show_passk:
        hdr += f" {f'pass^{k}':>16s}"
        if show_base_passk:
            hdr += f" {'base pass^k':>16s} {'Δ pp':>7s}"
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
        if show_passk:
            row += f" {pct(e['passk_ok'], e['n_tasks']):>16s}"
            if show_base_passk:
                b = base["per"].get(t)
                if b and b["n_tasks"]:
                    d = 100 * e["passk_ok"] / max(1, e["n_tasks"]) - 100 * b["passk_ok"] / b["n_tasks"]
                    row += f" {pct(b['passk_ok'], b['n_tasks']):>16s} {d:>+7.1f}"
                else:
                    row += f" {'–':>16s} {'–':>7s}"
        row += f" {e['turns'] / max(1, e['n']):>6.1f} {e['calls'] / max(1, e['n']):>6.1f}"
        print(row)
    print()


if __name__ == "__main__":
    main()

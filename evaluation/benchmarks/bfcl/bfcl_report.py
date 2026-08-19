#!/usr/bin/env python3
"""
evaluation/benchmarks/bfcl/bfcl_report.py
=========================================
Per-category report over one BFCL run (one BFCL_PROJECT_ROOT), optionally as a delta against a
baseline run. Counterpart to evaluation/eval_report.py for the in-domain eval.

Plain python3 — no bfcl imports, no venv needed. It reads what bfcl writes to disk:
    <root>/score/<model_key>/<group>/BFCL_v4_<cat>_score.json   line 1: accuracy/correct/total
    <root>/result/<model_key>/<group>/BFCL_v4_<cat>_result.json JSONL, one line per item

Why the diagnostic columns ride along instead of just the score:

  think%      The SFT predecessor (ep2) thought in 0 of 100 BFCL cases while the base thought
              everywhere. Every score comparison was therefore think-vs-nothink rather than a model
              comparison. Without this column the confound is invisible.
  ⌀ out_tok   That collapse on parallel/parallel_multiple was a one-call attractor: ONE clean call,
              then a hard stop at 31-64 output tokens. The token count shows it long before the
              score can be interpreted.
  zero_input  input_token_count == 0 means the request never went through (context blown or
              similar). That is a BROKEN RUN, not a weak model — in the old quick run 4 of 5
              multi_turn_long_context episodes silently scored 0 % that way. Reported as an error.
  at_cap      output_token_count >= 4096 = truncated at BFCL's max_tokens ceiling
              (min(4096, ctx - input - 2), base_oss_handler.py). The base never reached it (max
              3,170), so any occurrence is new and needs explaining.
  infra       bfcl stored a caught inference exception as the answer. Retryable, unlike a context
              overflow — see the report footer.

Usage:
    python3 evaluation/benchmarks/bfcl/bfcl_report.py --run data/generated/eval/bfcl/qwen3-4b_sft-ep3
    python3 evaluation/benchmarks/bfcl/bfcl_report.py --run  data/generated/eval/bfcl/qwen3-4b_sft-ep3 \
                                                      --baseline data/generated/eval/bfcl/qwen3-4b_base
"""
import argparse
import json
import sys
from pathlib import Path

AT_CAP_TOKENS = 4096  # BFCL's max_tokens ceiling (base_oss_handler.py), not ours

# bfcl catches every inference exception and stores its text AS the answer
# (multi_threaded_inference), so the item then scores as an ordinary failure. Two very different
# causes end up wearing the same jacket, and the marker alone does NOT tell them apart:
#   * the server rejected an over-long prompt  -> HTTP 400 "maximum context length is N tokens"
#   * the connection dropped / timed out       -> "Connection error." (APIConnectionError)
# Only the first is a real model limit; the second is retryable. Hence the second marker.
INFER_ERROR_MARK = "Error during inference"
CTX_ERROR_MARKS = ("maximum context length", "context_length_exceeded", "longer than the maximum")


def _leaves(x):
    """Every scalar out of arbitrarily nested lists — multi_turn yields lists of lists."""
    if isinstance(x, list):
        for y in x:
            yield from _leaves(y)
    elif x is not None:
        yield x


def _text(x) -> str:
    return " ".join(str(v) for v in _leaves(x))


def group_of(path: Path, root: Path, kind: str) -> str:
    """Group name derived from the storage path.

    bfcl does NOT nest uniformly two levels deep: non_live/live/multi_turn sit under
    `<kind>/<model>/<group>/`, but the agentic categories under
    `<kind>/<model>/agentic/memory/<backend>/` resp. `.../agentic/web_search/`. A `*/*` glob would
    miss the memory results entirely — the category would vanish from the report without comment.
    """
    rel = path.relative_to(root / kind).parts        # (<model>, <group>, [...], <file>)
    return rel[1] if len(rel) > 1 else "?"


def load_run(root: Path) -> dict:
    """-> {category: {group, n, acc, correct, total, think, out, inp, zero_input, at_cap, calls}}"""
    per = {}
    for score_file in sorted(root.glob("score/**/BFCL_v4_*_score.json")):
        cat = score_file.name[len("BFCL_v4_"):-len("_score.json")]
        with score_file.open() as f:
            head = json.loads(f.readline())
        per[cat] = {
            "group": group_of(score_file, root, "score"),
            "acc": head.get("accuracy"),
            "correct": head.get("correct_count"),
            "total": head.get("total_count"),
            "n": 0, "think": 0, "out": 0, "inp": 0, "zero_input": 0, "at_cap": 0, "calls": 0,
            "infra_err": 0, "ctx_err": 0,
        }

    for res_file in sorted(root.glob("result/**/BFCL_v4_*_result.json")):
        cat = res_file.name[len("BFCL_v4_"):-len("_result.json")]
        e = per.setdefault(cat, {"group": group_of(res_file, root, "result"), "acc": None,
                                 "correct": None, "total": None,
                                 "n": 0, "think": 0, "out": 0, "inp": 0, "zero_input": 0,
                                 "at_cap": 0, "calls": 0, "infra_err": 0, "ctx_err": 0})
        for line in res_file.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            e["n"] += 1
            if _text(r.get("reasoning_content")).strip():
                e["think"] += 1
            outs = [v for v in _leaves(r.get("output_token_count")) if isinstance(v, (int, float))]
            inps = [v for v in _leaves(r.get("input_token_count")) if isinstance(v, (int, float))]
            e["out"] += sum(outs)
            e["inp"] += sum(inps)
            e["calls"] += len(outs)
            e["at_cap"] += sum(1 for v in outs if v >= AT_CAP_TOKENS)
            # An episode counts as broken as soon as ONE request had 0 input tokens.
            if not inps or min(inps) == 0:
                e["zero_input"] += 1
            # Split the caught inference exceptions by what actually went wrong — see the marker
            # definitions above. Both look identical in the score; only one is worth retrying.
            res_text = _text(r.get("result"))
            if INFER_ERROR_MARK in res_text:
                low = res_text.lower()
                if any(mark in low for mark in CTX_ERROR_MARKS):
                    e["ctx_err"] += 1
                else:
                    e["infra_err"] += 1
    return per


GROUP_ORDER = ["non_live", "live", "multi_turn", "memory", "web_search", "format_sensitivity"]


def fmt_pct(x) -> str:
    return "   —  " if x is None else f"{100 * x:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="BFCL_PROJECT_ROOT of the run")
    ap.add_argument("--baseline", type=Path, help="comparison run for the delta column")
    a = ap.parse_args()

    if not a.run.is_dir():
        print(f"not a directory: {a.run}", file=sys.stderr)
        return 2
    run = load_run(a.run)
    base = load_run(a.baseline) if a.baseline else {}
    if not run:
        print(f"no results under {a.run} (neither score/ nor result/)", file=sys.stderr)
        return 2

    manifest = a.run / "run_manifest.json"
    missing = []
    if manifest.is_file():
        m = json.loads(manifest.read_text())
        print(f"Run       : {m.get('label')}  ({m.get('model')})")
        print(f"Checkpoint: {m.get('model_dir_resolved') or '—'}  fp={m.get('model_fingerprint') or '—'}")
        reg = m.get("registry") or {}
        print(f"Registry  : {m.get('model_key')} -> alias '{m.get('model_alias')}'"
              f"{'  (injected)' if reg.get('injected') else ''}"
              f" · max_model_len {(m.get('serving') or {}).get('max_model_len')}")
        print(f"bfcl {m.get('bfcl_version')} · repo {m.get('repo_commit')} · {m.get('created_utc')}")
        # Planned vs actually produced categories. Without this reconciliation a category that
        # produced NOTHING simply drops out of the table — and the report looks clean. That is
        # exactly how a memory probe ran silently empty while building this pipeline (wrong ID
        # prefix), without anything turning red.
        missing = [c for c in (m.get("categories") or []) if c not in run]
    if a.baseline:
        print(f"Baseline  : {a.baseline}")
    print()

    dw = 8 if a.baseline else 0
    dhead = f"{'Δ pp':>{dw}}" if a.baseline else ""
    print(f"{'Category':26s} {'n':>5} {'acc':>7}{dhead} {'think%':>7} {'⌀out':>6} {'⌀in':>7} "
          f"{'at_cap':>7} {'ctx_err':>8} {'infra':>6}")
    print("-" * (26 + 5 + 7 + dw + 7 + 6 + 7 + 7 + 8 + 6 + 9))

    broken, infra, ctx, totals = [], [], [], {"n": 0, "correct": 0, "total": 0}
    for group in GROUP_ORDER + sorted({v["group"] for v in run.values()} - set(GROUP_ORDER)):
        cats = sorted(c for c, v in run.items() if v["group"] == group)
        if not cats:
            continue
        print(f"— {group}")
        g = {"correct": 0, "total": 0}
        for cat in cats:
            e = run[cat]
            delta = ""
            if a.baseline:
                b = base.get(cat)
                if b and b.get("acc") is not None and e.get("acc") is not None:
                    d = 100 * (e["acc"] - b["acc"])
                    delta = f"{d:+7.1f}"
                else:
                    delta = f"{'—':>7}"
            calls = e["calls"] or 1
            n = e["n"] or 1
            print(f"  {cat:24s} {e['n']:5d} {fmt_pct(e['acc']):>7}{delta:>{dw}} "
                  f"{100 * e['think'] / n:6.0f}% {e['out'] / calls:6.0f} {e['inp'] / calls:7.0f} "
                  f"{e['at_cap']:7d} {e.get('ctx_err', 0):8d} {e.get('infra_err', 0):6d}")
            # Both error kinds leave 0 input tokens behind, so zero_input alone would count them
            # again. Report every item ONCE, under its actual cause; whatever zero_input is left
            # over after subtracting both is genuinely unexplained and worth its own section.
            if e.get("ctx_err"):
                ctx.append((cat, e["ctx_err"], e["n"]))
            if e.get("infra_err"):
                infra.append((cat, e["infra_err"], e["n"]))
            rest = e["zero_input"] - e.get("infra_err", 0) - e.get("ctx_err", 0)
            if rest > 0:
                broken.append((cat, rest, e["n"]))
            if e.get("correct") is not None:
                g["correct"] += e["correct"]
                g["total"] += e["total"]
            totals["n"] += e["n"]
        if g["total"]:
            totals["correct"] += g["correct"]
            totals["total"] += g["total"]
            print(f"  {'  Σ ' + group:24s} {'':5s} {fmt_pct(g['correct'] / g['total']):>7}"
                  f"  ({g['correct']}/{g['total']})")

    if totals["total"]:
        print("-" * 70)
        print(f"{'TOTAL (scored items)':26s} {totals['n']:5d} "
              f"{fmt_pct(totals['correct'] / totals['total']):>7}  ({totals['correct']}/{totals['total']})")

    rc = 0
    if missing:
        print(f"\n!! NO RESULTS — {len(missing)} planned categor(y|ies) produced nothing:")
        print(f"     {', '.join(missing)}")
        print("   Either not run yet (let it continue) or the ids do not match.")
        rc = 1
    if ctx:
        print("\n!! CONTEXT OVERFLOW — the server rejected the prompt (HTTP 400), scored as failure:")
        for cat, k, n in ctx:
            print(f"     {cat}: {k}/{n}")
        print("   NOT a model statement. Retrying changes nothing — either raise --max-model-len,")
        print("   or, if it already equals the model's max_position_embeddings, these items are")
        print("   beyond the model's context and must be excluded from interpretation, not read")
        print("   as 0 %.")
        rc = 1
    if infra:
        print("\n!! INFRASTRUCTURE ERRORS — bfcl stored an inference exception as the answer:")
        for cat, k, n in infra:
            print(f"     {cat}: {k}/{n}")
        print("   Connection drop/timeout, NOT a model statement — and unlike a context overflow")
        print("   this is fixed by retrying. A plain resume will NOT pick these up: the error item")
        print("   sits in the result file and therefore counts as done. Refetch them precisely")
        print("   (replaces ONLY those ids, everything else stays):")
        print("     BFCL_RUN_IDS=<file-with-the-ids> BFCL_ALLOW_OVERWRITE=1 bash ops/eval_bfcl.sh …")
        rc = 1
    if broken:
        print("\n!! UNEXPLAINED — input_token_count == 0 without a stored error message:")
        for cat, k, n in broken:
            print(f"     {cat}: {k}/{n}")
        print("   The request never went through and bfcl recorded no reason. Not a model result;")
        print("   inspect the affected items before reading the score of this category.")
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())

"""
evaluation/heldout_gate.py
==========================
Automatic go/no-go checks for heldout runs on a foreign host (ops/h100_open_models.sh), so an unattended
chain stops before it spends GPU days on a broken setup.

  smoke --input <db_traces_bakeoff_dev_<label>.jsonl>
      Does the model work in OUR harness (prompt-and-parse, inline <think>)? Only plumbing / format signals,
      never the score — a weak model must pass, a misconfigured one must not:
        errors      <= max(1, 2 %)  finish_reason teacher_error:* or error:* (HTTP 4xx/5xx, rejected sampling
                              params, context overflow, rollout exception). A config error hits EVERY rollout, a
                              single transient timeout must not cost a whole model
        truncated   <= 10 %   rollouts that hit the 3,072-token cap of a turn (think runs past the budget)
        think       >= 90 %   rollouts with >= 1 <think> block (thinking is really on). Per rollout, not per turn:
                              Harmony models (gpt-oss) think before every tool call but often write the final
                              answer without an analysis block (GB10 smoke 10.11.: 78 % of turns, 100 % of rollouts).
                              The per-turn share is still printed as info.
                              API runs (gen_params.backend, e.g. anthropic): INFO only, not a criterion — the model
                              decides per turn whether to think (adaptive) and only a summary comes back.
        tool calls  >= 90 %   rollouts with >= 1 parsed tool call (the call format is understood)
      Refusals (finish_reason refusal, API safety declines) are printed as info.

  calib --input <run on the new host> --reference <same model + protocol, known host>
      Paired bootstrap over the tasks both files cover completely (10k draws, seed 0): difference in mean
      accepted rate (= yield, the K-sample mean) and in pass^k. Passes if |delta yield| <= --max-delta-pp (3):
      roughly 3x the standard error of a K=8 difference over 480 tasks.

Exit 0 = pass, 1 = fail, 2 = unusable input. Plain python3; the acceptance rule comes from eval_report.py.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_report import accepted  # noqa: E402  (same accept gate as rollout.py / eval_report.py)

SMOKE_MAX_ERR = 0.02
SMOKE_MAX_TRUNC = 0.10
SMOKE_MIN_THINK = 0.90
SMOKE_MIN_TOOL = 0.90


def load(path: str, keep_messages: bool = False) -> list[dict]:
    """Records of a rollout JSONL; messages are dropped unless needed (a K=8 file is ~100 MB of traces)."""
    recs, bad = [], 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not keep_messages:
                r.pop("messages", None)
            recs.append(r)
    if bad:
        print(f"WARNUNG: {bad} ungueltige Zeile(n) in {path} uebersprungen")
    return recs


def is_error(rec: dict) -> bool:
    fr = str(rec.get("finish_reason") or "")
    return fr.startswith("teacher_error") or fr.startswith("error")


def smoke_stats(recs: list[dict]) -> dict:
    n = len(recs)
    turns = think_turns = with_think = with_tool = 0
    for r in recs:
        has_tool = has_think = False
        for m in r.get("messages") or []:
            if m.get("role") != "assistant":
                continue
            turns += 1
            t = (m.get("content") or "").lstrip().startswith("<think>")
            think_turns += t
            has_think = has_think or t
            has_tool = has_tool or bool(m.get("tool_calls"))
        with_think += has_think
        with_tool += has_tool
    backends = Counter((r.get("gen_params") or {}).get("backend") or "vllm" for r in recs)
    return {"n": n,
            "backend": backends.most_common(1)[0][0] if backends else "vllm",
            "refusals": sum(str(r.get("finish_reason") or "") == "refusal" for r in recs),
            "errors": sum(is_error(r) for r in recs),
            "truncated": sum(bool(r.get("truncated")) for r in recs) / max(1, n),
            "think": with_think / max(1, n),
            "think_turns": think_turns / max(1, turns),
            "tool": with_tool / max(1, n),
            "yield": sum(accepted(r) for r in recs) / max(1, n),
            "finish": Counter(str(r.get("finish_reason") or "")[:40] for r in recs)}


def smoke(args) -> int:
    recs = load(args.input, keep_messages=True)
    if not recs:
        print(f"SMOKE: {args.input} ist leer — unbrauchbar")
        return 2
    s = smoke_stats(recs)
    max_err = max(1, int(SMOKE_MAX_ERR * s["n"]))
    api = s["backend"] != "vllm"   # API model: adaptive thinking, summary only -> thinking is info, not a criterion
    checks = [("Fehler (teacher_error/error)", s["errors"] <= max_err, f"{s['errors']}", f"<= {max_err}"),
              ("truncated", s["truncated"] <= SMOKE_MAX_TRUNC, f"{s['truncated']:.1%}", f"<= {SMOKE_MAX_TRUNC:.0%}")]
    if not api:
        checks.append(("Rollouts mit Denken", s["think"] >= SMOKE_MIN_THINK, f"{s['think']:.1%}",
                       f">= {SMOKE_MIN_THINK:.0%}"))
    checks.append(("Rollouts mit Tool-Call", s["tool"] >= SMOKE_MIN_TOOL, f"{s['tool']:.1%}", f">= {SMOKE_MIN_TOOL:.0%}"))
    print(f"SMOKE {args.input}: {s['n']} Rollouts, Yield {s['yield']:.1%} (nur Info, kein Kriterium)"
          + (f" | Backend {s['backend']}" if api else ""))
    for name, ok, val, rule in checks:
        print(f"  {'OK ' if ok else 'XX '} {name:<28} {val:>7}   (Soll {rule})")
    if api:
        print(f"  info: Rollouts mit Denken {s['think']:.1%} (API: nur Info, Denken adaptiv + nur Zusammenfassung)")
    print(f"  info: Denken je Assistant-Zug {s['think_turns']:.1%}")
    print(f"  info: Ablehnungen (refusal) {s['refusals']}")
    print("  finish: " + ", ".join(f"{k} {v}" for k, v in s["finish"].most_common(6)))
    ok = all(c[1] for c in checks)
    print(f"SMOKE {'BESTANDEN' if ok else 'DURCHGEFALLEN'}")
    return 0 if ok else 1


def per_task(recs: list[dict]) -> dict:
    d = defaultdict(list)
    for r in recs:
        d[r["task_id"]].append(accepted(r))
    return d


def calib(args) -> int:
    new, ref = load(args.input), load(args.reference)
    if not new or not ref:
        print("CALIB: Eingabe leer — unbrauchbar")
        return 2
    tn, tr = per_task(new), per_task(ref)
    k = max(len(v) for v in tr.values())
    common = sorted(t for t in tn.keys() & tr.keys() if len(tn[t]) == k and len(tr[t]) == k)
    if len(common) < max(1, int(0.95 * len(tr))):
        print(f"CALIB: nur {len(common)} von {len(tr)} Referenz-Tasks mit vollen k={k} Samples in beiden — unbrauchbar")
        return 2
    yn = [sum(tn[t]) / k for t in common]
    yr = [sum(tr[t]) / k for t in common]
    pn = [float(all(tn[t])) for t in common]
    pr = [float(all(tr[t])) for t in common]
    n = len(common)
    d_y = (sum(yn) - sum(yr)) / n
    d_p = (sum(pn) - sum(pr)) / n
    rng = random.Random(args.seed)
    by, bp = [], []
    for _ in range(args.draws):
        idx = [rng.randrange(n) for _ in range(n)]
        by.append(sum(yn[i] - yr[i] for i in idx) / n)
        bp.append(sum(pn[i] - pr[i] for i in idx) / n)
    by.sort(); bp.sort()
    lo, hi = int(0.025 * args.draws), int(0.975 * args.draws) - 1

    def rate(recs, f):
        return sum(f(r) for r in recs) / len(recs)

    print(f"CALIB {n} Tasks x k={k} | neu: {args.input}\n                       Referenz: {args.reference}")
    print(f"  Yield (Mittel ueber k)  neu {sum(yn) / n:.1%}  Ref {sum(yr) / n:.1%}  "
          f"Delta {d_y * 100:+.2f} pp  KI95 [{by[lo] * 100:+.2f}, {by[hi] * 100:+.2f}]")
    print(f"  pass^{k}                 neu {sum(pn) / n:.1%}  Ref {sum(pr) / n:.1%}  "
          f"Delta {d_p * 100:+.2f} pp  KI95 [{bp[lo] * 100:+.2f}, {bp[hi] * 100:+.2f}]")
    print(f"  truncated               neu {rate(new, lambda r: bool(r.get('truncated'))):.1%}  "
          f"Ref {rate(ref, lambda r: bool(r.get('truncated'))):.1%}")
    print(f"  Fehler (teacher_error)  neu {rate(new, is_error):.1%}  Ref {rate(ref, is_error):.1%}")
    ok = abs(d_y) * 100 <= args.max_delta_pp
    print(f"CALIB {'BESTANDEN' if ok else 'DURCHGEFALLEN'} (|Delta Yield| {abs(d_y) * 100:.2f} pp, "
          f"Grenze {args.max_delta_pp} pp)")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke", help="Format-/Plumbing-Gate fuer einen K=1-Smoke")
    s.add_argument("--input", required=True)
    c = sub.add_parser("calib", help="Eichung: gleicher Lauf auf neuem vs. bekanntem Rechner")
    c.add_argument("--input", required=True)
    c.add_argument("--reference", required=True)
    c.add_argument("--max-delta-pp", type=float, default=3.0)
    c.add_argument("--draws", type=int, default=10_000)
    c.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return smoke(a) if a.cmd == "smoke" else calib(a)


if __name__ == "__main__":
    sys.exit(main())

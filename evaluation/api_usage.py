"""
evaluation/api_usage.py
=======================
Token usage + cost of an API rollout file (rollout.py --backend anthropic writes `api_calls` per record: one entry per
model call with input / cache-read / cache-write / output tokens and the stop reason).

    .venv-tau2/bin/python evaluation/api_usage.py --input data/generated/eval/db_traces_bakeoff_dev_api_<model>.jsonl \
        --project-rollouts 960 3840          # extrapolate the cost to a K=2 / K=8 heldout run (480 tasks)

Prints calls, token sums, cache hit share, the output tokens per call against the per-turn cap (p50/p95/max and the
share of calls that used >= 90 % of max_tokens — the headroom check before anyone raises the cap), stop reasons,
refusal categories and the cost. Prices are list prices per MTok (configuration, not results); a model without a
price entry gets tokens but no cost.
"""

import argparse
import json
import sys
from collections import Counter

# USD per 1M tokens: input (uncached), 5-min cache write, cache read, output — claude-api skill, as of 2026-11-12
PRICES = {
    "claude-sonnet-5-5": {"input": 2.00, "cache_write": 2.50, "cache_read": 0.10, "output": 10.00},
    "claude-haiku-5-5": {"input": 0.10, "cache_write": 0.125, "cache_read": 0.01, "output": 0.50},
    "claude-opus-5-5": {"input": 4.00, "cache_write": 5.00, "cache_read": 0.20, "output": 20.00},
}
NEAR_CAP = 0.90


def price_for(model: str | None) -> dict | None:
    """Exact id first, then the longest known prefix (dated snapshot ids)."""
    if not model:
        return None
    if model in PRICES:
        return PRICES[model]
    hits = [k for k in PRICES if model.startswith(k)]
    return PRICES[max(hits, key=len)] if hits else None


def call_cost(c: dict, price: dict) -> float:
    return (c.get("input_tokens", 0) * price["input"] + c.get("cache_creation_input_tokens", 0) * price["cache_write"]
            + c.get("cache_read_input_tokens", 0) * price["cache_read"] + c.get("output_tokens", 0) * price["output"]) / 1e6


def _pct(sorted_vals: list, q: float) -> int:
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))] if sorted_vals else 0


def summarize(records: list[dict]) -> dict:
    """records: dicts with `api_calls` (+ `gen_params` for model/max_tokens). Records without api_calls are skipped."""
    recs = [r for r in records if r.get("api_calls") is not None]
    s = Counter()
    outs, stops, refusals, models, unpriced = [], Counter(), Counter(), Counter(), set()
    cost, max_tokens = 0.0, None
    for r in recs:
        gp = r.get("gen_params") or {}
        max_tokens = max_tokens or gp.get("max_tokens")
        for c in r["api_calls"]:
            model = c.get("model") or gp.get("model")
            models[model] += 1
            s["calls"] += 1
            for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens"):
                s[k] += c.get(k, 0) or 0
            outs.append(c.get("output_tokens", 0) or 0)
            stops[c.get("stop_reason") or "?"] += 1
            if c.get("stop_reason") == "refusal":
                refusals[c.get("refusal_category") or "ohne Kategorie"] += 1
            price = price_for(model) or price_for(gp.get("model"))
            if price is None:
                unpriced.add(model)
            else:
                cost += call_cost(c, price)
    outs.sort()
    n, calls = len(recs), s["calls"]
    prompt = s["input_tokens"] + s["cache_read_input_tokens"] + s["cache_creation_input_tokens"]
    near = sum(o >= NEAR_CAP * max_tokens for o in outs) if max_tokens else 0
    return {"rollouts": n, "calls": calls, "calls_per_rollout": calls / max(1, n),
            "input_tokens": s["input_tokens"], "cache_read_input_tokens": s["cache_read_input_tokens"],
            "cache_creation_input_tokens": s["cache_creation_input_tokens"], "output_tokens": s["output_tokens"],
            "cache_hit_share": s["cache_read_input_tokens"] / max(1, prompt),
            "out_p50": _pct(outs, 0.50), "out_p95": _pct(outs, 0.95), "out_max": outs[-1] if outs else 0,
            "max_tokens": max_tokens, "near_cap_share": near / max(1, calls),
            "stop_reasons": dict(stops), "refusals": dict(refusals), "models": dict(models),
            "cost_usd": None if unpriced else cost, "cost_per_rollout_usd": None if unpriced else cost / max(1, n),
            "unpriced_models": sorted(m for m in unpriced if m)}


def mlflow_metrics(s: dict) -> dict:
    """Numeric subset for MLflow (rollout.py at the end of a run, log_heldout_mlflow.py after the fact)."""
    m = {"api_calls": s["calls"], "api_calls_per_rollout": s["calls_per_rollout"],
         "api_input_tokens": s["input_tokens"], "api_cache_read_tokens": s["cache_read_input_tokens"],
         "api_cache_write_tokens": s["cache_creation_input_tokens"], "api_output_tokens": s["output_tokens"],
         "api_cache_hit_share": s["cache_hit_share"], "api_out_p95": s["out_p95"],
         "api_near_cap_share": s["near_cap_share"], "api_refusals": sum(s["refusals"].values())}
    if s["cost_usd"] is not None:
        m["api_cost_usd"] = s["cost_usd"]
        m["api_cost_per_rollout_usd"] = s["cost_per_rollout_usd"]
    return m


def load(path: str) -> list[dict]:
    recs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r.pop("messages", None)
                recs.append(r)
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--project-rollouts", type=int, nargs="*", default=[],
                    help="extrapolate the cost per rollout to N rollouts (e.g. 960 = heldout K=2, 3840 = K=8)")
    a = ap.parse_args(argv)
    s = summarize(load(a.input))
    if not s["rollouts"]:
        print(f"API-USAGE {a.input}: keine Datensaetze mit api_calls (kein API-Lauf?)")
        return 2
    print(f"API-USAGE {a.input}: {s['rollouts']} Rollouts, {s['calls']} Aufrufe ({s['calls_per_rollout']:.2f} je Rollout)"
          f" | Modell(e): {', '.join(f'{k} {v}' for k, v in s['models'].items())}")
    print(f"  Token   Input {s['input_tokens']:,} | Cache-Lesen {s['cache_read_input_tokens']:,} | "
          f"Cache-Schreiben {s['cache_creation_input_tokens']:,} | Output {s['output_tokens']:,}")
    print(f"  Cache-Trefferquote (Lesen / alle Prompt-Token)  {s['cache_hit_share']:.1%}")
    cap = s["max_tokens"]
    print(f"  Output je Aufruf  p50 {s['out_p50']} | p95 {s['out_p95']} | max {s['out_max']}"
          + (f"  — Grenze {cap}, Aufrufe >= {NEAR_CAP:.0%} davon: {s['near_cap_share']:.1%}" if cap else ""))
    print("  stop_reason: " + ", ".join(f"{k} {v}" for k, v in sorted(s["stop_reasons"].items(), key=lambda x: -x[1])))
    if s["refusals"]:
        print("  Ablehnungen: " + ", ".join(f"{k} {v}" for k, v in s["refusals"].items()))
    if s["cost_usd"] is None:
        print(f"  Kosten: kein Preis fuer {s['unpriced_models']} (PRICES ergaenzen)")
        return 0
    print(f"  Kosten  {s['cost_usd']:.4f} $ gesamt | {s['cost_per_rollout_usd']:.5f} $ je Rollout (Listenpreise)")
    for n in a.project_rollouts:
        print(f"  Hochrechnung {n} Rollouts: {s['cost_per_rollout_usd'] * n:.2f} $")
    return 0


if __name__ == "__main__":
    sys.exit(main())

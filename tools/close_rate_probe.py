#!/usr/bin/env python3
"""Close-Rate-Probe fuer trainierte thinking-Studenten.

Misst, was die EX-Eval NICHT speichert: terminiert das Modell unter reinem greedy
sauber? Schickt die Testfragen an einen laufenden vLLM-Serve und zaehlt pro Antwort:
  - </think>-CLOSE-RATE  (Reasoning sauber geschlossen)
  - hit max_tokens       (lief ins Limit = Loop-Verdacht)
  - trigram-rep          (Repetitions-Signal)
Kein repetition_penalty -> ehrlicher Lackmustest, ob die Daten das Looping behoben haben.

Lauf auf dem Host gegen den gemappten Port:
    python3 tools/close_rate_probe.py --api-base http://localhost:8000/v1 \
        --n 100 --max-tokens 4096 --out data/final/eval/<merged>/close_rate.json
"""
import argparse, json, re, sys, time, urllib.request, collections, statistics
from concurrent.futures import ThreadPoolExecutor

SYS_THINK = """You are an expert SQL query writer. Given a natural language question and a database schema, write the correct SQL query that answers the question.

Think through the problem step by step before writing the SQL:
1. Identify which tables are needed
2. Determine what joins are required
3. Figure out what filters, aggregations, or ordering to apply
4. Write the final SQL

Output your reasoning in <think>...</think> tags, then the SQL query."""

def trig(t):
    w = re.findall(r"\w+", t.lower())
    if len(w) < 6: return 1
    return max(collections.Counter(tuple(w[i:i+3]) for i in range(len(w)-2)).values())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None, help="served-model-name; default = erstes aus /v1/models")
    ap.add_argument("--test-file", default="data/final/test_clean.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default, honest loop test)")
    ap.add_argument("--repetition-penalty", type=float, default=None, help="loop-fix: e.g. 1.1; None = vLLM default (off)")
    ap.add_argument("--top-p", type=float, default=None, help="loop-fix: e.g. 0.95; None = vLLM default")
    ap.add_argument("--top-k", type=int, default=None, help="loop-fix: e.g. 20; None = vLLM default")
    ap.add_argument("--label", default="", help="run label for the printout/report")
    ap.add_argument("--out", default=None, help="optional: JSON-Report rausschreiben")
    args = ap.parse_args()

    base = args.api_base.rstrip("/")
    model = args.model
    if not model:
        d = json.load(urllib.request.urlopen(base + "/models", timeout=30))
        model = d["data"][0]["id"]
    print(f"model = {model}")

    rows = [json.loads(l) for l in open(args.test_file) if l.strip()][:args.n]

    def probe(r):
        q = r.get("question") or r.get("query") or ""
        sc = r.get("schema", "")
        payload = {"model": model, "messages": [
            {"role": "system", "content": SYS_THINK},
            {"role": "user", "content": f"Database schema:\n{sc}\n\nQuestion: {q}"}],
            "max_tokens": args.max_tokens, "temperature": args.temperature,
            "chat_template_kwargs": {"enable_thinking": True}}
        if args.repetition_penalty is not None: payload["repetition_penalty"] = args.repetition_penalty
        if args.top_p is not None: payload["top_p"] = args.top_p
        if args.top_k is not None: payload["top_k"] = args.top_k
        body = json.dumps(payload).encode()
        try:
            req = urllib.request.Request(base + "/chat/completions", data=body,
                headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
            t0 = time.perf_counter()
            d = json.load(urllib.request.urlopen(req, timeout=600))
            dt = time.perf_counter() - t0
            ch = d["choices"][0]; out = ch["message"]["content"]; fr = ch["finish_reason"]
            ctok = (d.get("usage") or {}).get("completion_tokens")
            return {"fr": fr, "closed": "</think>" in out, "len": len(out), "trig": trig(out),
                    "dt": round(dt, 2), "ctok": ctok}
        except Exception as e:
            return {"err": str(e)[:80]}

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        res = list(ex.map(probe, rows))
    wall = time.perf_counter() - t_start
    ok = [r for r in res if "err" not in r]
    err = [r for r in res if "err" in r]
    n = len(ok)
    if n == 0:
        print("ALLE requests fehlgeschlagen:", err[:3]); sys.exit(1)
    closed = sum(r["closed"] for r in ok)
    hit_cap = sum(r["fr"] == "length" for r in ok)
    loops = sum((not r["closed"]) or r["fr"] == "length" or r["trig"] > 15 for r in ok)
    ctoks = [r["ctok"] for r in ok if r.get("ctok") is not None]
    dts = [r["dt"] for r in ok if r.get("dt") is not None]
    rep = {
        "label": args.label,
        "sampling": {"temperature": args.temperature, "repetition_penalty": args.repetition_penalty,
                     "top_p": args.top_p, "top_k": args.top_k, "max_tokens": args.max_tokens},
        "n_ok": n, "errs": len(err),
        "close_rate": round(closed / n, 3),
        "hit_cap_rate": round(hit_cap / n, 3),
        "loop_rate": round(loops / n, 3),
        "len_p50": int(statistics.median(r["len"] for r in ok)),
        "len_max": max(r["len"] for r in ok),
        "ctok_p50": int(statistics.median(ctoks)) if ctoks else None,
        "ctok_max": max(ctoks) if ctoks else None,
        "trig_p50": int(statistics.median(r["trig"] for r in ok)),
        "trig_max": max(r["trig"] for r in ok),
        "gen_s_per_req_p50": round(statistics.median(dts), 2) if dts else None,
        "gen_s_per_req_max": round(max(dts), 2) if dts else None,
        "wall_s_total": round(wall, 1),
        "wall_s_per_prompt": round(wall / n, 2),
    }
    print(f"\n=== CLOSE-RATE {args.label} ===")
    print(f"  sampling: temp={args.temperature} rep_pen={args.repetition_penalty} top_p={args.top_p} top_k={args.top_k} cap={args.max_tokens}")
    print(f"  </think>-CLOSE: {closed}/{n} = {100*rep['close_rate']:.0f}%")
    print(f"  hit max_tokens: {hit_cap}/{n} = {100*rep['hit_cap_rate']:.0f}%")
    print(f"  loop-rate:      {loops}/{n} = {100*rep['loop_rate']:.0f}%")
    print(f"  ctok p50/max: {rep['ctok_p50']}/{rep['ctok_max']}   trig p50/max: {rep['trig_p50']}/{rep['trig_max']}")
    print(f"  gen-time: {rep['wall_s_total']}s total, {rep['wall_s_per_prompt']}s/prompt (conc {args.concurrency}); per-req p50/max {rep['gen_s_per_req_p50']}/{rep['gen_s_per_req_max']}s")
    if args.out:
        json.dump(rep, open(args.out, "w"), indent=1)
        print(f"  -> {args.out}")

if __name__ == "__main__":
    main()

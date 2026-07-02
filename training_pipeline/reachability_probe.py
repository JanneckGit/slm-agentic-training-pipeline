"""
training_pipeline/reachability_probe.py
=============================
Stage 1 reachability filter for the RLVR-GRPO pilot (paper Prop 3.3 + GRPO needs intra-group
variance, else advantage=0). Scores each weak-category candidate with the SFT-14B-thinking model
(the merged checkpoint, served via the validated vLLM image): k rollouts at temperature 1.0,
reward = loose-EX (the SAME evaluation.reward.score_sql used in training). Keep only prompts with
0 < success_rate < 1 (all-pass = nothing to learn, all-fail = no signal / off-task gold), BIASED
toward ~50% (k/2 = max intra-group variance; 1/k and (k-1)/k give weak gradient). Emits the
reachable set + a probe report with the success-rate histogram and the variance>0 assertion (R4).

Run inside the training container, pointed at the vLLM service:
    python3 training_pipeline/reachability_probe.py --api-base http://vllm:8000/v1 \
        --k 8 --max-tokens 3072 --target-per-cat 300
Uses the EXACT thinking prompt the 14B-thinking student was SFT'd on (SYS_THINK + enable_thinking).
"""
import argparse
import json
import statistics
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from evaluation.reward import score_sql

# EXACT system prompt the thinking student was trained on (see adapter data_snapshot).
SYS_THINK = """You are an expert SQL query writer. Given a natural language question and a database schema, write the correct SQL query that answers the question.

Think through the problem step by step before writing the SQL:
1. Identify which tables are needed
2. Determine what joins are required
3. Figure out what filters, aggregations, or ordering to apply
4. Write the final SQL

Output your reasoning in <think>...</think> tags, then the SQL query."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://vllm:8000/v1")
    ap.add_argument("--config", default="config/pipeline_config.yaml",
                    help="pulls max_tokens / reward_timeout from the grpo: block so the probe "
                         "matches TRAINING conditions (CLI flags still override)")
    ap.add_argument("--model", default=None, help="served-model-name; default = first /v1/models")
    ap.add_argument("--candidates", default="data/final/grpo/weak_candidates.jsonl")
    ap.add_argument("--out", default="data/final/grpo/weak_prompts_reachable.jsonl")
    ap.add_argument("--report", default="data/final/grpo/weak_prompts_probe_report.json")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="default: grpo.max_completion_length from --config (match training)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.1, help="loop-fix (Pre-Pilot [1]); 1.0 = off")
    ap.add_argument("--top-p", type=float, default=0.95, help="loop-fix: Qwen3 default (1.0 = permissive)")
    ap.add_argument("--top-k", type=int, default=20, help="loop-fix: Qwen3 default (-1 = permissive)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--target-per-cat", type=int, default=300)
    ap.add_argument("--reward-timeout", type=float, default=None,
                    help="default: grpo.reward_timeout_s from --config (match training)")
    ap.add_argument("--limit", type=int, default=0, help="cap candidates (0=all) for a bounded run")
    ap.add_argument("--checkpoint", default=None,
                    help="resumable per-candidate JSONL (default <out>.partial.jsonl); a restart skips "
                         "candidates already scored here and continues. Delete it to start fresh.")
    args = ap.parse_args()

    # Match TRAINING conditions: max_tokens + reward_timeout come from the grpo: block unless the CLI
    # overrides them. A prompt judged "reachable" at a looser cap/timeout than training is a mismatch
    # (reachable@5s/4096 can still fail@3s/2048), so the probe defaults to the grpo training values.
    grpo = (yaml.safe_load(open(args.config)) if Path(args.config).exists() else {}).get("grpo") or {}
    if args.max_tokens is None:
        args.max_tokens = int(grpo.get("max_completion_length", 2048))
    if args.reward_timeout is None:
        args.reward_timeout = float(grpo.get("reward_timeout_s", 3))
    print(f"[config] max_tokens={args.max_tokens} reward_timeout={args.reward_timeout}s "
          f"(from grpo: block of {args.config}; matches training)")

    base = args.api_base.rstrip("/")
    model = args.model
    if not model:
        d = json.load(urllib.request.urlopen(base + "/models", timeout=30))
        model = d["data"][0]["id"]
    print(f"model = {model}")

    cands = [json.loads(l) for l in open(args.candidates) if l.strip()]
    if args.limit:
        bycat = defaultdict(list)
        for r in cands:
            bycat[r["complexity"]].append(r)
        per = max(1, args.limit // max(1, len(bycat)))
        cands = [r for rs in bycat.values() for r in rs[:per]]
    print(f"candidates: {len(cands)}  {dict(Counter(r['complexity'] for r in cands))}")

    # resumable checkpoint: each scored candidate is appended as it completes; on restart, everything
    # already scored here is skipped and the run continues from where it stopped (Pre-Pilot [4]).
    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path(args.out + ".partial.jsonl")
    def _ckey(r): return (r.get("question", ""), r.get("gold_sql", ""))
    scored = []
    if ckpt_path.exists():
        scored = [json.loads(l) for l in open(ckpt_path) if l.strip()]
        done = {_ckey(s) for s in scored}
        before = len(cands)
        cands = [r for r in cands if _ckey(r) not in done]
        print(f"[resume] {ckpt_path}: {len(scored)} already scored -> skip {before - len(cands)}, "
              f"{len(cands)} remaining")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_f = open(ckpt_path, "a")

    def probe(r):
        q, sc = r.get("question", ""), r.get("schema", "")
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": SYS_THINK},
                {"role": "user", "content": f"Database schema:\n{sc}\n\nQuestion: {q}"},
            ],
            "max_tokens": args.max_tokens, "temperature": args.temperature, "n": args.k,
            "repetition_penalty": args.repetition_penalty,   # loop-fix control (Pre-Pilot [1])
            "top_p": args.top_p, "top_k": args.top_k,
            "chat_template_kwargs": {"enable_thinking": True},
        }).encode()
        try:
            req = urllib.request.Request(
                base + "/chat/completions", data=body,
                headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
            d = json.load(urllib.request.urlopen(req, timeout=1800))
            comps = [c["message"]["content"] for c in d["choices"]]
            n_term = sum((c.get("finish_reason") == "stop") for c in d["choices"])  # eos, not cap
            passes = sum(score_sql(c, r.get("gold_sql", ""), sc, args.reward_timeout) for c in comps)
            return {**r, "k": len(comps), "n_pass": int(passes), "n_term": int(n_term),
                    "success_rate": (passes / len(comps)) if comps else 0.0}
        except Exception as e:
            return {**r, "err": str(e)[:140]}

    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(probe, r) for r in cands]
        for j, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            scored.append(res)
            ckpt_f.write(json.dumps(res, ensure_ascii=False) + "\n"); ckpt_f.flush()   # resumable
            if j % 10 == 0 or j == len(cands):
                done_reach = sum(1 for s in scored if "err" not in s and 0 < s.get("n_pass", 0) < s.get("k", 1))
                print(f"  progress {j}/{len(cands)} this-run  reachable_total={done_reach}  scored_total={len(scored)}", flush=True)
    ckpt_f.close()

    ok = [s for s in scored if "err" not in s]
    errs = [s for s in scored if "err" in s]
    allpass = [s for s in ok if s["n_pass"] == s["k"]]
    allfail = [s for s in ok if s["n_pass"] == 0]
    reach = [s for s in ok if 0 < s["n_pass"] < s["k"]]

    # 50%-bias selection: order reachable by |success_rate-0.5| ascending, keep per-cat target.
    bycat = defaultdict(list)
    for s in reach:
        bycat[s["complexity"]].append(s)
    selected = []
    for cat, rs in bycat.items():
        rs.sort(key=lambda s: abs(s["success_rate"] - 0.5))
        selected += rs[:args.target_per_cat]

    # R4 gate: every selected group has 0<n_pass<k (variance>0). Mean Bernoulli var p(1-p).
    assert all(0 < s["n_pass"] < s["k"] for s in selected), "zero-variance prompt selected!"
    mean_var = statistics.mean(s["success_rate"] * (1 - s["success_rate"]) for s in selected) if selected else 0.0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for s in selected:
            f.write(json.dumps({k: s[k] for k in (
                "question", "schema", "gold_sql", "complexity", "domain",
                "k", "n_pass", "success_rate")}, ensure_ascii=False) + "\n")

    hist = Counter(s["n_pass"] for s in ok)
    # termination rate (eos, not cap) + per-category success histogram — loop-fix health (Pre-Pilot [4])
    term_by_cat, tot_by_cat, hist_by_cat = defaultdict(int), defaultdict(int), defaultdict(Counter)
    for s in ok:
        c = s["complexity"]
        term_by_cat[c] += s.get("n_term", 0)
        tot_by_cat[c] += s["k"]
        hist_by_cat[c][s["n_pass"]] += 1
    termination_rate_per_cat = {c: round(term_by_cat[c] / tot_by_cat[c], 3) for c in tot_by_cat}
    n_pass_histogram_per_cat = {
        c: {str(i): hist_by_cat[c].get(i, 0) for i in range(args.k + 1)} for c in hist_by_cat}
    n_term_total, n_gen_total = sum(term_by_cat.values()), sum(tot_by_cat.values())
    report = {
        "model": model, "k": args.k, "temperature": args.temperature,
        "sampling": {"repetition_penalty": args.repetition_penalty, "top_p": args.top_p,
                     "top_k": args.top_k, "max_tokens": args.max_tokens},
        "n_candidates": len(scored), "n_ok": len(ok), "n_err": len(errs),
        "n_all_pass_dropped": len(allpass), "n_all_fail_dropped": len(allfail),
        "n_reachable": len(reach), "n_selected": len(selected),
        "candidates_per_cat": dict(Counter(s["complexity"] for s in ok)),
        "selected_per_cat": dict(Counter(s["complexity"] for s in selected)),
        "reachable_per_cat": dict(Counter(s["complexity"] for s in reach)),
        "termination_rate_overall": round(n_term_total / n_gen_total, 3) if n_gen_total else 0.0,
        "termination_rate_per_cat": termination_rate_per_cat,
        "n_pass_histogram": {str(i): hist.get(i, 0) for i in range(args.k + 1)},
        "n_pass_histogram_per_cat": n_pass_histogram_per_cat,
        "mean_intra_group_variance_selected": round(mean_var, 4),
        "variance_gate_pass": bool(selected) and mean_var > 0,
    }
    json.dump(report, open(args.report, "w"), indent=1)
    print("\n=== REACHABILITY REPORT ===")
    print(json.dumps(report, indent=1))
    if errs:
        print("sample err:", errs[0].get("err"))


if __name__ == "__main__":
    main()

"""
data_pipeline/backfill_toolace_think.py
=======================================
STAGE 2 of the ToolACE leg: generate the missing REASONING for the preselection, verify every generation
against ToolACE's gold call, and emit the verified subset for the SFT mix.

Why this exists: all 10,649 parseable ToolACE rows end on an assistant turn WITHOUT thinking. The Qwen3
template answers that with its canonical empty `<think>\\n\\n</think>` wrapper (the `loop.last` branch),
and that span IS trained (ToolACE has one user turn, so everything after it counts). The leg therefore
taught "open thinking, close it immediately" on every record — in the API-call setting, which is exactly
the BFCL setting where ep2 reasoned in 0 of 100 cases.

METHOD — free rollout, then gold verification (NOT rationalization):
The teacher sees only the system prompt and the question, thinks, and emits its calls; we keep the record
only if its calls match the gold. Showing the teacher the answer first would produce post-hoc
justification — reasoning that argues for a conclusion instead of deriving it, which is precisely the kind
of thinking that collapses at eval time. ToolACE has no executable environment (its observations are
invented), so the gold call is the ONLY signal — but it is a deterministic one, of the same kind as
BFCL's AST check. Failures are dropped, not repaired: with 4,057 parallel rows for ~2,500 slots we can
afford it, and a row a thinking 35B teacher cannot hit is usually ambiguous anyway.

STRICTNESS — `matches_exact` = the shared `_matches` PLUS key-set equality. Measured: 79.4% of the 17,775
gold calls already use every schema parameter, so strict and lenient are identical there; the difference
only exists for the remaining 20.6%. db_bahn can afford the lenient subset match because its state
verifier (DB comparison + action checks) catches whatever the call comparison lets through — ToolACE has
no such backstop, so the leniency would arrive without its safety net. `extra_keys_only` in the taxonomy
below is the measurement that decides whether to relax this later.

THREE ARTIFACTS (mirrors the db_bahn chain gen_tasks -> scored traces -> format_traj):
    toolace_preselect.jsonl      gold calls, no thinking      (input, from convert_toolace.py)
    toolace_backfill_raw.jsonl   EVERY run incl. failures     (the evidence; --filter-only reads it)
    toolace_chat.jsonl           verified only, with thinking (goes into the mix)

Usage (tau2 venv — the only one carrying mlflow; the rollout helpers import cleanly there):
    PYTHONPATH=. .venv-tau2/bin/python data_pipeline/backfill_toolace_think.py --mlflow
    ... --per-class 25 --classes multi,irrelevance,parallel2,parallel4plus   # stratified pilot
    ... --filter-only                                                       # re-filter, no GPU needed
"""

import argparse
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

try:  # soft dep, same pattern as rollout.py: tracking must never break a run
    import mlflow
except ImportError:
    mlflow = None

from data_pipeline.common import args_dict, load_config, write_jsonl
from sdg_pipeline.db_bahn.rollout import (_matches, extract_think, make_teacher_call, parse_tool_calls,
                                          resolve_teacher)

# Failure taxonomy — one label per turn. `ok` is the only one that survives into the chat file.
VERDICTS = ("ok", "no_think", "no_call", "asked_back", "unexpected_call", "wrong_tool", "wrong_args",
            "extra_keys_only", "partial_parallel", "extra_call", "truncated", "call_error")


# --------------------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------------------
def _norm(v):
    """Comparison normal form. Gold comes from ast.literal_eval (native types), the teacher answers in
    JSON — so "5" vs 5 and "true" vs True are formatting drift, not disagreement."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        try:
            return float(s)
        except ValueError:
            return s
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _pairs(calls: list[dict]) -> list[tuple[str, dict]]:
    return [(c["function"]["name"], {k: _norm(v) for k, v in args_dict(c["function"]["arguments"]).items()})
            for c in calls or []]


def matches_exact(name: str, args: dict, ref: dict) -> bool:
    """The shared `_matches` (same-tool + gold keys agree) PLUS key-set equality, so a teacher-invented
    extra argument is not silently accepted. See the module docstring for why ToolACE needs the tighter
    form while db_bahn does not."""
    return _matches(name, args, ref) and set(args) == set(ref["arguments"])


def _all_match(got: list, gold: list, exact: bool) -> bool:
    """Order-insensitive: a parallel bundle is a SET of calls, the model may emit them in any order."""
    pool = list(gold)
    for name, args in got:
        ref = next((g for g in pool
                    if (matches_exact if exact else _matches)(name, args, {"name": g[0], "arguments": g[1]})),
                   None)
        if ref is None:
            return False
        pool.remove(ref)
    return not pool


def classify_turn(got_calls: list, gold_calls: list, visible: str) -> str:
    """One verdict per turn — the label ordering is chosen so the FIRST thing that went wrong wins."""
    got, gold = _pairs(got_calls), _pairs(gold_calls)
    if not gold:                                    # irrelevance row: the correct move is NOT to call
        return "ok" if not got else "unexpected_call"
    if not got:
        return "asked_back" if visible.strip() else "no_call"
    if Counter(n for n, _ in got) != Counter(n for n, _ in gold):
        if {n for n, _ in got} <= {n for n, _ in gold} and len(got) < len(gold):
            return "partial_parallel"               # emitted 2 of 4 — the bundling failure we watch for
        return "wrong_tool"
    if len(got) != len(gold):
        return "partial_parallel" if len(got) < len(gold) else "extra_call"
    if _all_match(got, gold, exact=True):
        return "ok"
    if _all_match(got, gold, exact=False):
        return "extra_keys_only"                    # the strict-vs-lenient measurement
    return "wrong_args"


# --------------------------------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------------------------------
def trainable_turns(messages: list[dict]) -> list[int]:
    """Assistant turns AFTER the last user message — the only ones the collator takes gradient on
    (training_pipeline/collator_multiturn.py, final_turns_only). Generating thinking for the earlier
    turns of the 377 multi-user records would be wasted: the template drops it and the mask hides it."""
    lastq = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant" and i > lastq]


def _wire(m: dict) -> dict:
    """Our on-disk format keeps `function.arguments` as a DICT (as db_bahn and AReaL do); the OpenAI wire
    format demands a JSON STRING. Replaying a prior tool-call turn verbatim made vLLM answer HTTP 400
    ('Input should be a valid string') — which killed exactly the multi-turn records."""
    if not m.get("tool_calls"):
        return m
    return {**m, "tool_calls": [
        {**tc, "function": {**tc["function"],
                            "arguments": tc["function"]["arguments"]
                            if isinstance(tc["function"]["arguments"], str)
                            else json.dumps(tc["function"]["arguments"], ensure_ascii=False)}}
        for tc in m["tool_calls"]]}


def prompt_upto(messages: list[dict], k: int, thinks: dict[int, str]) -> list[dict]:
    """Context for generating turn k: the gold prefix, with thinking already generated for earlier turns
    inlined — so the teacher sees the same context shape the student will see at inference."""
    out = []
    for i, m in enumerate(messages[:k]):
        if i in thinks:
            out.append(_wire({**m, "content": f"<think>\n{thinks[i]}\n</think>\n\n{m.get('content') or ''}"}))
        else:
            out.append(_wire(m))
    return out


def backfill_record(ex: dict, call) -> dict:
    """Generate + verify every trainable turn of one record. Aborts at the first failing turn (the record
    is lost anyway — no point paying for the rest)."""
    msgs = [dict(m) for m in ex["messages"]]
    targets = trainable_turns(msgs)
    thinks, turns = {}, []
    for k in targets:
        gold = msgs[k].get("tool_calls") or []
        t0 = time.time()
        try:
            content, native, finish = call(prompt_upto(msgs, k, thinks), [])
        except Exception as e:  # noqa: BLE001 — a dead turn must not kill the pass
            turns.append({"turn": k, "verdict": "call_error", "error": f"{type(e).__name__}: {e}"[:200]})
            break
        think, _ = extract_think(content)
        text, got, tail = parse_tool_calls(content, native)
        verdict = "truncated" if finish == "length" else classify_turn(got, gold, text)
        if verdict == "ok" and not think:
            verdict = "no_think"                    # a record without reasoning is the bug, not the fix
        turns.append({"turn": k, "verdict": verdict, "n_gold": len(gold), "n_got": len(got),
                      "think_chars": len(think), "tail_chars": tail, "secs": round(time.time() - t0, 1)})
        if verdict != "ok":
            turns[-1]["got_text"] = text[:1500]     # evidence for the post-mortem
            turns[-1]["got_calls"] = got[:8]
            break
        thinks[k] = think
        msgs[k] = {**msgs[k], "content": f"<think>\n{think}\n</think>\n\n{msgs[k].get('content') or ''}"}

    verified = len(turns) == len(targets) and all(t["verdict"] == "ok" for t in turns)
    return {"messages": msgs, "turns": turns,
            "_meta": {**ex["_meta"], "verified": verified,
                      "verdict": "ok" if verified else (turns[-1]["verdict"] if turns else "no_target")}}


# --------------------------------------------------------------------------------------------------
# filter (also reachable standalone via --filter-only — no GPU needed)
# --------------------------------------------------------------------------------------------------
def read_raw(path: str) -> tuple[list[dict], int]:
    """Read the raw file, SKIPPING unparseable lines. A hard kill mid-write leaves a truncated last
    line; a strict reader would then die on the very restart that is supposed to recover the run."""
    rows, broken = [], 0
    for line in open(path):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            broken += 1
    return rows, broken


def rescue_prefix(rec: dict) -> list[dict] | None:
    """Salvage a failed record by cutting behind its last verified turn.

    backfill_record aborts at the FIRST failing turn, so the `ok` turns are always a prefix — everything
    up to and including the last of them carries generated thinking and passed gold verification. The
    shortened record ends on an assistant turn with <think>, either a call turn or a text answer; both
    are the normal shape for this leg (all raw rows end on an assistant turn). Same idea as
    trim_option_a in convert_areal.py. Costs no GPU time: those generations are already paid for.
    Returns None if not even the first turn verified."""
    ok = [t["turn"] for t in rec["turns"] if t["verdict"] == "ok"]
    return rec["messages"][:ok[-1] + 1] if ok else None


def filter_raw(raw_path: str, out_path: str) -> tuple[int, int, int, Counter]:
    """raw -> chat file. Returns (n_raw, n_verified, n_rescued, verdict counts)."""
    rows, broken = read_raw(raw_path)
    if broken:
        print(f"  [filter] skipped {broken} unparseable raw line(s)")
    verdicts = Counter(r["_meta"]["verdict"] for r in rows)
    keep, n_verified, n_rescued = [], 0, 0
    for r in rows:
        meta = {k: v for k, v in r["_meta"].items() if k not in ("verified", "verdict")}
        if r["_meta"]["verified"]:
            msgs, n_verified = r["messages"], n_verified + 1
        else:
            msgs = rescue_prefix(r)
            if msgs is None:
                continue
            meta["rescued"] = True
            n_rescued += 1
        meta["n_turns_kept"] = len(trainable_turns(msgs))
        keep.append({"messages": msgs, "_meta": meta})
    write_jsonl(keep, out_path)
    return len(rows), n_verified, n_rescued, verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline_config.yaml")
    ap.add_argument("--preselect", default="data/generated/toolace_preselect.jsonl")
    ap.add_argument("--raw", default="data/generated/toolace_backfill_raw.jsonl")
    ap.add_argument("--out", default="data/generated/toolace_chat.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N records after shuffling")
    ap.add_argument("--per-class", type=int, default=None, help="pilot: N records per sel_class")
    ap.add_argument("--classes", default=None, help="comma-separated sel_class filter (with --per-class)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--circuit", type=int, default=25,
                    help="abort after N consecutive call_error records (dead teacher); 0 = off")
    ap.add_argument("--filter-only", action="store_true", help="rebuild --out from --raw, no teacher")
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--mlflow-experiment", default="toolace_think_backfill")
    ap.add_argument("--mlflow-run-name", default=None)
    args = ap.parse_args()

    if args.filter_only:
        n, ver, resc, verdicts = filter_raw(args.raw, args.out)
        print(f"filter-only: {n} raw -> {ver} verified + {resc} rescued = {ver+resc} "
              f"({100*(ver+resc)/max(1,n):.1f}%) -> {args.out}")
        print("  verdicts:", dict(verdicts.most_common()))
        return

    import random
    pool = [json.loads(l) for l in open(args.preselect) if l.strip()]
    rng = random.Random(args.seed)
    if args.per_class:
        wanted = set((args.classes or "").split(",")) if args.classes else None
        by_class = {}
        for ex in pool:
            by_class.setdefault(ex["_meta"]["sel_class"], []).append(ex)
        sel = []
        for cls, items in sorted(by_class.items()):
            if wanted and cls not in wanted:
                continue
            rng.shuffle(items)
            sel += items[:args.per_class]
        pool = sel
    else:
        rng.shuffle(pool)
        if args.limit:
            pool = pool[:args.limit]

    done = set()
    if Path(args.raw).exists():  # resume: append-only, keyed by the stable preselect id
        prev, broken = read_raw(args.raw)
        done = {r["_meta"]["id"] for r in prev}
        if broken:
            print(f"  [resume] skipped {broken} unparseable raw line(s) — those records rerun")
    todo = [ex for ex in pool if ex["_meta"]["id"] not in done]
    print(f"backfill: {len(pool)} selected, {len(done)} already done -> {len(todo)} to generate")
    if not todo:
        print("  nothing to do")
        return

    cfg = resolve_teacher(load_config(args.config), args.api_base, args.model)
    call = make_teacher_call(cfg)
    print(f"  teacher {cfg['model']} @ {cfg['api_base']} | temp {cfg['temperature']} "
          f"top_p {cfg['top_p']} top_k {cfg['top_k']} presence {cfg['presence_penalty']} "
          f"thinking {cfg['enable_thinking']} max_tokens {cfg['max_tokens']}")

    lock, results, t0 = threading.Lock(), [], time.time()
    Path(args.raw).parent.mkdir(parents=True, exist_ok=True)
    fh = open(args.raw, "a")
    # CIRCUIT BREAKER. A dead vLLM server does not look like a failure from the outside: every record
    # turns into a call_error within seconds, so the raw file GROWS and any progress-based watchdog
    # stays quiet — the run would "finish" with 6,631 failures and exit 0. Counting CONSECUTIVE
    # call_errors keeps single HTTP hiccups harmless while a dead server costs at most --circuit records.
    tripped = threading.Event()
    streak = [0]

    def work(ex):
        if tripped.is_set():
            return
        rec = backfill_record(ex, call)
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            results.append(rec)
            n = len(results)
            if rec["_meta"]["verdict"] == "call_error":
                streak[0] += 1
                if args.circuit and streak[0] >= args.circuit and not tripped.is_set():
                    tripped.set()
                    err = next((t.get("error") for t in reversed(rec["turns"]) if t.get("error")), "?")
                    print(f"!!!! CIRCUIT BREAKER — {streak[0]} consecutive call_error, teacher looks dead.\n"
                          f"!!!!   last error: {err}\n"
                          f"!!!!   aborting; rerun resumes from the raw file", flush=True)
            else:
                streak[0] = 0
            if n % 25 == 0 or n == len(todo):
                ok = sum(1 for r in results if r["_meta"]["verified"])
                print(f"  {n}/{len(todo)} | verified {ok} ({100*ok/n:.0f}%) | "
                      f"{n/((time.time()-t0)/3600):.0f} rec/h", flush=True)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool_exec:
        list(pool_exec.map(work, todo))
    fh.close()
    if tripped.is_set():
        filter_raw(args.raw, args.out)  # keep whatever was good before the server died
        raise SystemExit(3)

    # ---- summary ----
    wall_h = (time.time() - t0) / 3600
    verdicts = Counter(r["_meta"]["verdict"] for r in results)
    per_class = Counter(r["_meta"]["sel_class"] for r in results)
    ok_class = Counter(r["_meta"]["sel_class"] for r in results if r["_meta"]["verified"])
    n_ok = sum(ok_class.values())
    think_chars = [t["think_chars"] for r in results for t in r["turns"] if t["verdict"] == "ok"]
    print(f"\n=== BACKFILL {len(results)} records in {wall_h:.2f} h ({len(results)/max(wall_h,1e-9):.0f} rec/h) ===")
    print(f"  verified {n_ok} = {100*n_ok/len(results):.1f}%")
    print("  verdicts:", dict(verdicts.most_common()))
    for cls in sorted(per_class):
        print(f"    {cls:14s} {ok_class[cls]:5d}/{per_class[cls]:5d} = {100*ok_class[cls]/per_class[cls]:5.1f}%")
    if think_chars:
        print(f"  think chars/turn: mean {sum(think_chars)/len(think_chars):.0f} max {max(think_chars)}")

    n_raw, n_ver, n_resc, _ = filter_raw(args.raw, args.out)
    print(f"  -> {args.out}: {n_ver} verified + {n_resc} rescued = {n_ver+n_resc} records "
          f"(raw file holds all {n_raw})")

    if args.mlflow:
        if mlflow is None:
            print("  [mlflow] --mlflow set but mlflow not importable — skipping")
        else:
            try:
                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
                mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI")
                                        or Path("mlruns").resolve().as_uri())
                mlflow.set_experiment(args.mlflow_experiment)
                with mlflow.start_run(run_name=args.mlflow_run_name or f"backfill_{len(results)}"):
                    mlflow.log_params({"teacher": cfg["model"], "temperature": cfg["temperature"],
                                       "top_p": cfg["top_p"], "top_k": cfg["top_k"],
                                       "presence_penalty": cfg["presence_penalty"],
                                       "max_tokens": cfg["max_tokens"], "strictness": "exact_keyset",
                                       "n_selected": len(pool), "n_generated": len(results),
                                       "concurrency": args.concurrency, "seed": args.seed,
                                       "per_class": args.per_class or "", "limit": args.limit or ""})
                    # strict vs. rescue kept apart — otherwise the effect of the prefix salvage would
                    # disappear inside a single number
                    mlflow.log_metrics({"yield_strict": n_ok / len(results),
                                        "yield_with_rescue": (n_ver + n_resc) / max(1, n_raw),
                                        "n_rescued": n_resc, "n_verified": n_ok,
                                        "records_per_hour": len(results) / max(wall_h, 1e-9),
                                        "n_generated": len(results)})
                    for cls in per_class:
                        mlflow.log_metric(f"yield_{cls}", ok_class[cls] / per_class[cls])
                    for v in VERDICTS:
                        mlflow.log_metric(f"rate_{v}", verdicts.get(v, 0) / len(results))
                    if think_chars:
                        mlflow.log_metric("think_chars_mean", sum(think_chars) / len(think_chars))
                print(f"  [mlflow] logged -> {args.mlflow_experiment}")
            except Exception as e:  # noqa: BLE001
                print(f"  [mlflow] logging failed ({type(e).__name__}: {e}) — continuing")


def _selftest():
    """Every taxonomy label must be reachable and unambiguous — run with --selftest (no GPU, no server)."""
    def c(name, **args):
        return {"id": "x", "type": "function", "function": {"name": name, "arguments": args}}

    gold2 = [c("weather", city="Boston"), c("weather", city="NYC")]
    cases = [
        ("exact single",      [c("weather", city="Boston")], [c("weather", city="Boston")], "", "ok"),
        ("parallel swapped",  [c("weather", city="NYC"), c("weather", city="Boston")], gold2, "", "ok"),
        ("type drift 5/'5'",  [c("page", n="5")], [c("page", n=5)], "", "ok"),
        ("missing call",      [c("weather", city="Boston")], gold2, "", "partial_parallel"),
        ("one call too many", gold2 + [c("weather", city="LA")], gold2, "", "wrong_tool"),
        ("wrong arg value",   [c("weather", city="Berlin")], [c("weather", city="Boston")], "", "wrong_args"),
        ("extra key only",    [c("weather", city="Boston", unit="c")], [c("weather", city="Boston")], "",
         "extra_keys_only"),
        ("wrong tool",        [c("news", city="Boston")], [c("weather", city="Boston")], "", "wrong_tool"),
        ("asked back",        [], [c("weather", city="Boston")], "Which city?", "asked_back"),
        ("silent no call",    [], [c("weather", city="Boston")], "", "no_call"),
        ("irrelevance ok",    [], [], "No function fits.", "ok"),
        ("irrelevance called", [c("weather", city="Boston")], [], "", "unexpected_call"),
    ]
    for label, got, gold, text, want in cases:
        have = classify_turn(got, gold, text)
        assert have == want, f"{label}: expected {want}, got {have}"

    # trainable_turns must ignore role:"tool" (Qwen renders those as user turns — a token-level scan
    # would collapse the rule to "only the very last turn"; here we work in message space)
    msgs = [{"role": "system", "content": ""}, {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [c("f")]},
            {"role": "tool", "tool_call_id": "x", "content": "obs"},
            {"role": "assistant", "content": "done"}]
    assert trainable_turns(msgs) == [2, 4], trainable_turns(msgs)
    msgs2 = msgs + [{"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"}]
    assert trainable_turns(msgs2) == [6], trainable_turns(msgs2)

    assert set(x[4] for x in cases) <= set(VERDICTS)

    # prefix salvage: the ok-turns are always a prefix (backfill_record aborts at the first failure)
    two_turn = [{"role": "system", "content": ""}, {"role": "user", "content": "q"},
                {"role": "assistant", "content": "<think>\nA\n</think>\n\n", "tool_calls": [c("f")]},
                {"role": "tool", "tool_call_id": "x", "content": "obs"},
                {"role": "assistant", "content": "gold answer"}]
    late = {"messages": two_turn, "turns": [{"turn": 2, "verdict": "ok"}, {"turn": 4, "verdict": "wrong_tool"}]}
    kept = rescue_prefix(late)
    assert kept is not None and len(kept) == 3, kept
    assert kept[-1]["role"] == "assistant" and "<think>" in kept[-1]["content"], "must end on a thinking turn"
    assert trainable_turns(kept) == [2], trainable_turns(kept)

    early = {"messages": two_turn, "turns": [{"turn": 2, "verdict": "no_call"}]}
    assert rescue_prefix(early) is None, "nothing verified -> nothing to salvage"

    print(f"backfill self-test OK — {len(cases)} verdict cases + trainable-turn selection + prefix salvage")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        main()

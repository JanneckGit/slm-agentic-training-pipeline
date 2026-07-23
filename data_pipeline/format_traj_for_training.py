"""
data_pipeline/format_traj_for_training.py
=========================================
Phase 5 of Plan (B): convert VERIFIED db_bahn trajectories into the multi-turn chat training JSONL.

Input : data/generated/sdg/db_traces_sft_train_<teacher>.jsonl (rollout records with score + messages)
Output: data/generated/legs/db_traces_chat.jsonl — one {"messages": [...], "_meta": {...}} per line.

Keeps ONLY verified (score==1.0) records. Messages stay OpenAI-style (system with embedded tool schemas,
user ticket, assistant turns with content "<plan>…</plan>" + tool_calls, role:"tool" observations, final
German answer) — the chat template renders tool_calls natively at tokenization time. The assistant-only
loss mask is applied later by training_pipeline/collator_multiturn.py.

Sanity gates per record: >=1 assistant turn with tool_calls, a non-empty final answer, valid roles.

With --dropped-out (needs --split-file) a second JSONL lists the split tasks that made it into NO chat
trace — task-wise, so a task counts as missing only if none of its rollouts survived. Those tasks are
training-unseen and are therefore candidates for the Stage-2 GRPO pool (build_grpo_pool.py resolves a
bare task_id against tasks.json / answer_keys.json, which hold all splits).

Usage:
    PYTHONPATH=. python3 data_pipeline/format_traj_for_training.py \
        --input data/generated/sdg/db_traces_sft_train_q36-35b-a3b.jsonl \
        --output data/generated/legs/db_traces_chat.jsonl \
        --split-file data/raw/db_sandbox/split_tasks.json --split sft_train \
        --dropped-out data/generated/sdg/db_failed-for-SFT_rl-candidates.jsonl
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from data_pipeline.common import args_dict, final_answer

# wave-3 degeneration thresholds — keep in sync with sdg_pipeline/db_bahn/rollout.py
# (duplicated on purpose: this module must not drag the tau2 env import chain in)
DEGEN_MAX_THINK_CHARS = 12_000
DEGEN_MAX_DUP8_RATIO = 0.5

# gen_tasks.py:202 sets expected_calls = len(oracle_calls) — those are the STATE-CHANGING calls only
# (they exist for the verifier's gold replay). Templates whose ticket demands a fallback check
# ("first choice X, else Y") also need READ calls to make that choice, and those are not in
# oracle_calls. The >=3x filter below compares against n_tool_calls, which counts ALL calls — so
# correct traces were dropped (measured on the wave-3.5 run: 102 tasks, 77 crew_doppelt + 25 crew,
# with an observed maximum of 4 resp. 3 calls, i.e. no flailing at all).
# Values = the minimal sensible path INCLUDING reads. If expected_calls is ever fixed at the source
# in gen_tasks.py, this table MUST go — otherwise the correction applies twice and the filter goes slack.
EXPECTED_CALLS_OVERRIDE = {
    "t_action_crew_doppelt": 3,   # check current crew + check first choice + assign
    "t_action_crew":         2,   # check + assign
}

VALID_ROLES = {"system", "user", "assistant", "tool"}


def convert(rec: dict) -> dict | None:
    # NOTE: the verified-only gate (score==1.0) lives in main() — records arriving here passed it.
    msgs = rec.get("messages") or []
    if not msgs or any(m.get("role") not in VALID_ROLES for m in msgs):
        return None
    has_call = any(m.get("role") == "assistant" and m.get("tool_calls") for m in msgs)
    if not has_call or not final_answer(msgs):
        return None
    # normalize: drop empty-content keys, keep tool_call_id on tool turns.
    # tool_call arguments -> DICT (the Qwen chat template renders arguments as a mapping, not a JSON string).
    out = []
    for m in msgs:
        e = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls"):
            e["tool_calls"] = [{"id": tc.get("id", ""), "type": "function",
                                "function": {"name": tc["function"]["name"],
                                             "arguments": args_dict(tc["function"].get("arguments"))}}
                               for tc in m["tool_calls"]]
        if m.get("tool_call_id"):
            e["tool_call_id"] = m["tool_call_id"]
        out.append(e)
    return {
        "messages": out,
        "_meta": {
            "task_id": rec["task_id"], "template": rec["template"],
            "injected": rec["injected"], "fault": rec.get("fault", "none"), "teacher": rec["teacher"],
            "expected_calls": rec.get("expected_calls", 0),
            "turns": rec["score"]["turns_used"], "n_tool_calls": rec["score"]["n_tool_calls"],
            "replan": bool(rec["score"].get("replan_occurred")),
            # B0/B2: self-correction. A B2-harvested trace ("recovery_mode":"harvest") kept the mistake +
            # its recovery — count it even when the mistake was a wrong-but-clean search (n_tool_errors==0,
            # which the verifier's self_recovery misses).
            "recovery_mode": rec.get("recovery_mode", "direct"),
            "self_recovery": bool(rec["score"].get("self_recovery")) or rec.get("recovery_mode") == "harvest",
            "emergent_recovery": (bool(rec["score"].get("self_recovery")) or rec.get("recovery_mode") == "harvest")
                                 and rec.get("fault", "none") in ("none", "state"),
            "source": "db_bahn_verified_trace", "lang": "de",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/generated/sdg/db_traces_sft_train_q36-35b-a3b.jsonl")
    ap.add_argument("--output", default="data/generated/legs/db_traces_chat.jsonl")
    ap.add_argument("--split-file", default=None,
                    help="split_tasks.json; if set, keep only records whose task_id is in --split")
    ap.add_argument("--split", default="sft_train",
                    help="split name to filter to (only with --split-file) — guards against leaking "
                         "tasks that moved to rl/heldout after a split regen")
    ap.add_argument("--dropped-out", default=None,
                    help="JSONL of split tasks that made it into NO chat trace (task-wise, not "
                         "record-wise): they are training-unseen and thus RL pool candidates. "
                         "Requires --split-file.")
    args = ap.parse_args()

    keep_ids = None
    if args.split_file:
        keep_ids = set(json.load(open(args.split_file))[args.split])
    if args.dropped_out and keep_ids is None:
        ap.error("--dropped-out needs --split-file (the task universe to diff against)")

    n_in = n_out = 0
    drops = {"unverified": 0, "structure": 0, "off_split": 0,
             "truncated": 0, "degen": 0, "inefficient": 0}
    written_ids = set()
    # per task: the LEAST-bad drop reason seen + its record (a task may have several rollouts —
    # top-up, restart — and only counts as missing if none of them survived)
    rejected: dict[str, tuple[str, dict]] = {}
    RANK = {"unverified": 0, "truncated": 1, "degen": 2, "inefficient": 3, "structure": 4}

    def reject(rec: dict, reason: str):
        drops[reason] += 1
        tid = rec.get("task_id")
        if tid is None:
            return
        prev = rejected.get(tid)
        if prev is None or RANK[reason] > RANK[prev[0]]:  # keep the closest-to-passing attempt
            rejected[tid] = (reason, rec)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for line in open(args.input):
            n_in += 1
            rec = json.loads(line)
            if keep_ids is not None and rec.get("task_id") not in keep_ids:
                drops["off_split"] += 1
                continue
            if rec.get("score", {}).get("score") != 1.0:
                reject(rec, "unverified")
                continue
            # --- wave-3 hard gates (the verifier is outcome-based and blind to these) -------
            if rec.get("truncated"):  # turn hit the token cap -> almost always a think loop
                reject(rec, "truncated")
                continue
            d = rec.get("degen") or {}
            if d.get("think_ngram_dup_ratio", 0.0) > DEGEN_MAX_DUP8_RATIO \
                    or d.get("max_think_chars", 0) > DEGEN_MAX_THINK_CHARS:
                reject(rec, "degen")
                continue
            exp = EXPECTED_CALLS_OVERRIDE.get(rec.get("template")) or int(rec.get("expected_calls") or 0)
            if exp > 0 and rec["score"]["n_tool_calls"] >= 3 * exp:  # flail: >=3x the oracle path
                reject(rec, "inefficient")
                continue
            conv = convert(rec)
            if conv is None:
                reject(rec, "structure")
                continue
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            written_ids.add(rec["task_id"])
            n_out += 1
    print(f"in {n_in} -> chat {n_out} (drops: {drops}) -> {out_path}")

    if args.dropped_out:
        missing = sorted(keep_ids - written_ids)
        n_att = Counter()
        for line in open(args.input):  # attempts per task (cheap second pass, ids only)
            r = json.loads(line)
            if r.get("task_id") in keep_ids:
                n_att[r["task_id"]] += 1
        dpath = Path(args.dropped_out)
        dpath.parent.mkdir(parents=True, exist_ok=True)
        by_reason = Counter()
        with open(dpath, "w") as f:
            for tid in missing:
                hit = rejected.get(tid)
                if hit is None:  # task has no rollout at all (aborted run) — still a candidate
                    by_reason["no_rollout"] += 1
                    f.write(json.dumps({"task_id": tid, "drop_reason": "no_rollout",
                                        "n_attempts": 0}, ensure_ascii=False) + "\n")
                    continue
                reason, rec = hit
                by_reason[reason] += 1
                sc = rec.get("score") or {}
                f.write(json.dumps({
                    "task_id": tid,                       # first field: greppable without a parser
                    "template": rec.get("template"),
                    "drop_reason": reason,
                    "n_attempts": n_att.get(tid, 0),
                    "best_score": sc.get("score"),
                    "n_tool_calls": sc.get("n_tool_calls"),
                    "expected_calls": rec.get("expected_calls"),
                    "fault": rec.get("fault", "none"),
                    "failed_components": sorted(k for k, v in (sc.get("components") or {}).items() if not v),
                }, ensure_ascii=False) + "\n")
        print(f"   RL candidates: {len(missing)} split tasks in no chat trace "
              f"({dict(by_reason)}) -> {dpath}")


if __name__ == "__main__":
    main()

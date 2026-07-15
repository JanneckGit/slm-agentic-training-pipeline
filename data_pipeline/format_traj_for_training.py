"""
data_pipeline/format_traj_for_training.py
=========================================
Phase 5 of Plan (B): convert VERIFIED db_bahn trajectories into the multi-turn chat training JSONL.

Input : data/generated/db_traces_sft_train_<teacher>.jsonl (rollout records with score + messages)
Output: data/final/db_traces_chat.jsonl — one {"messages": [...], "_meta": {...}} per line.

Keeps ONLY verified (score==1.0) records. Messages stay OpenAI-style (system with embedded tool schemas,
user ticket, assistant turns with content "<plan>…</plan>" + tool_calls, role:"tool" observations, final
German answer) — the chat template renders tool_calls natively at tokenization time. The assistant-only
loss mask is applied later by training_pipeline/collator_multiturn.py.

Sanity gates per record: >=1 assistant turn with tool_calls, a non-empty final answer, valid roles.

Usage:
    PYTHONPATH=. python3 data_pipeline/format_traj_for_training.py \
        --input data/generated/db_traces_sft_train_q36-35b-a3b.jsonl \
        --output data/final/db_traces_chat.jsonl
"""

import argparse
import json
from pathlib import Path

from data_pipeline.common import args_dict, final_answer

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
    ap.add_argument("--input", default="data/generated/db_traces_sft_train_q36-35b-a3b.jsonl")
    ap.add_argument("--output", default="data/final/db_traces_chat.jsonl")
    ap.add_argument("--split-file", default=None,
                    help="split_tasks.json; if set, keep only records whose task_id is in --split")
    ap.add_argument("--split", default="sft_train",
                    help="split name to filter to (only with --split-file) — guards against leaking "
                         "tasks that moved to rl/heldout after a split regen")
    args = ap.parse_args()

    keep_ids = None
    if args.split_file:
        keep_ids = set(json.load(open(args.split_file))[args.split])

    n_in = n_out = 0
    drops = {"unverified": 0, "structure": 0, "off_split": 0}
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
                drops["unverified"] += 1
                continue
            conv = convert(rec)
            if conv is None:
                drops["structure"] += 1
                continue
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"in {n_in} -> chat {n_out} (drops: {drops}) -> {out_path}")


if __name__ == "__main__":
    main()

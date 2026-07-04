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
    python3 data_pipeline/format_traj_for_training.py \
        --input data/generated/db_traces_sft_train_q36-35b-a3b.jsonl \
        --output data/final/db_traces_chat.jsonl
"""

import argparse
import json
from pathlib import Path

VALID_ROLES = {"system", "user", "assistant", "tool"}


def convert(rec: dict) -> dict | None:
    if rec.get("score", {}).get("score") != 1.0:
        return None
    msgs = rec.get("messages") or []
    if not msgs or any(m.get("role") not in VALID_ROLES for m in msgs):
        return None
    has_call = any(m.get("role") == "assistant" and m.get("tool_calls") for m in msgs)
    final = next((m for m in reversed(msgs)
                  if m.get("role") == "assistant" and not m.get("tool_calls")
                  and (m.get("content") or "").strip()), None)
    if not has_call or final is None:
        return None
    # normalize: drop empty-content keys, keep tool_call_id on tool turns.
    # tool_call arguments -> DICT (the Qwen chat template renders arguments as a mapping, not a JSON string).
    out = []
    for m in msgs:
        e = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = tc["function"]
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args or "{}")
                    except json.JSONDecodeError:
                        args = {}
                tcs.append({"id": tc.get("id", ""), "type": "function",
                            "function": {"name": fn["name"], "arguments": args}})
            e["tool_calls"] = tcs
        if m.get("tool_call_id"):
            e["tool_call_id"] = m["tool_call_id"]
        out.append(e)
    return {
        "messages": out,
        "_meta": {
            "task_id": rec["task_id"], "template": rec["template"],
            "injected": rec["injected"], "teacher": rec["teacher"],
            "turns": rec["score"]["turns_used"], "n_tool_calls": rec["score"]["n_tool_calls"],
            "replan": bool(rec["score"].get("replan_occurred")),
            "source": "db_bahn_verified_trace", "lang": "de",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/generated/db_traces_sft_train_q36-35b-a3b.jsonl")
    ap.add_argument("--output", default="data/final/db_traces_chat.jsonl")
    args = ap.parse_args()

    n_in = n_out = 0
    drops = {"unverified": 0, "structure": 0}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for line in open(args.input):
            n_in += 1
            rec = json.loads(line)
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

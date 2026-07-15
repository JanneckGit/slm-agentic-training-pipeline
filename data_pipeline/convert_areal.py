"""
data_pipeline/convert_areal.py
==============================
Convert the AReaL tau2 SFT data (per-turn rows) into the unified db_bahn chat format for the 4-leg mix.

AReaL ships 33,531 per-turn rows: each row = {messages: prior context, answer: the target turn, metadata}.
~12 rows of one dialog share a growing context. We REASSEMBLE one full episode per dialog (last row's
messages + its answer) so every dialog is trained ONCE (no 12x prefix duplication).

Pipeline per episode:
  1. EPISODE-LEVEL correctness filter: keep only dialogs where EVERY turn has metadata.correct == 1.
  2. Reassemble: messages(last row) + [answer as final assistant].
  3. Reasoning -> `<think>…</think>` prefix in content (Qwen3 native; template drops a separate field).
  4. tool_calls flat {name,arguments} -> OpenAI {id,type,function:{name,arguments:DICT}}; tool msgs get
     matched tool_call_id (by order after their assistant turn).
  5. Inject the domain's tau2 tool schemas into the system prompt (db_bahn <tools> style) — AReaL policies
     ship WITHOUT tool defs, so the student would otherwise call unseen tool names.
  6. Option-A trim @max-len: if the episode exceeds the budget, cut at the last assistant turn WITHOUT
     tool_calls (= a final message = clean sub-task boundary) whose token prefix fits. Never ends on an
     open tool_call; nothing is dropped unless even the first sub-task exceeds the budget.

Runs in the TRAINING container (needs the Qwen3-4B tokenizer for the exact trim). Reads the tool schemas
from data/raw/areal/tau2_tools_blocks.json (pre-dumped from the tau2 package, which lives only in .venv-tau2).

Usage (training container):
    python3 data_pipeline/convert_areal.py --max-seq-len 12288 \
        --out data/generated/areal_chat.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

TOOLS_SUFFIX = (
    "\n\n# Tools\n\nYou are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{tools_block}\n</tools>"
)


def domain_of(dialog_id: str) -> str:
    head = dialog_id.split("_")[0]
    return head if head in ("airline", "retail") else "telecom"  # telecom ids are numeric


def norm_tool_calls(tcs, ctr):
    """flat AReaL {name,arguments} -> OpenAI {id,type,function}; returns (calls, ids) with fresh ids."""
    out, ids = [], []
    for tc in tcs or []:
        fn = tc.get("function", tc)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                args = {}
        cid = f"call_{ctr[0]}"
        ctr[0] += 1
        out.append({"id": cid, "type": "function",
                    "function": {"name": fn["name"], "arguments": args if isinstance(args, dict) else {}}})
        ids.append(cid)
    return out, ids


def reasoning_of(m: dict) -> str:
    # context assistant turns use "reasoning"; the final answer uses "thinking"
    return (m.get("thinking") or m.get("reasoning") or "").strip()


def build_messages(rows_last, tools_block):
    """Reassemble one episode into unified chat messages. rows_last = the max-turn_index row of the dialog."""
    src = list(rows_last["messages"])
    a = rows_last["answer"]
    src.append({"role": "assistant", "content": a.get("content") or "",
                "thinking": a.get("thinking"), "tool_calls": a.get("tool_calls")})
    ctr = [1]
    msgs, pending_ids = [], []
    for m in src:
        role = m["role"]
        if role == "system":
            msgs.append({"role": "system", "content": (m.get("content") or "")
                         + TOOLS_SUFFIX.format(tools_block=tools_block)})
        elif role == "user":
            msgs.append({"role": "user", "content": m.get("content") or ""})
        elif role == "tool":
            # attach the next pending tool_call id (parallel calls resolve in order)
            tcid = pending_ids.pop(0) if pending_ids else f"call_orphan_{ctr[0]}"
            if not pending_ids and tcid.startswith("call_orphan"):
                ctr[0] += 1
            msgs.append({"role": "tool", "tool_call_id": tcid, "content": m.get("content") or ""})
        elif role == "assistant":
            r = reasoning_of(m)
            content = (f"<think>\n{r}\n</think>\n\n" if r else "") + (m.get("content") or "")
            calls, ids = norm_tool_calls(m.get("tool_calls"), ctr)
            e = {"role": "assistant", "content": content}
            if calls:
                e["tool_calls"] = calls
                pending_ids = list(ids)  # the following tool turns bind to these
            msgs.append(e)
    return msgs


def toklen(tok, msgs, max_len):
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)


def trim_option_a(tok, msgs, max_len):
    """Cut at the last assistant turn WITHOUT tool_calls whose prefix fits max_len. Returns (msgs|None, trimmed)."""
    if toklen(tok, msgs, max_len) <= max_len:
        return msgs, False
    best = 0
    for i in range(1, len(msgs) + 1):
        m = msgs[i - 1]
        if m["role"] == "assistant" and not m.get("tool_calls"):
            if toklen(tok, msgs[:i], max_len) <= max_len:
                best = i
            else:
                break  # prefixes only grow
    if best == 0:
        return None, True  # even the first final-message sub-task exceeds budget -> drop
    return msgs[:best], True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="data/raw/areal/tau2_sft_train.jsonl")
    ap.add_argument("--tools", default="data/raw/areal/tau2_tools_blocks.json")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-seq-len", type=int, default=12288)
    ap.add_argument("--out", default="data/generated/areal_chat.jsonl")
    args = ap.parse_args()

    tools_blocks = json.load(open(args.tools))
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # pass 1: group rows by dialog -> correctness flags + the max-turn_index row
    flags = defaultdict(list)
    last_row = {}
    for line in open(args.sft):
        r = json.loads(line)
        m = r["metadata"]
        did = m["source_dialog_id"]
        flags[did].append(m["correct"])
        if did not in last_row or m["turn_index"] > last_row[did]["metadata"]["turn_index"]:
            last_row[did] = r
    full_ok = [d for d, cs in flags.items() if all(c == 1 for c in cs)]

    out, stats = [], defaultdict(int)
    for did in sorted(full_ok):
        dom = domain_of(did)
        msgs = build_messages(last_row[did], tools_blocks[dom])
        kept, trimmed = trim_option_a(tok, msgs, args.max_seq_len)
        if kept is None:
            stats["dropped_overlength"] += 1
            continue
        stats[f"dom_{dom}"] += 1
        stats["trimmed"] += int(trimmed)
        out.append({"messages": kept,
                    "_meta": {"source": "areal", "domain": dom, "dialog_id": did,
                              "lang": "en", "n_turns": len(kept), "trimmed": trimmed}})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"AReaL: {len(flags)} dialogs, {len(full_ok)} full-correct -> {len(out)} episodes "
          f"(dropped {stats['dropped_overlength']} overlength, trimmed {stats['trimmed']}) -> {args.out}")
    print("  domains:", {k[4:]: v for k, v in sorted(stats.items()) if k.startswith("dom_")})


if __name__ == "__main__":
    main()

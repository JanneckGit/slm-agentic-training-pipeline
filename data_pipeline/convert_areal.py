"""
data_pipeline/convert_areal.py
==============================
Convert the AReaL tau2 SFT data (per-turn rows) into the unified db_bahn chat format for the 3-leg mix.

AReaL ships 33,531 per-turn rows: each row = {messages: prior context, answer: the target turn, metadata}.
~12 rows of one dialog share a growing context. We REASSEMBLE one full episode per dialog (last row's
messages + its answer) so every dialog is trained ONCE (no 12x prefix duplication).

NOTE the cost of that fold: the Qwen3 template keeps <think> only for assistant turns AFTER the last user
message, so in a reassembled multi-turn dialog 78% of the assistant turns render think-less. Per-turn
expansion would preserve them but costs 9.8x forward tokens on this leg (measured). We keep the fold and
mask the context turns out of the loss instead — see training_pipeline/collator_multiturn.py
(final_turns_only). Expansion stays available as a sampled knob if the tau2 eval shows multi-turn weakness.

Two legs only: airline + retail. TELECOM IS DROPPED (2026-07-22) — it is structurally think-less
(94.8% of its turns carry no reasoning), the all-correct filter eats almost only telecom (500->180),
and 333/500 of its tasks are verbatim official tau2-bench task ids (7 of them in the test set) =
eval contamination.

Pipeline per episode:
  0. Domain filter: skip SKIP_DOMAINS (telecom).
  1. EPISODE-LEVEL correctness filter: keep only dialogs where EVERY turn has metadata.correct == 1.
  2. Reassemble: messages(last row) + [answer as final assistant].
  3. Reasoning -> `<think>…</think>` prefix in content (Qwen3 native; template drops a separate field).
  4. tool_calls flat {name,arguments} -> OpenAI {id,type,function:{name,arguments:DICT}}; tool msgs get
     matched tool_call_id (by order after their assistant turn).
  5. Inject the domain's tau2 tool schemas into the system prompt (db_bahn <tools> style) — AReaL policies
     ship WITHOUT tool defs, so the student would otherwise call unseen tool names.
  6. sanitize_system(): strip the PARALLEL-CALL BAN from the airline prompt (see below).
  7. Option-A trim @max-len: if the episode exceeds the budget, cut at the last assistant turn WITHOUT
     tool_calls (= a final message = clean sub-task boundary) whose token prefix fits. Never ends on an
     open tool_call; nothing is dropped unless even the first sub-task exceeds the budget.
     Runs AFTER the sanitizer so the trim measures the final text (the shorter prompt keeps more turns).

PROMPT SURGERY (sanitize_system, 2026-07-22) — why the airline prompt is edited:
AReaL's airline system prompt bans parallel tool calls twice (an own `CRITICAL RULES` block AND the
official policy sentence), while its own trajectories show >=2 calls per turn in 1,324 airline turns.
db_bahn instructs the exact opposite in 100% of its rows ("bundle independent queries") and lives it in
6,376 turns. The student does not learn "one call" or "many calls" from that — it learns that
instructions about call count are unreliable. That is the axis on which ep2 lost 9/20 BFCL parallel
cases (single-call attractor).

The ban is NOT a tau2-bench requirement: the official `AGENT_INSTRUCTION` (tau2/agent/llm_agent.py) says
nothing about call count, the harness executes multiple calls per turn natively (_execute_tool_calls
loops, MultiToolMessage exists for exactly this), and no reward component counts tool calls. AReaL even
deleted the clause from the *retail* policy themselves and only forgot airline. So we do not invent a
sentence — we put airline on the wording that already exists in the corpus (= retail's = the official
one). Rule B ("never a message AND a tool call in the same turn") is REAL and stays: a mixed message is
routed to the environment, so its text never reaches the user simulator.

Runs in the TRAINING container (needs the Qwen3-4B tokenizer for the exact trim). Reads the tool schemas
from data/raw/areal/tau2_tools_blocks.json (pre-dumped from the tau2 package, which lives only in .venv-tau2).
If that file is lost (data/raw is gitignored), reconstruct it exactly from a produced areal_chat.jsonl:
one <tools>…</tools> block per domain sits verbatim in each record's system prompt (done 2026-07-15).

Usage (training container):
    python3 data_pipeline/convert_areal.py --max-seq-len 12288 \
        --out data/generated/areal_chat.jsonl
"""

import argparse
import json
from collections import defaultdict

from transformers import AutoTokenizer

from data_pipeline.common import STUDENT_MODEL_DEFAULT, TOOLS_BLOCK_TMPL, args_dict, write_jsonl

TOOLS_SUFFIX = "\n\n" + TOOLS_BLOCK_TMPL

SKIP_DOMAINS = ("telecom",)  # think-less by construction + contaminates the official tau2 test set

# --- sanitize_system: exact literals, verified against the raw file (one variant per domain) -----------
# TARGET = tau2's official AGENT_INSTRUCTION, which retail already carries byte-identically (11,395 rows)
# and which the eval harness itself builds. No hand-written text.
AGENT_INSTRUCTION_BLOCK = (
    "<instructions>\n"
    "You are a customer service agent that helps the user according to the <policy> provided below.\n"
    "In each turn you can either:\n"
    "- Send a message to the user.\n"
    "- Make a tool call.\n"
    "You cannot do both at the same time.\n"
    "\n"
    "Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.\n"
    "</instructions>"
)
# SOURCE = AReaL's own addition on top of the airline prompt (12,842 rows, exactly one variant)
AIRLINE_INSTRUCTION_BLOCK = (
    "<instructions>\n"
    "You are a customer service agent that helps the user according to the <policy> provided below.\n"
    "\n"
    "CRITICAL RULES (you MUST follow these):\n"
    "1. In each turn, you can ONLY do ONE of these:\n"
    "   - Send a message to the user, OR\n"
    "   - Make exactly ONE tool call\n"
    "2. You CANNOT make multiple tool calls in a single turn - only ONE tool call per turn!\n"
    "3. You CANNOT send a message and make a tool call at the same time.\n"
    "\n"
    "Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.\n"
    "</instructions>"
)
# The policy sentence bundles BOTH rules; drop the leading parallel-ban clause, keep rule B — this is
# verbatim the cut AReaL made in their own retail policy ("...at a time, and if you take" -> "If you take").
PARALLEL_CLAUSE = "You should only make one tool call at a time, and if you make a tool call,"
PARALLEL_CLAUSE_FIXED = "If you make a tool call,"
# Post-condition: none of these may survive in ANY emitted system prompt (3rd = the official retail wording,
# which AReaL already stripped — guards against a future raw-data refresh reintroducing it).
BANNED_PHRASES = ("only ONE tool call per turn", "only make one tool call at a time",
                  "at most make one tool call at a time")


def domain_of(dialog_id: str) -> str:
    head = dialog_id.split("_")[0]
    return head if head in ("airline", "retail") else "telecom"  # telecom ids are numeric


def sanitize_system(content: str, dom: str, stats=None) -> str:
    """Remove the parallel-call ban from the system prompt; keep rule B. Literal replacements only —
    a missing/duplicated pattern is a HARD ERROR, so a raw-data change can never make this a silent no-op."""
    if dom == "airline":
        for src, dst, tag in ((AIRLINE_INSTRUCTION_BLOCK, AGENT_INSTRUCTION_BLOCK, "instructions"),
                              (PARALLEL_CLAUSE, PARALLEL_CLAUSE_FIXED, "policy")):
            n = content.count(src)
            if n != 1:
                raise ValueError(f"sanitize_system[{dom}/{tag}]: expected exactly 1 occurrence, found {n}")
            content = content.replace(src, dst)
            if stats is not None:
                stats[f"sanitized_{dom}_{tag}"] += 1
    for phrase in BANNED_PHRASES:                       # post-condition, checked for EVERY domain
        if phrase in content:
            raise ValueError(f"sanitize_system[{dom}]: parallel-call ban survived: {phrase!r}")
    if AGENT_INSTRUCTION_BLOCK not in content:          # rule B must still be stated
        raise ValueError(f"sanitize_system[{dom}]: official <instructions> block missing after sanitize")
    return content


def assert_target_matches_corpus(last_row: dict) -> None:
    """G0: the TARGET literal must equal the <instructions> block retail actually ships, so a typo in the
    constant above cannot slip through. Called once with any retail row."""
    src = last_row["messages"][0]["content"]
    i, j = src.index("<instructions>"), src.index("</instructions>") + len("</instructions>")
    if src[i:j] != AGENT_INSTRUCTION_BLOCK:
        raise ValueError("G0 FAILED: AGENT_INSTRUCTION_BLOCK != retail's <instructions> block in the raw data")


def norm_tool_calls(tcs, ctr):
    """flat AReaL {name,arguments} -> OpenAI {id,type,function}; returns (calls, ids) with fresh ids."""
    out, ids = [], []
    for tc in tcs or []:
        fn = tc.get("function", tc)
        cid = f"call_{ctr[0]}"
        ctr[0] += 1
        out.append({"id": cid, "type": "function",
                    "function": {"name": fn["name"], "arguments": args_dict(fn.get("arguments"))}})
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
            # attach the next pending tool_call id (parallel calls resolve in order).
            # call_orphan_* is DELIBERATE: telecom has USER-side tools (status bar, phone checks) whose
            # observations follow a user turn with no assistant tool_call (~175/2052 episodes).
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
    ap.add_argument("--model", default=STUDENT_MODEL_DEFAULT)
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
    retail_row = next((last_row[d] for d in sorted(full_ok) if domain_of(d) == "retail"), None)
    if retail_row is None:
        raise ValueError("G0 FAILED: no retail dialog in the raw data to verify the target literal against")
    assert_target_matches_corpus(retail_row)

    for did in sorted(full_ok):
        dom = domain_of(did)
        if dom in SKIP_DOMAINS:
            stats[f"skipped_{dom}"] += 1
            continue
        msgs = build_messages(last_row[did], tools_blocks[dom])
        msgs[0]["content"] = sanitize_system(msgs[0]["content"], dom, stats)  # BEFORE the trim
        kept, trimmed = trim_option_a(tok, msgs, args.max_seq_len)
        if kept is None:
            stats["dropped_overlength"] += 1
            continue
        stats[f"dom_{dom}"] += 1
        stats["trimmed"] += int(trimmed)
        out.append({"messages": kept,
                    "_meta": {"source": "areal", "domain": dom, "dialog_id": did,
                              "lang": "en", "n_turns": len(kept), "trimmed": trimmed}})

    write_jsonl(out, args.out)
    print(f"AReaL: {len(flags)} dialogs, {len(full_ok)} full-correct -> {len(out)} episodes "
          f"(skipped {sum(stats[f'skipped_{d}'] for d in SKIP_DOMAINS)} in {list(SKIP_DOMAINS)}, "
          f"dropped {stats['dropped_overlength']} overlength, trimmed {stats['trimmed']}) -> {args.out}")
    print("  domains:", {k[4:]: v for k, v in sorted(stats.items()) if k.startswith("dom_")})
    print("  sanitized:", {k: v for k, v in sorted(stats.items()) if k.startswith("sanitized_")})


if __name__ == "__main__":
    main()

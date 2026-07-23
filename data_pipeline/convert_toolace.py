"""
data_pipeline/convert_toolace.py
================================
Convert ToolACE (ShareGPT, bracket-DSL tool calls) into the unified db_bahn chat format — STAGE 1 of the
ToolACE leg. The output is a PRESELECTION with gold calls and no thinking; the thinking is generated
afterwards by data_pipeline/backfill_toolace_think.py, which verifies every generation against these gold
calls and only then writes data/generated/legs/toolace_chat.jsonl for the mix.

ToolACE's value = API-SCHEMA BREADTH (26,507 distinct tools -> generalization to unseen schemas, pays into
BFCL) + IRRELEVANCE rows ("if none of the functions can be used, point it out") + a large reservoir of
PARALLEL calls. Weakness: 93% single-turn, English, invented observations.

Per row:
  1. Extract tool schemas from the system prompt (ToolACE dialect: parameters.type "dict"; outer required:null)
     -> OpenAI {type:function, function:{...}} and rebuild the system in the db_bahn <tools> style, KEEPING the
     "if none applies, point it out" irrelevance instruction.
  2. Walk conversations: an assistant turn is a TOOL-CALL turn iff the next turn is `tool`. Parse its bracket
     DSL `[Name(k=v, ...), Name2(...)]` -> OpenAI tool_calls (depth-aware split; values via ast.literal_eval).
     Text assistant turns -> plain content. `tool` turns -> role:"tool" with matched tool_call_id.
  3. A row whose bracket DSL fails to parse is SKIPPED whole (counted) — a broken call would corrupt training.

SELECTION (reworked 2026-07-22) — priority follows what the leg is FOR, not what is easy to count:
  1. PARALLEL rows      (>=2 calls in ONE turn)   4,001 in the single pool + 56 already inside multi
  2. IRRELEVANCE rows   (0 calls)                 1,538 — the mix has no other source for "do not call"
  3. MULTI-TURN rows    (>=2 call TURNS)            638 — nice to have; AReaL now covers dialogue better
  4. plain single calls                             500 (seed 42) — so the simple one-call shape is not
                                                          unlearned; NOT for breadth (the parallel rows
                                                          already span 4,001 distinct API sets)
  = 6,677 records.

The old sampler counted call TURNS, so it preferred the 638 multi-turn rows — of which only 56 are
parallel — and then filled randomly from the single pool, catching 43% of the parallel rows by accident:
1,757 in the mix, 2,300 left on the floor. That is the direct repair material for the BFCL parallel
regression (ep2 lost 9/20), so it now comes first.

Usage:  PYTHONPATH=. python3 data_pipeline/convert_toolace.py     # -> data/generated/sdg/toolace_preselect.jsonl
"""

import argparse
import ast
import json
import random
import re
from collections import Counter

from data_pipeline.common import TOOLS_BLOCK_TMPL, write_jsonl

TOOLS_HDR = "Here is a list of functions in JSON format that you can invoke:"
# CALL-FORMAT BLOCK — mirrors db_bahn's "Arbeitsweise" section, in English. Two jobs:
#   1. It states the SAME bundling convention db_bahn instructs and that we cleared out of the AReaL
#      airline prompt, so all three legs now agree about call count. Being a PERMISSION, it does not
#      contradict the 62% single-call rows.
#   2. It names the WIRE FORMAT. Without it the teacher improvises a dialect — the first smoke produced
#      `<tool_code> print(f(...)) </tool_code>` and `<tool_calls><tool>{...}</tool></tool_calls>` with
#      semantically CORRECT arguments that our parser could not read, so 7 of 10 records failed
#      verification for a formatting reason. ToolACE's original prompt never specified a format because
#      its calls were never generated; ours are.
# Permanently in the prompt, not just at generation time — otherwise we would train on a prompt the
# teacher never saw, and the student would learn the convention from a different source than the data.
BUNDLING_CLAUSE = ("Call tools in exactly this format (one block per call, arguments as JSON); "
                   "bundle INDEPENDENT queries as SEVERAL blocks in the same turn:")
CALL_FORMAT = (BUNDLING_CLAUSE
               + '\n<tool_call>\n{"name": "tool_name", "arguments": {"argument": "value"}}\n</tool_call>')
# NOTE: CALL_FORMAT is appended AFTER .format() (see convert_system) — its JSON example is full of
# braces and would otherwise have to be brace-escaped, which the next edit would silently break.
SYS_PREAMBLE = (
    "You are a helpful assistant that can call functions to answer the user's question. "
    "If none of the functions can be used, or a required parameter is missing, say so instead of guessing.\n\n"
    + TOOLS_BLOCK_TMPL
)


def _split_top_level(s: str, sep: str) -> list[str]:
    """Split on `sep` only at bracket/brace/paren depth 0, respecting quotes."""
    out, buf, depth, quote, esc = [], [], 0, None, False
    for ch in s:
        if esc:
            buf.append(ch); esc = False; continue
        if ch == "\\":
            buf.append(ch); esc = True; continue
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'":
            quote = ch; buf.append(ch); continue
        if ch in "([{":
            depth += 1; buf.append(ch); continue
        if ch in ")]}":
            depth -= 1; buf.append(ch); continue
        if ch == sep and depth == 0:
            out.append("".join(buf)); buf = []; continue
        buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def parse_bracket_dsl(value: str, ctr: list) -> tuple[list, list] | None:
    """`[Name(k=v, ...), N2(...)]` -> (tool_calls, ids). None if it isn't a well-formed call block."""
    v = value.strip()
    if not (v.startswith("[") and v.endswith("]")):
        return None
    inner = v[1:-1].strip()
    if not inner:
        return None
    calls, ids = [], []
    for seg in _split_top_level(inner, ","):
        seg = seg.strip()
        m = re.match(r"^(.*?)\((.*)\)$", seg, re.S)  # name up to first '(' ; args inside outer parens
        if not m:
            return None
        name, argstr = m.group(1).strip(), m.group(2).strip()
        args = {}
        if argstr:
            for kv in _split_top_level(argstr, ","):
                if "=" not in kv:
                    return None
                k, _, val = kv.partition("=")
                try:
                    args[k.strip()] = ast.literal_eval(val.strip())
                except (ValueError, SyntaxError):
                    return None
        cid = f"call_{ctr[0]}"; ctr[0] += 1
        calls.append({"id": cid, "type": "function", "function": {"name": name, "arguments": args}})
        ids.append(cid)
    return calls, ids


def convert_system(system: str) -> str | None:
    """Extract ToolACE tool defs -> OpenAI <tools> block in db_bahn style."""
    i = system.find(TOOLS_HDR)
    if i < 0:
        return None
    raw = system[i + len(TOOLS_HDR):].strip()
    try:
        defs, _ = json.JSONDecoder().raw_decode(raw)  # parse only the leading JSON array; ignore trailing text
    except json.JSONDecodeError:
        return None
    if not isinstance(defs, list):
        return None
    lines = []
    for d in defs:
        params = d.get("parameters") or {"type": "object", "properties": {}}
        if params.get("type") == "dict":
            params = {**params, "type": "object"}
        lines.append(json.dumps({"type": "function", "function": {
            "name": d.get("name", ""), "description": d.get("description", ""), "parameters": params}},
            ensure_ascii=False))
    return SYS_PREAMBLE.format(tools_block="\n".join(lines)) + "\n\n" + CALL_FORMAT


def schema_names(system: str) -> set[str]:
    """Tool names the row's own system prompt declares — the ground truth for the gate in convert_row."""
    i = system.find(TOOLS_HDR)
    if i < 0:
        return set()
    try:
        defs, _ = json.JSONDecoder().raw_decode(system[i + len(TOOLS_HDR):].strip())
    except json.JSONDecodeError:
        return set()
    return {d.get("name", "") for d in defs} if isinstance(defs, list) else set()


def convert_row(row: dict) -> tuple[dict, int] | None:
    """One ToolACE row -> (unified example, n_call_turns). None if unparseable."""
    sysmsg = convert_system(row.get("system", ""))
    if sysmsg is None:
        return None
    known = schema_names(row.get("system", ""))
    conv = row["conversations"]
    msgs = [{"role": "system", "content": sysmsg}]
    ctr, pending, n_calls = [1], [], 0
    role_map = {"user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant", "tool": "tool"}
    for i, m in enumerate(conv):
        role = role_map.get(m.get("from"), m.get("from"))
        val = m.get("value") or ""
        if role == "user":
            msgs.append({"role": "user", "content": val})
        elif role == "tool":
            if not pending:  # tool turn without a preceding assistant call -> malformed row, drop it
                return None
            msgs.append({"role": "tool", "tool_call_id": pending.pop(0), "content": val})
        elif role == "assistant":
            v = val.strip()
            if v.startswith("[") and v.endswith("]"):  # ToolACE call block (may be final = no execution)
                parsed = parse_bracket_dsl(val, ctr)
                if parsed is None:
                    return None  # bracket-enclosed but corrupt -> drop whole row
                calls, ids = parsed
                msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
                pending = list(ids); n_calls += 1
            else:
                msgs.append({"role": "assistant", "content": val})
        else:
            return None
    if not any(m["role"] == "assistant" for m in msgs):
        return None
    # GATE: every parsed call name must exist in the row's own schema. Tool names containing parentheses
    # ("Vehicle Identification Number (VIN) Lookup") make the bracket DSL ambiguous — parse_bracket_dsl
    # splits at the first "(" and mangles BOTH the name and the first argument key, e.g.
    #   [Get Earnings Before Interest and Taxes (EBIT)(symbol="TSLA")]
    #   -> name "Get Earnings Before Interest and Taxes", args {"EBIT)(symbol": "TSLA"}
    # 70 rows / 98 calls (0.66%) are affected. Found because the teacher answered CORRECTLY and the
    # backfill flagged it as wrong_tool — our gold was the broken side. Dropping is proportionate here
    # (the leg has surplus); recovering them would mean matching the longest schema-name prefix inside
    # the DSL parser, which is the most fragile function in this file.
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            if tc["function"]["name"] not in known:
                return None
    return {"messages": msgs, "_meta": {"source": "toolace", "lang": "en", "n_call_turns": n_calls}}, n_calls


def max_calls_per_turn(ex: dict) -> int:
    """Widest assistant turn of a record — 0 = irrelevance, 1 = plain call, >=2 = parallel."""
    return max([len(m.get("tool_calls") or []) for m in ex["messages"] if m["role"] == "assistant"] or [0])


def sel_class(ex: dict) -> str:
    """Selection class — drives the pool priority, the stratified pilot and the per-class yield metrics."""
    par, turns = max_calls_per_turn(ex), ex["_meta"]["n_call_turns"]
    if par >= 4:
        return "parallel4plus"
    if par == 3:
        return "parallel3"
    if par == 2:
        return "parallel2"
    if turns == 0:
        return "irrelevance"
    return "multi" if turns >= 2 else "single"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/toolace/toolace.jsonl")
    ap.add_argument("--n-plain-single", type=int, default=500,
                    help="plain 1-call rows kept so the simple one-call shape is not unlearned")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/generated/sdg/toolace_preselect.jsonl")
    args = ap.parse_args()

    parallel, irrelevance, multi, plain_single, skipped = [], [], [], [], 0
    for idx, line in enumerate(open(args.data)):
        res = convert_row(json.loads(line))
        if res is None:
            skipped += 1; continue
        ex, nc = res
        # stable id: the raw file is pinned, so the row index survives a re-run (resume key downstream)
        ex["_meta"]["id"] = f"toolace_{idx}"
        ex["_meta"]["max_calls_per_turn"] = max_calls_per_turn(ex)
        ex["_meta"]["sel_class"] = sel_class(ex)
        if ex["_meta"]["max_calls_per_turn"] >= 2:
            parallel.append(ex)
        elif nc == 0:
            irrelevance.append(ex)
        elif nc >= 2:
            multi.append(ex)
        else:
            plain_single.append(ex)

    rng = random.Random(args.seed)
    rng.shuffle(plain_single)
    kept = parallel + irrelevance + multi + plain_single[:args.n_plain_single]
    rng.shuffle(kept)

    write_jsonl(kept, args.out)
    parsed = len(parallel) + len(irrelevance) + len(multi) + len(plain_single)
    print(f"ToolACE: parsed {parsed} / skipped {skipped} ({100*skipped/(parsed+skipped):.1f}% unparseable)")
    print(f"  pools: parallel={len(parallel)} irrelevance={len(irrelevance)} multi={len(multi)} "
          f"plain_single={len(plain_single)} (kept {min(len(plain_single), args.n_plain_single)})")
    print(f"  preselect {len(kept)} -> {args.out}")
    print("  by class:", dict(sorted(Counter(e["_meta"]["sel_class"] for e in kept).items())))


if __name__ == "__main__":
    main()

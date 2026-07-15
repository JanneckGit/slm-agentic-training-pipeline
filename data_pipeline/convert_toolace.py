"""
data_pipeline/convert_toolace.py
================================
Convert ToolACE (ShareGPT, bracket-DSL tool calls) into the unified db_bahn chat format for the 3-leg mix.

ToolACE's value = API-SCHEMA BREADTH (26,507 distinct tools -> generalization to unseen schemas, pays into
BFCL) + IRRELEVANCE rows ("if none of the functions can be used, point it out"). Weakness: 93% single-turn,
English, invented observations. We keep all multi-tool-call convos + a sample of single/irrelevance rows.

Per row:
  1. Extract tool schemas from the system prompt (ToolACE dialect: parameters.type "dict"; outer required:null)
     -> OpenAI {type:function, function:{...}} and rebuild the system in the db_bahn <tools> style, KEEPING the
     "if none applies, point it out" irrelevance instruction.
  2. Walk conversations: an assistant turn is a TOOL-CALL turn iff the next turn is `tool`. Parse its bracket
     DSL `[Name(k=v, ...), Name2(...)]` -> OpenAI tool_calls (depth-aware split; values via ast.literal_eval).
     Text assistant turns -> plain content. `tool` turns -> role:"tool" with matched tool_call_id.
  3. A row whose bracket DSL fails to parse is SKIPPED whole (counted) — a broken call would corrupt training.

Sampling: keep ALL rows with >=2 tool-call turns; fill to --n-total (default 4800) with single-call rows
(seed 42), capping pure no-tool (irrelevance) rows at --n-irrelevance so English chat can't flood the mix.

Usage:  PYTHONPATH=. python3 data_pipeline/convert_toolace.py --n-total 4800 --out data/generated/toolace_chat.jsonl
"""

import argparse
import ast
import json
import random
import re
from data_pipeline.common import TOOLS_BLOCK_TMPL, write_jsonl

TOOLS_HDR = "Here is a list of functions in JSON format that you can invoke:"
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
    return SYS_PREAMBLE.format(tools_block="\n".join(lines))


def convert_row(row: dict) -> tuple[dict, int] | None:
    """One ToolACE row -> (unified example, n_call_turns). None if unparseable."""
    sysmsg = convert_system(row.get("system", ""))
    if sysmsg is None:
        return None
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
    return {"messages": msgs, "_meta": {"source": "toolace", "lang": "en", "n_call_turns": n_calls}}, n_calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw/toolace/toolace.jsonl")
    ap.add_argument("--n-total", type=int, default=4800)
    ap.add_argument("--n-irrelevance", type=int, default=500, help="cap on kept 0-tool-call (chat/refusal) rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/generated/toolace_chat.jsonl")
    args = ap.parse_args()

    multi, single, none_, skipped = [], [], [], 0
    for line in open(args.data):
        r = json.loads(line)
        res = convert_row(r)
        if res is None:
            skipped += 1; continue
        ex, nc = res
        (multi if nc >= 2 else single if nc == 1 else none_).append(ex)

    rng = random.Random(args.seed)
    rng.shuffle(single); rng.shuffle(none_)
    kept = list(multi)
    kept += none_[:args.n_irrelevance]
    kept += single[:max(0, args.n_total - len(kept))]
    if len(kept) < args.n_total:  # top up from remaining none_ if single ran out
        kept += none_[args.n_irrelevance:args.n_irrelevance + (args.n_total - len(kept))]
    rng.shuffle(kept)

    write_jsonl(kept, args.out)
    parsed_total = len(multi) + len(single) + len(none_)
    print(f"ToolACE: parsed {parsed_total} / skipped {skipped} "
          f"({100*skipped/(parsed_total+skipped):.1f}% unparseable)")
    print(f"  pools: multi(>=2 calls)={len(multi)} single(1)={len(single)} none(0)={len(none_)}")
    print(f"  kept {len(kept)} -> {args.out}  (multi {len(multi)} + irrelevance {min(len(none_),args.n_irrelevance)} + single fill)")


if __name__ == "__main__":
    main()

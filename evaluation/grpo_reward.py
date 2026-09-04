"""verl reward seam for the db_bahn tool-agent loop → evaluation/trajectory_reward.py.

verl's reward manager hands `compute_score` the DECODED response text of the whole episode
(`tokenizer.decode(response_ids, skip_special_tokens=True)`, naive.py:75) — one flat string in Hermes
shape: think text, <tool_call>{...}</tool_call>, then the template-rendered tool turn
(`user\n<tool_response>\n…\n</tool_response>` — one block per tool result, consecutive results share
one `user` turn), then the generation prompt remnant `assistant\n`, and so on. The verifier, however,
scores a message list (it replays the tool calls on a fresh env for the DB hash and builds the
grounding corpus from EVERY tool result). This module is that translation and nothing else:
parse → messages → delegate.

Contract (verl custom_reward_function): extra_info carries `task` / `answer_key` as JSON strings
(parquet-safe, see training_pipeline/build_grpo_pool.py). Never raises — a parse failure is a
0.0-reward episode, not a dead training run.

Stage-2 extras:
  * Ledger: if extra_info carries `ledger_dir`, every scored rollout appends one JSON line
    {task_id, score, ts, pid} to <ledger_dir>/<pid>.jsonl (one file per reward worker process, so
    concurrent Ray actors never interleave). training_pipeline/grpo_dataset.py reads the ledger to
    re-weight the task pool; validation rows carry no `ledger_dir` and therefore leave no trace.
  * Round-trip self-test: `python3 evaluation/grpo_reward.py --roundtrip <rollout.jsonl>` renders our
    eval traces exactly the way verl decodes an episode, parses them back and re-scores — every score
    must equal the stored one. Run it in the tau2 venv (the verifier replays on the db_bahn env).

Parser note: with N parallel tool calls an earlier version gave all N tool messages the FIRST response's
text and dropped the rest → grounding false negatives. Responses are paired with calls in order; the
round-trip self-test guards this.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # run by path (self-test) → make `evaluation.*` importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.trajectory_reward import compute_score as _score_messages  # noqa: E402

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_RESP_RE = re.compile(r"<tool_response>\n?(.*?)\n?</tool_response>", re.DOTALL)
_SEGMENT_RE = re.compile(r"<tool_call>\s*\{.*?\}\s*</tool_call>|<tool_response>.*?</tool_response>",
                         re.DOTALL)
# role words that survive `skip_special_tokens` around the template-rendered tool turn:
#   "</tool_call>\nuser\n<tool_response>" and "</tool_response>\nassistant\n<think>"
_ROLE_BEFORE_RESP_RE = re.compile(r"\s*\buser\s*(?=<tool_response>)")
_ROLE_AFTER_RESP_RE = re.compile(r"(?<=</tool_response>)\s*\bassistant\s*\n?")


def hermes_to_messages(text: str) -> list[dict]:
    """Flat Hermes episode text -> OpenAI-style messages (what the verifier consumes).

    Assistant turns keep their think text as content; each <tool_call> becomes a tool_calls entry and
    the i-th <tool_response> of the following tool turn becomes the role:"tool" message of the i-th
    call. The tail after the last tag is the final answer (the turn the verifier checks for grounding).
    """
    text = _ROLE_AFTER_RESP_RE.sub("", _ROLE_BEFORE_RESP_RE.sub("", text or ""))
    messages: list[dict] = []
    pending_calls: list[dict] = []
    pending_resps: list[str] = []
    buf: list[str] = []
    idx = 0
    pos = 0

    def flush_assistant(content: str, calls: list[dict]):
        if content.strip() or calls:
            msg = {"role": "assistant", "content": content.strip()}
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)

    def close_tool_turn():
        """Pair the collected responses with the pending calls (in order) and emit the tool messages."""
        nonlocal pending_calls, pending_resps
        if not pending_calls and not pending_resps:
            return
        flush_assistant("".join(buf), pending_calls)
        buf.clear()
        for i, resp in enumerate(pending_resps):
            call_id = pending_calls[i]["id"] if i < len(pending_calls) else f"orphan_{i}"
            messages.append({"role": "tool", "tool_call_id": call_id, "content": resp})
        pending_calls, pending_resps = [], []

    for m in _SEGMENT_RE.finditer(text):
        between = text[pos:m.start()]
        pos = m.end()
        seg = m.group(0)
        call = _TOOL_CALL_RE.fullmatch(seg)
        if call:
            if pending_resps:            # a new call after a tool turn -> the previous turn is complete
                close_tool_turn()
            buf.append(between)
            try:
                payload = json.loads(call.group(1))
            except Exception:
                continue  # malformed call: the model failed to emit valid JSON -> not a tool call
            idx += 1
            pending_calls.append({
                "id": f"call_{idx}", "type": "function",
                "function": {"name": payload.get("name", ""),
                             "arguments": json.dumps(payload.get("arguments") or {}, ensure_ascii=False)},
            })
            continue
        resp = _TOOL_RESP_RE.fullmatch(seg)
        if not pending_resps:            # first response of this tool turn: text before it is assistant text
            buf.append(between)
        pending_resps.append(resp.group(1) if resp else "")

    if pending_resps:
        close_tool_turn()
        buf.append(text[pos:])
        flush_assistant("".join(buf), [])
    else:
        buf.append(text[pos:])
        flush_assistant("".join(buf), pending_calls)   # trailing calls without a response still count
    return messages


def render_verl_response(messages: list[dict]) -> str:
    """Render a message list the way verl's reward manager sees an episode: the generated assistant
    text (think included) plus the Qwen3-template tool turns, with special tokens dropped. Mirrors
    chat_template.jinja (assistant: content + '\\n<tool_call>…</tool_call>' per call, then '<|im_end|>\\n';
    tool: '<|im_start|>user' once per block, '\\n<tool_response>\\n{c}\\n</tool_response>' per result,
    '<|im_end|>\\n'; then the generation prompt '<|im_start|>assistant\\n'). Used by the round-trip test."""
    out: list[str] = []
    body = [m for m in messages if m.get("role") in ("assistant", "tool")]
    for i, m in enumerate(body):
        if m["role"] == "assistant":
            content = m.get("content") or ""
            calls = m.get("tool_calls") or []
            out.append(content)
            for j, tc in enumerate(calls):
                if (j == 0 and content) or j > 0:
                    out.append("\n")
                fn = tc.get("function", tc)
                args = fn.get("arguments")
                args = args if isinstance(args, str) else json.dumps(args or {})
                out.append('<tool_call>\n{"name": "' + str(fn.get("name", "")) + '", "arguments": ' + args
                           + "}\n</tool_call>")
            out.append("\n")                                   # <|im_end|>\n
        else:
            if i == 0 or body[i - 1]["role"] != "tool":
                out.append("user")                             # <|im_start|>user
            out.append("\n<tool_response>\n" + (m.get("content") or "") + "\n</tool_response>")
            if i + 1 >= len(body) or body[i + 1]["role"] != "tool":
                out.append("\n")                               # <|im_end|>\n
                if i + 1 < len(body):
                    out.append("assistant\n")                  # generation prompt of the next turn
    return "".join(out)


def _ledger_append(extra_info: dict, ground_truth, score: float) -> None:
    ledger_dir = extra_info.get("ledger_dir")
    if not ledger_dir:
        return
    try:
        d = Path(ledger_dir)
        d.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"task_id": ground_truth, "score": float(score), "ts": time.time(),
                           "pid": os.getpid()}) + "\n"
        with open(d / f"{os.getpid()}.jsonl", "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:  # the ledger is an optimisation, never a reason to lose a reward
        print(f"[grpo_reward] ledger append failed: {type(e).__name__}: {e}", file=sys.stderr)


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kw):
    extra_info = extra_info or {}
    try:
        messages = hermes_to_messages(solution_str if isinstance(solution_str, str) else "")
        task = extra_info.get("task")
        key = extra_info.get("answer_key")
        info = {"task": json.loads(task) if isinstance(task, str) else (task or {}),
                "answer_key": json.loads(key) if isinstance(key, str) else key}
        out = _score_messages(data_source=data_source, solution_str=messages,
                              ground_truth=ground_truth, extra_info=info)
        out["n_msgs"] = float(len(messages))
    except Exception as e:  # a broken adapter must not kill the run
        out = {"score": 0.0, "adapter_error": 1.0, "error_str": f"{type(e).__name__}: {e}"}
    _ledger_append(extra_info, ground_truth, out.get("score", 0.0))
    return out


# --- round-trip self-test ------------------------------------------------------------------------
def _roundtrip(path: str, limit: int | None, verbose: int) -> int:
    from collections import Counter

    from evaluation.trajectory_reward import score_trajectory

    root = Path(__file__).resolve().parents[1] / "data/raw/db_sandbox"
    tasks = {t["id"]: t for t in json.loads((root / "tasks.json").read_text())}
    keys = json.loads((root / "answer_keys.json").read_text())
    n = ok = 0
    mism = Counter()
    examples = []
    t0 = time.time()
    with open(path) as f:
        for line in f:
            if limit and n >= limit:
                break
            r = json.loads(line)
            n += 1
            text = render_verl_response(r["messages"])
            msgs = hermes_to_messages(text)
            res = score_trajectory(tasks[r["task_id"]], msgs, keys.get(r["task_id"]))
            ref = r["score"]
            same = res["score"] == ref["score"] and res["components"] == ref["components"]
            if same:
                ok += 1
            else:
                diff = [c for c in set(res["components"]) | set(ref["components"])
                        if res["components"].get(c) != ref["components"].get(c)]
                mism[tuple(sorted(diff)) or ("score",)] += 1
                if len(examples) < verbose:
                    examples.append((r["task_id"], r.get("sample_idx"), ref["score"], res["score"], diff,
                                     res.get("error"), ref.get("error")))
            if n % 500 == 0:
                print(f"  {n} … identical {ok}", flush=True)
    print(f"\nround-trip {path}: {ok}/{n} identical ({100 * ok / max(1, n):.2f} %), {time.time() - t0:.0f}s")
    if mism:
        print("mismatch components:", dict(mism.most_common()))
        for ex in examples:
            print("  ", ex)
    return 0 if ok == n else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="verl reward seam; --roundtrip runs the parser self-test")
    ap.add_argument("--roundtrip", metavar="ROLLOUT_JSONL")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verbose", type=int, default=5, help="mismatch examples to print")
    a = ap.parse_args()
    if not a.roundtrip:
        ap.error("nothing to do (use --roundtrip <jsonl>)")
    sys.exit(_roundtrip(a.roundtrip, a.limit, a.verbose))

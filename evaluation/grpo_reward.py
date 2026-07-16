"""verl reward seam for the db_bahn tool-agent loop → evaluation/trajectory_reward.py.

verl's reward manager hands `compute_score` the DECODED response text of the whole episode — one flat
string in Hermes shape (think text, <tool_call>{...}</tool_call>, <tool_response>...</tool_response>, …),
not a message list. The verifier, however, scores a message list (it replays the tool calls on a fresh env
for the DB hash). This module is that translation and nothing else: parse → messages → delegate.

Contract (verl custom_reward_function): extra_info carries `task` / `answer_key` as JSON strings
(parquet-safe, see training_pipeline/build_grpo_pool.py). Never raises — a parse failure is a
0.0-reward episode, not a dead training run.
"""

import json
import re

from evaluation.trajectory_reward import compute_score as _score_messages

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_RESP_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_SEGMENT_RE = re.compile(r"<tool_call>\s*\{.*?\}\s*</tool_call>|<tool_response>\s*.*?\s*</tool_response>",
                         re.DOTALL)


def hermes_to_messages(text: str) -> list[dict]:
    """Flat Hermes episode text -> OpenAI-style messages (what the verifier consumes).

    Assistant turns keep their think/plan text as content; each <tool_call> becomes a tool_calls entry
    and each <tool_response> the matching role:"tool" message. The tail after the last tag is the final
    answer (the turn the verifier checks for grounding).
    """
    messages: list[dict] = []
    pending_calls: list[dict] = []
    buf: list[str] = []
    idx = 0
    pos = 0

    def flush_assistant(content: str, calls: list[dict]):
        if content.strip() or calls:
            msg = {"role": "assistant", "content": content.strip()}
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)

    for m in _SEGMENT_RE.finditer(text or ""):
        buf.append(text[pos:m.start()])
        pos = m.end()
        seg = m.group(0)
        call = _TOOL_CALL_RE.fullmatch(seg)
        if call:
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
        # tool_response: closes the assistant turn that requested it
        resp = _TOOL_RESP_RE.fullmatch(seg)
        if pending_calls:
            flush_assistant("".join(buf), pending_calls)
            buf = []
            for c in pending_calls:
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": resp.group(1) if resp else ""})
            pending_calls = []

    tail = "".join(buf) + text[pos:]
    flush_assistant(tail, pending_calls)   # trailing calls without a response still count as calls
    return messages


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
        return out
    except Exception as e:  # a broken adapter must not kill the run
        return {"score": 0.0, "adapter_error": 1.0, "error_str": f"{type(e).__name__}: {e}"}

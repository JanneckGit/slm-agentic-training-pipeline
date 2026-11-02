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
  * Constant keys: every episode returns exactly REWARD_KEYS, in that order — verl takes the reward-extra
    keys from the first episode of a batch and indexes all others with them (agent_loop `_postprocess`),
    then asserts equal key lists when it merges the workers' batches. One episode with other keys (the
    old adapter-error dict) is a KeyError/AssertionError that kills the run. Failures are flags instead:
    `adapter_error` (this module raised — parse/JSON) and `verifier_error` (the verifier could not judge,
    trajectory_reward.VERIFIER_ERROR_PREFIXES); both score 0.0 and land in the dumps + val-aux metrics.
  * Round-trip self-test: `python3 evaluation/grpo_reward.py --roundtrip <rollout.jsonl>` renders our
    eval traces exactly the way verl decodes an episode, parses them back and re-scores — every score
    must equal the stored one, and the full seam (compute_score with parquet-shaped extra_info) must
    return the same score with exactly REWARD_KEYS. Run it in the tau2 venv (the verifier replays on the
    db_bahn env); `--limit N --include-fails` = the first N rows plus every failed one.

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

# the verifier's numeric fields (trajectory_reward.score_trajectory, in its order) + this seam's fields
REWARD_KEYS = ("score", "task_solved", "turns_used", "n_tool_calls", "n_tool_errors", "tool_calls_valid",
               "n_plan_turns", "replan_occurred", "self_recovery", "parallel_max_calls_per_turn",
               "efficiency_expected_calls", "efficiency_call_ratio", "efficiency_within_3x",
               "n_msgs", "adapter_error", "verifier_error")
_INT_KEYS = frozenset({"turns_used", "n_tool_calls", "n_tool_errors", "n_plan_turns",
                       "parallel_max_calls_per_turn", "efficiency_expected_calls"})

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


def _default(key: str):
    return 0 if key in _INT_KEYS else 0.0


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
        out["adapter_error"] = 0.0
    except Exception as e:  # a broken adapter must not kill the run: 0.0, flagged, same keys
        print(f"[grpo_reward] ADAPTER_ERROR task={ground_truth}: {type(e).__name__}: {e}", file=sys.stderr)
        out = {"score": 0.0, "adapter_error": 1.0}
    out = {k: out.get(k, _default(k)) for k in REWARD_KEYS}
    _ledger_append(extra_info, ground_truth, out["score"])
    return out


# --- round-trip self-test ------------------------------------------------------------------------
def _typed(key: str, value) -> bool:
    return type(value) is (int if key in _INT_KEYS else float)


def roundtrip_row(rec: dict, task: dict, key: dict | None) -> list[str]:
    """One stored eval trace through the reward path the way verl sees it. Returns the problems ([] = ok):
    (a) render as verl decodes it → parse → score_trajectory: score AND components equal the stored ones;
    (b) the whole seam, compute_score with parquet-shaped extra_info (JSON strings): same score, exactly
    REWARD_KEYS in order, typed values, adapter_error 0, verifier_error == the verifier's own verdict."""
    from evaluation.trajectory_reward import VERIFIER_ERROR_PREFIXES, score_trajectory

    problems = []
    text = render_verl_response(rec["messages"])
    res = score_trajectory(task, hermes_to_messages(text), key)
    ref = rec["score"]
    if res["score"] != ref["score"] or res["components"] != ref["components"]:
        diff = sorted(c for c in set(res["components"]) | set(ref["components"])
                      if res["components"].get(c) != ref["components"].get(c))
        problems.append("components:" + (",".join(diff) or "score"))
    out = compute_score(data_source="db_bahn", solution_str=text, ground_truth=rec["task_id"],
                        extra_info={"task": json.dumps(task, ensure_ascii=False),
                                    "answer_key": json.dumps(key, ensure_ascii=False)})
    if tuple(out) != REWARD_KEYS:
        problems.append("keys")
    elif bad := [k for k in REWARD_KEYS if not _typed(k, out[k])]:
        problems.append("types:" + ",".join(bad))
    if out.get("score") != res["score"]:
        problems.append("seam_score")
    if out.get("adapter_error") != 0.0:
        problems.append("adapter_error")
    if out.get("verifier_error") != (1.0 if (res.get("error") or "").startswith(VERIFIER_ERROR_PREFIXES) else 0.0):
        problems.append("verifier_error")
    return problems


def roundtrip_file(path, limit: int | None = None, include_fails: bool = False, verbose: int = 0,
                   progress: bool = False) -> dict:
    """roundtrip_row over a K-rollout trace file: the first `limit` rows (all if None), plus — with
    include_fails — every row whose stored score is < 1 (the reward's error-prone half)."""
    from collections import Counter

    root = Path(__file__).resolve().parents[1] / "data/raw/db_sandbox"
    tasks = {t["id"]: t for t in json.loads((root / "tasks.json").read_text())}
    keys = json.loads((root / "answer_keys.json").read_text())
    n = ok = fails = 0
    kinds: Counter = Counter()
    examples = []
    t0 = time.time()
    with open(path) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            failed = r["score"]["score"] < 1
            if not (limit is None or i < limit or (include_fails and failed)):
                continue
            n += 1
            fails += failed
            problems = roundtrip_row(r, tasks[r["task_id"]], keys.get(r["task_id"]))
            if not problems:
                ok += 1
            else:
                kinds.update(problems)
                if len(examples) < verbose:
                    examples.append((r["task_id"], r.get("sample_idx"), r["score"]["score"], problems))
            if progress and n % 500 == 0:
                print(f"  {n} … identical {ok}", flush=True)
    return {"n": n, "ok": ok, "failed_rows": fails, "problems": dict(kinds.most_common()),
            "examples": examples, "seconds": round(time.time() - t0)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="verl reward seam; --roundtrip runs the parser self-test")
    ap.add_argument("--roundtrip", metavar="ROLLOUT_JSONL")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-fails", action="store_true", help="with --limit: also every row with stored score < 1")
    ap.add_argument("--verbose", type=int, default=5, help="mismatch examples to print")
    a = ap.parse_args()
    if not a.roundtrip:
        ap.error("nothing to do (use --roundtrip <jsonl>)")
    rt = roundtrip_file(a.roundtrip, a.limit, a.include_fails, a.verbose, progress=True)
    print(f"\nround-trip {a.roundtrip}: {rt['ok']}/{rt['n']} identical ({100 * rt['ok'] / max(1, rt['n']):.2f} %; "
          f"{rt['failed_rows']} failed rows), {rt['seconds']}s")
    if rt["problems"]:
        print("problems:", rt["problems"])
        for ex in rt["examples"]:
            print("  ", ex)
    sys.exit(0 if rt["ok"] == rt["n"] else 1)

"""
evaluation/trajectory_reward.py
===============================
Deterministic trajectory verifier for the `db_bahn` grounded-synthesis pipeline (Plan B, Phase 4).
verl-compatible contract: never raises, returns a dict whose "score" is the binary
reward and whose other keys are free aux metrics (verl-compatible shape for Stage-2).

Scoring (all deterministic — the P0-1/P0-2 fixes; do NOT trust tau2's read-only reward alone):
  ACTION tasks: replay the trajectory's tool calls on a fresh env (init actions applied) →
      db_match      : predicted DB hash == gold DB hash (gold = init + reference actions)  [tau2 semantics]
      asserts_pass  : all task env_assertions pass on the predicted env
  INFO tasks:
      no_write      : predicted DB hash == init-only hash (an info task must not mutate state)
  BOTH:
      actions_pass  : answer-key expected_tools ⊆ called tools (normalized set-membership, order-free)
      communicate   : every communicate_info string is a (case-insensitive) substring of assistant text
      grounding     : every entity-like token in the FINAL answer (ids, times, dates, Zugnummern)
                      appears in the grounding corpus = ticket + tool observations (+ tool-call args)
                      → hallucinated ids/times score 0.

score = 1.0 iff every applicable component passes. Aux: turns_used, n_tool_calls, n_tool_errors,
tool_calls_valid, n_plan_turns, replan_occurred (injected & ≥2 planning turns; for runtime-fault
tasks: ≥1 rejected tool call & ≥2 planning turns), components dict.

Trajectory format (produced by sdg_pipeline/db_bahn/rollout.py): OpenAI-style messages —
assistant turns may carry tool_calls=[{id,type,function:{name,arguments:<json str>}}], observations are
role:"tool" turns. Requires the tau2 venv (imports the db_bahn domain). Scenario suite (21 cases):
    tests/test_trajectory_reward.py — `bash ops/run_tests.sh`; running this file delegates to pytest.
"""

import json
import re
import sys

from data_pipeline.common import args_dict, final_answer

DEFAULT_MAX_REPLAY_CALLS = 64

_ENTITY_PATTERNS = [
    re.compile(r"\bMA-\d+\b"),                       # employee ids
    re.compile(r"\bWO-\d{4}-\d+\b"),                 # maintenance order ids
    re.compile(r"\bAS-\d+\b"),                       # assignment ids
    re.compile(r"\b\d{1,2}:\d{2}\b"),                # times
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),            # dates
    re.compile(r"\b(?:ICE|IC|EC|ECE|RJ|EN)\s?\d+\b"),  # train numbers
    re.compile(r"\b[A-Za-z0-9]+-9\d{3}\b"),          # vehicle ids (…-9xxx)
]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").lower())


def _tool_calls_of(msg: dict) -> list[dict]:
    return msg.get("tool_calls") or []


THINK_RE = re.compile(r"<(?:seed:)?think>.*?</(?:seed:)?think>\s*", re.DOTALL)
THINK_BRACKET_RE = re.compile(r"\[THINK\].*?\[/THINK\]\s*", re.DOTALL)


def _strip_think(text) -> str:
    """Drop reasoning blocks before grading: speculative values in the reasoning must not count as
    hallucination (grounding) or as communicated info — judged is only the visible answer.
    Wave 3: full dialect parity with rollout.strip_think (<seed:think>, [THINK], unpaired closes),
    so ad-hoc evals of foreign thinking models grade identically."""
    text = THINK_RE.sub("", text or "")
    text = THINK_BRACKET_RE.sub("", text)
    for close in ("</seed:think>", "</think>", "[/THINK]"):
        if close in text:
            text = text.rsplit(close, 1)[1]
    return text


def _assistant_text(messages: list[dict]) -> str:
    return "\n".join(_strip_think(m.get("content")) for m in messages if m.get("role") == "assistant")


def _final_answer(messages: list[dict]) -> str:
    ans = _strip_think(final_answer(messages))
    if ans.strip():
        return ans
    for m in reversed(messages):  # fallback (verifier-only): last assistant turn even WITH tool_calls
        if m.get("role") == "assistant" and _strip_think(m.get("content")).strip():
            return _strip_think(m["content"])
    return ""


def _grounding_corpus(messages: list[dict], ticket: str) -> str:
    parts = [ticket or ""]
    for m in messages:
        if m.get("role") == "tool":
            parts.append(m.get("content") or "")
        for tc in _tool_calls_of(m):
            fn = tc.get("function", {})
            parts.append(fn.get("name", "") + " " + str(fn.get("arguments", "")))
    raw = "\n".join(parts)
    # DERIVED-TIME grounding (calibration, 2026-07-03): a time computed as observed-time ± observed-delay
    # is legitimate reasoning, not hallucination (e.g. planned 18:30 + 45 min -> 19:15). Augment the corpus
    # with all such derivations. Times only — ids/dates stay strict.
    times = {(int(h), int(mi)) for h, mi in re.findall(r"\b(\d{1,2}):(\d{2})\b", raw)}
    delays = {int(d) for d in re.findall(r'"verspaetung_minuten"\s*:\s*(\d+)', raw)}
    delays |= {int(d) for d in re.findall(r"\+(\d+)\s*Min", raw)}
    derived = []
    for h, mi in times:
        for d in delays:
            if d == 0:
                continue
            for total in ((h * 60 + mi + d) % 1440, (h * 60 + mi - d) % 1440):
                derived.append(f"{total // 60:02d}:{total % 60:02d}")
    return _norm(raw + "\n" + " ".join(derived))


def _grounding_pass(final_answer: str, corpus_norm: str) -> tuple[bool, list[str]]:
    """Every entity-like token in the final answer must appear in the grounding corpus."""
    missing = []
    for pat in _ENTITY_PATTERNS:
        for tok in pat.findall(final_answer or ""):
            if _norm(tok) not in corpus_norm:
                missing.append(tok)
    return (len(missing) == 0), missing


def score_trajectory(task, messages: list[dict], answer_key: dict | None = None,
                     get_env=None, max_replay_calls: int = DEFAULT_MAX_REPLAY_CALLS) -> dict:
    """Binary verified/not-verified for ONE trajectory. Never raises.

    Args:
        task: tau2 Task (pydantic) or plain dict with the same schema.
        messages: OpenAI-style message list (the full rollout).
        answer_key: our side-channel entry (expected_tools/kind/injected); optional but recommended.
        get_env: callable(solo_mode=True) -> fresh tau2 Environment. Defaults to the db_bahn domain.
    """
    out = {"score": 0.0, "task_solved": 0.0, "turns_used": 0, "n_tool_calls": 0,
           "n_tool_errors": 0, "tool_calls_valid": 0.0, "n_plan_turns": 0,
           "replan_occurred": 0.0, "self_recovery": 0.0,
           # wave 3 soft aux (never gating): efficiency vs the oracle path + parallel emission
           "parallel_max_calls_per_turn": 0, "efficiency_expected_calls": 0,
           "efficiency_call_ratio": 0.0, "efficiency_within_3x": 1.0,
           "components": {}, "error": None}
    try:
        if get_env is None:
            from sdg_pipeline.db_bahn.tau2_domain import get_environment as get_env  # tau2 venv only

        t = task if isinstance(task, dict) else task.model_dump()
        crit = t.get("evaluation_criteria") or {}
        init_actions = ((t.get("initial_state") or {}).get("initialization_actions")) or []
        ticket = t.get("ticket") or ""
        key = answer_key or {}
        kind = key.get("kind") or ("action" if "DB" in (crit.get("reward_basis") or []) else "info")

        from tau2.data_model.tasks import EnvAssertion, EnvFunctionCall

        def fresh(apply_init: bool):
            env = get_env(solo_mode=True)
            if apply_init and init_actions:
                env.run_env_function_calls(
                    [EnvFunctionCall.model_validate(a) if isinstance(a, dict) else a for a in init_actions])
            return env

        # --- replay the trajectory's tool calls on a fresh env -----------------------------
        pred_env = fresh(apply_init=True)
        called, n_err, n_calls = [], 0, 0
        for m in messages:
            if m.get("role") != "assistant":
                continue
            tcs = _tool_calls_of(m)
            if tcs:
                out["n_plan_turns"] += 1
                out["parallel_max_calls_per_turn"] = max(out["parallel_max_calls_per_turn"], len(tcs))
            for tc in tcs:
                if n_calls >= max_replay_calls:
                    break
                fn = tc.get("function", {})
                name, args = fn.get("name", ""), args_dict(fn.get("arguments"))
                n_calls += 1
                called.append(name)
                try:
                    pred_env.use_tool(name, **args)
                except Exception:
                    n_err += 1
        out["n_tool_calls"], out["n_tool_errors"] = n_calls, n_err
        out["tool_calls_valid"] = (n_calls - n_err) / n_calls if n_calls else 0.0
        out["turns_used"] = sum(1 for m in messages if m.get("role") in ("assistant", "tool"))
        # wave 3: SOFT efficiency vs the oracle call count (expected_calls = len(oracle_calls)).
        # Deliberately NOT a component — the hard >=3x gate lives in format_traj (training data),
        # so teacher yield and heldout comparability stay outcome-based.
        exp = int(key.get("expected_calls") or 0)
        out["efficiency_expected_calls"] = exp
        out["efficiency_call_ratio"] = round(n_calls / exp, 3) if exp else 0.0
        out["efficiency_within_3x"] = 1.0 if (not exp or n_calls < 3 * exp) else 0.0

        comp = {}

        # --- deterministic DB-state component ----------------------------------------------
        gold_env = fresh(apply_init=True)
        for a in (crit.get("actions") or []):
            try:
                a = a if isinstance(a, dict) else a.model_dump()
                gold_env.use_tool(a["name"], **a["arguments"])
            except Exception as e:  # gold must replay — else the task itself is broken
                out["error"] = f"gold_replay_failed: {e}"
                return out
        if kind == "action":
            comp["db_match"] = pred_env.tools.db.get_hash() == gold_env.tools.db.get_hash()
            ok_asserts = True
            for a in (crit.get("env_assertions") or []):
                ea = EnvAssertion.model_validate(a) if isinstance(a, dict) else a
                ok_asserts &= bool(pred_env.run_env_assertion(ea, raise_assertion_error=False))
            comp["asserts_pass"] = ok_asserts
        else:
            comp["no_write"] = pred_env.tools.db.get_hash() == gold_env.tools.db.get_hash()

        # --- normalized action set-membership ----------------------------------------------
        expected = set(key.get("expected_tools") or [])
        comp["actions_pass"] = expected.issubset(set(called)) if expected else True
        # wave 3 (refusal tasks): tools the agent must NOT call (e.g. all WRITE tools). Empty
        # or absent => True, so every pre-wave-3 task is unaffected.
        forb = set(key.get("forbidden_tools") or [])
        comp["no_forbidden"] = forb.isdisjoint(set(called)) if forb else True

        # --- communicate substrings ---------------------------------------------------------
        atext = _assistant_text(messages).lower()
        comm = crit.get("communicate_info") or []
        comp["communicate"] = all((c or "").lower() in atext for c in comm)

        # --- anti-hallucination grounding ---------------------------------------------------
        final = _final_answer(messages)
        gp, missing = _grounding_pass(final, _grounding_corpus(messages, ticket))
        comp["grounding"] = gp
        if missing:
            out["error"] = f"ungrounded_tokens: {missing[:5]}"
        comp["has_final_answer"] = bool(final.strip())

        out["components"] = comp
        solved = all(comp.values())
        out["task_solved"] = out["score"] = 1.0 if solved else 0.0
        injected = bool(key.get("injected")) or bool(init_actions)
        # runtime-fault tasks (wave 2): the surprise is a REJECTED tool call, not a state injection
        runtime_fault = key.get("fault") in ("runtime", "state+runtime")
        out["replan_occurred"] = 1.0 if (
            (injected and out["n_plan_turns"] >= 2)
            or (runtime_fault and out["n_tool_errors"] >= 1 and out["n_plan_turns"] >= 2)
        ) else 0.0
        # self-recovery (B0 detector): a verified trace that hit >=1 tool error mid-way and still
        # reached 1.0 — i.e. the agent recovered from a failed call. Emergent-vs-scripted is split
        # downstream by `fault` (fault∈{none,state} = emergent; {runtime,state+runtime} = scripted).
        out["self_recovery"] = 1.0 if (solved and out["n_tool_errors"] >= 1) else 0.0
        return out
    except Exception as e:
        # contract: never raise — but a verifier bug must not pass as a silent 0.0 either
        out["error"] = f"verifier_exception: {type(e).__name__}: {e}"
        print(f"[trajectory_reward] WARNING {out['error']}", file=sys.stderr)
        return out


# --- verl reward-manager adapter (Stage-2 seam) -----------------------------------------------
# errors after which the verifier could not judge the episode at all (exception, unreplayable gold) —
# NOT `ungrounded_tokens`, which is a regular verdict about the answer
VERIFIER_ERROR_PREFIXES = ("verifier_exception", "gold_replay_failed")


def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kw):
    """Stage-2 GRPO seam: extra_info carries {'task': ..., 'answer_key': ...}; solution_str is the
    trajectory messages as JSON. Returns dict{score, ...aux} (verl custom_reward_function contract).
    `verifier_error` = 1.0 when the verifier could not score the episode (its 0.0 says nothing about
    the policy) — counted per run by training_pipeline/grpo_metrics.py."""
    extra_info = extra_info or {}
    try:
        messages = json.loads(solution_str) if isinstance(solution_str, str) else (solution_str or [])
    except Exception:
        messages = []
    res = score_trajectory(extra_info.get("task") or {}, messages, extra_info.get("answer_key"))
    flat = {k: v for k, v in res.items() if isinstance(v, (int, float))}
    flat["score"] = res["score"]
    err = res.get("error") or ""
    flat["verifier_error"] = 1.0 if err.startswith(VERIFIER_ERROR_PREFIXES) else 0.0
    if err.startswith("gold_replay_failed"):  # score_trajectory warns only for its own exceptions
        print(f"[trajectory_reward] WARNING {err}", file=sys.stderr)
    return flat


if __name__ == "__main__":
    # The scenario suite (21 cases, formerly `_selftest` here) lives in tests/test_trajectory_reward.py and
    # needs the tau2 venv + the generated db_bahn pool; this entry point keeps the historical command working.
    import os
    try:
        import pytest
    except ImportError:
        sys.exit("the verifier suite runs under pytest — .venv-tau2 (requirements/tau2.lock.txt) has it")
    sys.exit(pytest.main(["-q", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests",
                                             "test_trajectory_reward.py")]))

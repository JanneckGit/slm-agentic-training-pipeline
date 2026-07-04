"""
evaluation/trajectory_reward.py
===============================
Deterministic trajectory verifier for the `db_bahn` grounded-synthesis pipeline (Plan B, Phase 4).
Mirrors evaluation/reward.py's contract: never raises, returns a dict whose "score" is the binary
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
tool_calls_valid, n_plan_turns, replan_occurred (injected & ≥2 planning turns), components dict.

Trajectory format (produced by sdg_pipeline/db_bahn/rollout.py): OpenAI-style messages —
assistant turns may carry tool_calls=[{id,type,function:{name,arguments:<json str>}}], observations are
role:"tool" turns. Requires the tau2 venv (imports the db_bahn domain). Self-test:
    PYTHONPATH=. <tau2-venv>/bin/python evaluation/trajectory_reward.py   # exits non-zero on failure
"""

import json
import re

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


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw or "{}")
    except Exception:
        return {}


def _assistant_text(messages: list[dict]) -> str:
    return "\n".join(m.get("content") or "" for m in messages if m.get("role") == "assistant")


def _final_answer(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "assistant" and (m.get("content") or "").strip() and not _tool_calls_of(m):
            return m["content"]
    for m in reversed(messages):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            return m["content"]
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
           "replan_occurred": 0.0, "components": {}, "error": None}
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
            for tc in tcs:
                if n_calls >= max_replay_calls:
                    break
                fn = tc.get("function", {})
                name, args = fn.get("name", ""), _parse_args(fn.get("arguments"))
                n_calls += 1
                called.append(name)
                try:
                    pred_env.use_tool(name, **args)
                except Exception:
                    n_err += 1
        out["n_tool_calls"], out["n_tool_errors"] = n_calls, n_err
        out["tool_calls_valid"] = (n_calls - n_err) / n_calls if n_calls else 0.0
        out["turns_used"] = sum(1 for m in messages if m.get("role") in ("assistant", "tool"))

        comp = {}

        # --- deterministic DB-state component ----------------------------------------------
        gold_env = fresh(apply_init=True)
        for a in (crit.get("actions") or []):
            a = a if isinstance(a, dict) else a
            try:
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
        out["replan_occurred"] = 1.0 if (injected and out["n_plan_turns"] >= 2) else 0.0
        return out
    except Exception as e:
        out["error"] = f"verifier_exception: {type(e).__name__}: {e}"
        return out


# --- verl reward-manager adapter (Stage-2 seam; same dict contract as evaluation/reward.py) --
def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kw):
    """Stage-2 GRPO seam: extra_info carries {'task': ..., 'answer_key': ...}; solution_str is the
    trajectory messages as JSON. Returns dict{score, ...aux} exactly like evaluation/reward.py."""
    extra_info = extra_info or {}
    try:
        messages = json.loads(solution_str) if isinstance(solution_str, str) else (solution_str or [])
    except Exception:
        messages = []
    res = score_trajectory(extra_info.get("task") or {}, messages, extra_info.get("answer_key"))
    flat = {k: v for k, v in res.items() if isinstance(v, (int, float))}
    flat["score"] = res["score"]
    return flat


# --- self-test (Phase-4 smoke: known-good → 1.0, wrong-write / hallucination → 0.0) ----------
def _mk_call(i, name, **args):
    return {"id": f"call_{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def _selftest():
    import sdg_pipeline.db_bahn.tau2_domain as dom

    tasks = {t["id"]: t for t in json.load(open("data/raw/db_sandbox/tasks.json"))}
    keys = json.load(open("data/raw/db_sandbox/answer_keys.json"))

    # 1) good ACTION trace (reference actions echoed with a grounded confirmation) -> 1.0
    tid = next(i for i, k in keys.items() if k["template"] == "t_action_crew")
    task, key = tasks[tid], keys[tid]
    ref = task["evaluation_criteria"]["actions"][0]
    obs_env = dom.get_environment(solo_mode=True)
    obs = obs_env.use_tool(ref["name"], **ref["arguments"])
    msgs = [
        {"role": "system", "content": "…"},
        {"role": "user", "content": task["ticket"]},
        {"role": "assistant", "content": "<plan>Zuteilung ausführen.</plan>",
         "tool_calls": [_mk_call(1, ref["name"], **ref["arguments"])]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs, ensure_ascii=False, default=str)},
        {"role": "assistant", "content":
            f"Erledigt: {key['facts']['emp_id']} ist dem Zug als {key['facts']['rolle']} zugeteilt."},
    ]
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0, f"good ACTION should pass: {r}"

    # 2) ACTION trace with the WRONG write -> 0.0 (db hash mismatch)
    bad = json.loads(json.dumps(msgs))
    bad[2]["tool_calls"] = [_mk_call(1, "wartung_einplanen",
                                     fahrzeug_id="EC-9006", typ="Inspektion", faellig_am="2026-07-05 06:00")]
    r2 = score_trajectory(task, bad, key)
    assert r2["score"] == 0.0 and not r2["components"]["db_match"], f"wrong write must fail: {r2}"

    # 3) good INFO trace (tool called, facts grounded) -> 1.0
    tid3 = next(i for i, k in keys.items() if k["template"] == "t_info_crew")
    task3, key3 = tasks[tid3], keys[tid3]
    zug = key3["facts"]["lokfuehrer"][0]
    env3 = dom.get_environment(solo_mode=True)
    zn = task3["ticket"].split("auf ")[1].split(" als")[0]
    obs3 = env3.use_tool("mitarbeiter_info", zugnummer=zn)
    msgs3 = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task3["ticket"]},
        {"role": "assistant", "content": "<plan>Besatzung abfragen.</plan>",
         "tool_calls": [_mk_call(1, "mitarbeiter_info", zugnummer=zn)]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs3, ensure_ascii=False)},
        {"role": "assistant", "content": f"Als Lokführer ist {zug['name']} ({zug['mitarbeiter_id']}) eingeteilt."},
    ]
    r3 = score_trajectory(task3, msgs3, key3)
    assert r3["score"] == 1.0, f"good INFO should pass: {r3}"

    # 4) hallucinated INFO trace (no tool call, invented id but correct-looking answer) -> 0.0
    msgs4 = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task3["ticket"]},
        {"role": "assistant", "content": f"Als Lokführer ist Max Muster (MA-99999) eingeteilt. "
                                         f"{zug['name']} {zug['mitarbeiter_id']}"},
    ]
    r4 = score_trajectory(task3, msgs4, key3)
    assert r4["score"] == 0.0 and not r4["components"]["grounding"], f"hallucination must fail: {r4}"
    assert not r4["components"]["actions_pass"], "expected-tools check must fail without tool calls"

    # 5) fault-injected task: init actions apply during replay; replan metric fires with >=2 plan turns
    tid5 = next(i for i, k in keys.items() if k["template"] == "t_action_ersatz")
    task5, key5 = tasks[tid5], keys[tid5]
    ref5 = task5["evaluation_criteria"]["actions"][0]
    from tau2.data_model.tasks import EnvFunctionCall
    env5 = dom.get_environment(solo_mode=True)
    env5.run_env_function_calls([EnvFunctionCall.model_validate(a)
                                 for a in task5["initial_state"]["initialization_actions"]])
    zn5 = ref5["arguments"]["zugnummer"]
    obs5a = env5.use_tool("mitarbeiter_info", zugnummer=zn5)
    obs5b = env5.use_tool(ref5["name"], **ref5["arguments"])
    msgs5 = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task5["ticket"]},
        {"role": "assistant", "content": "<plan>Erst Besatzung prüfen.</plan>",
         "tool_calls": [_mk_call(1, "mitarbeiter_info", zugnummer=zn5)]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs5a, ensure_ascii=False)},
        {"role": "assistant", "content": "<plan>Kein Lokführer eingeteilt – Ersatz zuweisen.</plan>",
         "tool_calls": [_mk_call(2, ref5["name"], **ref5["arguments"])]},
        {"role": "tool", "tool_call_id": "call_2", "content": json.dumps(obs5b, ensure_ascii=False, default=str)},
        {"role": "assistant", "content": f"{key5['facts']['ersatz_id']} ist als Ersatz-Lokführer zugeteilt."},
    ]
    r5 = score_trajectory(task5, msgs5, key5)
    assert r5["score"] == 1.0 and r5["replan_occurred"] == 1.0, f"injected ACTION should pass with replan: {r5}"

    print("trajectory_reward.py self-test OK "
          f"(good-action 1.0 | wrong-write 0.0 | good-info 1.0 | hallucination 0.0 | injected+replan 1.0)")


if __name__ == "__main__":
    _selftest()

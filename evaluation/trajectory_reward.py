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
role:"tool" turns. Requires the tau2 venv (imports the db_bahn domain). Self-test:
    PYTHONPATH=. <tau2-venv>/bin/python evaluation/trajectory_reward.py   # exits non-zero on failure
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
def compute_score(data_source=None, solution_str=None, ground_truth=None, extra_info=None, **kw):
    """Stage-2 GRPO seam: extra_info carries {'task': ..., 'answer_key': ...}; solution_str is the
    trajectory messages as JSON. Returns dict{score, ...aux} (verl custom_reward_function contract)."""
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
def _mk_call(i, _name, **args):
    # positional is _name so tool arguments literally named `name` (mitarbeiter_suchen, wave 3)
    # don't collide with the parameter
    return {"id": f"call_{i}", "type": "function",
            "function": {"name": _name, "arguments": json.dumps(args, ensure_ascii=False)}}


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
    assert r["self_recovery"] == 0.0, f"clean trace must not flag self_recovery: {r}"

    # 1b) SAME trace, final answer prefixed with a <think> block containing a bogus ID -> still 1.0
    #     (thinking-mode eval: speculative values in the reasoning must be stripped, not graded)
    think = json.loads(json.dumps(msgs))
    think[-1]["content"] = ("<think>\nVielleicht ist auch MA-99999 verfügbar? Nein, laut Tool nicht.\n</think>\n\n"
                            + think[-1]["content"])
    r1b = score_trajectory(task, think, key)
    assert r1b["score"] == 1.0, f"<think> with bogus ID must be stripped, not graded: {r1b}"

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

    # 6) runtime-fault roundtrip (wave 2): invalid crew_zuweisen -> error observation -> search ->
    #    valid assignment -> grounded final. Must pass with n_tool_errors==1 and replan_occurred==1.
    tid6 = next(i for i, k in keys.items() if k["template"] == "t_action_ersatz_quali")
    task6, key6 = tasks[tid6], keys[tid6]
    ref6 = task6["evaluation_criteria"]["actions"][0]
    zn6 = ref6["arguments"]["zugnummer"]
    proposed6 = key6["facts"]["vorschlag_ungueltig"]
    env6 = dom.get_environment(solo_mode=True)
    env6.run_env_function_calls([EnvFunctionCall.model_validate(a)
                                 for a in task6["initial_state"]["initialization_actions"]])
    obs6a = env6.use_tool("mitarbeiter_info", zugnummer=zn6)
    try:
        env6.use_tool("crew_zuweisen", zugnummer=zn6, mitarbeiter_id=proposed6, rolle="Lokführer")
        raise AssertionError("invalid assignment must be rejected by the qualification/role gate")
    except ValueError as e:
        err6 = json.dumps({"error": f"ValueError: {e}"}, ensure_ascii=False)
    obs6b = env6.use_tool("mitarbeiter_suchen", **key6["oracle_calls"][1]["arguments"])
    obs6c = env6.use_tool(ref6["name"], **ref6["arguments"])
    msgs6 = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task6["ticket"]},
        {"role": "assistant", "content": "<plan>Besatzung prüfen, dann Vorschlag zuteilen.</plan>",
         "tool_calls": [_mk_call(1, "mitarbeiter_info", zugnummer=zn6),
                        _mk_call(2, "crew_zuweisen", zugnummer=zn6,
                                 mitarbeiter_id=proposed6, rolle="Lokführer")]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs6a, ensure_ascii=False)},
        {"role": "tool", "tool_call_id": "call_2", "content": err6},
        {"role": "assistant", "content": "<plan>Abgelehnt – qualifizierten Ersatz suchen und den "
                                         "ersten Treffer zuteilen.</plan>",
         "tool_calls": [_mk_call(3, "mitarbeiter_suchen", **key6["oracle_calls"][1]["arguments"]),
                        _mk_call(4, ref6["name"], **ref6["arguments"])]},
        {"role": "tool", "tool_call_id": "call_3", "content": json.dumps(obs6b, ensure_ascii=False)},
        {"role": "tool", "tool_call_id": "call_4", "content": json.dumps(obs6c, ensure_ascii=False, default=str)},
        {"role": "assistant", "content": f"{proposed6} wurde abgelehnt (fehlende Qualifikation); "
                                         f"{key6['facts']['ersatz_id']} ist jetzt als Lokführer zugeteilt."},
    ]
    r6 = score_trajectory(task6, msgs6, key6)
    assert r6["score"] == 1.0 and r6["n_tool_errors"] == 1 and r6["replan_occurred"] == 1.0, \
        f"runtime-fault roundtrip should pass with replan: {r6}"
    assert r6["self_recovery"] == 1.0, f"recovered-from-error trace must flag self_recovery: {r6}"

    # 7) rejection IGNORED (only the invalid attempt, then a confident claim) -> 0.0
    msgs7 = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task6["ticket"]},
        {"role": "assistant", "content": "<plan>Vorschlag direkt zuteilen.</plan>",
         "tool_calls": [_mk_call(1, "crew_zuweisen", zugnummer=zn6,
                                 mitarbeiter_id=proposed6, rolle="Lokführer")]},
        {"role": "tool", "tool_call_id": "call_1", "content": err6},
        {"role": "assistant", "content": f"Erledigt: {proposed6} ist {zn6} als Lokführer zugeteilt."},
    ]
    r7 = score_trajectory(task6, msgs7, key6)
    assert r7["score"] == 0.0 and not r7["components"]["db_match"], f"ignored rejection must fail: {r7}"

    # 8) search-INFO trace via the key's oracle_calls (wave-2 search tool) -> 1.0
    tid8 = next(i for i, k in keys.items() if k["template"] == "t_info_mitarbeiter_suche")
    task8, key8 = tasks[tid8], keys[tid8]
    env8 = dom.get_environment(solo_mode=True)
    call8 = key8["oracle_calls"][0]
    obs8 = env8.use_tool(call8["name"], **call8["arguments"])
    msgs8 = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task8["ticket"]},
        {"role": "assistant", "content": "<plan>Mitarbeitersuche mit den Filtern aus dem Ticket.</plan>",
         "tool_calls": [_mk_call(1, call8["name"], **call8["arguments"])]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs8, ensure_ascii=False)},
        {"role": "assistant", "content": "Im Dienst: " + ", ".join(key8["facts"]["emp_ids"]) + "."},
    ]
    r8 = score_trajectory(task8, msgs8, key8)
    assert r8["score"] == 1.0, f"good search-INFO should pass: {r8}"

    # ------------------------------- wave-3 cases 9-15 -------------------------------------
    # 9) refusal: correct check-first refusal -> 1.0; forbidden write attempted -> 0.0 via
    #    no_forbidden (even though the terminal-status rejection mutated nothing); an actually
    #    MUTATING write on a refusal task -> 0.0 via no_write.
    tid9 = next(i for i, k in keys.items() if k["template"] == "t_refusal_policy")
    task9, key9 = tasks[tid9], keys[tid9]
    oid9 = key9["facts"]["auftrag_id"]
    zn9 = key9["oracle_calls"][0]["arguments"]["kennung"]
    env9 = dom.get_environment(solo_mode=True)
    obs9 = env9.use_tool("wartung_status", kennung=zn9)
    msgs9 = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task9["ticket"]},
        {"role": "assistant", "content": "<think>\nErst Status prüfen.\n</think>\n\n",
         "tool_calls": [_mk_call(1, "wartung_status", kennung=zn9)]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs9, ensure_ascii=False)},
        {"role": "assistant", "content": f"{oid9} ist bereits abgeschlossen (Endstatus) — laut "
                                         f"Richtlinie keine Änderung möglich; ich habe nichts geändert."},
    ]
    r9 = score_trajectory(task9, msgs9, key9)
    assert r9["score"] == 1.0 and r9["components"]["no_write"] and r9["components"]["no_forbidden"], \
        f"correct refusal should pass: {r9}"
    bad9 = json.loads(json.dumps(msgs9))
    bad9[2]["tool_calls"].append(_mk_call(2, "wartung_status_setzen", auftrag_id=oid9, status="geplant"))
    bad9.insert(4, {"role": "tool", "tool_call_id": "call_2",
                    "content": json.dumps({"error": "ValueError: Endstatus"}, ensure_ascii=False)})
    r9b = score_trajectory(task9, bad9, key9)
    assert r9b["score"] == 0.0 and not r9b["components"]["no_forbidden"], \
        f"forbidden write attempt must fail via no_forbidden: {r9b}"
    tid9c = next(i for i, k in keys.items() if k["template"] == "t_refusal_nicht_machbar")
    task9c, key9c = tasks[tid9c], keys[tid9c]
    zn9c = key9c["facts"]["zugnummer"]
    env9c = dom.get_environment(solo_mode=True)
    obs9c1 = env9c.use_tool("fahrplan", zugnummer=zn9c)
    vid9c = env9c.use_tool("wartung_status", kennung=zn9c)["fahrzeug_id"]
    obs9c2 = env9c.use_tool("wartung_einplanen", fahrzeug_id=vid9c, typ="Inspektion",
                            faellig_am="2026-07-05 06:00")
    msgs9c = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task9c["ticket"]},
        {"role": "assistant", "content": "",
         "tool_calls": [_mk_call(1, "fahrplan", zugnummer=zn9c),
                        _mk_call(2, "wartung_einplanen", fahrzeug_id=vid9c, typ="Inspektion",
                                 faellig_am="2026-07-05 06:00")]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs9c1, ensure_ascii=False)},
        {"role": "tool", "tool_call_id": "call_2", "content": json.dumps(obs9c2, ensure_ascii=False,
                                                                         default=str)},
        {"role": "assistant", "content": f"{zn9c} stornier ich nicht, stattdessen Wartung angelegt."},
    ]
    r9c = score_trajectory(task9c, msgs9c, key9c)
    assert r9c["score"] == 0.0 and not r9c["components"]["no_write"], \
        f"mutating write on refusal task must fail via no_write: {r9c}"

    # 10) batch-parallel: THREE calls in ONE assistant turn -> 1.0, parallel aux == 3
    tid10 = next(i for i, k in keys.items() if k["template"] == "t_info_batch_verspaetung")
    task10, key10 = tasks[tid10], keys[tid10]
    env10 = dom.get_environment(solo_mode=True)
    inits10 = (task10.get("initial_state") or {}).get("initialization_actions") or []
    env10.run_env_function_calls([EnvFunctionCall.model_validate(a) for a in inits10])
    calls10 = [_mk_call(i + 1, c["name"], **c["arguments"])
               for i, c in enumerate(key10["oracle_calls"])]
    obs10 = [env10.use_tool(c["name"], **c["arguments"]) for c in key10["oracle_calls"]]
    comm10 = task10["evaluation_criteria"]["communicate_info"]
    msgs10 = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task10["ticket"]},
        {"role": "assistant", "content": "<think>\nDrei unabhängige Abfragen — bündeln.\n</think>\n\n",
         "tool_calls": calls10},
    ] + [{"role": "tool", "tool_call_id": f"call_{i + 1}",
          "content": json.dumps(o, ensure_ascii=False)} for i, o in enumerate(obs10)] + [
        {"role": "assistant", "content": "Lage: " + "; ".join(comm10) + "."},
    ]
    r10 = score_trajectory(task10, msgs10, key10)
    assert r10["score"] == 1.0 and r10["parallel_max_calls_per_turn"] == 3, \
        f"parallel batch should pass with parallel_max==3: {r10}"

    # 11) transient: fail -> identical retry -> 1.0 with n_tool_errors==1; skipping the retry and
    #     claiming a halt anyway -> 0.0 (the halt name is grounded nowhere)
    tid11 = next(i for i, k in keys.items() if k["template"] == "t_info_transient")
    task11, key11 = tasks[tid11], keys[tid11]
    zn11 = key11["oracle_calls"][0]["arguments"]["zugnummer"]
    env11 = dom.get_environment(solo_mode=True)
    env11.run_env_function_calls([EnvFunctionCall.model_validate(a)
                                  for a in task11["initial_state"]["initialization_actions"]])
    try:
        env11.use_tool("zugstandort", zugnummer=zn11)
        raise AssertionError("transient fault must fire on the first call")
    except ValueError as e:
        err11 = json.dumps({"error": f"ValueError: {e}"}, ensure_ascii=False)
    obs11 = env11.use_tool("zugstandort", zugnummer=zn11)
    halt11 = key11["facts"]["naechster_halt"]
    msgs11 = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task11["ticket"]},
        {"role": "assistant", "content": "", "tool_calls": [_mk_call(1, "zugstandort", zugnummer=zn11)]},
        {"role": "tool", "tool_call_id": "call_1", "content": err11},
        {"role": "assistant", "content": "<think>\nVorübergehend — genau ein Retry.\n</think>\n\n",
         "tool_calls": [_mk_call(2, "zugstandort", zugnummer=zn11)]},
        {"role": "tool", "tool_call_id": "call_2", "content": json.dumps(obs11, ensure_ascii=False)},
        {"role": "assistant", "content": f"{zn11} ist unterwegs, nächster Halt {halt11}."},
    ]
    r11 = score_trajectory(task11, msgs11, key11)
    assert r11["score"] == 1.0 and r11["n_tool_errors"] == 1, f"transient retry should pass: {r11}"
    # skipping the retry and GUESSING a halt fails via communicate (the exact halt out of 576
    # stations must be named; station names are deliberately not a grounding pattern, so only
    # a 1/576-lucky exact guess would slip through — accepted residual)
    msgs11b = msgs11[:4] + [{"role": "assistant",
                             "content": f"{zn11} ist unterwegs, nächster Halt Musterstadt Hbf."}]
    r11b = score_trajectory(task11, msgs11b, key11)
    assert r11b["score"] == 0.0 and not r11b["components"]["communicate"], \
        f"guessed wrong halt without the retry must fail communicate: {r11b}"

    # 12) efficiency is SOFT: 7 calls at expected 1 -> metrics flag it, score stays 1.0
    msgs12 = json.loads(json.dumps(msgs3))
    msgs12[2]["tool_calls"] = [_mk_call(i, "mitarbeiter_info", zugnummer=zn) for i in range(1, 8)]
    r12 = score_trajectory(task3, msgs12, key3)
    assert r12["score"] == 1.0 and r12["efficiency_within_3x"] == 0.0 \
        and r12["efficiency_call_ratio"] == 7.0, f"efficiency must stay soft: {r12}"

    # 13) data gap: honest gap answer -> 1.0; invented time -> 0.0 (grounding)
    tid13 = next(i for i, k in keys.items()
                 if k["template"] == "t_info_datenluecke" and k["facts"].get("flavor") == "standort")
    task13, key13 = tasks[tid13], keys[tid13]
    zn13 = key13["oracle_calls"][0]["arguments"]["zugnummer"]
    env13 = dom.get_environment(solo_mode=True)
    env13.run_env_function_calls([EnvFunctionCall.model_validate(a)
                                  for a in task13["initial_state"]["initialization_actions"]])
    obs13 = env13.use_tool("zugstandort", zugnummer=zn13)
    msgs13 = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task13["ticket"]},
        {"role": "assistant", "content": "", "tool_calls": [_mk_call(1, "zugstandort", zugnummer=zn13)]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs13, ensure_ascii=False)},
        {"role": "assistant", "content": f"Für {zn13} liegt keine aktuelle Positionsmeldung vor."},
    ]
    r13 = score_trajectory(task13, msgs13, key13)
    assert r13["score"] == 1.0, f"honest gap answer should pass: {r13}"
    bad13 = json.loads(json.dumps(msgs13))
    bad13[-1]["content"] = f"{zn13} passiert um 13:37 den nächsten Meldepunkt."
    r13b = score_trajectory(task13, bad13, key13)
    assert r13b["score"] == 0.0 and not r13b["components"]["grounding"], \
        f"invented time on a data gap must fail grounding: {r13b}"

    # 14) name-filter search: ambiguous name -> refine by base -> 1.0
    tid14 = next(i for i, k in keys.items() if k["template"] == "t_info_name_suche")
    task14, key14 = tasks[tid14], keys[tid14]
    env14 = dom.get_environment(solo_mode=True)
    o14 = [env14.use_tool(c["name"], **c["arguments"]) for c in key14["oracle_calls"]]
    msgs14 = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task14["ticket"]},
        {"role": "assistant", "content": "",
         "tool_calls": [_mk_call(1, key14["oracle_calls"][0]["name"],
                                 **key14["oracle_calls"][0]["arguments"])]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(o14[0], ensure_ascii=False)},
        {"role": "assistant", "content": "",
         "tool_calls": [_mk_call(2, key14["oracle_calls"][1]["name"],
                                 **key14["oracle_calls"][1]["arguments"])]},
        {"role": "tool", "tool_call_id": "call_2", "content": json.dumps(o14[1], ensure_ascii=False)},
        {"role": "assistant", "content": f"Gefunden: {key14['facts']['emp_id']} "
                                         f"({key14['facts']['rolle']})."},
    ]
    r14 = score_trajectory(task14, msgs14, key14)
    assert r14["score"] == 1.0, f"name search should pass: {r14}"

    # 15) think-strip parity: <seed:think> dialect with a bogus id -> still 1.0
    seed15 = json.loads(json.dumps(msgs))
    seed15[-1]["content"] = ("<seed:think>vielleicht MA-99999?</seed:think>" + seed15[-1]["content"])
    r15 = score_trajectory(task, seed15, key)
    assert r15["score"] == 1.0, f"<seed:think> must be stripped like <think>: {r15}"

    # ------------------------------- wave-3.5 cases 16-21 ----------------------------------
    def _oracle_trace(task_x, key_x):
        """Replay oracle_calls on a fresh initialized env -> (messages sans final, env)."""
        env_x = dom.get_environment(solo_mode=True)
        inits_x = (task_x.get("initial_state") or {}).get("initialization_actions") or []
        if inits_x:
            env_x.run_env_function_calls([EnvFunctionCall.model_validate(a) for a in inits_x])
        ms = [{"role": "system", "content": "…"}, {"role": "user", "content": task_x["ticket"]}]
        for i, c in enumerate(key_x["oracle_calls"], 1):
            ms.append({"role": "assistant", "content": "",
                       "tool_calls": [_mk_call(i, c["name"], **c["arguments"])]})
            try:
                obs_x = env_x.use_tool(c["name"], **c["arguments"])
                body = json.dumps(obs_x, ensure_ascii=False, default=str)
            except Exception as e:
                body = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
            ms.append({"role": "tool", "tool_call_id": f"call_{i}", "content": body})
        return ms

    # 16) K1 conjunction: full pass -> 1.0; SAME trace minus the "bereits"-part in the final
    #     answer -> 0.0 via communicate (nails down that comm gates action tasks)
    tid16 = next(i for i, k in keys.items() if k["template"] == "t_action_lagebericht")
    task16, key16 = tasks[tid16], keys[tid16]
    comm16 = task16["evaluation_criteria"]["communicate_info"]
    msgs16 = _oracle_trace(task16, key16) + [
        {"role": "assistant", "content": "Lage: " + "; ".join(comm16) + "."}]
    r16 = score_trajectory(task16, msgs16, key16)
    assert r16["score"] == 1.0, f"K1 full conjunction should pass: {r16}"
    bad16 = json.loads(json.dumps(msgs16))
    bad16[-1]["content"] = "Lage: " + "; ".join(c for c in comm16 if c != "bereits") + ". Zuteilung erledigt."
    r16b = score_trajectory(task16, bad16, key16)
    assert r16b["score"] == 0.0 and not r16b["components"]["communicate"], \
        f"K1 without the 'bereits' acknowledgement must fail communicate: {r16b}"

    # 17) K2: 2 writes + rejected terminal write -> 1.0 with error+replan; only 1 write -> 0.0
    tid17 = next(i for i, k in keys.items() if k["template"] == "t_action_batch_konflikt")
    task17, key17 = tasks[tid17], keys[tid17]
    comm17 = task17["evaluation_criteria"]["communicate_info"]
    msgs17 = _oracle_trace(task17, key17) + [
        {"role": "assistant", "content": "Erledigt: " + "; ".join(comm17) + "."}]
    r17 = score_trajectory(task17, msgs17, key17)
    assert r17["score"] == 1.0 and r17["n_tool_errors"] == 1 and r17["replan_occurred"] == 1.0, \
        f"K2 batch with rejection should pass: {r17}"
    only_one = json.loads(json.dumps(msgs17))
    # drop the SECOND valid write (assistant+tool pair) -> db hash mismatch
    a_open = key17["facts"]["offen"][1]
    idx17 = next(j for j, m in enumerate(only_one) if m.get("tool_calls")
                 and json.loads(m["tool_calls"][0]["function"]["arguments"]).get("auftrag_id") == a_open)
    del only_one[idx17:idx17 + 2]
    r17b = score_trajectory(task17, only_one, key17)
    assert r17b["score"] == 0.0 and not r17b["components"]["db_match"], \
        f"K2 with a missing write must fail db_match: {r17b}"

    # 18) K3: honest phantom flag -> 1.0; silently skipping the phantom -> 0.0 communicate
    tid18 = next(i for i, k in keys.items() if k["template"] == "t_info_batch_phantom")
    task18, key18 = tasks[tid18], keys[tid18]
    comm18 = task18["evaluation_criteria"]["communicate_info"]
    msgs18 = _oracle_trace(task18, key18) + [
        {"role": "assistant", "content": "Lage: " + "; ".join(comm18) + "."}]
    r18 = score_trajectory(task18, msgs18, key18)
    assert r18["score"] == 1.0, f"K3 honest phantom report should pass: {r18}"
    bad18 = json.loads(json.dumps(msgs18))
    bad18[-1]["content"] = "Lage: " + "; ".join(c for c in comm18 if "gefunden" not in c) + "."
    r18b = score_trajectory(task18, bad18, key18)
    assert r18b["score"] == 0.0 and not r18b["components"]["communicate"], \
        f"K3 without the phantom flag must fail: {r18b}"

    # 19) T1: full iteration + the correct write -> 1.0; write on an UNAFFECTED train -> 0.0
    tid19 = next(i for i, k in keys.items() if k["template"] == "t_action_iteration_ersatz")
    task19, key19 = tasks[tid19], keys[tid19]
    comm19 = task19["evaluation_criteria"]["communicate_info"]
    msgs19 = _oracle_trace(task19, key19) + [
        {"role": "assistant", "content": "Betroffen: " + "; ".join(comm19) + "."}]
    r19 = score_trajectory(task19, msgs19, key19)
    assert r19["score"] == 1.0, f"T1 iteration+write should pass: {r19}"
    wrong19 = json.loads(json.dumps(msgs19))
    del wrong19[-3:-1]  # drop the final write (assistant+tool pair) -> asserts/db_match fail
    r19b = score_trajectory(task19, wrong19, key19)
    assert r19b["score"] == 0.0, f"T1 without the write must fail: {r19b}"

    # 20) T2: transient fail -> retry -> assign -> 1.0 with self_recovery; skipping the retry
    #     and guessing the minutes -> 0.0 (grounding/communicate)
    tid20 = next(i for i, k in keys.items() if k["template"] == "t_action_doppelfault")
    task20, key20 = tasks[tid20], keys[tid20]
    comm20 = task20["evaluation_criteria"]["communicate_info"]
    msgs20 = _oracle_trace(task20, key20) + [
        {"role": "assistant", "content": "Erledigt: " + "; ".join(comm20) + "."}]
    r20 = score_trajectory(task20, msgs20, key20)
    assert r20["score"] == 1.0 and r20["n_tool_errors"] == 1 and r20["self_recovery"] == 1.0, \
        f"T2 retry roundtrip should pass: {r20}"
    skip20 = _oracle_trace(task20, {"oracle_calls": key20["oracle_calls"][:1]})  # only the failed call
    skip20 += [{"role": "assistant", "content": "Erledigt: " + "; ".join(comm20) + "."}]
    r20b = score_trajectory(task20, skip20, key20)
    assert r20b["score"] == 0.0, f"T2 claiming results without retry/write must fail: {r20b}"

    # 21) inspektion_bedingt clean branch after H5: free wording (no dictated phrase) -> 1.0
    tid21 = next((i for i, k in keys.items()
                  if k["template"] == "t_action_inspektion_bedingt"
                  and not keys[i]["injected"]), None)
    if tid21 is not None:  # clean-branch tasks exist only at fault_rate < 1
        task21, key21 = tasks[tid21], keys[tid21]
        msgs21 = _oracle_trace(task21, key21) + [
            {"role": "assistant", "content": "Der Zug liegt unter der Schwelle, ich habe nichts eingeplant."}]
        r21 = score_trajectory(task21, msgs21, key21)
        assert r21["score"] == 1.0, f"H5: free clean-branch wording must pass now: {r21}"

    print("trajectory_reward.py self-test OK "
          "(good-action 1.0 | wrong-write 0.0 | good-info 1.0 | hallucination 0.0 | injected+replan 1.0 | "
          "runtime-fault roundtrip 1.0 +self_recovery | ignored rejection 0.0 | search-info 1.0 | "
          "refusal 1.0/0.0/0.0 | batch-parallel 1.0 (max=3) | transient 1.0/0.0 | efficiency-soft | "
          "data-gap 1.0/0.0 | name-search 1.0 | seed-think parity 1.0 | "
          "w35: K1-conjunction 1.0/0.0 | K2-batch-konflikt 1.0/0.0 | K3-phantom 1.0/0.0 | "
          "T1-iteration 1.0/0.0 | T2-doppelfault 1.0/0.0 | H5-clean-frei 1.0)")


if __name__ == "__main__":
    _selftest()

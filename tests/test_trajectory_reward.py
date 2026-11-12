"""
tests/test_trajectory_reward.py
===============================
The verifier's scenario suite (formerly `_selftest` inside evaluation/trajectory_reward.py): known-good
traces -> 1.0; wrong write, hallucination, ignored rejection, skipped retry -> 0.0. Runs against the
generated db_bahn pool (see conftest) in the tau2 venv: `bash ops/run_tests.sh` or
`.venv-tau2/bin/python -m pytest tests/test_trajectory_reward.py`.

Case numbers follow the original suite (1-21: wave 2 / 3 / 3.5; 22: typography 12.11.2026). Shared preparations are module-scoped
fixtures; variants mutate JSON copies, never the fixture objects.
"""
import json

import pytest

from evaluation.trajectory_reward import score_trajectory
from tests.db_bahn_helpers import copy, env_for, mk_call, oracle_trace


# ------------------------------------------------------------------ shared scenarios (wave 2)
@pytest.fixture(scope="module")
def action_crew(dom, tasks, keys, pick):
    """Case 1: good ACTION trace on t_action_crew — reference action echoed with a grounded confirmation."""
    tid = pick("t_action_crew")
    task, key = tasks[tid], keys[tid]
    ref = task["evaluation_criteria"]["actions"][0]
    obs = dom.get_environment(solo_mode=True).use_tool(ref["name"], **ref["arguments"])
    msgs = [
        {"role": "system", "content": "…"},
        {"role": "user", "content": task["ticket"]},
        {"role": "assistant", "content": "<plan>Zuteilung ausführen.</plan>",
         "tool_calls": [mk_call(1, ref["name"], **ref["arguments"])]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs, ensure_ascii=False, default=str)},
        {"role": "assistant", "content":
            f"Erledigt: {key['facts']['emp_id']} ist dem Zug als {key['facts']['rolle']} zugeteilt."},
    ]
    return task, key, msgs


@pytest.fixture(scope="module")
def info_crew(dom, tasks, keys, pick):
    """Case 3: good INFO trace on t_info_crew — tool called, facts grounded."""
    tid = pick("t_info_crew")
    task, key = tasks[tid], keys[tid]
    zug = key["facts"]["lokfuehrer"][0]
    zn = task["ticket"].split("auf ")[1].split(" als")[0]
    obs = dom.get_environment(solo_mode=True).use_tool("mitarbeiter_info", zugnummer=zn)
    msgs = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task["ticket"]},
        {"role": "assistant", "content": "<plan>Besatzung abfragen.</plan>",
         "tool_calls": [mk_call(1, "mitarbeiter_info", zugnummer=zn)]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs, ensure_ascii=False)},
        {"role": "assistant", "content": f"Als Lokführer ist {zug['name']} ({zug['mitarbeiter_id']}) eingeteilt."},
    ]
    return task, key, zug, zn, msgs


@pytest.fixture(scope="module")
def ersatz_quali(dom, tasks, keys, pick):
    """Case 6: runtime-fault roundtrip on t_action_ersatz_quali — invalid crew_zuweisen -> error
    observation -> search -> valid assignment -> grounded final."""
    tid = pick("t_action_ersatz_quali")
    task, key = tasks[tid], keys[tid]
    ref = task["evaluation_criteria"]["actions"][0]
    zn = ref["arguments"]["zugnummer"]
    proposed = key["facts"]["vorschlag_ungueltig"]
    env = env_for(dom, task)
    obs_a = env.use_tool("mitarbeiter_info", zugnummer=zn)
    try:
        env.use_tool("crew_zuweisen", zugnummer=zn, mitarbeiter_id=proposed, rolle="Lokführer")
        raise AssertionError("invalid assignment must be rejected by the qualification/role gate")
    except ValueError as e:
        err = json.dumps({"error": f"ValueError: {e}"}, ensure_ascii=False)
    obs_b = env.use_tool("mitarbeiter_suchen", **key["oracle_calls"][1]["arguments"])
    obs_c = env.use_tool(ref["name"], **ref["arguments"])
    msgs = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task["ticket"]},
        {"role": "assistant", "content": "<plan>Besatzung prüfen, dann Vorschlag zuteilen.</plan>",
         "tool_calls": [mk_call(1, "mitarbeiter_info", zugnummer=zn),
                        mk_call(2, "crew_zuweisen", zugnummer=zn, mitarbeiter_id=proposed, rolle="Lokführer")]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs_a, ensure_ascii=False)},
        {"role": "tool", "tool_call_id": "call_2", "content": err},
        {"role": "assistant", "content": "<plan>Abgelehnt – qualifizierten Ersatz suchen und den "
                                         "ersten Treffer zuteilen.</plan>",
         "tool_calls": [mk_call(3, "mitarbeiter_suchen", **key["oracle_calls"][1]["arguments"]),
                        mk_call(4, ref["name"], **ref["arguments"])]},
        {"role": "tool", "tool_call_id": "call_3", "content": json.dumps(obs_b, ensure_ascii=False)},
        {"role": "tool", "tool_call_id": "call_4", "content": json.dumps(obs_c, ensure_ascii=False, default=str)},
        {"role": "assistant", "content": f"{proposed} wurde abgelehnt (fehlende Qualifikation); "
                                         f"{key['facts']['ersatz_id']} ist jetzt als Lokführer zugeteilt."},
    ]
    return task, key, zn, proposed, err, msgs


# ------------------------------------------------------------------ wave 2: cases 1-8
def test_01_good_action_passes(action_crew):
    task, key, msgs = action_crew
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0, f"good ACTION should pass: {r}"
    assert r["self_recovery"] == 0.0, f"clean trace must not flag self_recovery: {r}"


def test_01b_think_block_with_bogus_id_is_stripped(action_crew):
    # thinking-mode eval: speculative values in the reasoning must be stripped, not graded
    task, key, msgs = action_crew
    think = copy(msgs)
    think[-1]["content"] = ("<think>\nVielleicht ist auch MA-99999 verfügbar? Nein, laut Tool nicht.\n</think>\n\n"
                            + think[-1]["content"])
    r = score_trajectory(task, think, key)
    assert r["score"] == 1.0, f"<think> with bogus ID must be stripped, not graded: {r}"


def test_02_wrong_write_fails_db_match(action_crew):
    task, key, msgs = action_crew
    bad = copy(msgs)
    bad[2]["tool_calls"] = [mk_call(1, "wartung_einplanen",
                                    fahrzeug_id="EC-9006", typ="Inspektion", faellig_am="2026-07-05 06:00")]
    r = score_trajectory(task, bad, key)
    assert r["score"] == 0.0 and not r["components"]["db_match"], f"wrong write must fail: {r}"


def test_03_good_info_passes(info_crew):
    task, key, _zug, _zn, msgs = info_crew
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0, f"good INFO should pass: {r}"


def test_04_hallucinated_info_fails_grounding(info_crew):
    # no tool call, invented id but correct-looking answer
    task, key, zug, _zn, _msgs = info_crew
    msgs = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task["ticket"]},
        {"role": "assistant", "content": f"Als Lokführer ist Max Muster (MA-99999) eingeteilt. "
                                         f"{zug['name']} {zug['mitarbeiter_id']}"},
    ]
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 0.0 and not r["components"]["grounding"], f"hallucination must fail: {r}"
    assert not r["components"]["actions_pass"], "expected-tools check must fail without tool calls"


def test_05_injected_action_passes_with_replan(dom, tasks, keys, pick):
    # fault-injected task: init actions apply during replay; replan metric fires with >=2 plan turns
    tid = pick("t_action_ersatz")
    task, key = tasks[tid], keys[tid]
    ref = task["evaluation_criteria"]["actions"][0]
    env = env_for(dom, task)
    zn = ref["arguments"]["zugnummer"]
    obs_a = env.use_tool("mitarbeiter_info", zugnummer=zn)
    obs_b = env.use_tool(ref["name"], **ref["arguments"])
    msgs = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task["ticket"]},
        {"role": "assistant", "content": "<plan>Erst Besatzung prüfen.</plan>",
         "tool_calls": [mk_call(1, "mitarbeiter_info", zugnummer=zn)]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs_a, ensure_ascii=False)},
        {"role": "assistant", "content": "<plan>Kein Lokführer eingeteilt – Ersatz zuweisen.</plan>",
         "tool_calls": [mk_call(2, ref["name"], **ref["arguments"])]},
        {"role": "tool", "tool_call_id": "call_2", "content": json.dumps(obs_b, ensure_ascii=False, default=str)},
        {"role": "assistant", "content": f"{key['facts']['ersatz_id']} ist als Ersatz-Lokführer zugeteilt."},
    ]
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0 and r["replan_occurred"] == 1.0, f"injected ACTION should pass with replan: {r}"


def test_06_runtime_fault_roundtrip_passes_with_self_recovery(ersatz_quali):
    task, key, _zn, _proposed, _err, msgs = ersatz_quali
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0 and r["n_tool_errors"] == 1 and r["replan_occurred"] == 1.0, \
        f"runtime-fault roundtrip should pass with replan: {r}"
    assert r["self_recovery"] == 1.0, f"recovered-from-error trace must flag self_recovery: {r}"


def test_07_ignored_rejection_fails(ersatz_quali):
    # only the invalid attempt, then a confident claim
    task, key, zn, proposed, err, _msgs = ersatz_quali
    msgs = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task["ticket"]},
        {"role": "assistant", "content": "<plan>Vorschlag direkt zuteilen.</plan>",
         "tool_calls": [mk_call(1, "crew_zuweisen", zugnummer=zn, mitarbeiter_id=proposed, rolle="Lokführer")]},
        {"role": "tool", "tool_call_id": "call_1", "content": err},
        {"role": "assistant", "content": f"Erledigt: {proposed} ist {zn} als Lokführer zugeteilt."},
    ]
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 0.0 and not r["components"]["db_match"], f"ignored rejection must fail: {r}"


def test_08_search_info_via_oracle_calls_passes(dom, tasks, keys, pick):
    tid = pick("t_info_mitarbeiter_suche")
    task, key = tasks[tid], keys[tid]
    call = key["oracle_calls"][0]
    obs = dom.get_environment(solo_mode=True).use_tool(call["name"], **call["arguments"])
    msgs = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task["ticket"]},
        {"role": "assistant", "content": "<plan>Mitarbeitersuche mit den Filtern aus dem Ticket.</plan>",
         "tool_calls": [mk_call(1, call["name"], **call["arguments"])]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs, ensure_ascii=False)},
        {"role": "assistant", "content": "Im Dienst: " + ", ".join(key["facts"]["emp_ids"]) + "."},
    ]
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0, f"good search-INFO should pass: {r}"


# ------------------------------------------------------------------ wave 3: cases 9-15
@pytest.fixture(scope="module")
def refusal_policy(dom, tasks, keys, pick):
    """Case 9: check-first refusal on t_refusal_policy — terminal status, change nothing, justify."""
    tid = pick("t_refusal_policy")
    task, key = tasks[tid], keys[tid]
    oid = key["facts"]["auftrag_id"]
    zn = key["oracle_calls"][0]["arguments"]["kennung"]
    obs = dom.get_environment(solo_mode=True).use_tool("wartung_status", kennung=zn)
    msgs = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task["ticket"]},
        {"role": "assistant", "content": "<think>\nErst Status prüfen.\n</think>\n\n",
         "tool_calls": [mk_call(1, "wartung_status", kennung=zn)]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs, ensure_ascii=False)},
        {"role": "assistant", "content": f"{oid} ist bereits abgeschlossen (Endstatus) — laut "
                                         f"Richtlinie keine Änderung möglich; ich habe nichts geändert."},
    ]
    return task, key, oid, msgs


def test_09_correct_refusal_passes(refusal_policy):
    task, key, _oid, msgs = refusal_policy
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0 and r["components"]["no_write"] and r["components"]["no_forbidden"], \
        f"correct refusal should pass: {r}"


def test_09b_forbidden_write_attempt_fails_no_forbidden(refusal_policy):
    # the terminal-status rejection mutated nothing — the ATTEMPT is what fails
    task, key, oid, msgs = refusal_policy
    bad = copy(msgs)
    bad[2]["tool_calls"].append(mk_call(2, "wartung_status_setzen", auftrag_id=oid, status="geplant"))
    bad.insert(4, {"role": "tool", "tool_call_id": "call_2",
                   "content": json.dumps({"error": "ValueError: Endstatus"}, ensure_ascii=False)})
    r = score_trajectory(task, bad, key)
    assert r["score"] == 0.0 and not r["components"]["no_forbidden"], \
        f"forbidden write attempt must fail via no_forbidden: {r}"


def test_09c_mutating_write_on_refusal_task_fails_no_write(dom, tasks, keys, pick):
    tid = pick("t_refusal_nicht_machbar")
    task, key = tasks[tid], keys[tid]
    zn = key["facts"]["zugnummer"]
    env = dom.get_environment(solo_mode=True)
    obs1 = env.use_tool("fahrplan", zugnummer=zn)
    vid = env.use_tool("wartung_status", kennung=zn)["fahrzeug_id"]
    obs2 = env.use_tool("wartung_einplanen", fahrzeug_id=vid, typ="Inspektion", faellig_am="2026-07-05 06:00")
    msgs = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task["ticket"]},
        {"role": "assistant", "content": "",
         "tool_calls": [mk_call(1, "fahrplan", zugnummer=zn),
                        mk_call(2, "wartung_einplanen", fahrzeug_id=vid, typ="Inspektion",
                                faellig_am="2026-07-05 06:00")]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs1, ensure_ascii=False)},
        {"role": "tool", "tool_call_id": "call_2", "content": json.dumps(obs2, ensure_ascii=False, default=str)},
        {"role": "assistant", "content": f"{zn} stornier ich nicht, stattdessen Wartung angelegt."},
    ]
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 0.0 and not r["components"]["no_write"], \
        f"mutating write on refusal task must fail via no_write: {r}"


def test_10_batch_parallel_three_calls_in_one_turn(dom, tasks, keys, pick):
    tid = pick("t_info_batch_verspaetung")
    task, key = tasks[tid], keys[tid]
    env = env_for(dom, task)
    calls = [mk_call(i + 1, c["name"], **c["arguments"]) for i, c in enumerate(key["oracle_calls"])]
    obs = [env.use_tool(c["name"], **c["arguments"]) for c in key["oracle_calls"]]
    comm = task["evaluation_criteria"]["communicate_info"]
    msgs = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task["ticket"]},
        {"role": "assistant", "content": "<think>\nDrei unabhängige Abfragen — bündeln.\n</think>\n\n",
         "tool_calls": calls},
    ] + [{"role": "tool", "tool_call_id": f"call_{i + 1}", "content": json.dumps(o, ensure_ascii=False)}
         for i, o in enumerate(obs)] + [
        {"role": "assistant", "content": "Lage: " + "; ".join(comm) + "."},
    ]
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0 and r["parallel_max_calls_per_turn"] == 3, \
        f"parallel batch should pass with parallel_max==3: {r}"


@pytest.fixture(scope="module")
def transient(dom, tasks, keys, pick):
    """Case 11: transient fault on t_info_transient — the first zugstandort call fails, the identical retry succeeds."""
    tid = pick("t_info_transient")
    task, key = tasks[tid], keys[tid]
    zn = key["oracle_calls"][0]["arguments"]["zugnummer"]
    env = env_for(dom, task)
    try:
        env.use_tool("zugstandort", zugnummer=zn)
        raise AssertionError("transient fault must fire on the first call")
    except ValueError as e:
        err = json.dumps({"error": f"ValueError: {e}"}, ensure_ascii=False)
    obs = env.use_tool("zugstandort", zugnummer=zn)
    halt = key["facts"]["naechster_halt"]
    msgs = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task["ticket"]},
        {"role": "assistant", "content": "", "tool_calls": [mk_call(1, "zugstandort", zugnummer=zn)]},
        {"role": "tool", "tool_call_id": "call_1", "content": err},
        {"role": "assistant", "content": "<think>\nVorübergehend — genau ein Retry.\n</think>\n\n",
         "tool_calls": [mk_call(2, "zugstandort", zugnummer=zn)]},
        {"role": "tool", "tool_call_id": "call_2", "content": json.dumps(obs, ensure_ascii=False)},
        {"role": "assistant", "content": f"{zn} ist unterwegs, nächster Halt {halt}."},
    ]
    return task, key, zn, msgs


def test_11_transient_retry_passes(transient):
    task, key, _zn, msgs = transient
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0 and r["n_tool_errors"] == 1, f"transient retry should pass: {r}"


def test_11b_skipped_retry_with_guessed_halt_fails_communicate(transient):
    # the exact halt out of 576 stations must be named; station names are deliberately not a grounding
    # pattern, so only a 1/576-lucky exact guess would slip through — accepted residual
    task, key, zn, msgs = transient
    guess = msgs[:4] + [{"role": "assistant", "content": f"{zn} ist unterwegs, nächster Halt Musterstadt Hbf."}]
    r = score_trajectory(task, guess, key)
    assert r["score"] == 0.0 and not r["components"]["communicate"], \
        f"guessed wrong halt without the retry must fail communicate: {r}"


def test_12_efficiency_is_soft(info_crew):
    # 7 calls at expected 1 -> metrics flag it, score stays 1.0
    task, key, _zug, zn, msgs = info_crew
    many = copy(msgs)
    many[2]["tool_calls"] = [mk_call(i, "mitarbeiter_info", zugnummer=zn) for i in range(1, 8)]
    r = score_trajectory(task, many, key)
    assert r["score"] == 1.0 and r["efficiency_within_3x"] == 0.0 and r["efficiency_call_ratio"] == 7.0, \
        f"efficiency must stay soft: {r}"


@pytest.fixture(scope="module")
def data_gap(dom, tasks, keys, pick):
    """Case 13: data gap (flavor standort) on t_info_datenluecke — the honest gap answer."""
    tid = pick("t_info_datenluecke", flavor="standort")
    task, key = tasks[tid], keys[tid]
    zn = key["oracle_calls"][0]["arguments"]["zugnummer"]
    obs = env_for(dom, task).use_tool("zugstandort", zugnummer=zn)
    msgs = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task["ticket"]},
        {"role": "assistant", "content": "", "tool_calls": [mk_call(1, "zugstandort", zugnummer=zn)]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs, ensure_ascii=False)},
        {"role": "assistant", "content": f"Für {zn} liegt keine aktuelle Positionsmeldung vor."},
    ]
    return task, key, zn, msgs


def test_13_honest_gap_answer_passes(data_gap):
    task, key, _zn, msgs = data_gap
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0, f"honest gap answer should pass: {r}"


def test_13b_invented_time_on_gap_fails_grounding(data_gap):
    task, key, zn, msgs = data_gap
    bad = copy(msgs)
    bad[-1]["content"] = f"{zn} passiert um 13:37 den nächsten Meldepunkt."
    r = score_trajectory(task, bad, key)
    assert r["score"] == 0.0 and not r["components"]["grounding"], \
        f"invented time on a data gap must fail grounding: {r}"


def test_14_ambiguous_name_search_refined_by_base_passes(dom, tasks, keys, pick):
    tid = pick("t_info_name_suche")
    task, key = tasks[tid], keys[tid]
    env = dom.get_environment(solo_mode=True)
    calls = key["oracle_calls"]
    obs = [env.use_tool(c["name"], **c["arguments"]) for c in calls]
    msgs = [
        {"role": "system", "content": "…"}, {"role": "user", "content": task["ticket"]},
        {"role": "assistant", "content": "", "tool_calls": [mk_call(1, calls[0]["name"], **calls[0]["arguments"])]},
        {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(obs[0], ensure_ascii=False)},
        {"role": "assistant", "content": "", "tool_calls": [mk_call(2, calls[1]["name"], **calls[1]["arguments"])]},
        {"role": "tool", "tool_call_id": "call_2", "content": json.dumps(obs[1], ensure_ascii=False)},
        {"role": "assistant", "content": f"Gefunden: {key['facts']['emp_id']} ({key['facts']['rolle']})."},
    ]
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0, f"name search should pass: {r}"


def test_15_seed_think_dialect_is_stripped_like_think(action_crew):
    task, key, msgs = action_crew
    seed = copy(msgs)
    seed[-1]["content"] = "<seed:think>vielleicht MA-99999?</seed:think>" + seed[-1]["content"]
    r = score_trajectory(task, seed, key)
    assert r["score"] == 1.0, f"<seed:think> must be stripped like <think>: {r}"


# ------------------------------------------------------------------ wave 3.5: cases 16-21
def _scenario(dom, tasks, keys, pick, template, prefix):
    """oracle_calls replayed on a fresh env + a final answer that states every communicate_info item."""
    tid = pick(template)
    task, key = tasks[tid], keys[tid]
    comm = task["evaluation_criteria"]["communicate_info"]
    msgs = oracle_trace(dom, task, key) + [{"role": "assistant", "content": prefix + "; ".join(comm) + "."}]
    return task, key, comm, msgs


@pytest.fixture(scope="module")
def k1_lagebericht(dom, tasks, keys, pick):
    return _scenario(dom, tasks, keys, pick, "t_action_lagebericht", "Lage: ")


@pytest.fixture(scope="module")
def k2_batch_konflikt(dom, tasks, keys, pick):
    return _scenario(dom, tasks, keys, pick, "t_action_batch_konflikt", "Erledigt: ")


@pytest.fixture(scope="module")
def k3_batch_phantom(dom, tasks, keys, pick):
    return _scenario(dom, tasks, keys, pick, "t_info_batch_phantom", "Lage: ")


@pytest.fixture(scope="module")
def t1_iteration_ersatz(dom, tasks, keys, pick):
    return _scenario(dom, tasks, keys, pick, "t_action_iteration_ersatz", "Betroffen: ")


@pytest.fixture(scope="module")
def t2_doppelfault(dom, tasks, keys, pick):
    return _scenario(dom, tasks, keys, pick, "t_action_doppelfault", "Erledigt: ")


def test_16_k1_full_conjunction_passes(k1_lagebericht):
    task, key, _comm, msgs = k1_lagebericht
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0, f"K1 full conjunction should pass: {r}"


def test_16b_k1_without_bereits_acknowledgement_fails_communicate(k1_lagebericht):
    # nails down that communicate gates action tasks
    task, key, comm, msgs = k1_lagebericht
    bad = copy(msgs)
    bad[-1]["content"] = "Lage: " + "; ".join(c for c in comm if c != "bereits") + ". Zuteilung erledigt."
    r = score_trajectory(task, bad, key)
    assert r["score"] == 0.0 and not r["components"]["communicate"], \
        f"K1 without the 'bereits' acknowledgement must fail communicate: {r}"


def test_17_k2_batch_with_rejected_terminal_write_passes(k2_batch_konflikt):
    task, key, _comm, msgs = k2_batch_konflikt
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0 and r["n_tool_errors"] == 1 and r["replan_occurred"] == 1.0, \
        f"K2 batch with rejection should pass: {r}"


def test_17b_k2_missing_second_write_fails_db_match(k2_batch_konflikt):
    task, key, _comm, msgs = k2_batch_konflikt
    only_one = copy(msgs)
    # drop the SECOND valid write (assistant+tool pair) -> db hash mismatch
    a_open = key["facts"]["offen"][1]
    idx = next(j for j, m in enumerate(only_one) if m.get("tool_calls")
               and json.loads(m["tool_calls"][0]["function"]["arguments"]).get("auftrag_id") == a_open)
    del only_one[idx:idx + 2]
    r = score_trajectory(task, only_one, key)
    assert r["score"] == 0.0 and not r["components"]["db_match"], f"K2 with a missing write must fail db_match: {r}"


def test_18_k3_honest_phantom_flag_passes(k3_batch_phantom):
    task, key, _comm, msgs = k3_batch_phantom
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0, f"K3 honest phantom report should pass: {r}"


def test_18b_k3_silently_skipping_phantom_fails_communicate(k3_batch_phantom):
    task, key, comm, msgs = k3_batch_phantom
    bad = copy(msgs)
    bad[-1]["content"] = "Lage: " + "; ".join(c for c in comm if "gefunden" not in c) + "."
    r = score_trajectory(task, bad, key)
    assert r["score"] == 0.0 and not r["components"]["communicate"], f"K3 without the phantom flag must fail: {r}"


def test_19_t1_full_iteration_with_write_passes(t1_iteration_ersatz):
    task, key, _comm, msgs = t1_iteration_ersatz
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0, f"T1 iteration+write should pass: {r}"


def test_19b_t1_without_the_write_fails(t1_iteration_ersatz):
    task, key, _comm, msgs = t1_iteration_ersatz
    wrong = copy(msgs)
    del wrong[-3:-1]  # drop the final write (assistant+tool pair) -> asserts/db_match fail
    r = score_trajectory(task, wrong, key)
    assert r["score"] == 0.0, f"T1 without the write must fail: {r}"


def test_20_t2_transient_retry_then_assign_passes_with_self_recovery(t2_doppelfault):
    task, key, _comm, msgs = t2_doppelfault
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0 and r["n_tool_errors"] == 1 and r["self_recovery"] == 1.0, \
        f"T2 retry roundtrip should pass: {r}"


def test_20b_t2_claiming_results_without_retry_or_write_fails(dom, t2_doppelfault):
    task, key, comm, _msgs = t2_doppelfault
    skip = oracle_trace(dom, task, {"oracle_calls": key["oracle_calls"][:1]})  # only the failed call
    skip += [{"role": "assistant", "content": "Erledigt: " + "; ".join(comm) + "."}]
    r = score_trajectory(task, skip, key)
    assert r["score"] == 0.0, f"T2 claiming results without retry/write must fail: {r}"


def test_21_h5_clean_branch_free_wording_passes(dom, tasks, keys, pick):
    # inspektion_bedingt clean branch after H5: free wording (no dictated phrase) -> 1.0
    try:
        tid = pick("t_action_inspektion_bedingt", injected=False)
    except AssertionError:
        pytest.skip("clean-branch tasks exist only at fault_rate < 1")
    task, key = tasks[tid], keys[tid]
    msgs = oracle_trace(dom, task, key) + [
        {"role": "assistant", "content": "Der Zug liegt unter der Schwelle, ich habe nichts eingeplant."}]
    r = score_trajectory(task, msgs, key)
    assert r["score"] == 1.0, f"H5: free clean-branch wording must pass now: {r}"


# ------------------------------------------------------------------ 12.11.2026: typographic variants
def test_22_typographic_spaces_and_hyphens_are_normalized(info_crew):
    # gpt-oss writes narrow no-break spaces / non-breaking hyphens ("EC 209", "MA‑123"):
    # the same characters to a reader -> the communicate substrings must match them
    task, key, _zug, _zn, msgs = info_crew
    comm = task["evaluation_criteria"]["communicate_info"]
    assert any(" " in c or "-" in c for c in comm), "case needs a communicate item with a space or hyphen"
    typo = copy(msgs)
    typo[-1]["content"] = typo[-1]["content"].replace(" ", " ").replace("-", "‑")
    r = score_trajectory(task, typo, key)
    assert r["score"] == 1.0, f"typographic spaces/hyphens must not fail a correct answer: {r}"


def test_22b_invented_id_with_nonbreaking_hyphen_fails_grounding(info_crew):
    # the other side of the same fix: an id written with U+2011 is no longer invisible to grounding
    task, key, _zug, _zn, msgs = info_crew
    bad = copy(msgs)
    bad[-1]["content"] += " Vertretung wäre MA‑99999."
    r = score_trajectory(task, bad, key)
    assert r["score"] == 0.0 and not r["components"]["grounding"], f"invented MA‑99999 must fail: {r}"

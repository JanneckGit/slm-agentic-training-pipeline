"""GRPO reward seam (evaluation/grpo_reward.py): constant reward-extra keys, error flags, ledger, and the
round-trip of stored eval traces through the whole seam.

Why the keys matter: verl takes the reward-extra keys from the first episode of a batch and indexes every
other episode with them, then asserts equal key sets/lists when it merges worker batches — one episode
with a different key set (the pre-r3 adapter-error dict) killed the run. `_verl_merge` below replays
those three steps (experimental/agent_loop/agent_loop.py `_postprocess`, utils/py_functional.py
`list_of_dict_to_dict_of_list`, protocol.py `DataProto.concat` meta_info check).
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

import evaluation.trajectory_reward as tr
from evaluation.grpo_reward import REWARD_KEYS, _INT_KEYS, compute_score, roundtrip_row

REPO = Path(__file__).resolve().parents[1]
EP3_TRACES = REPO / "data/generated/eval/db_traces_heldout_eval_after_qwen3-4b_ep3_k8.jsonl"


def _verl_merge(worker_batches: list[list[dict]]) -> dict:
    """verl's handling of reward_extra_info across episodes and workers (raises like verl does)."""
    per_worker = []
    for infos in worker_batches:
        keys = list(infos[0].keys())                                   # agent_loop: keys of the FIRST episode
        cols = {k: np.array([info[k] for info in infos]) for k in keys}  # KeyError on a missing key
        for info in infos:                                             # list_of_dict_to_dict_of_list
            for k in info:
                assert k in cols, f"Key '{k}' is not present in the keys of the first dictionary"
        per_worker.append((keys, cols))
    first_keys = per_worker[0][0]
    for keys, _ in per_worker:                                         # DataProto.concat meta_info
        assert keys == first_keys, "Conflicting values for meta_info key 'reward_extra_keys'"
    return {k: np.concatenate([c[k] for _, c in per_worker]) for k in first_keys}


def _fake_score(error=None, score=1.0):
    def fake(task, messages, answer_key=None, **kw):
        return {"score": score, "task_solved": score, "turns_used": 3, "n_tool_calls": 1, "n_tool_errors": 0,
                "tool_calls_valid": 1.0, "n_plan_turns": 1, "replan_occurred": 0.0, "self_recovery": 0.0,
                "parallel_max_calls_per_turn": 1, "efficiency_expected_calls": 1, "efficiency_call_ratio": 1.0,
                "efficiency_within_3x": 1.0, "components": {"grounding": error is None}, "error": error}
    return fake


EXTRA = {"task": json.dumps({"id": "t1"}), "answer_key": json.dumps({"kind": "info"})}


def test_normal_row_has_exact_keys_and_types(monkeypatch):
    monkeypatch.setattr(tr, "score_trajectory", _fake_score())
    out = compute_score("db_bahn", "Antwort", "t1", EXTRA)
    assert tuple(out) == REWARD_KEYS
    assert out["score"] == 1.0 and out["adapter_error"] == 0.0 and out["verifier_error"] == 0.0
    assert all(type(out[k]) is (int if k in _INT_KEYS else float) for k in REWARD_KEYS)


def test_adapter_error_is_flagged_not_fatal(capsys):
    out = compute_score("db_bahn", "Antwort", "t-broken", {"task": "{not json", "answer_key": "null"})
    assert tuple(out) == REWARD_KEYS
    assert out["score"] == 0.0 and out["adapter_error"] == 1.0 and out["verifier_error"] == 0.0
    assert all(type(out[k]) is (int if k in _INT_KEYS else float) for k in REWARD_KEYS)
    assert "ADAPTER_ERROR task=t-broken" in capsys.readouterr().err


@pytest.mark.parametrize("error,flag", [("verifier_exception: TypeError: x", 1.0), ("gold_replay_failed: y", 1.0),
                                        ("ungrounded_tokens: ['16:51']", 0.0), (None, 0.0)])
def test_verifier_error_flag(monkeypatch, capsys, error, flag):
    monkeypatch.setattr(tr, "score_trajectory", _fake_score(error, score=0.0 if error else 1.0))
    out = compute_score("db_bahn", "Antwort", "t1", EXTRA)
    assert tuple(out) == REWARD_KEYS and out["verifier_error"] == flag and out["adapter_error"] == 0.0
    if error and error.startswith("gold_replay_failed"):
        assert "gold_replay_failed" in capsys.readouterr().err


def test_verifier_fields_are_reward_keys(tasks, keys):
    """A new numeric verifier field must be added to REWARD_KEYS deliberately, not dropped silently."""
    tid = next(iter(tasks))
    raw = tr.compute_score(solution_str=[{"role": "assistant", "content": "x"}],
                           extra_info={"task": tasks[tid], "answer_key": keys.get(tid)})
    assert set(raw) <= set(REWARD_KEYS)
    assert set(raw) | {"n_msgs", "adapter_error"} == set(REWARD_KEYS)


def test_verl_merge_survives_every_order(monkeypatch):
    rows = []
    monkeypatch.setattr(tr, "score_trajectory", _fake_score())
    rows.append(compute_score("db_bahn", "ok", "t1", EXTRA))
    monkeypatch.setattr(tr, "score_trajectory", _fake_score("verifier_exception: boom", 0.0))
    rows.append(compute_score("db_bahn", "v", "t2", EXTRA))
    rows.append(compute_score("db_bahn", "a", "t3", {"task": "{", "answer_key": "null"}))
    for perm in itertools.permutations(rows):
        merged = _verl_merge([list(perm[:2]), list(perm[2:])])
        assert list(merged) == list(REWARD_KEYS) and merged["adapter_error"].sum() == 1
        assert merged["verifier_error"].sum() == 1


def test_old_error_dict_would_have_crashed(monkeypatch):
    """Regression proof: the pre-r3 error dict next to a normal row is fatal in verl's merge."""
    monkeypatch.setattr(tr, "score_trajectory", _fake_score())
    ok = compute_score("db_bahn", "ok", "t1", EXTRA)
    old = {"score": 0.0, "adapter_error": 1.0, "error_str": "TypeError: x"}
    with pytest.raises(KeyError):          # normal row first: the error row lacks task_solved, …
        _verl_merge([[ok, old]])
    with pytest.raises(KeyError):          # error row first: the normal row lacks error_str
        _verl_merge([[old, ok]])
    with pytest.raises(AssertionError):    # same keys inside each worker, different lists across workers
        _verl_merge([[ok], [old]])


def test_ledger_gets_one_line_per_rollout(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "score_trajectory", _fake_score())
    compute_score("db_bahn", "ok", "t1", {**EXTRA, "ledger_dir": str(tmp_path)})
    compute_score("db_bahn", "a", "t2", {"task": "{", "answer_key": "null", "ledger_dir": str(tmp_path)})
    lines = [json.loads(x) for f in tmp_path.glob("*.jsonl") for x in f.read_text().splitlines()]
    assert sorted((x["task_id"], x["score"]) for x in lines) == [("t1", 1.0), ("t2", 0.0)]


def test_model_caused_verifier_exception(tasks, keys):
    """A tool call with name null (the model emitted it) crashes the verifier's corpus builder ->
    verifier_exception -> flagged, still a constant-key row."""
    tid = next(iter(tasks))
    msgs = [{"role": "user", "content": tasks[tid].get("ticket", "")},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                                  "function": {"name": None, "arguments": "{}"}}]},
            {"role": "assistant", "content": "fertig"}]
    raw = tr.compute_score(solution_str=msgs, extra_info={"task": tasks[tid], "answer_key": keys.get(tid)})
    assert raw["verifier_error"] == 1.0 and raw["score"] == 0.0


def test_roundtrip_failed_and_accepted_rows(tasks, keys):
    """Every failed row (the error-prone half) + 20 accepted rows of the ep3 K=8 traces through
    render -> parse -> verifier AND the full seam: identical scores, constant keys, flags 0."""
    if not EP3_TRACES.exists():
        pytest.skip(f"{EP3_TRACES.name} not present (generated eval data)")
    n_ok = n_fail = 0
    problems = []
    with open(EP3_TRACES) as f:
        for line in f:
            r = json.loads(line)
            failed = r["score"]["score"] < 1
            if not failed and n_ok >= 20:
                continue
            n_fail += failed
            n_ok += not failed
            if p := roundtrip_row(r, tasks[r["task_id"]], keys.get(r["task_id"])):
                problems.append((r["task_id"], r.get("sample_idx"), p))
    assert n_fail >= 50 and not problems, problems[:5]

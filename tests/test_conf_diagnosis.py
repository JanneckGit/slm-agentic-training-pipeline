"""Confidence diagnosis (evaluation/conf_diagnosis.py): the torch-free parts — wrong-time detection, digit
alignment, segment spans — and the pre-registered decision rule at all of its boundaries."""
import json
from pathlib import Path

import pytest

from evaluation.conf_diagnosis import (TAUS, assistant_segments, decide, first_diff, is_truncated, locate_time,
                                       pick_correct, time_verdict, wrong_time)

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "data/generated/eval"


def _rec(answer, think="", ankunft="15:20", delay=31, template="t_info_ankunft_prognose", grounded=False, score=0.0):
    tool = json.dumps({"zug": "ICE 7", "ankunft": ankunft, "verspaetung_minuten": delay})
    return {"template": template, "score": {"score": score, "components": {"grounding": grounded}},
            "messages": [{"role": "user", "content": "Wann kommt ICE 7 an?"},
                         {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                                                                              "function": {"name": "zug", "arguments": "{}"}}]},
                         {"role": "tool", "tool_call_id": "c1", "content": tool},
                         {"role": "assistant", "content": f"<think>\n{think}\n</think>\n\n{answer}"}]}


def test_wrong_time_found_and_correct_derived():
    assert wrong_time(_rec("ICE 7 kommt um 16:51 an.")) == ("16:51", {"15:51"})


@pytest.mark.parametrize("rec", [
    _rec("ICE 7 kommt um 15:51 an."),                          # correct sum -> verifier artefact, not wrong
    _rec("ICE 7 kommt um 16:51 an.", grounded=True),           # grounding passed
    _rec("ICE 7 kommt um 16:51 an.", score=1.0),               # accepted
    _rec("ICE 7 kommt um 16:51 an.", template="t_info_standort"),
])
def test_wrong_time_ignores_other_cases(rec):
    assert wrong_time(rec) is None


@pytest.mark.parametrize("answer", [
    "ICE 7 kommt planmäßig um 09:11 an, mit 5 Minuten Verspätung um 09:16.",
    "ICE 7 kommt um 33:11 (09:11 am nächsten Tag) an, +5 Minuten: 09:16.",
])
def test_wrap_of_an_observed_time_is_an_artefact_not_an_error(answer):
    # GTFS writes times after midnight as 33:11; 09:11 is right, only the verifier's corpus lacks the wrap
    rec = _rec(answer, ankunft="33:11", delay=5)
    assert time_verdict(rec) == {"kind": "artefact", "written": "09:11"}
    assert wrong_time(rec) is None


@pytest.mark.parametrize("answer", ["ICE 7 kommt um 10:16 an.", "ICE 7 kommt um 09:11 an, also 10:16."])
def test_wrong_sum_after_midnight_is_still_an_error(answer):
    assert wrong_time(_rec(answer, ankunft="33:11", delay=5)) == ("10:16", {"09:16"})


def test_is_truncated():
    assert not is_truncated(_rec("ICE 7 kommt um 15:51 an."))
    assert is_truncated({**_rec(""), "truncated": True})
    runaway = _rec("")
    runaway["messages"][-1]["content"] = "<think>\nerst 15:20, dann plus 31, also"
    assert is_truncated(runaway)


@pytest.mark.parametrize("written,want,correct", [
    ("16:51", {"15:51"}, "15:51"), ("9:05", {"08:55"}, "8:55"), ("09:05", {"08:55"}, "08:55"),
    ("00:10", {"23:50", "12:00"}, "23:50"),                    # circular distance across midnight
])
def test_pick_correct(written, want, correct):
    assert pick_correct(written, want) == correct


@pytest.mark.parametrize("a,b,i", [("16:51", "15:51", 1), ("16:11", "15:51", 1), ("14:25", "14:35", 3),
                                   ("9:45", "10:45", 0), ("15:51", "15:51", None), ("1:05", "13:05", 1)])
def test_first_diff(a, b, i):
    assert first_diff(a, b) == i


TEXT = ("<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nQ<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nplan\n</think>\n\nich frage nach\n<tool_call>\n{\"name\": \"zug\"}\n</tool_call><|im_end|>\n"
        "<|im_start|>user\n<tool_response>\n{\"ankunft\": \"15:20\"}\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n<think>\n15:20 + 31 = 16:51\n</think>\n\nAnkunft 16:51.<|im_end|>\n")


def test_assistant_segments_cover_every_assistant_char():
    spans = assistant_segments(TEXT)
    assert [(seg, t) for _, _, seg, t in spans] == [("think", 0), ("text", 0), ("tool_call", 0), ("end", 0),
                                                     ("think", 1), ("answer", 1), ("end", 1)]
    assert TEXT[spans[5][0]:spans[5][1]] == "\n\nAnkunft 16:51."
    for s, e, seg, _ in spans:
        if seg == "think":
            assert TEXT[s:e].startswith("<think>") and TEXT[s:e].endswith("</think>")
        if seg == "end":
            assert TEXT[s:e] == "<|im_end|>"


def test_locate_time_prefers_the_requested_segment():
    spans = assistant_segments(TEXT)
    i, w = locate_time(TEXT, spans, (16, 51), turn=1, segment="think")
    assert w == "16:51" and TEXT[i:i + 5] == "16:51" and TEXT.rfind("<think>") < i < TEXT.rfind("</think>")
    j, _ = locate_time(TEXT, spans, (16, 51), turn=1, segment="answer")
    assert j > TEXT.rfind("</think>")
    assert locate_time(TEXT, spans, (15, 20), turn=1, segment="answer") is None


# --- the decision rule -------------------------------------------------------------------------------------
def _meas(p_errs, pcts, below_share=0.1, rank_gt20=0, segment="think", label="ep3"):
    n = 10000
    return {"label": label,
            "errors": [{"task_id": f"t{i % 3}", "p_err": p, "percentile": q, "segment": segment}
                       for i, (p, q) in enumerate(zip(p_errs, pcts))],
            "segments": {s: {"n": n, "rank_gt20": rank_gt20 if s == segment else 0} for s in ("think", "answer", "text")},
            "failed_tokens": {"n": n, "below": {str(t): int(below_share * n) for t in TAUS}}}


def test_decide_A_enabled_with_smallest_tau():
    r = decide(_meas([0.2, 0.3, 0.4, 0.55, 0.65], [0.01, 0.02, 0.03, 0.04, 0.2]))
    assert r["verdict"] == "A" and r["enabled"] and r["tau"] == 0.6          # 4/5 = 80 % below 0.6
    assert r["n_errors"] == 5 and r["selectivity_source"] == "ep3"


def test_decide_A_not_selective_and_corridor_source():
    m = _meas([0.2, 0.3, 0.4], [0.01, 0.01, 0.01])
    r = decide(m, corridor=_meas([], [], below_share=0.35, label="ep3_corridor"))
    assert r["verdict"] == "A" and not r["enabled"] and r["tau"] == 0.5 and r["selectivity_source"] == "ep3_corridor"


def test_decide_boundaries():
    assert decide(_meas([0.6, 0.6, 0.6], [0.05, 0.05, 0.05]))["verdict"] == "A"            # inclusive
    assert decide(_meas([0.61, 0.6, 0.62], [0.01, 0.01, 0.01]))["verdict"] == "C"
    assert decide(_meas([0.5, 0.5, 0.5], [0.06, 0.06, 0.06]))["verdict"] == "C"            # dent, not a low one
    assert decide(_meas([0.9, 0.95, 0.99], [0.5, 0.6, 0.7]))["verdict"] == "B"
    r = decide(_meas([0.55, 0.58, 0.59, 0.95, 0.96], [0.01] * 5))                          # median 0.59 but tail
    assert r["verdict"] == "A" and r["tau"] is None and not r["enabled"]                  # no tau keeps 80 %


def test_decide_unreliable_and_empty():
    r = decide(_meas([0.2, 0.3], [0.01, 0.01], rank_gt20=60))                               # 0.6 % > 0.5 %
    assert r["verdict"] == "unreliable" and not r["enabled"]
    assert decide(_meas([], []))["verdict"] == "C"


@pytest.mark.parametrize("name,wrong,tasks,artefacts,truncated", [
    ("db_traces_heldout_eval_after_qwen3-4b_ep3_k8.jsonl", 8, 4, 3, 2),
    ("db_traces_heldout_eval_after_grpo_db_bahn_r2_step24_k8.jsonl", 17, 9, 3, 1),
])
def test_time_verdicts_on_the_k8_traces(name, wrong, tasks, artefacts, truncated):
    f = EVAL / name
    if not f.exists():
        pytest.skip("K=8 heldout traces not present")
    recs = [json.loads(line) for line in open(f)]
    verdicts = [(r, time_verdict(r)) for r in recs]
    real = [r for r, v in verdicts if v and v["kind"] == "wrong"]
    assert len(real) == wrong and len({r["task_id"] for r in real}) == tasks
    assert sum(1 for _, v in verdicts if v and v["kind"] == "artefact") == artefacts
    assert sum(map(is_truncated, recs)) == truncated

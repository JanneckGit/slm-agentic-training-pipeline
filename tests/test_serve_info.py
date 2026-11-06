"""evaluation/serve_info.py: identity per role, start events, the resume gate. No docker, no network."""
import json

from evaluation.serve_info import mem_available_gib, record

V21 = {"vllm_version": "0.21.0", "image_id": "sha256:f023269abe06aaaa", "image_ref": "vllm/vllm-openai@sha256:a2"}
V30 = {"vllm_version": "0.30.0", "image_id": "sha256:91d9b077589ebbbb", "image_ref": "vllm/vllm-openai:v0.30.0"}
EV = {"gpu_util": 0.22, "mem_avail_gib": 80.5}


def _load(p):
    return json.loads(p.read_text())


def test_fresh_file_gets_identity_and_one_event(tmp_path):
    p = tmp_path / "serve_info.json"
    ok, msg = record(p, "agent", V21, EV, gate=True)
    d = _load(p)
    assert ok and msg == ""
    assert d["agent"] == V21
    assert len(d["events"]) == 1
    ev = d["events"][0]
    assert ev["role"] == "agent" and ev["vllm_version"] == "0.21.0" and ev["gpu_util"] == 0.22
    assert ev["mem_avail_gib"] == 80.5 and "mismatch" not in ev


def test_same_identity_appends_event(tmp_path):
    p = tmp_path / "serve_info.json"
    record(p, "agent", V21, EV, gate=True)
    record(p, "user_sim", V21, {"gpu_util": 0.61, "mem_avail_gib": 5.0}, gate=True)
    ok, _ = record(p, "agent", V21, EV, gate=True)                 # resume, same server
    d = _load(p)
    assert ok
    assert [e["role"] for e in d["events"]] == ["agent", "user_sim", "agent"]
    assert d["agent"] == V21 and d["user_sim"] == V21


def test_gate_refuses_other_image_and_writes_nothing(tmp_path):
    p = tmp_path / "serve_info.json"
    record(p, "agent", V21, EV, gate=True)
    before = p.read_bytes()
    ok, msg = record(p, "agent", V30, EV, gate=True)
    assert not ok and "image_id" in msg and "vllm_version" in msg
    assert p.read_bytes() == before


def test_allow_change_overrides_gate_and_marks_event(tmp_path):
    p = tmp_path / "serve_info.json"
    record(p, "agent", V21, EV, gate=True)
    ok, msg = record(p, "agent", V30, EV, gate=True, allow_change=True)
    d = _load(p)
    assert ok and msg.startswith("WARN")
    assert d["agent"] == V21                                       # identity stays the first one
    assert d["events"][-1]["mismatch"] == ["image_id", "vllm_version"]


def test_without_gate_mismatch_only_warns(tmp_path):
    p = tmp_path / "serve_info.json"
    record(p, "server", V21, EV, gate=False)
    ok, msg = record(p, "server", V30, EV, gate=False)
    assert ok and msg.startswith("WARN")
    assert len(_load(p)["events"]) == 2


def test_unknown_version_compares_image_only(tmp_path):
    p = tmp_path / "serve_info.json"
    record(p, "agent", V21, EV, gate=True)
    same_image = {**V21, "vllm_version": None}
    assert record(p, "agent", same_image, EV, gate=True)[0]       # /version silent, same image: fine
    other_image = {**V30, "vllm_version": None}
    ok, msg = record(p, "agent", other_image, EV, gate=True)
    assert not ok and "image_id" in msg and "vllm_version" not in msg


def test_mem_available_parser():
    sample = "MemTotal:       125440000 kB\nMemFree:         1000000 kB\nMemAvailable:   112197632 kB\n"
    assert mem_available_gib(sample) == 107.0
    assert mem_available_gib("MemTotal: 1 kB\n") is None

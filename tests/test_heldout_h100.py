"""Heldout auf einem Fremdrechner (ops/h100_open_models.sh): Gates, MLflow-Nachtrag, Template-Schalter.

GPU-frei, ohne Docker, ohne Netz. Synthetische Rollout-Datensaetze im Format von sdg_pipeline/db_bahn/rollout.py.
"""
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load("heldout_gate_under_test", REPO / "evaluation/heldout_gate.py")
posthoc = _load("log_heldout_mlflow_under_test", REPO / "evaluation/log_heldout_mlflow.py")


def rec(task="t1", sample=0, ok=True, truncated=False, finish="final_answer", think=True, tool=True,
        template="tplA", fault="none", recovery=0.0, replan=0.0, turns=3, tail=0):
    content = ("<think>\nabwaegen\n</think>\n\n" if think else "") + "Antwort"
    asst = {"role": "assistant", "content": content}
    if tool:
        asst["tool_calls"] = [{"id": "n1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]
    return {"task_id": task, "sample_idx": sample, "split": "heldout_eval", "teacher": "lbl", "template": template,
            "fault": fault, "truncated": truncated, "degen": {"think_ngram_dup_ratio": 0.0, "max_think_chars": 10},
            "roleplay_tail_chars": tail, "finish_reason": finish,
            "score": {"score": 1.0 if ok else 0.0, "self_recovery": recovery, "replan_occurred": replan,
                      "turns_used": turns},
            "messages": [{"role": "user", "content": "x"}, asst]}


def write(path: Path, recs: list) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return str(path)


# ---- smoke gate ---------------------------------------------------------------------------------------------
def test_smoke_passes_a_weak_but_working_model(tmp_path):
    recs = [rec(task=f"t{i}", ok=(i % 3 == 0)) for i in range(48)]          # Yield 33 %: kein Kriterium
    assert gate.main(["smoke", "--input", write(tmp_path / "s.jsonl", recs)]) == 0


@pytest.mark.parametrize("bad", [
    {"finish": "teacher_error:HTTPStatusError: 400"},    # z. B. abgelehnter Sampling-Parameter -> trifft ALLE
    {"truncated": True},                                  # Denken laeuft ins Token-Limit
    {"think": False},                                     # Thinking nicht an
    {"tool": False},                                      # Tool-Format nicht verstanden
])
def test_smoke_fails_on_systematic_problems(tmp_path, bad):
    recs = [rec(task=f"t{i}", **(bad if i < 10 else {})) for i in range(48)]
    assert gate.main(["smoke", "--input", write(tmp_path / "s.jsonl", recs)]) == 1


def test_smoke_tolerates_one_transient_error(tmp_path):
    recs = [rec(task=f"t{i}", finish="teacher_error:ReadTimeout" if i == 0 else "final_answer") for i in range(48)]
    assert gate.main(["smoke", "--input", write(tmp_path / "s.jsonl", recs)]) == 0


def test_smoke_empty_file_is_unusable(tmp_path):
    assert gate.main(["smoke", "--input", write(tmp_path / "s.jsonl", [])]) == 2


# ---- calib gate ---------------------------------------------------------------------------------------------
def _k8(n_tasks: int, fail_every: int) -> list:
    return [rec(task=f"t{t}", sample=s, ok=((t * 8 + s) % fail_every != 0)) for t in range(n_tasks) for s in range(8)]


def test_calib_identical_runs_pass(tmp_path):
    a = write(tmp_path / "a.jsonl", _k8(60, 4))
    assert gate.main(["calib", "--input", a, "--reference", a, "--draws", "200"]) == 0


def test_calib_shifted_run_fails(tmp_path):
    a = write(tmp_path / "a.jsonl", _k8(60, 4))     # 75 % Yield
    b = write(tmp_path / "b.jsonl", _k8(60, 2))     # 50 % Yield
    assert gate.main(["calib", "--input", b, "--reference", a, "--draws", "200"]) == 1


def test_calib_incomplete_run_is_unusable(tmp_path):
    a = write(tmp_path / "a.jsonl", _k8(60, 4))
    b = write(tmp_path / "b.jsonl", _k8(30, 4))     # Haelfte der Tasks fehlt
    assert gate.main(["calib", "--input", b, "--reference", a, "--draws", "200"]) == 2


# ---- MLflow-Nachtrag ----------------------------------------------------------------------------------------
def test_posthoc_summary_matches_hand_count():
    recs = [rec(task="t1", ok=True, template="A", recovery=1.0, fault="runtime", replan=1.0, turns=4),
            rec(task="t2", ok=True, template="A", recovery=1.0, fault="none", turns=2),
            rec(task="t3", ok=False, template="B", tail=5, turns=6),
            rec(task="t4", ok=True, truncated=True, template="B", turns=4)]   # verifiziert, aber gekappt -> nicht akzeptiert
    m, y = posthoc.summary(recs)
    assert m["n_rollouts"] == 4
    assert m["verified_yield"] == pytest.approx(2 / 4)
    assert m["truncated_rate"] == pytest.approx(1 / 4)
    assert m["roleplay_tail_rate"] == pytest.approx(1 / 4)
    assert m["replan_rate"] == pytest.approx(1 / 4)
    assert m["avg_turns"] == pytest.approx(16 / 4)
    assert m["scripted_recovery_rate"] == pytest.approx(1 / 2)
    assert m["emergent_recovery_rate"] == pytest.approx(1 / 2)
    assert y == {"yield_A": 1.0, "yield_B": 0.0}


def test_posthoc_provenance_from_sidecars(tmp_path):
    out = tmp_path / "db_traces_heldout_eval_lbl_k8.jsonl"
    recs = [rec(task=f"t{t}", sample=s) for t in range(3) for s in range(8)]
    write(out, recs)
    (tmp_path / (out.name + ".meta.json")).write_text(json.dumps({
        "identity": {"model": "org/m", "label": "lbl", "split": "heldout_eval", "k": 8, "max_turns": 15,
                     "chat_template_kwargs": {"preserve_thinking": True}},
        "info": {"host": "h100", "serve": "venv"}}))
    (tmp_path / (out.name + ".serve.json")).write_text(json.dumps({"server": {"vllm_version": "0.33.0"}}))
    params, label, k = posthoc.provenance(str(out), posthoc.load(str(out)))
    assert (label, k) == ("lbl", 8)
    assert params["model"] == "org/m" and params["n_tasks"] == 3
    assert params["host"] == "h100" and params["serve"] == "venv" and params["vllm_version"] == "0.33.0"
    assert json.loads(params["chat_template_kwargs"]) == {"preserve_thinking": True}


# ---- rollout.py: Template-Schalter kommen beim Server an ---------------------------------------------------
@pytest.fixture
def stub_server():
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1", seen


def test_chat_template_kwargs_reach_the_server(stub_server, monkeypatch):
    pytest.importorskip("tau2")
    pytest.importorskip("httpx")
    for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(v, raising=False)
    from sdg_pipeline.db_bahn.rollout import make_teacher_call, resolve_teacher
    base, seen = stub_server
    cfg = resolve_teacher({}, base, "m")
    make_teacher_call(cfg)([{"role": "user", "content": "x"}], [])
    make_teacher_call({**cfg, "chat_template_kwargs": {"preserve_thinking": True}})([{"role": "user", "content": "x"}], [])
    assert seen[0]["chat_template_kwargs"] == {"enable_thinking": True}                  # Default unveraendert
    assert seen[1]["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}

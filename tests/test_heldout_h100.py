"""Heldout auf einem Fremdrechner (ops/h100_open_models.sh): Gates, MLflow-Nachtrag, Template-Schalter, nativer Tool-Modus.

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


def test_smoke_think_counts_per_rollout(tmp_path):
    # Harmony-Muster (gpt-oss): denkt vor jedem Tool-Call, Schlussantwort ohne Denkblock -> 50 % der Zuege, 100 % der Rollouts
    recs = []
    for i in range(48):
        r = rec(task=f"t{i}")
        r["messages"].append({"role": "tool", "tool_call_id": "n1", "content": "{}"})
        r["messages"].append({"role": "assistant", "content": "Schlussantwort ohne Denkblock"})
        recs.append(r)
    path = write(tmp_path / "s.jsonl", recs)
    s = gate.smoke_stats(gate.load(path, keep_messages=True))
    assert s["think"] == 1.0 and s["think_turns"] == pytest.approx(0.5)
    assert gate.main(["smoke", "--input", path]) == 0


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
                     "temperature": 1.0, "top_k": -1, "vllm_extra": "", "tool_mode": "native",
                     "chat_template_kwargs": {"preserve_thinking": True}},
        "info": {"host": "h100", "serve": "venv", "eval_conc": 12}}))
    (tmp_path / (out.name + ".serve.json")).write_text(json.dumps({"server": {"vllm_version": "0.33.0"}}))
    params, label, k = posthoc.provenance(str(out), posthoc.load(str(out)))
    assert (label, k) == ("lbl", 8)
    assert params["model"] == "org/m" and params["n_tasks"] == 3
    assert params["host"] == "h100" and params["serve"] == "venv" and params["vllm_version"] == "0.33.0"
    assert json.loads(params["chat_template_kwargs"]) == {"preserve_thinking": True}
    assert params["temperature"] == 1.0 and params["top_k"] == -1 and params["eval_conc"] == 12
    assert params["tool_mode"] == "native"
    assert "vllm_extra" not in params and "top_p" not in params   # leer bzw. fehlend -> nicht geloggt
    assert "rescored" not in params


def test_posthoc_provenance_carries_the_last_rescore(tmp_path):
    out = tmp_path / "db_traces_heldout_eval_lbl_k8.jsonl"
    write(out, [rec(task="t0", sample=s) for s in range(8)])
    (tmp_path / (out.name + ".meta.json")).write_text(json.dumps({
        "identity": {"label": "lbl", "k": 8},
        "rescored": [{"at": "2026-11-01T10:00:00", "reason": "alt", "verifier_sha256": "aaa", "changed_records": 1},
                     {"at": "2026-11-12T09:30:00", "reason": "typo-norm", "verifier_sha256": "bbb",
                      "changed_records": 7}]}))
    params, _, _ = posthoc.provenance(str(out), posthoc.load(str(out)))
    assert params["rescored"] == "2026-11-12 typo-norm (verifier bbb, 7 geaendert)"


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


# ---- nativer Tool-Modus (Runde 2: gpt-oss, Gemma 4, Muse) ---------------------------------------------------
def _no_proxy(monkeypatch):
    for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(v, raising=False)


@pytest.fixture
def reply_server():
    """Stub-vLLM: merkt sich jeden Request, antwortet mit der naechsten vorgegebenen `message`."""
    seen, replies = [], []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            msg = replies.pop(0) if replies else {"content": "ok"}
            body = json.dumps({"choices": [{"message": msg, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1", seen, replies


TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]
HISTORY = [{"role": "system", "content": "S"}, {"role": "user", "content": "x"},
           {"role": "assistant", "content": "<think>\nalt\n</think>\n\nsichtbar",
            "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
           {"role": "tool", "tool_call_id": "a", "content": "{}"}]


def test_native_mode_sends_tools_and_moves_think_into_reasoning(reply_server, monkeypatch):
    pytest.importorskip("tau2")
    pytest.importorskip("httpx")
    _no_proxy(monkeypatch)
    from sdg_pipeline.db_bahn.rollout import make_teacher_call, resolve_teacher
    base, seen, replies = reply_server
    replies.append({"content": "", "reasoning": "erst f fragen",
                    "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]})
    history = json.loads(json.dumps(HISTORY))
    content, native, _ = make_teacher_call({**resolve_teacher({}, base, "m"), "native_tools": True})(history, TOOLS)
    assert content.startswith("<think>\nerst f fragen\n</think>")             # Denkfeld -> kanonisch inline
    assert native[0]["function"]["name"] == "f"                                # Server-Calls werden durchgereicht
    req = seen[0]
    assert req["tools"] == TOOLS and req["tool_choice"] == "auto"
    assert "strict" not in json.dumps(req["tools"])                            # keine Grammatik
    asst = req["messages"][2]
    assert asst["content"] == "sichtbar" and asst["reasoning"] == "alt" and asst["reasoning_content"] == "alt"
    assert history == HISTORY                                                  # Datensatz-Kopie bleibt inline


def test_prompt_mode_payload_unchanged_but_reasoning_field_is_kept(reply_server, monkeypatch):
    pytest.importorskip("tau2")
    pytest.importorskip("httpx")
    _no_proxy(monkeypatch)
    from sdg_pipeline.db_bahn.rollout import make_teacher_call, resolve_teacher
    base, seen, replies = reply_server
    call = make_teacher_call(resolve_teacher({}, base, "m"))
    replies.append({"content": "Antwort", "reasoning_content": "nachgedacht"})     # Server mit Reasoning-Parser
    replies.append({"content": "<think>\ninline\n</think>\n\nAntwort", "reasoning": "doppelt"})
    first, _, _ = call(HISTORY, TOOLS)
    second, _, _ = call(HISTORY, TOOLS)
    assert "tools" not in seen[0] and "tool_choice" not in seen[0]
    assert seen[0]["messages"] == HISTORY                                      # Verlauf 1:1 wie bisher
    assert first == "<think>\nnachgedacht\n</think>\n\nAntwort"
    assert second == "<think>\ninline\n</think>\n\nAntwort"                    # nie doppelt einwickeln


def test_native_system_prompt_and_markup_detector():
    pytest.importorskip("tau2")
    import sdg_pipeline.db_bahn.rollout as ro
    nat = ro.SYSTEM_TEMPLATE_NATIVE
    assert nat.startswith("{policy}\n\n## Arbeitsweise") and "<tool_call>" not in nat and "{tools_block}" not in nat
    assert nat.endswith(ro.SYSTEM_TEMPLATE[ro.SYSTEM_TEMPLATE.index("- Nach jedem Werkzeug-Ergebnis"):])
    for leak in ("<|tool_call>call:f{a:1}<tool_call|>",                       # Gemma 4
                 '<atem:function_calls>\n<atem:invoke name="f">',               # Muse (ATEM)
                 'to=functions.f <|constrain|>json{"a": 1}',                   # gpt-oss (Harmony)
                 '<tool_call>{"name": "f", "argume'):                          # abgerissenes Hermes-JSON
        assert ro.TOOL_MARKUP_RE.search(leak), leak
    assert not ro.TOOL_MARKUP_RE.search("Der ICE 123 fährt um 10:15 ab.")


def _scripted(*turns):
    turns = list(turns)
    return lambda messages, tools: turns.pop(0)


def test_native_rollout_uses_native_prompt_and_flags_format_errors(tasks, keys):
    from sdg_pipeline.db_bahn.rollout import run_rollout
    tid = sorted(tasks)[0]
    leak = ("<think>\nplane\n</think>\n\n<|tool_call>call:f{a:1}<tool_call|>", [], "stop")
    nat = run_rollout(tasks[tid], keys[tid], _scripted(leak), 5, 60.0, native=True)
    assert nat["finish_reason"] == "tool_format_error"
    sys_nat = nat["messages"][0]["content"]
    assert "<tools>" not in sys_nat and "Funktionsaufruf-Schnittstelle" in sys_nat
    old = run_rollout(tasks[tid], keys[tid], _scripted(leak), 5, 60.0)        # Prompt-Modus: Verhalten wie bisher
    assert old["finish_reason"] == "final_answer" and "<tools>" in old["messages"][0]["content"]


def test_native_rollout_executes_server_parsed_calls(tasks, keys):
    from sdg_pipeline.db_bahn.rollout import run_rollout
    tid = sorted(tasks)[0]
    seen_tools = []

    def teacher(messages, tools):
        seen_tools.append(tools)
        if len(seen_tools) == 1:
            call = {"id": "s1", "type": "function", "function": {"name": tools[0]["function"]["name"], "arguments": "{}"}}
            return "<think>\nnachsehen\n</think>\n\n", [call], "tool_calls"
        return "<think>\nfertig\n</think>\n\nErledigt.", [], "stop"

    res = run_rollout(tasks[tid], keys[tid], teacher, 5, 60.0, native=True)
    msgs = res["messages"]
    assert res["finish_reason"] == "final_answer"
    assert msgs[2]["tool_calls"][0]["id"] == "s1" and msgs[2]["content"].startswith("<think>")
    assert msgs[3]["role"] == "tool" and msgs[3]["tool_call_id"] == "s1"

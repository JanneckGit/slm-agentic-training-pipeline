"""Claude-API-Backend (rollout.py --backend anthropic): Request-Form, unveraenderte Rueckgabe der signierten Denkbloecke,
kanonische Datensaetze, Abbruchregeln, Gate/Kosten/pass^n.

GPU-frei, ohne Netz: ein lokaler Stub emuliert POST /v1/messages, das ECHTE anthropic-SDK zeigt per base_url darauf.
"""
import importlib.util
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from math import comb
from pathlib import Path

import pytest

pytest.importorskip("anthropic")
REPO = Path(__file__).resolve().parents[1]
MODEL = "claude-sonnet-5-5"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load("heldout_gate_api_test", REPO / "evaluation/heldout_gate.py")
posthoc = _load("log_heldout_mlflow_api_test", REPO / "evaluation/log_heldout_mlflow.py")
usage = _load("api_usage_test", REPO / "evaluation/api_usage.py")
report = _load("eval_report_api_test", REPO / "evaluation/eval_report.py")


# ---- Stub der Messages-API ----------------------------------------------------------------------------------
def think(text, sig):
    return {"type": "thinking", "thinking": text, "signature": sig}


def text(t):
    return {"type": "text", "text": t}


def tool(i, name, args):
    return {"type": "tool_use", "id": i, "name": name, "input": args}


def msg(blocks, stop="end_turn", stop_details=None, **use):
    return {"id": "msg_x", "type": "message", "role": "assistant", "model": MODEL, "content": blocks,
            "stop_reason": stop, "stop_sequence": None, "stop_details": stop_details,
            "usage": {"input_tokens": 120, "output_tokens": 80, "cache_read_input_tokens": 4000,
                      "cache_creation_input_tokens": 0, **use}}


def err(status, etype, message):
    return status, {"type": "error", "error": {"type": etype, "message": message}}


class Script(list):
    """Antworten der Reihe nach; ist die Liste leer, kommt `fallback` (Default: Schlussantwort) bei jeder Anfrage."""
    fallback = None


@pytest.fixture
def api_stub(monkeypatch):
    """-> (base_url, seen requests, script). Script item = Message-dict (200) oder (status, body); leer = fallback."""
    for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-a-key")
    seen, script = [], Script()

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            item = script.pop(0) if script else (script.fallback or msg([text("Erledigt.")]))
            status, body = item if isinstance(item, tuple) else (200, item)
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("request-id", f"req_{len(seen)}")
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen, script
    srv.shutdown()


def cfg(base):
    return {"backend": "anthropic", "model": MODEL, "api_base": base, "max_tokens": 3072, "effort": "high",
            "native_tools": True, "api_max_retries": 0}


def first_task(tasks, keys):
    tid = sorted(tasks)[0]
    return tasks[tid], keys[tid], keys[tid]["oracle_calls"]


# ---- Request-Form + Replay ----------------------------------------------------------------------------------
def test_request_shape_and_signed_thinking_is_replayed_unchanged(api_stub, tasks, keys):
    from sdg_pipeline.db_bahn.rollout import make_anthropic_call, run_rollout
    base, seen, script = api_stub
    task, key, oracle = first_task(tasks, keys)
    c0 = oracle[0]
    r1 = msg([think("Erst den Auftrag pruefen.", "sig-1"), text("Ich pruefe."), tool("toolu_1", c0["name"], c0["arguments"])],
             stop="tool_use", cache_creation_input_tokens=4000, cache_read_input_tokens=0)
    script += [r1, msg([think("Fertig.", "sig-2"), text("Erledigt: WO-2026-1145.")])]
    res = run_rollout(task, key, make_anthropic_call(cfg(base)), 5, 60.0, native=True)

    q1, q2 = seen
    assert not {"temperature", "top_p", "top_k"} & set(q1)                     # keine Sampling-Parameter
    assert q1["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert q1["output_config"] == {"effort": "high"} and q1["max_tokens"] == 3072 and q1["model"] == MODEL
    assert q1["cache_control"] == {"type": "ephemeral"}                         # automatischer Breakpoint (Verlauf)
    assert q1["system"][0]["cache_control"] == {"type": "ephemeral"}            # fester Breakpoint (Tools + System)
    sys_text = q1["system"][0]["text"]
    assert "Funktionsaufruf-Schnittstelle" in sys_text and "<tools>" not in sys_text    # SYSTEM_TEMPLATE_NATIVE
    assert q1["tool_choice"] == {"type": "auto"}
    assert all({"name", "description", "input_schema"} <= set(t) for t in q1["tools"])
    # append-only: Aufruf 2 = Aufruf 1 + Anhang; System und Tools identisch
    assert q2["system"] == q1["system"] and q2["tools"] == q1["tools"] and q2["messages"][:1] == q1["messages"]
    asst = q2["messages"][1]
    assert asst["role"] == "assistant" and asst["content"][0] == r1["content"][0]          # Signatur unveraendert
    assert [b["type"] for b in asst["content"]] == ["thinking", "text", "tool_use"]
    assert asst["content"][2]["id"] == "toolu_1" and asst["content"][2]["input"] == c0["arguments"]
    tr = q2["messages"][2]
    assert tr["role"] == "user" and tr["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": res["messages"][3]["content"]}]

    m = res["messages"]
    assert m[2]["content"] == "<think>\nErst den Auftrag pruefen.\n</think>\n\nIch pruefe."   # kanonisch inline
    assert m[2]["tool_calls"][0]["id"] == "toolu_1"
    assert json.loads(m[2]["tool_calls"][0]["function"]["arguments"]) == c0["arguments"]
    assert m[3]["role"] == "tool" and m[3]["tool_call_id"] == "toolu_1"
    assert res["finish_reason"] == "final_answer" and "score" in res["score"]
    assert [c["cache_creation_input_tokens"] for c in res["api_calls"]] == [4000, 0]
    assert res["api_calls"][0]["request_id"] == "req_1" and res["api_calls"][0]["stop_reason"] == "tool_use"


def test_parallel_tool_uses_become_one_user_message(api_stub, tasks, keys):
    from sdg_pipeline.db_bahn.rollout import make_anthropic_call, run_rollout
    base, seen, script = api_stub
    task, key, oracle = first_task(tasks, keys)
    a, b = oracle[0], oracle[1]
    script += [msg([think("Beide zugleich.", "s"), tool("toolu_a", a["name"], a["arguments"]),
                    tool("toolu_b", b["name"], b["arguments"])], stop="tool_use")]
    res = run_rollout(task, key, make_anthropic_call(cfg(base)), 5, 60.0, native=True)
    results = seen[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["toolu_a", "toolu_b"] and len(seen[1]["messages"]) == 3
    assert [m["role"] for m in res["messages"]] == ["system", "user", "assistant", "tool", "tool", "assistant"]


@pytest.mark.parametrize("stop,finish,truncated", [("max_tokens", "truncated", True), ("refusal", "refusal", False)])
def test_cap_and_refusal_end_the_rollout(api_stub, tasks, keys, stop, finish, truncated):
    from sdg_pipeline.db_bahn.rollout import make_anthropic_call, run_rollout
    base, seen, script = api_stub
    task, key, _ = first_task(tasks, keys)
    script += [msg([think("Lange Ueberlegung ...", "s")], stop=stop,
                   stop_details={"type": "refusal", "category": "general_harms"} if stop == "refusal" else None)]
    res = run_rollout(task, key, make_anthropic_call(cfg(base)), 5, 60.0, native=True)
    assert res["finish_reason"] == finish and res["truncated"] is truncated and len(seen) == 1
    if stop == "refusal":
        assert res["api_calls"][0]["refusal_category"] == "general_harms"


# ---- Abbruchregeln ------------------------------------------------------------------------------------------
@pytest.mark.parametrize("error", [err(401, "authentication_error", "invalid x-api-key"),
                                   err(400, "invalid_request_error",
                                       "Your credit balance is too low to access the Anthropic API."),
                                   err(404, "not_found_error", "model: claude-nope")])
def test_account_level_errors_are_fatal(api_stub, tasks, keys, error):
    from sdg_pipeline.db_bahn.rollout import FatalApiError, make_anthropic_call, run_rollout
    base, seen, script = api_stub
    task, key, _ = first_task(tasks, keys)
    script.append(error)
    with pytest.raises(FatalApiError):
        run_rollout(task, key, make_anthropic_call(cfg(base)), 5, 60.0, native=True)


def test_other_request_errors_stay_per_rollout(api_stub, tasks, keys):
    from sdg_pipeline.db_bahn.rollout import make_anthropic_call, run_rollout
    base, seen, script = api_stub
    task, key, _ = first_task(tasks, keys)
    script.append(err(400, "invalid_request_error", "messages.1: something specific"))
    res = run_rollout(task, key, make_anthropic_call(cfg(base)), 5, 60.0, native=True)
    assert res["finish_reason"].startswith("teacher_error:BadRequestError")


def _cli(base, out, *extra):
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    env.update(PYTHONPATH=str(REPO), ANTHROPIC_API_KEY="sk-test-not-a-key", LOGURU_LEVEL="ERROR")
    return subprocess.run([sys.executable, "sdg_pipeline/db_bahn/rollout.py", "--backend", "anthropic", "--api-base", base,
                           "--model", MODEL, "--teacher-name", "api_test", "--split", "bakeoff_dev", "--n-tasks", "3",
                           "--max-regen", "0", "--max-turns", "3", "--max-tokens-per-turn", "3072", "--concurrency", "1",
                           "--api-max-retries", "0", "--output", str(out), *extra],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=120)


def test_cli_writes_api_records_without_sampling_params(api_stub, tmp_path, tasks):
    base, seen, script = api_stub
    out = tmp_path / "ok.jsonl"
    p = _cli(base, out, "--effort", "high")
    assert p.returncode == 0, p.stdout + p.stderr
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(recs) == 3
    gp = recs[0]["gen_params"]
    assert gp == {"backend": "anthropic", "model": MODEL, "effort": "high",
                  "thinking": {"type": "adaptive", "display": "summarized"}, "enable_thinking": True,
                  "max_tokens": 3072, "tool_mode": "native"}
    assert len(recs[0]["api_calls"]) == 1 and recs[0]["finish_reason"] == "final_answer"
    assert list(recs[0]).index("api_calls") == list(recs[0]).index("wall_s") + 1
    assert "sk-test" not in out.read_text() and "sk-test" not in p.stdout       # der Key landet nirgends


def test_cli_fatal_error_stops_the_run_without_records(api_stub, tmp_path, tasks):
    base, seen, script = api_stub
    script += [err(401, "authentication_error", "invalid x-api-key")] * 5
    out = tmp_path / "fatal.jsonl"
    p = _cli(base, out, "--effort", "high")
    assert p.returncode == 3, p.stdout + p.stderr
    assert "ABBRUCH" in p.stdout and (not out.exists() or out.read_text() == "")
    assert len(seen) == 1                                                       # danach keine weiteren Anfragen


@pytest.mark.parametrize("extra", [["--effort", "high", "--temperature", "0.6"],    # Sampling: API lehnt ab
                                   [],                                              # effort fehlt
                                   ["--effort", "high", "--chat-template-kwargs", "{}"]])
def test_cli_rejects_invalid_api_flags(api_stub, tmp_path, extra):
    base, seen, _ = api_stub
    p = _cli(base, tmp_path / "x.jsonl", *extra)
    assert p.returncode == 2 and not seen


# ---- Gate, Kosten, pass^n, MLflow ---------------------------------------------------------------------------
def api_rec(task="t1", sample=0, ok=True, think_=False, calls=None, finish="final_answer"):
    content = ("<think>\nkurz\n</think>\n\n" if think_ else "") + "Antwort"
    return {"task_id": task, "sample_idx": sample, "template": "tplA", "fault": "none", "truncated": False,
            "degen": {"think_ngram_dup_ratio": 0.0, "max_think_chars": 5}, "roleplay_tail_chars": 0,
            "finish_reason": finish, "gen_params": {"backend": "anthropic", "model": MODEL, "max_tokens": 3072},
            "score": {"score": 1.0 if ok else 0.0, "turns_used": 2},
            "api_calls": calls if calls is not None else [{"input_tokens": 100, "cache_read_input_tokens": 0,
                                                           "cache_creation_input_tokens": 0, "output_tokens": 50,
                                                           "stop_reason": "end_turn", "model": MODEL}],
            "messages": [{"role": "user", "content": "x"},
                         {"role": "assistant", "content": content,
                          "tool_calls": [{"id": "a", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}]}


def write(path: Path, recs: list) -> str:
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return str(path)


def test_smoke_gate_treats_thinking_as_info_for_api_runs(tmp_path, capsys):
    recs = [api_rec(task=f"t{i}", think_=(i < 10)) for i in range(48)]          # nur 21 % mit Denken
    recs[0]["finish_reason"] = "refusal"
    assert gate.main(["smoke", "--input", write(tmp_path / "s.jsonl", recs)]) == 0
    out = capsys.readouterr().out
    assert "Backend anthropic" in out and "nur Info" in out and "Ablehnungen (refusal) 1" in out
    vllm = [{**r, "gen_params": {"temperature": 0.6}} for r in recs]            # vLLM: weiter hartes Kriterium
    assert gate.main(["smoke", "--input", write(tmp_path / "v.jsonl", vllm)]) == 1


def test_api_usage_cost_matches_hand_count():
    calls = [{"input_tokens": 1000, "cache_read_input_tokens": 4000, "cache_creation_input_tokens": 0,
              "output_tokens": 3000, "stop_reason": "max_tokens", "model": MODEL},
             {"input_tokens": 500, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 4000,
              "output_tokens": 200, "stop_reason": "end_turn", "model": MODEL}]
    s = usage.summarize([api_rec(calls=calls), {"task_id": "vllm-record-without-api-calls"}])
    hand = (1500 * 2.00 + 4000 * 2.50 + 4000 * 0.10 + 3200 * 10.00) / 1e6
    assert s["rollouts"] == 1 and s["calls"] == 2 and s["cost_usd"] == pytest.approx(hand)
    assert s["near_cap_share"] == pytest.approx(0.5) and s["out_max"] == 3000      # 3000 >= 0.9 * 3072
    assert s["cache_hit_share"] == pytest.approx(4000 / 9500)
    assert usage.mlflow_metrics(s)["api_cost_usd"] == pytest.approx(hand)
    unknown = api_rec(calls=[{**calls[0], "model": "unbekannt"}])
    unknown["gen_params"]["model"] = "unbekannt"
    assert usage.summarize([unknown])["cost_usd"] is None                      # ohne Preis: Token ja, Kosten nein


def test_passk_at_is_the_unbiased_estimator(tmp_path):
    recs = [api_rec(task=t, sample=s, ok=s < c) for t, c in (("a", 8), ("b", 6), ("c", 1)) for s in range(8)]
    run = report.load(write(tmp_path / "k8.jsonl", recs))
    est, n = report.passk_at(run["per"], run["meta"]["k"], 2)
    assert n == 3 and est == pytest.approx((1 + comb(6, 2) / comb(8, 2) + 0) / 3)
    assert report.passk_at(run["per"], 1, 2) is None


def test_posthoc_mlflow_carries_backend_and_cost(tmp_path):
    path = write(tmp_path / "db_traces_heldout_eval_api_x_k2.jsonl", [api_rec(task=f"t{i}", sample=s)
                                                                       for i in range(3) for s in range(2)])
    Path(path + ".meta.json").write_text(json.dumps({
        "identity": {"model": MODEL, "label": "api_x", "split": "heldout_eval", "k": 2, "backend": "anthropic",
                     "effort": "high", "thinking": "adaptive/summarized", "tool_mode": "native", "max_turns": 15},
        "info": {"serve": "none", "eval_conc": 24}}))
    recs = posthoc.load(path)
    params, label, k = posthoc.provenance(path, recs)
    assert params["backend"] == "anthropic" and params["effort"] == "high" and params["thinking"] == "adaptive/summarized"
    assert "temperature" not in params and label == "api_x" and k == 2
    metrics, _ = posthoc.summary(recs)
    assert metrics["api_calls"] == 6 and metrics["api_cost_usd"] == pytest.approx(6 * (100 * 2 + 50 * 10) / 1e6)


# ---- Runde fuer Runde + Fehler-Bremse ------------------------------------------------------------------------
def _pairs(path):
    p = Path(path)
    return [(r["task_id"], r["sample_idx"]) for r in map(json.loads, p.read_text().splitlines())] if p.exists() else []


def _finishes(path):
    return [json.loads(line)["finish_reason"] for line in Path(path).read_text().splitlines()]


def test_round_order_abort_leaves_complete_rounds_and_resume_fills_the_rest(api_stub, tmp_path, tasks):
    base, seen, script = api_stub
    script += [msg([text("Erledigt.")])] * 9                                   # 9 Rollouts klappen ...
    script.fallback = err(401, "authentication_error", "invalid x-api-key")    # ... dann ist der Account weg
    out, run = tmp_path / "round.jsonl", ("--effort", "high", "--n-tasks", "4", "--k", "3", "--order", "round")
    p = _cli(base, out, *run)
    assert p.returncode == 3, p.stdout + p.stderr
    pairs = _pairs(out)
    assert [s for _, s in pairs] == [0] * 4 + [1] * 4 + [2]                    # Runde 0 und 1 komplett, Runde 2 begonnen
    assert [t for t, _ in pairs[:4]] == [t for t, _ in pairs[4:8]]             # jede Runde = alle Tasks
    assert "vollstaendige Runden: 2 von 3" in p.stdout and "--max-sample 2" in p.stdout
    clean = report.load(str(out), max_sample=2)                                # sauberes pass^2 ueber alle 4 Tasks
    assert clean["meta"]["k"] == 2 and clean["meta"]["incomplete"] == 0 and clean["meta"]["n"] == 8
    raw = report.load(str(out))                                                # ohne Filter: k=3, 3 Tasks unvollstaendig
    assert raw["meta"]["k"] == 3 and raw["meta"]["incomplete"] == 3
    script.fallback = None                                                     # Guthaben wieder da -> Resume
    p2 = _cli(base, out, *run)
    assert p2.returncode == 0, p2.stdout + p2.stderr
    pairs = _pairs(out)
    assert len(pairs) == 12 and len(set(pairs)) == 12                          # genau der Rest, keine Duplikate


def test_task_order_stays_the_default(api_stub, tmp_path, tasks):
    base, seen, script = api_stub
    out = tmp_path / "task.jsonl"
    assert _cli(base, out, "--effort", "high", "--n-tasks", "2", "--k", "2").returncode == 0
    pairs = _pairs(out)
    assert [s for _, s in pairs] == [0, 1, 0, 1] and pairs[0][0] == pairs[1][0] != pairs[2][0]


def test_error_brake_stops_an_outage_with_unknown_wording_without_error_rows(api_stub, tmp_path, tasks):
    base, seen, script = api_stub
    script += [msg([text("Erledigt.")])] * 2
    script.fallback = err(400, "invalid_request_error", "a wording nobody anticipated")
    out = tmp_path / "brake.jsonl"
    p = _cli(base, out, "--effort", "high", "--n-tasks", "20", "--k", "1")
    assert p.returncode == 3, p.stdout + p.stderr
    assert "Fehlerserie" in p.stdout and len(seen) == 2 + 10                   # stoppt nach 10 Fehlern in Folge
    assert _finishes(out) == ["final_answer", "final_answer"]                 # keine Fehlerzeile geschrieben


@pytest.mark.parametrize("pattern", ["middle", "end"])
def test_isolated_errors_are_still_recorded(api_stub, tmp_path, tasks, pattern):
    base, seen, script = api_stub
    ok, bad = msg([text("Erledigt.")]), err(400, "invalid_request_error", "messages.1: something specific")
    script += [ok, bad, ok, ok] if pattern == "middle" else [ok, ok, ok, bad]
    out = tmp_path / f"iso_{pattern}.jsonl"
    p = _cli(base, out, "--effort", "high", "--n-tasks", "4", "--k", "1")
    assert p.returncode == 0, p.stdout + p.stderr
    fin = _finishes(out)
    assert len(fin) == 4 and sum(f.startswith("teacher_error:BadRequestError") for f in fin) == 1
    assert fin.index(next(f for f in fin if f.startswith("teacher_error"))) == (1 if pattern == "middle" else 3)

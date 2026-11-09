"""Konfiguration 'τ² + strict' (TAU2_STRICT_TOOLS=1): Shim-Patch, litellm-Pfad, Report-Hygiene, Preflight-Gate.

GPU-frei, ohne Docker. Der litellm-Test schickt den echten tau2-generate-Pfad an einen Stub-HTTP-Server auf
127.0.0.1 und prueft den Request-Body: "strict": true muss beim Server ankommen (Provider openai/), und die
Negativkontrolle hosted_vllm/ belegt, dass litellm es dort still loescht.
"""
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation/benchmarks/tau2"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


shim = _load("tau2_shim_under_test", BENCH / "run_tau2.py")
report = _load("tau2_report_under_test", BENCH / "tau2_report.py")


def get_customer_by_id(customer_id: str) -> str:
    """Look up a customer.

    Args:
        customer_id: The customer ID.
    """
    return customer_id


@pytest.fixture
def la(monkeypatch):
    """tau2.agent.llm_agent; jede Aenderung an la.generate wird nach dem Test zurueckgesetzt."""
    pytest.importorskip("tau2")
    import tau2.agent.llm_agent as mod
    monkeypatch.setattr(mod, "generate", mod.generate)
    return mod


@pytest.fixture
def tool():
    pytest.importorskip("tau2")
    from tau2.environment.tool import as_tool
    return as_tool(get_customer_by_id)


STRICT = {"agent": {"strict_tools": True}}


def test_without_strict_nothing_is_patched(la):
    orig = la.generate
    assert shim.apply_strict_tools({"agent": {}}) is False
    assert shim.apply_strict_tools({}) is False
    assert la.generate is orig


def test_strict_wraps_agent_tools_only(la, tool, monkeypatch):
    seen = []
    monkeypatch.setattr(la, "generate", lambda *a, **kw: seen.append((a, kw)) or "msg")
    assert shim.apply_strict_tools(STRICT) is True
    assert shim.apply_strict_tools(STRICT) is True                       # idempotent: kein Doppel-Wrapper
    assert la.generate(model="m", messages=[], tools=[tool]) == "msg"   # Keyword-Aufruf wie LLMAgent
    la.generate("m", [], [tool])                                        # positional
    for a, kw in seen:
        tools = kw["tools"] if "tools" in kw else a[2]
        (t,) = tools
        assert t.openai_schema["function"]["strict"] is True
        assert t.openai_schema["function"]["name"] == "get_customer_by_id"
        assert t.name == "get_customer_by_id"                             # Rest geht ans Original
    assert "strict" not in tool.openai_schema["function"]                  # Original nicht mutiert
    import tau2.user.user_simulator as us
    assert not getattr(us.generate, "_strict_tools", False)               # User-Sim unberuehrt


class _Stub(BaseHTTPRequestHandler):
    bodies: list = []

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        _Stub.bodies.append(json.loads(self.rfile.read(n)))
        out = json.dumps({
            "id": "chatcmpl-stub", "object": "chat.completion", "created": 0, "model": "stub",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):  # still
        pass


@pytest.fixture
def stub(monkeypatch):
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    _Stub.bodies = []
    srv = HTTPServer(("127.0.0.1", 0), _Stub)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()
    srv.server_close()


def test_strict_reaches_the_server_via_litellm(la, tool, stub):
    from tau2.data_model.message import UserMessage
    shim.apply_strict_tools(STRICT)
    msgs = [UserMessage(role="user", content="hi")]
    la.generate(model="openai/stub", messages=msgs, tools=[tool], api_base=stub, api_key="dummy", num_retries=0)
    la.generate(model="hosted_vllm/stub", messages=msgs, tools=[tool], api_base=stub, api_key="dummy",
                num_retries=0)
    assert len(_Stub.bodies) == 2
    via_openai, via_hosted = _Stub.bodies
    assert via_openai["tool_choice"] == "auto"
    assert [t["function"]["strict"] for t in via_openai["tools"]] == [True]
    # Die Falle, die der Agent-Provider openai/ umgeht: hosted_vllm/ strippt strict still.
    assert "strict" not in via_hosted["tools"][0]["function"]


def test_tool_hygiene_counts_agent_side_only():
    tc = lambda name, **args: SimpleNamespace(name=name, arguments=args)  # noqa: E731
    msgs = [
        SimpleNamespace(role="assistant", tool_calls=[tc("get_customer_by_id", customer_id="C1")]),
        SimpleNamespace(role="tool", requestor="assistant", error=False, content="{...}"),
        SimpleNamespace(role="assistant", tool_calls=[tc("check_status_bar"), tc("get_customer_by_id",
                                                                                customer_id="C1")]),
        SimpleNamespace(role="tool", requestor="assistant", error=True,
                        content="Error: Tool 'check_status_bar' not found."),
        SimpleNamespace(role="tool", requestor="assistant", error=False, content="{...}"),
        SimpleNamespace(role="assistant", tool_calls=[tc("get_customer_by_id", customer_id="C2")]),
        SimpleNamespace(role="tool", requestor="assistant", error=True, content="Error: Customer C2 not found"),
        SimpleNamespace(role="user", tool_calls=[tc("toggle_airplane_mode")]),
        SimpleNamespace(role="tool", requestor="user", error=True, content="Error: Tool 'x' not found."),
    ]
    assert report.tool_hygiene(msgs) == {"agent_tool_calls": 4, "unknown_tool_calls": 1, "repeat_tool_calls": 1}
    as_dicts = [{k: v for k, v in vars(m).items()} for m in msgs]          # results.json-Form
    assert report.tool_hygiene(as_dicts)["unknown_tool_calls"] == 1
    assert report.tool_hygiene(None) == {"agent_tool_calls": 0, "unknown_tool_calls": 0, "repeat_tool_calls": 0}


def test_strict_tools_gate():
    pytest.importorskip("yaml")
    pf = _load("tau2_preflight_under_test", BENCH / "preflight.py")
    old = {"agent": {"llm": "openai/x"}}                                    # Alt-Manifest ohne Feld
    assert pf.strict_tools_check(None, False, "qwen3-4b_base")[0] == "ok"
    assert pf.strict_tools_check(None, True, "qwen3-4b_x_vllm030-strict")[0] == "ok"
    assert pf.strict_tools_check(old, False, "qwen3-4b_base")[0] == "ok"
    assert pf.strict_tools_check(old, True, "qwen3-4b_base-strict")[0] == "fail"   # Resume mit anderem Schalter
    assert pf.strict_tools_check({"agent": {"strict_tools": True}}, False, "a-strict")[0] == "fail"
    assert pf.strict_tools_check({"agent": {"strict_tools": True}}, True, "a-strict")[0] == "ok"
    assert pf.strict_tools_check(None, True, "qwen3-4b_grpo-r3-step32")[0] == "fail"  # strict ohne Label-Hinweis

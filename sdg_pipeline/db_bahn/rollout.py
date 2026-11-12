"""
sdg_pipeline/db_bahn/rollout.py
===============================
Phase 3 of Plan (B): multi-turn teacher rollout harness against the tau2 `db_bahn` sandbox.

Forked from sdg_pipeline/trace_capture.py's production scaffolding (ThreadPoolExecutor + write-lock,
append-only + flush, resume by key, regenerate-before-drop) with the single-shot SQL call replaced by a
manual agent loop: teacher → parse tool calls → execute on the env (real observations) → append
role:"tool" messages → repeat until final answer or caps. Every rollout is scored inline by
evaluation/trajectory_reward.py; ALL rollouts are written with their score (verified-yield is measured
downstream — the bake-off metric), so nothing is silently dropped.

Tool-calling = prompt-and-parse (plan decision): we pass `tools=[...]` in the request so the model uses
its native trained syntax, but we parse `<tool_call>{...}</tool_call>` blocks ourselves (also accepting a
server-parsed `tool_calls` field if present) — no dependency on vLLM's tool parser. `<think>` blocks are
stripped from the running context (loop hygiene, base lesson); `<plan>` stays (Variante C supervision).

NATIVE tool mode (--native-tools, for models whose home format is NOT <think>/<tool_call>, e.g. gpt-oss/Harmony,
Gemma 4, Muse Glimmer): the server gets `tools` (+ tool_choice auto, no `strict` -> no grammar, as in prompt mode)
and splits calls + reasoning with its own parsers (vLLM --enable-auto-tool-choice --tool-call-parser X
--reasoning-parser Y); the system prompt drops our tools block + <tool_call> example (the template renders the
tools in the model's format). Records stay identical in shape: reasoning is stored inline as <think>…</think>.

ANTHROPIC backend (--backend anthropic, Claude API via the official `anthropic` SDK): the same native protocol
(SYSTEM_TEMPLATE_NATIVE, tools as a request parameter, tool_choice auto) — but NO sampling parameters (the API rejects
them), adaptive thinking at an explicit --effort, thinking returned as a SUMMARY (display "summarized") and stored
inline as <think> like every other model's. The signed thinking blocks of a turn must be replayed byte-for-byte (the
API rejects an edited history), so the per-rollout closure keeps the raw response blocks and the history is
append-only. Records carry per-call token usage in `api_calls` (cost: evaluation/api_usage.py). Account-level errors
(key, balance, permissions, unknown model) stop the WHOLE run without writing a record (exit 3) -> RESUME continues;
so do API_ERROR_STREAK error rollouts in a row (error brake: error rows are held until a good rollout follows, so an
outage with an unknown error wording leaves no error rows either). With --order round an abort leaves complete rounds
over all tasks (clean pass^r, eval_report.py --max-sample r).

Runs in the tau2 venv (needs the domain + verifier):
    PYTHONPATH=. <tau2-venv>/bin/python sdg_pipeline/db_bahn/rollout.py \
        --config config/pipeline_config.yaml --split bakeoff_dev --k 3

CPU smoke without any GPU/teacher:  --dry-run  (scripted oracle exercises the FULL loop incl. parser)
and --dry-run-broken (hallucinating oracle; must score 0.0).
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


try:  # soft dep: eval-metric tracking is opt-in (--mlflow); absence must never break a rollout
    import mlflow
except ImportError:
    mlflow = None

from sdg_pipeline.db_bahn.tau2_domain import get_environment
from sdg_pipeline.db_bahn.tau2_domain.environment import DATA_DIR
from data_pipeline.common import TOOLS_BLOCK_TMPL, load_config
from evaluation.trajectory_reward import score_trajectory

TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Qwen format drift: models also emit the XML-style call syntax -> parse both.
#   <tool_call><function=name><parameter=key>value</parameter>...</function></tool_call>
FUNC_XML_RE = re.compile(r"<function=([\w-]+)>(.*?)</function>", re.DOTALL)
PARAM_XML_RE = re.compile(r"<parameter=([\w-]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
ZUG_RE = re.compile(r"\b(?:ICE|IC|EC|ECE|RJ|EN)\s\d+\b")
MAX_TOOL_CONTENT = 4000

SYSTEM_TEMPLATE = "{policy}\n\n" + TOOLS_BLOCK_TMPL + """

## Arbeitsweise (wichtig)

- Denke vor jedem Schritt gründlich nach (Denkmodus), halte die SICHTBARE Ausgabe aber knapp:
  keine Selbstzweifel, kein „Warte“/„Eigentlich“, keine Wiederholungen.
- Rufe Werkzeuge in genau diesem Format auf (ein Block pro Aufruf, Argumente als JSON);
  UNABHÄNGIGE Abfragen bündelst du als MEHRERE Blöcke im selben Zug:
<tool_call>
{{"name": "werkzeug_name", "arguments": {{"argument": "wert"}}}}
</tool_call>
- Nach jedem Werkzeug-Ergebnis: kurz prüfen, ob dein Vorgehen noch passt; bei Überraschungen umplanen.
- Wenn die Aufgabe gelöst ist: KEIN Tool-Aufruf mehr, sondern eine kurze deutsche Schlussantwort mit den
  belegten Fakten (nur Werte, die ein Werkzeug geliefert hat)."""

# NATIVE tool mode: same prompt minus our tools block and the <tool_call> format rule (the chat template
# renders the `tools` param in the model's own format). Derived from SYSTEM_TEMPLATE so the rest can't drift.
_PROMPT_FORMAT_RULE = SYSTEM_TEMPLATE[SYSTEM_TEMPLATE.index("- Rufe Werkzeuge"):
                                      SYSTEM_TEMPLATE.index("</tool_call>") + len("</tool_call>")]
SYSTEM_TEMPLATE_NATIVE = (SYSTEM_TEMPLATE
                          .replace("\n\n" + TOOLS_BLOCK_TMPL, "", 1)
                          .replace(_PROMPT_FORMAT_RULE,
                                   "- Rufe Werkzeuge über die Funktionsaufruf-Schnittstelle auf (ein Aufruf je Werkzeug);\n"
                                   "  UNABHÄNGIGE Abfragen bündelst du als MEHRERE Aufrufe im selben Zug.", 1))
# home-format tool markup that survived WITHOUT a parsed call (native mode) = a format error, not an answer
TOOL_MARKUP_RE = re.compile(r"<\|tool_call>|<atem:|to=functions\.|<tool_call>|<function=")
THINK_TAG_RE = re.compile(r"<(?:seed:)?think>|</(?:seed:)?think>|\[THINK\]|\[/THINK\]")


def resolve_teacher(config: dict, api_base=None, model=None) -> dict:
    t = (config.get("teacher") or {}).get("vllm_local", {})
    traj = config.get("trajectory") or {}
    # wave-3 defaults = the official Qwen3.6 thinking-mode recipe (model card): temp 1.0,
    # top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5 (the anti-loop knob), thinking ON.
    return {"api_base": api_base or t.get("api_base", "http://localhost:8000/v1"),
            "model": model or t.get("model", ""),
            "api_key": t.get("api_key", "token-local"),
            "max_tokens": int(traj.get("max_tokens_per_turn", t.get("max_tokens", 2048))),
            "temperature": float(traj.get("temperature", 1.0)),
            "top_p": float(traj.get("top_p", 0.95)),
            "top_k": int(traj.get("top_k", 20)),
            "min_p": float(traj.get("min_p", 0.0)),
            "presence_penalty": float(traj.get("presence_penalty", 1.5)),
            "enable_thinking": bool(traj.get("enable_thinking", True)),
            "legacy_stop": bool(traj.get("legacy_stop", False))}


def to_wire(messages: list[dict]) -> list[dict]:
    """NATIVE mode: move the inline <think> of assistant turns into the reasoning field, so the template renders
    it in the model's own channel. Both names: vLLM (0.21–0.33) reads `reasoning` and hands the template
    `reasoning` + `reasoning_content`; Harmony reads only `reasoning`, the Muse template only `reasoning_content`.
    The records keep the canonical inline form — this copy only goes over the wire."""
    out = []
    for m in messages:
        if m.get("role") == "assistant":
            think, rest = extract_think(m.get("content") or "")
            if think:
                m = {**m, "content": rest, "reasoning": think, "reasoning_content": think}
        out.append(m)
    return out


def make_teacher_call(cfg: dict, timeout: float = 600.0):
    import httpx
    client = httpx.Client(timeout=timeout)
    url = cfg["api_base"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}
    native = bool(cfg.get("native_tools"))

    def call(messages: list[dict], tools: list[dict]) -> tuple[str, list[dict], str]:
        # PROMPT-AND-PARSE (default): tool schemas live in the system prompt; the `tools` param is NOT sent
        # (vLLM 400s on tools without --enable-auto-tool-choice — exactly the server dependency we avoid).
        # NATIVE (--native-tools): `tools` + tool_choice auto, the server's parsers split calls and reasoning;
        # no `strict` on the tools -> vLLM applies no grammar (same freedom as prompt mode).
        payload = {"model": cfg["model"], "messages": to_wire(messages) if native else messages,
                   "temperature": cfg["temperature"], "max_tokens": cfg["max_tokens"],
                   # top_k/min_p are vLLM protocol extensions (accepted at the top level)
                   "top_p": cfg["top_p"], "top_k": cfg["top_k"], "min_p": cfg["min_p"],
                   "presence_penalty": cfg["presence_penalty"]}
        if native:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if cfg.get("legacy_stop"):
            # pre-wave-3 behavior: halt right after the FIRST tool-call block. This also made
            # multi-call turns (parallel batching) impossible; the role-played-tool-RESPONSE
            # problem it solved is now handled by parse_tool_calls' tail cut.
            payload["stop"] = ["</tool_call>", "</tools>", "</TOOLCALL>"]
            payload["include_stop_str_in_output"] = True
        if not cfg.get("omit_thinking_kwarg"):  # mistral-tokenizer models reject template kwargs
            # extra template switches per model family (e.g. preserve_thinking), merged over the thinking flag
            payload["chat_template_kwargs"] = {"enable_thinking": cfg["enable_thinking"],
                                               **(cfg.get("chat_template_kwargs") or {})}
        r = client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:  # include the body — vLLM puts the actual reason there
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        choice = r.json()["choices"][0]
        msg = choice["message"]
        content = msg.get("content") or ""
        # a server-side reasoning parser returns the thinking in its own field (vLLM >= 0.30 `reasoning`, older
        # `reasoning_content`) -> rewrap it inline so extract_think sees it like any other model's <think>.
        # Without a reasoning parser (all prompt-mode runs) the field is absent and nothing changes.
        reasoning = (msg.get("reasoning") or msg.get("reasoning_content") or "").strip()
        if reasoning and not THINK_TAG_RE.search(content):
            content = f"<think>\n{reasoning}\n</think>\n\n{content}"
        return content, msg.get("tool_calls") or [], choice.get("finish_reason", "")

    return call


# --- Claude API backend (--backend anthropic) -------------------------------------------------
class FatalApiError(RuntimeError):
    """Account-level API failure (key, balance, permissions, unknown model): stops the WHOLE run — every further
    request would fail the same way, and error records would be counted as done by a resume."""


API_THINKING = {"type": "adaptive", "display": "summarized"}   # thinking ON (adaptive), summary visible
API_ERROR_STREAK = 10   # error rollouts in a row (API backend) = systematic outage -> stop without writing them
_CACHE = {"type": "ephemeral"}                                  # 5-min prompt cache (kept warm by the eval traffic)
_ANTHROPIC_CLIENTS: dict = {}
_ANTHROPIC_LOCK = threading.Lock()


def api_gen_params(cfg: dict) -> dict:
    """gen_params of an API record: what was actually sent (no sampling parameters exist on this path)."""
    return {"backend": cfg["backend"], "model": cfg["model"], "effort": cfg["effort"], "thinking": API_THINKING,
            "enable_thinking": True, "max_tokens": cfg["max_tokens"], "tool_mode": "native"}


def anthropic_tools(tools: list[dict]) -> list[dict]:
    """OpenAI function schema -> Anthropic tool definition (same names, descriptions, JSON schemas and order)."""
    return [{"name": t["function"]["name"], "description": t["function"].get("description") or "",
             "input_schema": t["function"].get("parameters") or {"type": "object", "properties": {}}}
            for t in tools]


def to_anthropic(messages: list[dict], raw_turns: list) -> tuple[list[dict], list[dict]]:
    """Canonical messages -> (system blocks, Anthropic messages). Assistant turn i is replayed from raw_turns[i] —
    the exact blocks the API returned (signed thinking included, never rebuilt from the canonical text); the tool
    results of one turn go into ONE user message (plain content, no is_error — same signal the open models get).
    The system block carries the fixed cache breakpoint (cache prefix = tools + system, identical in every episode)."""
    system, out, n_asst = [], [], 0
    for m in messages:
        role = m.get("role")
        if role == "system":
            system = [{"type": "text", "text": m.get("content") or "", "cache_control": _CACHE}]
        elif role == "user":
            out.append({"role": "user", "content": m.get("content") or ""})
        elif role == "assistant":
            if n_asst >= len(raw_turns):
                raise RuntimeError("anthropic: assistant turn without raw response blocks — no faithful replay")
            out.append({"role": "assistant", "content": raw_turns[n_asst]})
            n_asst += 1
        elif role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.get("tool_call_id"), "content": m.get("content") or ""}
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return system, out


def _is_fatal_api_error(e) -> bool:
    """Errors that hit every request of the run alike: bad key, no permission, unknown model, empty balance or
    spend limit, organization on hold. Everything else (429/5xx after the SDK's retries, timeouts, a single 400)
    stays a per-rollout teacher_error, as with vLLM."""
    import anthropic
    if isinstance(e, (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError,
                      anthropic.EnforcedSpendLimitReachedError, anthropic.OrganizationOnHoldError)):
        return True
    if isinstance(e, anthropic.APIStatusError):
        text = f"{e.type or ''} {e.error_code or ''} {e.api_message or ''}".lower()
        return e.status_code == 402 or "billing" in text or "credit balance" in text
    return False


def _anthropic_client(cfg: dict, timeout: float):
    """One client per (endpoint, timeout, retries) and process — thread-safe, shared by all rollouts. The key comes
    from ANTHROPIC_API_KEY; the SDK retries 408/409/429/5xx/529 with backoff (honours retry-after)."""
    k = (cfg.get("api_base"), float(timeout), int(cfg.get("api_max_retries", 8)))
    with _ANTHROPIC_LOCK:
        if k not in _ANTHROPIC_CLIENTS:
            import anthropic
            _ANTHROPIC_CLIENTS[k] = anthropic.Anthropic(base_url=k[0] or None, timeout=k[1], max_retries=k[2])
        return _ANTHROPIC_CLIENTS[k]


def make_anthropic_call(cfg: dict, timeout: float = 600.0):
    """teacher_call for the Claude API — one closure per rollout: it keeps the raw response blocks (replayed
    unchanged, see to_anthropic) and the per-call usage (`call.api_calls`, copied into the record by run_rollout)."""
    import anthropic
    client = _anthropic_client(cfg, timeout)
    raw_turns: list = []
    api_calls: list[dict] = []
    tool_defs: list = []

    def call(messages: list[dict], tools: list[dict]) -> tuple[str, list[dict], str]:
        if not tool_defs:
            tool_defs.extend(anthropic_tools(tools))
        system, msgs = to_anthropic(messages, raw_turns)
        try:
            resp = client.messages.create(
                model=cfg["model"], max_tokens=cfg["max_tokens"], system=system, tools=tool_defs,
                tool_choice={"type": "auto"}, messages=msgs, thinking=API_THINKING,
                output_config={"effort": cfg["effort"]},
                cache_control=_CACHE)   # automatic breakpoint on the growing tail (2 of 4 breakpoints used)
        except anthropic.APIStatusError as e:
            if _is_fatal_api_error(e):
                raise FatalApiError(f"{type(e).__name__} (HTTP {e.status_code}): "
                                    f"{(e.api_message or str(e))[:200]}") from e
            raise
        raw_turns.append(resp.content)
        thinks, texts, calls = [], [], []
        for b in resp.content:
            if b.type == "thinking" and (b.thinking or "").strip():
                thinks.append(b.thinking.strip())
            elif b.type == "text" and (b.text or "").strip():
                texts.append(b.text.strip())
            elif b.type == "tool_use":
                calls.append({"id": b.id, "type": "function",
                              "function": {"name": b.name,
                                           "arguments": json.dumps(b.input or {}, ensure_ascii=False)}})
        u, sr = resp.usage, resp.stop_reason or ""
        rec = {"input_tokens": u.input_tokens, "cache_read_input_tokens": u.cache_read_input_tokens or 0,
               "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
               "output_tokens": u.output_tokens, "stop_reason": sr, "model": resp.model}
        if sr == "refusal":
            rec["refusal_category"] = getattr(resp.stop_details, "category", None)
        if getattr(resp, "_request_id", None):
            rec["request_id"] = resp._request_id
        api_calls.append(rec)
        think, text = "\n\n".join(thinks), "\n\n".join(texts)
        content = f"<think>\n{think}\n</think>\n\n{text}" if think else text
        fr = "length" if sr in ("max_tokens", "model_context_window_exceeded") else sr
        return content, calls, fr

    call.api_calls = api_calls
    return call


def strip_think(content: str) -> str:
    """Remove reasoning across model dialects: paired tags (<think>, <seed:think>, [THINK]) AND an
    unpaired trailing close (thinking-only models — the template injects the opening tag, so output is
    `reasoning</think>answer`; base extract_sql lesson)."""
    content = re.sub(r"<(?:seed:)?think>.*?</(?:seed:)?think>\s*", "", content or "", flags=re.DOTALL)
    content = re.sub(r"\[THINK\].*?\[/THINK\]\s*", "", content, flags=re.DOTALL)
    for close in ("</seed:think>", "</think>", "[/THINK]"):
        if close in content:
            content = content.rsplit(close, 1)[1]
    return content.strip()


def extract_think(content: str) -> tuple[str, str]:
    """Split reasoning from the visible output across model dialects — the KEEPING mirror of
    strip_think (wave 3: thinking is captured, canonically rewrapped as <think>…</think> and
    stored inline in the assistant content). Unpaired trailing close: everything before it is
    reasoning (thinking-only models — the template injects the opening tag)."""
    content = content or ""
    thinks: list[str] = []

    def grab(m):
        thinks.append(m.group(1).strip())
        return ""

    rest = re.sub(r"<(?:seed:)?think>(.*?)</(?:seed:)?think>\s*", grab, content, flags=re.DOTALL)
    rest = re.sub(r"\[THINK\](.*?)\[/THINK\]\s*", grab, rest, flags=re.DOTALL)
    for close in ("</seed:think>", "</think>", "[/THINK]"):
        if close in rest:
            head, rest = rest.rsplit(close, 1)
            thinks.append(head.strip())
    return "\n\n".join(t for t in thinks if t), rest.strip()


def parse_tool_calls(content: str, native: list[dict]) -> tuple[str, list[dict], int]:
    """Native server-parsed tool_calls win; else parse <tool_call> blocks from the (think-stripped)
    text. Third return: chars of role-played prose AFTER the last tool-call block (wave 3: the
    stop token is gone, so hallucinated tool RESPONSES are cut here instead — and measured)."""
    content = strip_think(content)
    if native:
        calls = [{"id": tc.get("id") or f"n{i:08d}", "type": "function",
                  "function": {"name": tc["function"]["name"],
                               "arguments": tc["function"].get("arguments") or "{}"}}
                 for i, tc in enumerate(native, 1)]
        return content, calls, 0
    tail_chars = 0
    blocks = list(TOOLCALL_RE.finditer(content))
    if blocks:
        tail_chars = len(content[blocks[-1].end():].strip())
        content = content[:blocks[-1].end()]
    calls = []
    for i, block in enumerate(TOOLCALL_RE.findall(content), 1):
        try:
            obj = json.loads(block)
            calls.append({"id": f"c{i:08d}", "type": "function",
                          "function": {"name": obj.get("name", ""),
                                       "arguments": json.dumps(obj.get("arguments") or {}, ensure_ascii=False)}})
        except json.JSONDecodeError:
            continue  # malformed block -> treated as prose; regen loop handles empty-progress turns
    for j, (name, body) in enumerate(FUNC_XML_RE.findall(content), 1):  # Qwen XML drift format
        args = {k: v.strip() for k, v in PARAM_XML_RE.findall(body)}
        calls.append({"id": f"x{j:08d}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    if not calls and "</tool_call>" in content and "<tool_call>" not in content:
        # template-injected opening tag (GLM): content starts directly with the JSON, then </tool_call>
        m = re.search(r"(\{.*?\})\s*</tool_call>", content, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                if obj.get("name"):
                    calls.append({"id": "g00000001", "type": "function",
                                  "function": {"name": obj["name"],
                                               "arguments": json.dumps(obj.get("arguments") or {},
                                                                       ensure_ascii=False)}})
                    content = content[:m.start()]
            except json.JSONDecodeError:
                pass
    if not calls:  # bare JSON call, no wrapper at all (Qwen3-30B-Thinking) -> first call only, cut tail
        m = re.search(r'(\{"name"\s*:\s*"[\w-]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}\})', content)
        if m:
            try:
                obj = json.loads(m.group(1))
                calls.append({"id": "b00000001", "type": "function",
                              "function": {"name": obj["name"],
                                           "arguments": json.dumps(obj.get("arguments") or {},
                                                                   ensure_ascii=False)}})
                content = content[:m.start()]
            except json.JSONDecodeError:
                pass
    if not calls:  # wrapper drift (<tools>, <TOOLCALL>[...]) + unclosed tail (stop-sequence cut)
        m = re.search(r"<(?:tools|tool_call|TOOLCALL)>\s*\[?\s*(\{.*?\})\s*\]?\s*"
                      r"(?:</(?:tools|tool_call|TOOLCALL)>|$)", content, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(1))
                if obj.get("name"):
                    calls.append({"id": "t00000001", "type": "function",
                                  "function": {"name": obj["name"],
                                               "arguments": json.dumps(obj.get("arguments") or {},
                                                                       ensure_ascii=False)}})
                    content = content[:m.start()]
            except json.JSONDecodeError:
                pass
    clean = FUNC_XML_RE.sub("", TOOLCALL_RE.sub("", content))
    clean = re.sub(r"<tool_call>\s*</tool_call>", "", clean).strip()
    return clean, calls, tail_chars


def openai_tools(env) -> list[dict]:
    return [{"type": "function", "function": t.openai_schema["function"]} for t in env.get_tools()]


# --- oracle teachers for GPU-free smoke ------------------------------------------------------
def make_oracle(task: dict, key: dict, broken: bool = False):
    """Scripted teacher: replays the answer key's `oracle_calls` (the exact valid path) as
    <tool_call> TEXT (exercises the parser), then a grounded final answer."""
    crit = task.get("evaluation_criteria") or {}
    m = ZUG_RE.search(task.get("ticket") or "")
    zugnummer = m.group(0) if m else ""
    plan = [(c["name"], c["arguments"]) for c in (key.get("oracle_calls") or [])]
    comm = crit.get("communicate_info") or []
    state = {"i": 0}

    def call(messages, tools):
        if broken:
            return (f"<plan>Ich beantworte das direkt.</plan>\n"
                    f"Als Zuständiger ist Max Muster (MA-99999) vermerkt. {' '.join(comm)}", [], "stop")
        i = state["i"]
        if i < len(plan):
            state["i"] += 1
            name, args = plan[i]
            return (f"<plan>Schritt {i + 1}: {name} aufrufen.</plan>\n"
                    f"<tool_call>\n{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}\n</tool_call>",
                    [], "stop")
        facts = " ".join(str(c) for c in comm) or "Aufgabe ausgeführt."
        return (f"Ergebnis: {facts} (Details siehe Werkzeug-Ausgaben zu {zugnummer}.)", [], "stop")

    return call


# --- the multi-turn rollout ------------------------------------------------------------------
def run_rollout(task: dict, key: dict, teacher_call, max_turns: int, rollout_timeout_s: float,
                resume_messages: list | None = None, native: bool = False) -> dict:
    """One multi-turn rollout, scored by the verifier.

    resume_messages (A1 branch-on-failure): if given, the env state is reconstructed by replaying the
    prefix's assistant tool calls (mirrors trajectory_reward's replay), messages start from a copy of the
    prefix, and the teacher continues the tail. Default (None) = today's from-scratch behavior, untouched.
    native (--native-tools): system prompt without our tools block / format rule; a turn without a parsed
    call but with leftover tool markup ends as finish 'tool_format_error' (the score stays the verifier's).
    """
    from tau2.data_model.tasks import EnvFunctionCall

    env = get_environment(solo_mode=True)  # fresh env per rollout (thread-safe by isolation)
    inits = (task.get("initial_state") or {}).get("initialization_actions") or []
    if inits:
        env.run_env_function_calls([EnvFunctionCall.model_validate(a) for a in inits])
    tools = openai_tools(env)
    if resume_messages is None:
        tools_block = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
        sys_prompt = (SYSTEM_TEMPLATE_NATIVE if native else SYSTEM_TEMPLATE).format(
            policy=env.get_policy(), tools_block=tools_block)
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": task["ticket"]}]
    else:
        messages = [dict(m) for m in resume_messages]  # continue from the verified prefix
        # reconstruct env state: replay the prefix's tool calls (order-faithful)
        obs_by_id = {m.get("tool_call_id"): (m.get("content") or "")
                     for m in messages if m.get("role") == "tool"}
        for m in messages:
            if m.get("role") != "assistant":
                continue
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                raw = fn.get("arguments")
                if isinstance(raw, dict):
                    args = raw
                else:
                    try:
                        args = json.loads(raw or "{}")
                    except json.JSONDecodeError:
                        continue  # the original call errored on parse too (no mutation) -> skip
                if not isinstance(args, dict):
                    continue
                try:
                    env.use_tool(fn.get("name", ""), **args)
                except Exception as e:
                    # a call the ORIGINAL rollout also failed mutated nothing -> harmless. But if
                    # the recorded observation (matched by tool_call_id) was a success, the replay
                    # diverged -> the branch would continue on a WRONG env state; surface it.
                    orig_obs = obs_by_id.get(tc.get("id"), "")
                    if not orig_obs.startswith('{"error"'):
                        print(f"warning: prefix replay diverged on {fn.get('name', '')} for task "
                              f"{task.get('id')}: {type(e).__name__}: {e}")
    t0 = time.time()
    finish = "max_turns"
    truncated = False
    tail_total = 0
    for _ in range(max_turns):
        if time.time() - t0 > rollout_timeout_s:
            finish = "timeout"
            break
        try:
            content, server_calls, fr = teacher_call(messages, tools)
        except FatalApiError:  # account-level (key/balance/model): stop the run, never a record
            raise
        except Exception as e:  # HTTP 4xx/5xx, context overflow, transient server error — end this
            finish = f"teacher_error:{type(e).__name__}: {str(e)[:120]}"  # rollout gracefully (score 0,
            break                                                        # keep partial trace, don't abort)
        # wave 3: capture the reasoning and store it INLINE (canonical <think> rewrap). The same
        # messages list is the next turn's context, so prior-turn think stays visible to the
        # teacher — matching both the Qwen template's single-query behavior and what the student
        # will see in training and at inference.
        think, visible = extract_think(content)
        clean, calls, tail_chars = parse_tool_calls(visible, server_calls)
        tail_total += tail_chars
        assistant = {"role": "assistant",
                     "content": (f"<think>\n{think}\n</think>\n\n" if think else "") + clean}
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)
        if fr == "length":  # wave 3: a turn that hit the token cap is almost always a think loop
            finish = "truncated"
            truncated = True
            break
        if fr == "refusal":  # Claude API safety decline (HTTP 200, stop_reason refusal) — not an answer
            finish = "refusal"
            break
        if not calls:
            if native and TOOL_MARKUP_RE.search(visible):  # the server's parser missed a home-format call
                finish = "tool_format_error"
            else:
                finish = "final_answer" if clean.strip() else f"empty({fr})"
            break
        for tc in calls:
            fn = tc["function"]
            try:
                obs = env.use_tool(fn["name"], **json.loads(fn["arguments"] or "{}"))
                content_s = json.dumps(obs, ensure_ascii=False, default=str)
            except Exception as e:
                content_s = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": content_s[:MAX_TOOL_CONTENT]})
    score = score_trajectory(task, messages, key)
    res = {"messages": messages, "score": score, "finish_reason": finish,
           "truncated": truncated, "degen": degen_stats(messages),
           "roleplay_tail_chars": tail_total,
           "wall_s": round(time.time() - t0, 2)}
    if hasattr(teacher_call, "api_calls"):  # API backend: per-call token usage (cost) — absent on every other path
        res["api_calls"] = list(teacher_call.api_calls)
    return res


# --- A1: branch-on-failure (answer-key-guided prefix reuse) -----------------------------------
def _agent_calls(messages: list) -> list:
    """(name, args_dict, obs_msg_index) for each assistant tool call, in order."""
    out = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            # observation for this call is the next tool message after this assistant turn
            obs_idx = next((j for j in range(i + 1, len(messages))
                            if messages[j].get("role") == "tool"), i)
            out.append((fn.get("name", ""), args, obs_idx))
    return out


def _matches(name: str, args: dict, ref: dict) -> bool:
    """Agent call matches a reference oracle_calls entry: same tool + args agree on the reference keys."""
    if name != ref.get("name"):
        return False
    return all(args.get(k) == v for k, v in (ref.get("arguments") or {}).items())


def choose_branch_point(messages: list, key: dict) -> list | None:
    """Longest prefix of the rollout that still tracks the gold path (key['oracle_calls']); cut BEFORE the
    first deviation (safe even for a clean-but-wrong WRITE, which has no error observation). Returns the
    message prefix to resume from, or None if there is no matching progress (→ full restart)."""
    oracle = key.get("oracle_calls") or []
    if not oracle:
        return None
    calls = _agent_calls(messages)
    j, last_obs = 0, None
    for name, args, obs_idx in calls:
        if j < len(oracle) and _matches(name, args, oracle[j]):
            j += 1
            last_obs = obs_idx
        else:
            break  # first deviation from the gold path
    if j == 0 or last_obs is None:
        return None
    return messages[:last_obs + 1]


_WRITE_TOOLS = {"crew_zuweisen", "wartung_einplanen", "wartung_status_setzen"}


def choose_harvest_point(messages: list, key: dict) -> list | None:
    """B2 recovery-harvest (opposite cut to choose_branch_point): keep the prefix UP TO AND INCLUDING the
    first divergent step + its observation, so the resampled continuation must RECOVER from the mistake
    → a self-correction trace ("tried X, wrong, replanned, tried Y"). Only valid if the mistake is
    NON-MUTATING (a READ, or a WRITE that was rejected/errored) — a mutating WRITE cannot be undone to reach
    the gold state. Returns the prefix (incl. the mistake), or None → fall back to yield-mode. Unlike
    choose_branch_point this also fires on a step-1 divergence (the hard search tails yield-mode can't reuse)."""
    oracle = key.get("oracle_calls") or []
    if not oracle:
        return None
    calls = _agent_calls(messages)
    j = 0
    for name, args, obs_idx in calls:
        if j < len(oracle) and _matches(name, args, oracle[j]):
            j += 1
            continue
        # calls[this] is the first divergence = the mistake
        obs = messages[obs_idx] if 0 <= obs_idx < len(messages) else {}
        obs_content = obs.get("content", "") if obs.get("role") == "tool" else ""
        rejected = '"error"' in obs_content
        if name in _WRITE_TOOLS and not rejected:
            return None  # mutating write executed → no undo → cannot harvest to gold
        return messages[:obs_idx + 1]  # keep prefix + mistake + its observation
    return None  # no divergence (everything matched) → nothing to harvest


def _graded(res: dict) -> float:
    """Fraction of passed verifier components — ranks failed attempts to pick the best branch base."""
    comp = (res.get("score") or {}).get("components") or {}
    return sum(1 for v in comp.values() if v) / len(comp) if comp else 0.0


# --- wave 3: degeneration gate (the verifier is outcome-based and blind to think rambling) ----
# CONSERVATIVE initial thresholds — calibrated against the smoke runs' P99 of healthy verified
# traces (S6); a hit means "regenerate", not "keep", so false positives only cost a retry.
DEGEN_MAX_THINK_CHARS = 12_000
DEGEN_MAX_DUP8_RATIO = 0.5
THINK_CAPTURE_RE = re.compile(r"<think>\n?(.*?)\n?</think>", re.DOTALL)


def degen_stats(messages: list) -> dict:
    """Worst-case think length + 8-gram duplication ratio across the trace's think blocks."""
    dup_max, think_max = 0.0, 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for th in THINK_CAPTURE_RE.findall(m.get("content") or ""):
            think_max = max(think_max, len(th))
            words = th.split()
            if len(words) >= 32:
                grams = [" ".join(words[i:i + 8]) for i in range(len(words) - 7)]
                dup_max = max(dup_max, 1.0 - len(set(grams)) / len(grams))
    return {"think_ngram_dup_ratio": round(dup_max, 3), "max_think_chars": think_max}


def is_degenerate(res: dict) -> bool:
    d = res.get("degen") or {}
    return (d.get("think_ngram_dup_ratio", 0.0) > DEGEN_MAX_DUP8_RATIO
            or d.get("max_think_chars", 0) > DEGEN_MAX_THINK_CHARS)


def accepted(res: dict) -> bool:
    """Wave-3 accept gate: verified AND not token-capped AND not degenerate. Replaces the bare
    score==1.0 checks in solve_task/_try_recovery, so branch/regen machinery re-rolls loops."""
    return (res["score"]["score"] == 1.0 and not res.get("truncated")
            and not is_degenerate(res))


def _try_recovery(base: dict, task: dict, key: dict, make_call, args) -> tuple:
    """One B2-priority recovery attempt on a failed rollout: harvest (keep the mistake) first, then
    yield-mode (clean) fallback. Returns (candidate, mode|None, n_resamples). `candidate` is the best
    result seen (verified if recovered, else the most-progressed)."""
    bump = getattr(args, "branch_temp_bump", 0.0)
    n = 0
    best = base
    for point_fn, mode in ((choose_harvest_point, "harvest"), (choose_branch_point, "clean")):
        prefix = point_fn(best["messages"], key)
        if prefix is None:
            continue
        cand = run_rollout(task, key, make_call(temp_bump=bump), args.max_turns,
                           args.rollout_timeout_s, resume_messages=prefix,
                           native=getattr(args, "native_tools", False))
        n += 1
        if accepted(cand):
            return cand, mode, n
        if _graded(cand) > _graded(best):
            best = cand
    return best, None, n


def solve_task(task: dict, key: dict, make_call, args, teacher_cfg) -> dict:
    """Per-item rollout policy. Without --branch-on-fail: plain best-of-N (--max-regen), unchanged.
    With --branch-on-fail: BRANCH-FIRST + B2-priority — on a failure try recovery (harvest self-correction
    first, then yield-mode clean) BEFORE a full restart; restarts are the fallback. Bounded by branch_attempts
    (recovery resamples) + max_regen (full restarts). Adds 'n_branch_attempts' and 'recovery_mode'
    (direct|harvest|clean|restart|failed). Module-level so it is unit-testable without a GPU."""
    is_oracle = args.dry_run or args.dry_run_broken
    branch_on = getattr(args, "branch_on_fail", False)

    native = getattr(args, "native_tools", False)
    best = run_rollout(task, key, make_call(), args.max_turns, args.rollout_timeout_s, native=native)
    n_branch = 0
    mode = "direct" if accepted(best) else None

    if not is_oracle:
        # range(N), NOT range(N+1): --max-regen is "re-sample a failed rollout up to N times", so 0
        # must mean exactly one attempt (what an eval wants). The old +1 made --max-regen 0 still fire
        # one restart; the argparse default moved 1 -> 2 so the generation path is unchanged.
        for _ in range(args.max_regen):
            if accepted(best):
                break
            if branch_on and n_branch < args.branch_attempts:  # branch-first: recovery before restart
                cand, m, n = _try_recovery(best, task, key, make_call, args)
                n_branch += n
                best = cand
                if accepted(best):
                    mode = m
                    break
            # recovery didn't verify (or no reusable prefix / budget spent) → fresh full sample
            fresh = run_rollout(task, key, make_call(), args.max_turns, args.rollout_timeout_s, native=native)
            if accepted(fresh):
                best, mode = fresh, "restart"
                break
            if _graded(fresh) > _graded(best):
                best = fresh

    best["n_branch_attempts"] = n_branch
    best["recovery_mode"] = mode or ("direct" if accepted(best) else "failed")
    return best


def main():
    ap = argparse.ArgumentParser(description="Multi-turn teacher rollouts against the db_bahn sandbox")
    ap.add_argument("--config", default="config/pipeline_config.yaml")
    ap.add_argument("--split", default="bakeoff_dev",
                    help="split name from split_tasks.json (validated after loading)")
    ap.add_argument("--task-ids-file", default=None,
                    help="restrict to these task ids (one per line); for k=2 top-up on a failed subset")
    ap.add_argument("--k", type=int, default=1, help="rollouts per task")
    ap.add_argument("--n-tasks", type=int, default=None, help="cap number of tasks (debug)")
    ap.add_argument("--stratify", action="store_true",
                    help="pick --n-tasks round-robin over templates (balanced bake-off subset)")
    ap.add_argument("--max-tokens-per-turn", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--enable-thinking", dest="enable_thinking", action="store_true", default=None,
                    help="override config trajectory.enable_thinking (wave-3 default: on)")
    ap.add_argument("--no-enable-thinking", dest="enable_thinking", action="store_false")
    ap.add_argument("--legacy-stop", action="store_true",
                    help="restore the pre-wave-3 stop list (halts after the FIRST tool call — "
                         "makes parallel multi-call turns impossible; A/B fallback only)")
    ap.add_argument("--omit-thinking-kwarg", action="store_true",
                    help="don't send chat_template_kwargs (mistral tokenizer mode)")
    ap.add_argument("--chat-template-kwargs", default=None, metavar="JSON",
                    help="extra chat-template switches as a JSON object, merged over enable_thinking "
                         "(e.g. '{\"preserve_thinking\": true}'); recorded in gen_params only when set")
    ap.add_argument("--native-tools", action="store_true",
                    help="NATIVE tool mode: send `tools` (tool_choice auto), the server parses calls + reasoning "
                         "(needs vLLM --enable-auto-tool-choice --tool-call-parser X --reasoning-parser Y); "
                         "system prompt without our tools block; recorded as gen_params.tool_mode=native")
    ap.add_argument("--backend", choices=("openai", "anthropic"), default="openai",
                    help="openai = OpenAI-compatible endpoint (vLLM, default); anthropic = Claude API via the "
                         "`anthropic` SDK (implies --native-tools, no sampling flags, needs --effort + ANTHROPIC_API_KEY; "
                         "--api-base overrides the SDK endpoint, e.g. a test stub)")
    ap.add_argument("--effort", choices=("low", "medium", "high", "xhigh", "max"), default=None,
                    help="--backend anthropic: output_config.effort, sent EXPLICITLY (the model default, e.g. "
                         "Sonnet 5.5 high / Haiku 5.5 medium, so a later default change can't shift a run)")
    ap.add_argument("--api-max-retries", type=int, default=8,
                    help="--backend anthropic: SDK retries for 429/5xx/529 (exponential backoff, honours retry-after)")
    ap.add_argument("--order", choices=("task", "round"), default="task",
                    help="schedule: task = all k samples of a task in a row (default, historical); round = sample 0 "
                         "for ALL tasks, then sample 1, ... -> an aborted run (e.g. API balance) leaves r complete "
                         "rounds over every task (clean pass^r via eval_report --max-sample r); RESUME fills the rest")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--max-regen", type=int, default=2, help="re-sample a failed rollout up to N times "
                                                            "(0 = single shot, for evals)")
    ap.add_argument("--branch-on-fail", action="store_true",
                    help="A1: on a failed rollout, keep the gold-path prefix and resample only the tail")
    ap.add_argument("--branch-attempts", type=int, default=2,
                    help="rewind+resample attempts after a full-rollout failure (only with --branch-on-fail)")
    ap.add_argument("--branch-temp-bump", type=float, default=0.2,
                    help="temperature increase for resampled tails (diversity, avoid repeating the mistake)")
    ap.add_argument("--rollout-timeout-s", type=float, default=300.0)
    ap.add_argument("--request-timeout-s", type=float, default=600.0,
                    help="HTTP read timeout per model call (s). Under high server concurrency a single long turn can "
                         "exceed it and is then counted as teacher_error; evals set it to the rollout budget")
    ap.add_argument("--teacher-name", default=None, help="label written into records (bake-off table)")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--dry-run", action="store_true", help="scripted oracle teacher (CPU smoke)")
    ap.add_argument("--dry-run-broken", action="store_true", help="hallucinating oracle (must score 0)")
    ap.add_argument("--mlflow", action="store_true", help="log summary eval metrics to MLflow (opt-in)")
    ap.add_argument("--mlflow-experiment", default="db_bahn_traj_eval")
    ap.add_argument("--mlflow-run-name", default=None, help="default: the teacher label")
    ap.add_argument("--mlflow-tracking-uri", default=None,
                    help="default: $MLFLOW_TRACKING_URI, else the repo-local mlruns/ dir")
    args = ap.parse_args()
    extra_kwargs = None
    if args.chat_template_kwargs:
        try:
            extra_kwargs = json.loads(args.chat_template_kwargs)
        except json.JSONDecodeError as e:
            ap.error(f"--chat-template-kwargs is not valid JSON: {e}")
        if not isinstance(extra_kwargs, dict):
            ap.error("--chat-template-kwargs must be a JSON object")
        if args.omit_thinking_kwarg:
            ap.error("--chat-template-kwargs and --omit-thinking-kwarg exclude each other")
    if args.backend == "anthropic" and not (args.dry_run or args.dry_run_broken):
        sampling = [f"--{f.replace('_', '-')}" for f in ("temperature", "top_p", "top_k", "min_p", "presence_penalty")
                    if getattr(args, f) is not None]
        if sampling:
            ap.error(f"--backend anthropic takes no sampling flags (the API rejects them): {' '.join(sampling)}")
        if args.chat_template_kwargs or args.omit_thinking_kwarg or args.legacy_stop:
            ap.error("--backend anthropic: --chat-template-kwargs/--omit-thinking-kwarg/--legacy-stop are vLLM-only")
        if args.enable_thinking is False:
            ap.error("--backend anthropic: thinking stays on (adaptive)")
        if not args.effort:
            ap.error("--backend anthropic needs --effort (send the model default explicitly)")
        if not args.model:
            ap.error("--backend anthropic needs --model (e.g. claude-sonnet-5-5)")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            ap.error("--backend anthropic: ANTHROPIC_API_KEY is not set")
        args.native_tools = True   # the API path IS the native protocol (tools parameter, SYSTEM_TEMPLATE_NATIVE)
    elif args.effort and args.backend != "anthropic":
        ap.error("--effort only applies to --backend anthropic")

    config =load_config(args.config) if Path(args.config).exists() else {}
    tasks = {t["id"]: t for t in json.load(open(DATA_DIR / "tasks.json"))}
    keys = json.load(open(DATA_DIR / "answer_keys.json"))
    splits = json.load(open(DATA_DIR / "split_tasks.json"))
    if args.split not in splits:
        ap.error(f"unknown split '{args.split}'; available: {sorted(splits)}")
    split_ids = splits[args.split]
    if args.task_ids_file:  # k=2 top-up: restrict to a subset (e.g. tasks that failed pass 1)
        wanted = [ln.strip() for ln in open(args.task_ids_file) if ln.strip()]
        split_ids = [t for t in wanted if t in tasks]
        missing = [t for t in wanted if t not in tasks]
        if missing:
            print(f"warning: {len(missing)} task-ids from {args.task_ids_file} not in tasks.json (skipped)")
        if not split_ids:
            ap.error(f"no valid task ids in {args.task_ids_file}")
        print(f"task-ids-file: {len(split_ids)} tasks selected (subset of --split {args.split})")
    if args.n_tasks and args.stratify:
        by_tpl = {}
        for tid in split_ids:
            by_tpl.setdefault(keys[tid]["template"], []).append(tid)
        for tpl in by_tpl:
            by_tpl[tpl].sort()
        picked, i = [], 0
        while len(picked) < args.n_tasks and any(by_tpl.values()):
            for tpl in sorted(by_tpl):
                if by_tpl[tpl] and len(picked) < args.n_tasks:
                    picked.append(by_tpl[tpl].pop(0))
            i += 1
        split_ids = picked
    elif args.n_tasks:
        split_ids = split_ids[:args.n_tasks]

    teacher_label = args.teacher_name or ("oracle" if args.dry_run else
                                          "oracle_broken" if args.dry_run_broken else
                                          resolve_teacher(config, args.api_base, args.model)["model"])
    out_path = Path(args.output or (Path("data/generated/sdg") /
                                    f"db_traces_{args.split}_{teacher_label.replace('/', '_')}.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # resume: skip (task_id, sample_idx) already written
    done = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["task_id"], r["sample_idx"]))
                except Exception:
                    pass

    # --order round: sample 0 for ALL tasks, then sample 1, ... -> an abort in round r+1 leaves r complete rounds
    # over every task (clean pass^r); task order (default) keeps the historical schedule byte-identical.
    pairs = ([(tid, s) for s in range(args.k) for tid in split_ids] if args.order == "round"
             else [(tid, s) for tid in split_ids for s in range(args.k)])
    todo = [p for p in pairs if p not in done]
    print(f"split={args.split} tasks={len(split_ids)} k={args.k} todo={len(todo)} "
          f"(resumed {len(done)}) teacher={teacher_label}" + (" order=round" if args.order == "round" else ""))

    teacher_cfg = None if (args.dry_run or args.dry_run_broken) else \
        resolve_teacher(config, args.api_base, args.model)
    if teacher_cfg and args.backend == "anthropic":
        # own config: resolve_teacher's Qwen sampling defaults must not leak into an API record
        teacher_cfg = {"backend": "anthropic", "model": args.model, "api_base": args.api_base,
                       "max_tokens": args.max_tokens_per_turn or teacher_cfg["max_tokens"], "effort": args.effort,
                       "native_tools": True, "api_max_retries": args.api_max_retries}
    elif teacher_cfg:
        if args.max_tokens_per_turn:
            teacher_cfg["max_tokens"] = args.max_tokens_per_turn
        if args.temperature is not None:
            teacher_cfg["temperature"] = args.temperature
        for flag in ("top_p", "top_k", "min_p", "presence_penalty"):
            v = getattr(args, flag)
            if v is not None:
                teacher_cfg[flag] = v
        if args.enable_thinking is not None:
            teacher_cfg["enable_thinking"] = args.enable_thinking
        if args.legacy_stop:
            teacher_cfg["legacy_stop"] = True
        teacher_cfg["omit_thinking_kwarg"] = args.omit_thinking_kwarg
        if extra_kwargs:
            teacher_cfg["chat_template_kwargs"] = extra_kwargs
        if args.native_tools:
            teacher_cfg["native_tools"] = True
    lock = threading.Lock()
    stats = {"n": 0, "verified": 0, "replan": 0, "turns": 0.0,
             "branched": 0, "emergent_rec": 0, "scripted_rec": 0,
             "truncated": 0, "degen": 0, "roleplay_tail": 0}
    tmpl_n, tmpl_verified = Counter(), Counter()  # per-template yield (eval breakdown)
    out_f = open(out_path, "a")

    def make_call_for(task, key):
        """Factory: make_call(temp_bump=0.0) -> a teacher_call closure (oracle ignores temp)."""
        if args.dry_run or args.dry_run_broken:
            return lambda temp_bump=0.0: make_oracle(task, key, broken=args.dry_run_broken)
        def factory(temp_bump=0.0):
            if teacher_cfg.get("backend") == "anthropic":   # no temperature on the API -> nothing to bump
                return make_anthropic_call(teacher_cfg, timeout=args.request_timeout_s)
            cfg = teacher_cfg if not temp_bump else {**teacher_cfg,
                                                     "temperature": teacher_cfg["temperature"] + temp_bump}
            return make_teacher_call(cfg, timeout=args.request_timeout_s)
        return factory

    stop, fatal = threading.Event(), []   # FatalApiError: stop scheduling, write nothing for the hit rollouts
    api_recs = []                         # API backend: per-record usage for the MLflow cost metrics
    written = set()                       # (task_id, sample_idx) written by THIS process (complete-rounds report)
    # API error brake: error rollouts are HELD until a non-error rollout follows (then written — isolated errors stay
    # in the data as before). API_ERROR_STREAK errors in a row = systematic failure (empty balance with an unknown
    # wording, 429 storm after all SDK retries, network down): drop the held rows, stop, exit 3 -> RESUME continues.
    brake = teacher_cfg is not None and teacher_cfg.get("backend") == "anthropic"
    pending_err = []

    def emit(rec, res):
        """Write one record + update the counters (caller holds the lock)."""
        sc, fault = rec["score"], rec["fault"]
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()
        written.add((rec["task_id"], rec["sample_idx"]))
        if "api_calls" in rec:
            api_recs.append({"api_calls": rec["api_calls"], "gen_params": rec["gen_params"]})
        stats["n"] += 1
        # wave-3 accept semantics: a verified-but-truncated/degenerate trace is NOT a yield
        verified = accepted(res)
        stats["verified"] += verified
        stats["truncated"] += rec["truncated"]
        stats["degen"] += is_degenerate(res)
        stats["roleplay_tail"] += rec["roleplay_tail_chars"] > 0
        stats["replan"] += sc["replan_occurred"]
        stats["turns"] += sc["turns_used"]
        stats["branched"] += res.get("n_branch_attempts", 0)
        tmpl_n[rec["template"]] += 1
        tmpl_verified[rec["template"]] += verified
        if verified and sc.get("self_recovery"):  # B0: split emergent vs scripted self-recovery
            if fault in ("runtime", "state+runtime"):
                stats["scripted_rec"] += 1
            else:
                stats["emergent_rec"] += 1
        if stats["n"] % 10 == 0 or stats["n"] == len(todo):
            print(f"  {stats['n']}/{len(todo)}  verified-yield={stats['verified'] / stats['n']:.0%}")

    def work(item):
        if stop.is_set():
            return
        tid, s = item
        task, key = tasks[tid], keys[tid]
        try:
            res = solve_task(task, key, make_call_for(task, key), args, teacher_cfg)
        except FatalApiError as e:  # account-level: every further request fails alike -> stop the run cleanly
            with lock:
                if not fatal:
                    fatal.append(str(e))
                    print(f"  ABBRUCH (API, Account-Ebene): {e}")
                pending_err.clear()   # held error rows of the same outage are never written
            stop.set()
            return
        except Exception as e:  # per-request isolation (trace_capture lesson): never kill the pool
            res = {"messages": [], "finish_reason": f"error:{type(e).__name__}: {str(e)[:200]}",
                   "wall_s": 0.0, "n_branch_attempts": 0, "recovery_mode": "failed",
                   "score": {"score": 0.0, "task_solved": 0.0, "turns_used": 0, "n_tool_calls": 0,
                             "n_tool_errors": 0, "tool_calls_valid": 0.0, "n_plan_turns": 0,
                             "replan_occurred": 0.0, "self_recovery": 0.0, "components": {},
                             "error": "rollout_exception"}}
        sc, fault = res["score"], key.get("fault", "none")
        rec = {"task_id": tid, "sample_idx": s, "split": args.split, "teacher": teacher_label,
               "template": key["template"], "injected": key["injected"], "fault": fault,
               "expected_calls": key.get("expected_calls", 0),  # wave 3: for the >=3x filter
               "n_branch_attempts": res.get("n_branch_attempts", 0),
               "recovery_mode": res.get("recovery_mode", "direct"),
               "truncated": bool(res.get("truncated")), "degen": res.get("degen") or {},
               "roleplay_tail_chars": res.get("roleplay_tail_chars", 0),
               "gen_params": None if teacher_cfg is None else
               api_gen_params(teacher_cfg) if teacher_cfg.get("backend") == "anthropic" else
               {**{k: teacher_cfg.get(k) for k in ("temperature", "top_p", "top_k", "min_p",
                                                   "presence_penalty", "enable_thinking",
                                                   "legacy_stop", "max_tokens")},
                # only when set -> records of runs without extra switches stay byte-identical
                **({"chat_template_kwargs": teacher_cfg["chat_template_kwargs"]}
                   if teacher_cfg.get("chat_template_kwargs") else {}),
                **({"tool_mode": "native"} if teacher_cfg.get("native_tools") else {})},
               "score": sc, "finish_reason": res["finish_reason"],
               "wall_s": res["wall_s"],
               **({"api_calls": res["api_calls"]} if "api_calls" in res else {}),
               "messages": res["messages"]}
        with lock:
            if brake and str(rec["finish_reason"]).startswith(("teacher_error", "error")):
                if stop.is_set():          # the run is aborting: an error row of the outage is never written
                    return
                pending_err.append((rec, res))
                if len(pending_err) >= API_ERROR_STREAK:
                    if not fatal:
                        fatal.append(f"{len(pending_err)} Rollouts in Folge mit API-Fehler, zuletzt: "
                                     f"{str(rec['finish_reason'])[:160]}")
                        print(f"  ABBRUCH (Fehlerserie): {fatal[0]}")
                    pending_err.clear()
                    stop.set()
                return
            for held in pending_err:       # a non-error rollout followed -> those were isolated errors: keep them
                emit(*held)
            pending_err.clear()
            emit(rec, res)

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(work, todo))
        if not fatal:
            for held in pending_err:       # trailing isolated errors of a run that ended normally
                emit(*held)
    finally:
        out_f.close()
    if fatal:
        have = done | written
        rounds = next((r for r in range(args.k) if any((t, r) not in have for t in split_ids)), args.k)
        print(f"\nABBRUCH -> {out_path}: {fatal[0]}\n  {stats['n']} Rollouts geschrieben, {len(todo) - stats['n']} offen "
              f"(keine Fehlerzeilen geschrieben -> RESUME=1 setzt nach der Ursache fort)\n"
              f"  vollstaendige Runden: {rounds} von {args.k} (alle Tasks mit sample_idx < {rounds})"
              + (f" -> eval_report.py --max-sample {rounds}" if 0 < rounds < args.k else ""))
        sys.exit(3)

    n = max(1, stats["n"])
    v = max(1, stats["verified"])
    print(f"\nDONE -> {out_path}")
    print(f"  rollouts          : {stats['n']}")
    print(f"  verified-yield    : {stats['verified'] / n:.1%}")
    print(f"  replan-rate       : {stats['replan'] / n:.1%}")
    print(f"  avg turns         : {stats['turns'] / n:.1f}")
    if args.branch_on_fail:
        print(f"  branch-attempts   : {stats['branched']} total")
    print(f"  wave-3 gates      : truncated {stats['truncated']}, degen {stats['degen']}, "
          f"roleplay-tails {stats['roleplay_tail']}")
    # B0 self-recovery detector (share of VERIFIED traces that recovered from >=1 tool error)
    print(f"  self-recovery     : emergent {stats['emergent_rec']}/{stats['verified']} "
          f"({stats['emergent_rec'] / v:.1%}), scripted {stats['scripted_rec']}/{stats['verified']} "
          f"({stats['scripted_rec'] / v:.1%})")

    # MLflow summary tracking (opt-in --mlflow; soft pattern — never
    # breaks the run). File-based logs/JSONL stay the source of truth; this is the cross-run dashboard.
    if args.mlflow:
        if mlflow is None:
            print("  [mlflow] --mlflow set but mlflow not importable — skipping (install mlflow-skinny)")
        else:
            try:
                # our store is a plain file dir (shared with GRPO + the UI service); mlflow>=3.14 gates
                # the file backend behind this opt-out flag — set it rather than migrate to sqlite.
                os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
                uri = args.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") \
                    or Path("mlruns").resolve().as_uri()
                mlflow.set_tracking_uri(uri)
                mlflow.set_experiment(args.mlflow_experiment)
                mlflow.start_run(run_name=args.mlflow_run_name or teacher_label)
                mlflow.log_params({"split": args.split, "teacher": teacher_label, "model": args.model,
                                   "k": args.k, "n_tasks": len(split_ids),
                                   "branch_on_fail": args.branch_on_fail, "max_turns": args.max_turns,
                                   **({"tool_mode": "native"} if args.native_tools else {}),
                                   **({"backend": "anthropic", "effort": args.effort}
                                      if args.backend == "anthropic" else {})})
                if api_recs:   # API cost/token metrics (same numbers as evaluation/api_usage.py prints)
                    from evaluation.api_usage import mlflow_metrics, summarize
                    mlflow.log_metrics(mlflow_metrics(summarize(api_recs)))
                mlflow.log_metrics({
                    "verified_yield": stats["verified"] / n, "replan_rate": stats["replan"] / n,
                    "avg_turns": stats["turns"] / n, "n_rollouts": stats["n"],
                    "emergent_recovery_rate": stats["emergent_rec"] / v,
                    "scripted_recovery_rate": stats["scripted_rec"] / v,
                    "truncated_rate": stats["truncated"] / n, "degen_rate": stats["degen"] / n,
                    "roleplay_tail_rate": stats["roleplay_tail"] / n})
                for tpl in sorted(tmpl_n):  # per-template yield = the eval breakdown (before/after)
                    mlflow.log_metric(f"yield_{tpl}", tmpl_verified[tpl] / max(1, tmpl_n[tpl]))
                print(f"  [mlflow] logged run '{args.mlflow_run_name or teacher_label}' "
                      f"-> {args.mlflow_experiment} @ {uri}")
            except Exception as e:  # noqa: BLE001 — tracking must never fail the run
                print(f"  [mlflow] logging failed ({type(e).__name__}: {e}) — continuing")
            finally:
                try:
                    mlflow.end_run()
                except Exception:
                    pass


if __name__ == "__main__":
    main()

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

Runs in the tau2 venv (needs the domain + verifier):
    PYTHONPATH=. <tau2-venv>/bin/python sdg_pipeline/db_bahn/rollout.py \
        --config config/pipeline_config.local.yaml --split bakeoff_dev --k 3

CPU smoke without any GPU/teacher:  --dry-run  (scripted oracle exercises the FULL loop incl. parser)
and --dry-run-broken (hallucinating oracle; must score 0.0).
"""

import argparse
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from sdg_pipeline.db_bahn.tau2_domain import get_environment
from sdg_pipeline.db_bahn.tau2_domain.environment import DATA_DIR
from evaluation.trajectory_reward import score_trajectory

TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Qwen format drift: models also emit the XML-style call syntax -> parse both.
#   <tool_call><function=name><parameter=key>value</parameter>...</function></tool_call>
FUNC_XML_RE = re.compile(r"<function=([\w-]+)>(.*?)</function>", re.DOTALL)
PARAM_XML_RE = re.compile(r"<parameter=([\w-]+)>\s*(.*?)\s*</parameter>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
ZUG_RE = re.compile(r"\b(?:ICE|IC|EC|ECE|RJ|EN)\s\d+\b")
MAX_TOOL_CONTENT = 4000

SYSTEM_TEMPLATE = """{policy}

# Tools

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_block}
</tools>

## Arbeitsweise (wichtig)

- Beginne jeden Schritt mit einem KURZEN Plan in <plan>…</plan> (1–3 Sätze, keine Selbstzweifel,
  kein „Warte“/„Eigentlich“, keine Wiederholungen).
- Rufe Werkzeuge in genau diesem Format auf (ein Block pro Aufruf, Argumente als JSON):
<tool_call>
{{"name": "werkzeug_name", "arguments": {{"argument": "wert"}}}}
</tool_call>
- Nach jedem Werkzeug-Ergebnis: kurz prüfen, ob der Plan noch passt; bei Überraschungen umplanen.
- Wenn die Aufgabe gelöst ist: KEIN Tool-Aufruf mehr, sondern eine kurze deutsche Schlussantwort mit den
  belegten Fakten (nur Werte, die ein Werkzeug geliefert hat)."""


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_teacher(config: dict, api_base=None, model=None) -> dict:
    t = (config.get("teacher") or {}).get("vllm_local", {})
    traj = config.get("trajectory") or {}
    return {"api_base": api_base or t.get("api_base", "http://localhost:8000/v1"),
            "model": model or t.get("model", ""),
            "api_key": t.get("api_key", "token-local"),
            "max_tokens": int(traj.get("max_tokens_per_turn", t.get("max_tokens", 2048))),
            "temperature": float(traj.get("temperature", 0.7)),
            "enable_thinking": bool(traj.get("enable_thinking", False))}


def make_teacher_call(cfg: dict, timeout: float = 600.0):
    import httpx
    client = httpx.Client(timeout=timeout)
    url = cfg["api_base"].rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {cfg['api_key']}"}

    def call(messages: list[dict], tools: list[dict]) -> tuple[str, list[dict], str]:
        # PROMPT-AND-PARSE: tool schemas live in the system prompt; the `tools` param is NOT sent
        # (vLLM 400s on tools without --enable-auto-tool-choice — exactly the server dependency we avoid).
        payload = {"model": cfg["model"], "messages": messages,
                   "temperature": cfg["temperature"], "max_tokens": cfg["max_tokens"],
                   # stop right after a tool-call block: prevents models role-playing the tool
                   # RESPONSE in the same turn (observed with Qwen3-Next hallucinating results)
                   "stop": ["</tool_call>", "</tools>", "</TOOLCALL>"],
                   "include_stop_str_in_output": True}
        if not cfg.get("omit_thinking_kwarg"):  # mistral-tokenizer models reject template kwargs
            payload["chat_template_kwargs"] = {"enable_thinking": cfg["enable_thinking"]}
        r = client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:  # include the body — vLLM puts the actual reason there
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        msg = r.json()["choices"][0]["message"]
        return msg.get("content") or "", msg.get("tool_calls") or [], \
            r.json()["choices"][0].get("finish_reason", "")

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


def parse_tool_calls(content: str, native: list[dict]) -> tuple[str, list[dict]]:
    """Native server-parsed tool_calls win; else parse <tool_call> blocks from the (think-stripped) text."""
    content = strip_think(content)
    if native:
        calls = [{"id": tc.get("id") or f"n{i:08d}", "type": "function",
                  "function": {"name": tc["function"]["name"],
                               "arguments": tc["function"].get("arguments") or "{}"}}
                 for i, tc in enumerate(native, 1)]
        return content, calls
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
    return clean, calls


def openai_tools(env) -> list[dict]:
    return [{"type": "function", "function": t.openai_schema["function"]} for t in env.get_tools()]


# --- oracle teachers for GPU-free smoke ------------------------------------------------------
def make_oracle(task: dict, key: dict, broken: bool = False):
    """Scripted teacher: emits <tool_call> TEXT (exercises the parser), then a grounded final answer."""
    crit = task.get("evaluation_criteria") or {}
    m = ZUG_RE.search(task.get("ticket") or "")
    zugnummer = m.group(0) if m else ""
    plan = []
    if key["kind"] == "action":
        for name in key["expected_tools"]:
            ref = next((a for a in (crit.get("actions") or []) if a["name"] == name), None)
            args = ref["arguments"] if ref else (
                {"kennung": zugnummer} if name == "wartung_status" else {"zugnummer": zugnummer})
            plan.append((name, args))
    else:
        for name in key["expected_tools"]:
            plan.append((name, {"kennung": zugnummer} if name == "wartung_status" else {"zugnummer": zugnummer}))
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
def run_rollout(task: dict, key: dict, teacher_call, max_turns: int, rollout_timeout_s: float) -> dict:
    from tau2.data_model.tasks import EnvFunctionCall

    env = get_environment(solo_mode=True)  # fresh env per rollout (thread-safe by isolation)
    inits = (task.get("initial_state") or {}).get("initialization_actions") or []
    if inits:
        env.run_env_function_calls([EnvFunctionCall.model_validate(a) for a in inits])
    tools = openai_tools(env)
    tools_block = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
    sys_prompt = SYSTEM_TEMPLATE.format(policy=env.get_policy(), tools_block=tools_block)
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": task["ticket"]}]
    t0 = time.time()
    finish = "max_turns"
    for _ in range(max_turns):
        if time.time() - t0 > rollout_timeout_s:
            finish = "timeout"
            break
        content, native, fr = teacher_call(messages, tools)
        clean, calls = parse_tool_calls(content, native)
        assistant = {"role": "assistant", "content": clean}
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)
        if not calls:
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
    return {"messages": messages, "score": score, "finish_reason": finish,
            "wall_s": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser(description="Multi-turn teacher rollouts against the db_bahn sandbox")
    ap.add_argument("--config", default="config/pipeline_config.yaml")
    ap.add_argument("--split", default="bakeoff_dev", choices=["bakeoff_dev", "heldout_eval", "sft_train"])
    ap.add_argument("--k", type=int, default=1, help="rollouts per task")
    ap.add_argument("--n-tasks", type=int, default=None, help="cap number of tasks (debug)")
    ap.add_argument("--stratify", action="store_true",
                    help="pick --n-tasks round-robin over templates (balanced bake-off subset)")
    ap.add_argument("--max-tokens-per-turn", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--omit-thinking-kwarg", action="store_true",
                    help="don't send chat_template_kwargs (mistral tokenizer mode)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=10)
    ap.add_argument("--max-regen", type=int, default=1, help="re-sample a failed rollout up to N times")
    ap.add_argument("--rollout-timeout-s", type=float, default=300.0)
    ap.add_argument("--teacher-name", default=None, help="label written into records (bake-off table)")
    ap.add_argument("--api-base", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--dry-run", action="store_true", help="scripted oracle teacher (CPU smoke)")
    ap.add_argument("--dry-run-broken", action="store_true", help="hallucinating oracle (must score 0)")
    args = ap.parse_args()

    config = load_config(args.config) if Path(args.config).exists() else {}
    tasks = {t["id"]: t for t in json.load(open(DATA_DIR / "tasks.json"))}
    keys = json.load(open(DATA_DIR / "answer_keys.json"))
    split_ids = json.load(open(DATA_DIR / "split_tasks.json"))[args.split]
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
    out_path = Path(args.output or (Path("data/generated") /
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

    todo = [(tid, s) for tid in split_ids for s in range(args.k) if (tid, s) not in done]
    print(f"split={args.split} tasks={len(split_ids)} k={args.k} todo={len(todo)} "
          f"(resumed {len(done)}) teacher={teacher_label}")

    teacher_cfg = None if (args.dry_run or args.dry_run_broken) else \
        resolve_teacher(config, args.api_base, args.model)
    if teacher_cfg:
        if args.max_tokens_per_turn:
            teacher_cfg["max_tokens"] = args.max_tokens_per_turn
        if args.temperature is not None:
            teacher_cfg["temperature"] = args.temperature
        teacher_cfg["omit_thinking_kwarg"] = args.omit_thinking_kwarg
    lock = threading.Lock()
    stats = {"n": 0, "verified": 0, "replan": 0, "turns": 0.0}
    out_f = open(out_path, "a")

    def work(item):
        tid, s = item
        task, key = tasks[tid], keys[tid]
        if args.dry_run or args.dry_run_broken:
            call = make_oracle(task, key, broken=args.dry_run_broken)
        else:
            call = make_teacher_call(teacher_cfg)
        res = None
        try:
            for _ in range(args.max_regen + 1):
                res = run_rollout(task, key, call, args.max_turns, args.rollout_timeout_s)
                if res["score"]["score"] == 1.0:
                    break
                if args.dry_run or args.dry_run_broken:
                    break  # oracles are deterministic; regen is pointless
                call = make_teacher_call(teacher_cfg)  # fresh sample
        except Exception as e:  # per-request isolation (trace_capture lesson): never kill the pool
            res = {"messages": [], "finish_reason": f"error:{type(e).__name__}: {str(e)[:200]}",
                   "wall_s": 0.0,
                   "score": {"score": 0.0, "task_solved": 0.0, "turns_used": 0, "n_tool_calls": 0,
                             "n_tool_errors": 0, "tool_calls_valid": 0.0, "n_plan_turns": 0,
                             "replan_occurred": 0.0, "components": {}, "error": "rollout_exception"}}
        rec = {"task_id": tid, "sample_idx": s, "split": args.split, "teacher": teacher_label,
               "template": key["template"], "injected": key["injected"],
               "score": res["score"], "finish_reason": res["finish_reason"],
               "wall_s": res["wall_s"], "messages": res["messages"]}
        with lock:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            stats["n"] += 1
            stats["verified"] += res["score"]["score"] == 1.0
            stats["replan"] += res["score"]["replan_occurred"]
            stats["turns"] += res["score"]["turns_used"]
            if stats["n"] % 10 == 0 or stats["n"] == len(todo):
                print(f"  {stats['n']}/{len(todo)}  verified-yield={stats['verified'] / stats['n']:.0%}")

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            list(pool.map(work, todo))
    finally:
        out_f.close()

    n = max(1, stats["n"])
    print(f"\nDONE -> {out_path}")
    print(f"  rollouts        : {stats['n']}")
    print(f"  verified-yield  : {stats['verified'] / n:.1%}")
    print(f"  replan-rate     : {stats['replan'] / n:.1%}")
    print(f"  avg turns       : {stats['turns'] / n:.1f}")


if __name__ == "__main__":
    main()

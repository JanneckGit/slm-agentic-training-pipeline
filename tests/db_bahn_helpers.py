"""Trace-building helpers shared by the db_bahn tests (moved out of evaluation/trajectory_reward.py)."""
import json


def mk_call(i, _name, **args):
    # positional is _name so tool arguments literally named `name` (mitarbeiter_suchen, wave 3)
    # don't collide with the parameter
    return {"id": f"call_{i}", "type": "function",
            "function": {"name": _name, "arguments": json.dumps(args, ensure_ascii=False)}}


def env_for(dom, task):
    """Fresh solo environment with the task's initialization_actions applied (fault injection)."""
    env = dom.get_environment(solo_mode=True)
    inits = (task.get("initial_state") or {}).get("initialization_actions") or []
    if inits:
        from tau2.data_model.tasks import EnvFunctionCall
        env.run_env_function_calls([EnvFunctionCall.model_validate(a) for a in inits])
    return env


def oracle_trace(dom, task, key):
    """Replay oracle_calls on a fresh initialized env -> messages sans final answer."""
    env = env_for(dom, task)
    ms = [{"role": "system", "content": "…"}, {"role": "user", "content": task["ticket"]}]
    for i, c in enumerate(key["oracle_calls"], 1):
        ms.append({"role": "assistant", "content": "", "tool_calls": [mk_call(i, c["name"], **c["arguments"])]})
        try:
            body = json.dumps(env.use_tool(c["name"], **c["arguments"]), ensure_ascii=False, default=str)
        except Exception as e:  # noqa: BLE001 — the error observation IS the trace content
            body = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
        ms.append({"role": "tool", "tool_call_id": f"call_{i}", "content": body})
    return ms


def copy(msgs):
    """Deep copy via JSON — fixtures hand out originals, tests mutate copies."""
    return json.loads(json.dumps(msgs))

"""verl BaseTool adapter for the 12 db_bahn tools — the Stage-2 rollout seam.

verl's ToolAgentLoop drives one instance per tool name (registered via grpo_tool_config.yaml);
this class is that instance for whichever `tool_name` its config carries. It owns NO domain logic:
every call is delegated to the SAME tau2 environment the SDG rollouts and the verifier use
(sdg_pipeline/db_bahn/tau2_domain), so rollout / eval / RL share one environment definition.

Two verl lifecycle facts shape this file (verl 0.8.0, experimental/agent_loop/tool_agent_loop.py):
  * create()/release() run PER TOOL CALL, not per episode -> per-episode state must NOT live in
    the instance_id dict. The env is cached on `agent_data`, which is the one object that lives
    for the whole trajectory.
  * agent_data.extra_fields is ray-pickled to the reward worker -> the env (unpicklable, and
    irrelevant there: the verifier replays on a fresh env) must never be put there.
"""

import json
import threading

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

MAX_TOOL_CONTENT = 4000  # mirrors sdg_pipeline/db_bahn/rollout.py:53 — same obs truncation as SDG/eval

_ENV_ATTR = "_db_bahn_env"
_schema_lock = threading.Lock()
_schemas: dict | None = None


def _flatten_optional(schema: dict) -> dict:
    """tau2's Optional[str] params render as `anyOf: [{type: str}, {type: null}]` (pydantic default),
    but verl's OpenAIFunctionPropertySchema requires a flat `type` (schemas.py:25) -> take the first
    non-null branch. Only the three search tools' optional filters are affected; SDG never hit this
    because it dumps the schemas into the prompt as text instead of validating them."""
    for prop in ((schema.get("function", {}).get("parameters", {}) or {}).get("properties", {}) or {}).values():
        branches = prop.pop("anyOf", None)
        if branches:
            first = next((b for b in branches if b.get("type") != "null"), branches[0])
            prop.update({k: v for k, v in first.items() if k not in prop})
    return schema


def _tool_schemas() -> dict:
    """{tool_name: openai_schema} from one throwaway env (schemas are static)."""
    global _schemas
    with _schema_lock:
        if _schemas is None:
            import copy

            from sdg_pipeline.db_bahn.tau2_domain import get_environment
            _schemas = {t.name: _flatten_optional(copy.deepcopy(t.openai_schema))
                        for t in get_environment(solo_mode=True).get_tools()}
    return _schemas


def _episode_env(agent_data, create_kwargs: dict):
    """The trajectory's env: built on first tool call, then reused for every later call."""
    env = getattr(agent_data, _ENV_ATTR, None)
    if env is None:
        from sdg_pipeline.db_bahn.tau2_domain import get_environment
        env = get_environment(solo_mode=True)
        init_actions = json.loads(create_kwargs.get("initialization_actions_json") or "[]")
        if init_actions:
            from tau2.data_model.tasks import EnvFunctionCall
            env.run_env_function_calls([EnvFunctionCall.model_validate(a) for a in init_actions])
        setattr(agent_data, _ENV_ATTR, env)
    return env


class DbBahnTool(BaseTool):
    """One db_bahn tool (config: {tool_name: <name>}); schema comes from the tau2 env itself."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None):
        self.name = (config or {}).get("tool_name")
        if self.name not in _tool_schemas():
            raise ValueError(f"unknown db_bahn tool {self.name!r}; have {sorted(_tool_schemas())}")
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema.model_validate(_tool_schemas()[self.name])
        super().__init__(config or {}, tool_schema)
        self._instances: dict[str, dict] = {}

    async def create(self, instance_id=None, create_kwargs=None, **kwargs):
        import uuid
        instance_id = instance_id or str(uuid.uuid4())
        self._instances[instance_id] = create_kwargs or {}   # per-call; the env lives on agent_data
        return instance_id, ToolResponse()

    async def execute(self, instance_id, parameters=None, agent_data=None, **kwargs):
        try:
            env = _episode_env(agent_data, self._instances.get(instance_id, {}))
            obs = env.use_tool(self.name, **(parameters or {}))
            text = json.dumps(obs, ensure_ascii=False, default=str)
        except Exception as e:
            # tau2 rejections (role gate, duplicate, terminal status) are legitimate observations the
            # agent must replan around — same shape as rollout.py:294, so the verifier sees what it expects.
            text = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)
        return ToolResponse(text=text[:MAX_TOOL_CONTENT]), 0.0, {}

    async def release(self, instance_id, **kwargs):
        self._instances.pop(instance_id, None)

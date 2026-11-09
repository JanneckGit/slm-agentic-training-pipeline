#!/usr/bin/env python3
"""Shim vor der tau2-CLI (Muster: evaluation/benchmarks/bfcl/run_bfcl.py).

    .venv-tau2bench/bin/python run_tau2.py --manifest <run_manifest.json> -- run <flags…>

Warum ein Shim noetig ist (Reihenfolge ist tragend):
1. tau2 bindet seine Judge-Modelle als Modul-KONSTANTEN zur Import-Zeit
   (`from tau2.config import DEFAULT_LLM_NL_ASSERTIONS` in evaluator_nl_assertions.py),
   Default = gpt-4.1. retail (112/114 Tasks mit NL_ASSERTION) braucht diesen Judge →
   wir patchen tau2.config UND die bereits importierte Modul-Bindung auf den lokalen
   Judge aus dem Manifest, BEVOR die CLI dispatcht.
2. DEFAULT_LLM_ENV_INTERFACE + DEFAULT_LLM_EVAL_USER_SIMULATOR werden auf einen
   Sentinel gebogen: ein unerwarteter Judge-Aufruf soll LAUT scheitern (litellm kennt
   den Provider nicht), statt still OpenAI zu versuchen.
3. Env-Wachen: TAU2_DATA_DIR muss stehen (tau2.utils liest es beim Import),
   OPENAI_API_KEY & Co. duerfen NICHT stehen (vergessene gpt-4.1-Defaults sollen
   crashen, nie Geld kosten).

Die Patch-Spec kommt IMMER aus dem Manifest (eine Quelle der Wahrheit, vom Preflight
gegated) — nie aus eigenen Flags.

Optional (Manifest agent.strict_tools, Env-Knopf TAU2_STRICT_TOOLS=1 in ops/eval_tau2.sh):
die Tool-Schemas des AGENTEN tragen "strict": true. vLLM >= 0.30 legt bei tool_choice="auto"
nur dann eine Grammatik auf die Tool-Calls (structural_tag_registry: kein strict-Tool -> keine
Grammatik); tau2 selbst setzt strict nie. Gepatcht wird nur die Modul-Bindung
tau2.agent.llm_agent.generate (LLMAgent + LLMGTAgent); der User-Sim bindet generate in
tau2.user.user_simulator selbst und bleibt unberuehrt. litellm-Falle: der Provider
`hosted_vllm/` loescht strict still (_remove_strict_from_schema) — der Agent laeuft ueber
`openai/<alias>`, dort bleibt es stehen (tests/test_tau2_strict.py prueft beides).
"""

import argparse
import json
import os
import sys
from pathlib import Path

FORBIDDEN_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")
SENTINEL = "disabled-local-sentinel/refuse"


def apply_patches(manifest: dict) -> None:
    import tau2.config as cfg

    judge = manifest.get("nl_judge") or {}
    if judge.get("model_litellm"):
        model = judge["model_litellm"]
        args = dict(judge.get("llm_args") or {})
        cfg.DEFAULT_LLM_NL_ASSERTIONS = model
        cfg.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args
        # Import-Zeit-Bindung im Evaluator nachziehen (`from … import KONSTANTE`)
        import tau2.evaluator.evaluator_nl_assertions as nle
        nle.DEFAULT_LLM_NL_ASSERTIONS = model
        nle.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args

    # Sentinels: alles, was wir NICHT nutzen, soll laut scheitern statt gpt-4.1 zu rufen.
    cfg.DEFAULT_LLM_ENV_INTERFACE = SENTINEL
    cfg.DEFAULT_LLM_ENV_INTERFACE_ARGS = {}
    cfg.DEFAULT_LLM_EVAL_USER_SIMULATOR = SENTINEL
    try:  # Modul-Bindung, falls vorhanden
        import tau2.environment.utils.interface_agent as ia
        for name in ("DEFAULT_LLM_ENV_INTERFACE", "DEFAULT_LLM_ENV_INTERFACE_ARGS"):
            if hasattr(ia, name):
                setattr(ia, name, SENTINEL if "ARGS" not in name else {})
    except ImportError:
        pass


class StrictTool:
    """Read-only-Proxy um ein tau2-Tool: openai_schema traegt "strict": true, sonst alles vom Original."""

    def __init__(self, tool):
        self._tool = tool

    def __getattr__(self, name):
        return getattr(self._tool, name)

    @property
    def openai_schema(self) -> dict:
        s = self._tool.openai_schema
        return {**s, "function": {**s["function"], "strict": True}}


def apply_strict_tools(manifest: dict) -> bool:
    """Agent-Tools mit "strict": true schicken, wenn das Manifest es verlangt. True = gepatcht."""
    if not (manifest.get("agent") or {}).get("strict_tools"):
        return False
    import tau2.agent.llm_agent as la
    inner = la.generate
    if getattr(inner, "_strict_tools", False):  # idempotent
        return True

    def generate(*args, **kwargs):
        if len(args) > 2:  # generate(model, messages, tools, ...) positional
            args = (*args[:2], [StrictTool(t) for t in args[2]] if args[2] else args[2], *args[3:])
        elif kwargs.get("tools"):
            kwargs["tools"] = [StrictTool(t) for t in kwargs["tools"]]
        return inner(*args, **kwargs)

    generate._strict_tools = True
    la.generate = generate
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("tau2_args", nargs=argparse.REMAINDER)
    a = ap.parse_args()

    if not os.getenv("TAU2_DATA_DIR"):
        print("run_tau2: TAU2_DATA_DIR ist nicht gesetzt — tau2 wuerde ins "
              "site-packages-Verzeichnis zeigen. ops/eval_tau2.sh exportiert es.",
              file=sys.stderr)
        return 2
    present = [k for k in FORBIDDEN_KEYS if os.getenv(k)]
    if present:
        print(f"run_tau2: {present} im Env — verboten (vergessene gpt-4.1-Defaults "
              "muessen laut crashen, nie still Geld kosten). Unset und neu starten.",
              file=sys.stderr)
        return 2

    manifest = json.loads(a.manifest.read_text())
    apply_patches(manifest)
    if apply_strict_tools(manifest):
        print("run_tau2: strict_tools — Agent-Tool-Schemas tragen \"strict\": true", file=sys.stderr)

    rest = a.tau2_args[1:] if a.tau2_args[:1] == ["--"] else a.tau2_args
    from tau2.cli import main as cli_main
    sys.argv = ["tau2"] + rest
    try:
        cli_main()
    except SystemExit as e:
        return int(e.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

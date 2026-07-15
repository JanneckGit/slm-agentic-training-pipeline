"""
data_pipeline/common.py
=======================
Shared helpers for the agentic pipeline (2026-07-15 dedup — these existed as 3-6 copies across
data_pipeline/, sdg_pipeline/db_bahn/ and evaluation/). Stdlib-only at import time (yaml is
imported lazily) so every venv/container (tau2 venv, sdg, training) can import it via the
repo-root PYTHONPATH / editable install.
"""

import json
from pathlib import Path

# The one student model this pipeline trains/serves (dense — NOT the Qwen3.5 MM hybrid).
STUDENT_MODEL_DEFAULT = "Qwen/Qwen3-4B"

# The shared "# Tools" system-prompt segment (Qwen3 <tools> style). Consumers wrap it with their
# own preamble/policy/suffix; the segment itself must stay byte-identical across legs so the
# student sees ONE convention (db_bahn rollouts, ToolACE leg, AReaL leg).
TOOLS_BLOCK_TMPL = (
    "# Tools\n\nYou are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{tools_block}\n</tools>"
)


def load_config(path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def write_jsonl(rows, path) -> Path:
    """One JSON object per line (ensure_ascii=False everywhere in this repo)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def args_dict(raw) -> dict:
    """Tool-call arguments -> dict. Accepts dict (as-is) or JSON string; {} on parse failure."""
    if isinstance(raw, dict):
        return raw
    try:
        out = json.loads(raw or "{}")
    except Exception:
        return {}
    return out if isinstance(out, dict) else {}


def final_answer(messages) -> str:
    """Content of the last assistant turn that has content and NO tool_calls ('' if none)."""
    for m in reversed(messages):
        if m.get("role") == "assistant" and (m.get("content") or "").strip() and not m.get("tool_calls"):
            return m["content"]
    return ""

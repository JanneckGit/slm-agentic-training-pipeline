#!/usr/bin/env python3
"""Runtime registry entry for bfcl_eval — the fix for the 262,144-context trap.

bfcl-eval has no registry key for the hybrid Qwen3-4B (the 4B slot only carries the later
`Qwen3-4B-Instruct-2507`, a DIFFERENT model with a 262,144-token config). Borrowing that key made
BFCL believe our served model accepts 262k context, so its own
`max_tokens = min(4096, max_context_length - input - 2)` never clamped — and every long prompt
died at the vLLM boundary as a silent 0 % (HTTP 400, `input_token_count = 0`).

Instead of borrowing, this module INJECTS an entry into `MODEL_CONFIG_MAPPING` at runtime whose
`model_name` is the model we actually serve. That one string is three things at once in bfcl:
the `model=` field of every API request (== the required vLLM `--served-model-name`), the source
for the host-side AutoConfig/AutoTokenizer load (which yields the TRUE `max_position_embeddings`),
and the identity in the leaderboard CSV. Injection is in-place mutation of the mapping dict —
verified against bfcl-eval 2026.3.23: all consumer modules bind the same dict object, every read
is a lazy runtime lookup, and generation runs in-process (ThreadPoolExecutor, no re-import).
The installed package is never patched, so this survives `pip install --upgrade`.

Hard rules encoded here (each learned from a concrete failure mode):
  * The key must contain NO underscores — bfcl recovers keys from result-directory names via
    `name.replace("_", "/")`, which replaces EVERY underscore. 0 of 177 stock keys carry one.
  * The key must end in `-FC` — `base_handler.py` routes on the SUBSTRING "FC" in the registry
    name (in addition to `is_fc_model`), and the preflight's format_sensitivity check keys on it.
  * Mutate `MODEL_CONFIG_MAPPING` in place; never rebind the name — importers hold the object.
  * The same injection must be active for `generate`, `evaluate` AND the preflight: `evaluate`
    walks every result subdir and looks its key up in the mapping (KeyError otherwise).
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

# Cloned template for injected entries. Any Qwen3 FC entry would do (identical QwenFCHandler,
# is_fc_model=True, underscore_to_dot=False); 8B is the architecturally closest sibling.
BASE_KEY = "Qwen/Qwen3-8B-FC"

# Namespace prefix for injected keys. No stock key uses `local/` (verified against 2026.3.23),
# so collisions with upstream entries are impossible by construction.
KEY_PREFIX = "local/"


def derive_slug(model: str) -> str:
    """`$MODEL` (HF id or serving path) -> a filesystem- and registry-safe slug.

    HF id ("Qwen/Qwen3-4B")            -> last component:      "qwen3-4b"
    path ("/app/.../merged_x/ep3")     -> last TWO components:  "merged-x-ep3"
    (two components because checkpoint leaves are generic names like `ep3` or `selected`)
    """
    parts = [p for p in model.strip("/").split("/") if p]
    raw = "-".join(parts[-2:]) if model.startswith(("/", ".")) else parts[-1]
    slug = re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9.-]", "-", raw.lower())).strip("-")
    if not slug:
        raise ValueError(f"cannot derive a slug from model {model!r}")
    return slug


def derive_key(model: str) -> str:
    key = f"{KEY_PREFIX}{derive_slug(model)}-FC"
    # An underscore would round-trip result/<dir> -> key as "/" and break `evaluate`.
    assert "_" not in key, f"derived key {key!r} contains an underscore"
    return key


def injection_spec(model: str, model_name: str) -> dict:
    """The injection description that goes into run_manifest.json.

    `model` is the model exactly as the vLLM CONTAINER sees it (arg 1 of ops/eval_bfcl.sh);
    `model_name` is the HOST-resolved source (HF id, or the host path of a checkpoint) — the
    string that must equal `--served-model-name` on vLLM.
    """
    slug = derive_slug(model)
    return {
        "injected": True,
        "model_key": derive_key(model),
        "model_name": model_name,
        "display_name": f"{slug} (local FC)",
        "base_key": BASE_KEY,
    }


def inject(spec: dict) -> None:
    """Insert the entry into bfcl's registry — in place, idempotent, collision-guarded."""
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

    key, model_name = spec["model_key"], spec["model_name"]
    existing = MODEL_CONFIG_MAPPING.get(key)
    if existing is not None:
        if existing.model_name == model_name:
            return  # already injected (idempotent re-entry, e.g. generate then evaluate)
        raise RuntimeError(
            f"registry key {key!r} already exists with model_name={existing.model_name!r} "
            f"(wanted {model_name!r}) — key collision, refusing to overwrite")

    base = MODEL_CONFIG_MAPPING[spec.get("base_key") or BASE_KEY]
    MODEL_CONFIG_MAPPING[key] = dataclasses.replace(
        base,
        model_name=model_name,
        display_name=spec["display_name"],
        url=(f"https://huggingface.co/{model_name}"
             if not model_name.startswith(("/", ".")) else f"file://{Path(model_name)}"),
    )

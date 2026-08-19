#!/usr/bin/env python3
"""Shim around the bfcl CLI: apply the run's registry injection, then delegate.

Replaces direct `bfcl generate|evaluate` calls in ops/eval_bfcl.sh. The injection spec is NOT
computed here — the preflight computed and validated it once and wrote it into run_manifest.json;
this shim only re-applies it. One source of truth, and generate/evaluate are guaranteed to run
under exactly the entry the preflight gated.

    .venv-bfcl/bin/python evaluation/benchmarks/bfcl/run_bfcl.py \
        --manifest "$ROOT/run_manifest.json" -- generate --model <key> ...

Order of operations is load-bearing:
  1. BFCL_PROJECT_ROOT must be set BEFORE bfcl_eval is imported — its eval_config mkdirs
     result/ + score/ at import time (unset, they would land in site-packages).
  2. Inject BEFORE invoking the CLI (same dict object, lazy lookups -> timing is the only rule).
  3. `cli` is a Typer app in click standalone mode: it ALWAYS raises SystemExit, including on
     success — catch it and propagate the code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path, help="run_manifest.json of this run")
    ap.add_argument("bfcl_args", nargs=argparse.REMAINDER,
                    help="everything after `--` goes to the bfcl CLI verbatim")
    a = ap.parse_args()

    if not os.getenv("BFCL_PROJECT_ROOT"):
        print("run_bfcl: BFCL_PROJECT_ROOT is not set — bfcl_eval would mkdir result/ and score/ "
              "inside site-packages. Export it first (ops/eval_bfcl.sh does).", file=sys.stderr)
        return 2

    manifest = json.loads(a.manifest.read_text())
    registry = manifest.get("registry") or {}
    if registry.get("injected"):
        import registry_inject
        registry_inject.inject(registry)

    rest = a.bfcl_args[1:] if a.bfcl_args[:1] == ["--"] else a.bfcl_args
    from bfcl_eval.__main__ import cli
    try:
        cli(rest)
    except SystemExit as e:
        return int(e.code or 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())

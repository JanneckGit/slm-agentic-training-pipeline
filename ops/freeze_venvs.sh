#!/usr/bin/env bash
# ops/freeze_venvs.sh — freeze the three host venvs into requirements/*.lock.txt (pip freeze + header).
#
# The locks are what the README installs with `pip install --no-deps -r requirements/<x>.lock.txt`.
# --no-deps on purpose: a freeze is complete, nothing needs resolving — and bfcl-eval's numpy==1.26.4
# pin vs the deliberate numpy>=2 in .venv-bfcl would make pip's resolver abort
# (evaluation/benchmarks/bfcl/requirements.txt, NUMPY block).
# tau2 is installed from a local checkout in both tau2 venvs; pip freeze renders that as
# `tau2 @ file:///…` (unreproducible once the checkout is gone) -> rewritten to the pinned git commit.
# Run after a deliberate upgrade of a venv; never edit the locks by hand.
#
# Usage: bash ops/freeze_venvs.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=requirements; mkdir -p "$OUT"
TAU2_REPO=https://github.com/sierra-research/tau2-bench

freeze() {  # venv  lock  tau2-commit(full sha, "" = venv without tau2)  note
  local venv=$1 lock=$2 sha=$3 note=$4 py="$1/bin/python"
  [ -x "$py" ] || { echo "== $venv: missing, skipped" >&2; return 0; }
  {
    echo "# $lock — frozen $venv ($("$py" --version 2>&1), $(uname -m), $(date +%F))"
    echo "# $note"
    echo "# install:    python3.12 -m venv $venv && $venv/bin/pip install --no-deps -r $OUT/$lock"
    echo "# regenerate: bash ops/freeze_venvs.sh   (after a deliberate upgrade — never edit by hand)"
    "$py" -m pip freeze \
      | grep -vE '^(-e |slm[-_]agentic[-_]training[-_]pipeline)' \
      | sed -E "s#^tau2 @ file://.*#tau2 @ git+${TAU2_REPO}@${sha}#"
  } > "$OUT/$lock"
  echo "== $venv -> $OUT/$lock ($(grep -vc '^#' "$OUT/$lock") pins)"
}

freeze .venv-tau2      tau2.lock.txt      1901a301961cbbe3fd11f3e84a2a376530c759e3 \
  "db_bahn domain runtime + tests: tau2 1.0.0 @ 1901a30 (built from GitHub), pytest. Pure-python deps — portable across x86/aarch64."
freeze .venv-tau2bench tau2bench.lock.txt fc0055dc4e0a316c3f83133267fbd6faaa770992 \
  "tau2-bench OOD harness: tau2 1.0.1 (tag v1.0.1) + rank_bm25. The checkout data/raw/tau2-bench stays required as TAU2_DATA_DIR (README, OOD section)."
freeze .venv-bfcl      bfcl.lock.txt      "" \
  "BFCL v4 harness. aarch64 snapshot incl. torch/nvidia wheels from PyPI — on x86 pip picks that platform's wheels for the same versions."

#!/usr/bin/env bash
# ops/run_tests.sh — the one command for the repo's tests.
#
#   1) pytest in the tau2 venv: the verifier scenario suite (tests/test_trajectory_reward.py, 31 cases
#      against the generated db_bahn pool), sampler math, ToolACE backfill verdicts. The collator test
#      skips here (no torch in that venv).
#   2) the collator loss-mask golden test in the training container (torch + transformers + Qwen3-4B
#      tokenizer) — started GPU-FREE via docker/docker-compose.cpu.yml: the shared GPU is never touched
#      by a test run, and the step refuses to run if torch can see a GPU. SKIP_CONTAINER=1 leaves it
#      out (no docker on this machine).
#
# Usage: bash ops/run_tests.sh [pytest args…]        e.g.  bash ops/run_tests.sh -k trajectory -x
# CI runs the same tests from requirements/tau2.lock.txt (.github/workflows/tests.yml).
set -uo pipefail   # KEIN -e: both stages must run; the exit code is their sum
cd "$(dirname "$0")/.."
TAU2PY=${TAU2PY:-.venv-tau2/bin/python}
COMPOSE_CPU="docker compose -f docker/docker-compose.yml -f docker/docker-compose.cpu.yml"   # training service WITHOUT the GPU
rc=0

echo "== 1/2 pytest ($TAU2PY)"
LOGURU_LEVEL=${LOGURU_LEVEL:-ERROR} "$TAU2PY" -m pytest -q "$@" || rc=1

if [ "${SKIP_CONTAINER:-0}" = 1 ]; then
  echo "== 2/2 collator golden test: skipped (SKIP_CONTAINER=1)"
else
  echo "== 2/2 collator golden test (training container, GPU-free)"
  $COMPOSE_CPU run --rm -T training bash -c '
    python3 -c "import sys, torch; sys.exit(int(torch.cuda.is_available()))" \
      || { echo "== ABORT: a GPU is visible inside the test container (docker/docker-compose.cpu.yml not in effect)"; exit 1; }
    python3 training_pipeline/collator_multiturn.py' || rc=1
fi

if [ "$rc" = 0 ]; then echo "== ALL GREEN"; else echo "== FAILURES (see above)"; fi
exit "$rc"

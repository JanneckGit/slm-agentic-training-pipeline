#!/usr/bin/env bash
# ops/run_tests.sh — the one command for the repo's tests.
#
#   1) pytest in the tau2 venv: the verifier scenario suite (tests/test_trajectory_reward.py, 31 cases
#      against the generated db_bahn pool), sampler math, ToolACE backfill verdicts. The collator test
#      skips here (no torch in that venv).
#   2) the collator loss-mask golden test + the GRPO confidence-weight self-test in the training
#      container (torch + transformers + Qwen3-4B tokenizer) — started GPU-FREE via
#      docker/docker-compose.cpu.yml: the shared GPU is never touched by a test run, and the step refuses
#      to run if torch can see a GPU. SKIP_CONTAINER=1 leaves steps 2 and 3 out (no docker on this machine).
#   3) the conf_weighted loss against verl's own vanilla loss in the grpo image (verl_check), GPU-free the
#      same way. SKIP_GRPO=1 leaves it out; skipped when the image is not on this machine.
#
# Usage: bash ops/run_tests.sh [pytest args…]        e.g.  bash ops/run_tests.sh -k trajectory -x
# CI runs the same tests from requirements/tau2.lock.txt (.github/workflows/tests.yml).
set -uo pipefail   # KEIN -e: both stages must run; the exit code is their sum
cd "$(dirname "$0")/.."
TAU2PY=${TAU2PY:-.venv-tau2/bin/python}
COMPOSE_CPU="docker compose -f docker/docker-compose.yml -f docker/docker-compose.cpu.yml"   # training/grpo services WITHOUT the GPU
rc=0

echo "== 1/3 pytest ($TAU2PY)"
LOGURU_LEVEL=${LOGURU_LEVEL:-ERROR} "$TAU2PY" -m pytest -q "$@" || rc=1

GPU_GUARD='python3 -c "import sys, torch; sys.exit(int(torch.cuda.is_available()))" || { echo "== ABORT: a GPU is visible inside the test container (docker/docker-compose.cpu.yml not in effect)"; exit 1; }'
if [ "${SKIP_CONTAINER:-0}" = 1 ]; then
  echo "== 2/3 collator golden test + conf-weight self-test: skipped (SKIP_CONTAINER=1)"
else
  echo "== 2/3 collator golden test + conf-weight self-test (training container, GPU-free)"
  $COMPOSE_CPU run --rm -T training bash -c "$GPU_GUARD
    python3 training_pipeline/collator_multiturn.py && python3 training_pipeline/grpo_conf_loss.py --selftest" || rc=1
fi

GRPO_IMAGE=${GRPO_IMAGE:-agentic-grpo:verl}
if [ "${SKIP_CONTAINER:-0}" = 1 ] || [ "${SKIP_GRPO:-0}" = 1 ]; then
  echo "== 3/3 conf_weighted loss vs verl: skipped (SKIP_CONTAINER/SKIP_GRPO)"
elif ! docker image inspect "$GRPO_IMAGE" >/dev/null 2>&1; then
  echo "== 3/3 conf_weighted loss vs verl: skipped (image $GRPO_IMAGE not on this machine)"
else
  echo "== 3/3 conf_weighted loss vs verl (grpo image, GPU-free)"
  GRPO_IMAGE=$GRPO_IMAGE $COMPOSE_CPU --profile grpo run --rm -T --entrypoint bash grpo -c "$GPU_GUARD
    python3 -c 'import sys, training_pipeline.grpo_conf_loss as m; p = m.verl_check(); print(\"verl_check:\", p or \"ok\"); sys.exit(1 if p else 0)'" || rc=1
fi

if [ "$rc" = 0 ]; then echo "== ALL GREEN"; else echo "== FAILURES (see above)"; fi
exit "$rc"

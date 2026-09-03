#!/usr/bin/env bash
# =============================================================================
# Stage-2 GRPO on db_bahn — profile runner for the frozen verl stack (config/grpo_db_bahn.yaml).
#
#   bash ops/grpo_run.sh smoke            # 2 steps, dynamic corridor sampler, 24-task validation
#   bash ops/grpo_run.sh smoke_static     # 1 step, static sampler
#   bash ops/grpo_run.sh r1 [RUN_NAME]    # the real run — in tmux, on explicit go
#
# What happens: (1) image check, (2) preflight INSIDE the container (CPU; pool, init fingerprint,
# registry, flags, imports, parser round-trip, run state), (3) flags rendered from the YAML profile,
# (4) verl main_ppo in the grpo container with the memory sampler running beside it, (5) summary.
# Re-running the identical command resumes (verl resume_mode=auto + the ledger/sampler state).
#
# Env: RUN (override run name), ROUNDTRIP_LIMIT (preflight sample, default 200), SKIP_PREFLIGHT=1,
#      DRY_RUN=1 (print the verl command, start nothing).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
PROFILE=${1:?usage: grpo_run.sh <profile> [RUN_NAME]}
RUN=${2:-${RUN:-}}
COMPOSE="docker compose -f docker/docker-compose.yml"
RUN_ARG=(); [ -n "$RUN" ] && RUN_ARG=(--run "$RUN")

# image + paths come from the YAML (host python3 + PyYAML suffice for the renderer)
eval "$(python3 training_pipeline/grpo_config.py "$PROFILE" "${RUN_ARG[@]}" --env)" || { echo "config render failed"; exit 1; }
export GRPO_IMAGE="$IMAGE"
mkdir -p logs
LOG="logs/${RUN}.log"; MEMLOG="logs/${RUN}.mem.log"

echo "==== GRPO $PROFILE run=$RUN image=$GRPO_IMAGE — $(date '+%F %T') ===="
if ! docker image inspect "$GRPO_IMAGE" >/dev/null 2>&1; then
  echo "== image $GRPO_IMAGE not found (docker images); abort"; exit 1
fi
if [ -z "${TMUX:-}" ] && [ "$PROFILE" != "smoke" ] && [ "$PROFILE" != "smoke_static" ]; then
  echo "== WARNING: not inside tmux — a dropped SSH session kills this run (see docs/stage2-RL.md)"
fi

# --- preflight (container, CPU) --------------------------------------------------------------------
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  $COMPOSE --profile grpo run --rm -T grpo training_pipeline/grpo_preflight.py "$PROFILE" "${RUN_ARG[@]}" \
      --roundtrip-limit "${ROUNDTRIP_LIMIT:-200}" 2>&1 | grep -E "^\[(OK|FAIL|INFO)|^preflight:" | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "== PREFLIGHT FAILED (rc=$rc); abort"; exit 1; }
fi

# --- flags ---------------------------------------------------------------------------------------
mapfile -t FLAGS < <(python3 training_pipeline/grpo_config.py "$PROFILE" "${RUN_ARG[@]}" --print-flags)
echo "== ${#FLAGS[@]} verl flags; ckpt=$CKPT_DIR run_dir=$RUN_DIR log=$LOG"
if [ "${DRY_RUN:-0}" = "1" ]; then
  printf '%s\n' "$COMPOSE --profile grpo run --rm -T -e GRPO_RUN_DIR=$RUN_DIR -e MLFLOW_ALLOW_FILE_STORE=true grpo -m verl.trainer.main_ppo" "${FLAGS[@]}"
  exit 0
fi

# --- memory sampler (unified memory: watch 'available', not 'free') ----------------------------------
( while true; do echo "$(date '+%F %T') avail_gb=$(free -g | awk '/^Speicher|^Mem/{print $7}')"; sleep 30; done ) >> "$MEMLOG" &
MEMPID=$!
trap 'kill $MEMPID 2>/dev/null' EXIT

# --- train -----------------------------------------------------------------------------------------
$COMPOSE --profile grpo run --rm -T -e GRPO_RUN_DIR="$RUN_DIR" -e MLFLOW_ALLOW_FILE_STORE=true \
    grpo -m verl.trainer.main_ppo "${FLAGS[@]}" 2>&1 | tee -a "$LOG" | grep -vE "^\s*$"
rc=${PIPESTATUS[0]}
kill $MEMPID 2>/dev/null
# the container runs as root: hand the run's files back to the host user (no sudo needed — same image)
docker run --rm --entrypoint bash -v "$REPO":/app "$GRPO_IMAGE" -c "chown -R $(id -u):$(id -g) /app/data/final/grpo /app/data/final/checkpoints /app/mlruns 2>/dev/null; true"
{
  echo "==== GRPO $PROFILE run=$RUN done rc=$rc — $(date '+%F %T') ===="
  echo "   ckpt: ${CKPT_DIR#/app/}   run: ${RUN_DIR#/app/}   log: $LOG   mem: $MEMLOG"
  echo "   next: python3 training_pipeline/grpo_metrics.py $PROFILE ${RUN_ARG[*]}"
} | tee -a "$LOG"
exit "$rc"

#!/usr/bin/env bash
# =============================================================================
# Stage-2 GRPO on db_bahn — profile runner for the frozen verl stack (config/grpo_db_bahn.yaml).
#
#   bash ops/grpo_run.sh smoke                    # 2 steps, dynamic corridor sampler, 24-task validation
#   bash ops/grpo_run.sh smoke_static             # 1 step, static sampler
#   bash ops/grpo_run.sh default <run-name>       # a real run — in tmux; the same command again resumes
#
# A profile is a configuration variant (config/grpo_db_bahn.yaml); the run name is the identity of the
# run and names its folders (checkpoints, ledger, dumps, MLflow run). What happens: (1) image check,
# (2) preflight INSIDE the container (CPU; pool, init fingerprint, registry, flags, imports, parser
# round-trip, run state), (3) the resolved configuration is written to runs/<run>/resolved_config.json
# (or verified against it on resume), (4) verl main_ppo in the grpo container with the memory sampler
# beside it, (5) summary. Metrics and export read the run's resolved config, not the YAML.
#
# Env: ROUNDTRIP_LIMIT (preflight sample, default 200), SKIP_PREFLIGHT=1, DRY_RUN=1 (print the verl
#      command, start nothing).
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
PROFILE=${1:?usage: grpo_run.sh <profile> [run-name]   (profiles: default, smoke, smoke_static)}
RUN=${2:-}
COMPOSE="docker compose -f docker/docker-compose.yml"
RUN_ARG=(); [ -n "$RUN" ] && RUN_ARG=(--run "$RUN")

# image + paths come from the YAML (host python3 + PyYAML suffice for the renderer)
ENV_LINES=$(python3 training_pipeline/grpo_config.py "$PROFILE" "${RUN_ARG[@]}" --env) \
  || { echo "config render failed (usage: grpo_run.sh <profile> [run-name]; a run name is required for profile 'default')"; exit 1; }
eval "$ENV_LINES"
export GRPO_IMAGE="$IMAGE"
mkdir -p logs
LOG="logs/${RUN}.log"; MEMLOG="logs/${RUN}.mem.log"

echo "==== GRPO $PROFILE run=$RUN image=$GRPO_IMAGE — $(date '+%F %T') ===="
if ! docker image inspect "$GRPO_IMAGE" >/dev/null 2>&1; then
  echo "== image $GRPO_IMAGE not found (docker images); abort"; exit 1
fi
if [ -z "${TMUX:-}" ] && [ "$PROFILE" != "smoke" ] && [ "$PROFILE" != "smoke_static" ]; then
  echo "== WARNING: not inside tmux — a dropped SSH session kills this run"
fi

# --- preflight (container, CPU) --------------------------------------------------------------------
if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  $COMPOSE --profile grpo run --rm -T grpo training_pipeline/grpo_preflight.py "$PROFILE" "${RUN_ARG[@]}" \
      --roundtrip-limit "${ROUNDTRIP_LIMIT:-200}" 2>&1 | grep -E "^\[(OK|FAIL|INFO)|^preflight:" | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "== PREFLIGHT FAILED (rc=$rc); abort"; exit 1; }
fi

# --- resolved configuration: created on the first start, verified on every resume -------------------
python3 training_pipeline/grpo_config.py "$PROFILE" "${RUN_ARG[@]}" --write-resolved "${RUN_DIR#/app/}/resolved_config.json" \
  || { echo "== resolved_config mismatch; abort"; exit 1; }

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
  echo "   next: python3 training_pipeline/grpo_metrics.py $RUN [--select]   (--mlflow only inside the grpo container)"
  echo "         bash ops/grpo_export.sh $RUN selected"
} | tee -a "$LOG"
exit "$rc"

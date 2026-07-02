#!/bin/bash
# =============================================================================
# Self-heal supervisor for the verl GRPO pilot on GB10  (Pilot-Prep [C])
# =============================================================================
# WHY: verl 0.8.0 has NO rollout timeout/watchdog, and its vLLM is in-process /
# colocated (workers/rollout/vllm_rollout/vllm_async_server.py: AsyncLLM.from_vllm_config,
# co-scheduled Ray actor) — NOT a separately-restartable server. So a wedged vLLM
# (the GB10 wedge: ~96% util / ~13 W / throughput->0) hangs the WHOLE run forever,
# and the only recovery is killing + restarting the entire verl process.
# NOTE (2026-06-30): the GB10 wedge root cause = the FlashInfer top-k/top-p sampler race (vLLM #43885),
# now FIXED by VLLM_USE_FLASHINFER_SAMPLER=0 (docker-compose grpo env). This supervisor is the residual
# wedge/crash recovery backstop, not the fix.
#
# HOW: the runner (training_pipeline/grpo_verl_runner.py) sets
#   trainer.save_freq=<grpo.save_freq>   (checkpoint every N train steps)
#   trainer.resume_mode=auto             (resume from the latest checkpoint on relaunch)
#   trainer.default_local_dir=data/final/grpo/verl_ckpt/<run_name>   (stable, host-mounted)
# verl auto-resume restores global_step + dataloader position + LoRA adapter + optimizer +
# LR scheduler + RNG (ray_trainer.py:_load_checkpoint). This script launches the runner,
# polls GPU power for the wedge, and on wedge kills the grpo container + relaunches (resume).
#
# MAX PROGRESS LOST PER RECOVERY: <= save_freq train steps + the in-flight step's rollout.
#   (save_freq=5, ~2 min/step => <= ~12 min lost per wedge.)
#
# Usage:   DATE=20260626 bash ops/grpo_pilot_supervised.sh
#   DATE MUST be stable across restarts — it sets experiment_name -> default_local_dir -> resume.
#   Gate the GPU yourself before first launch (shared GB10): no foreign proc > 40 GB.
# =============================================================================
set -u
cd "$(dirname "$0")/.."
COMPOSE="docker compose -f docker/docker-compose.yml"
DATE="${DATE:-pilot}"
CFG="${CFG:-config/pipeline_config.local.yaml}"
LOG="${LOG:-logs/grpo_pilot_${DATE}.log}"
RUNNAME="grpo_qwen3_4b_thinking_${DATE}"
TRACKER="data/final/grpo/verl_ckpt/${RUNNAME}/latest_checkpointed_iteration.txt"
POLL="${POLL:-60}"            # watchdog poll seconds
WEDGE_W="${WEDGE_W:-25}"      # power.draw below this (W) = idle/wedged (healthy train/rollout >> 40 W)
WEDGE_STALL="${WEDGE_STALL:-5}"  # ... sustained this many polls (~5 min) -> declare wedge
ARM_W="${ARM_W:-40}"         # only arm wedge-detection AFTER power first exceeds this (skip startup)
MAXWAIT_H="${MAXWAIT_H:-24}"
mkdir -p logs
log(){ echo "[$(date +%F\ %T)] $*" | tee -a "$LOG"; }
kill_grpo(){ docker ps -q --filter ancestor=text2sql-grpo:verl | xargs -r docker kill >/dev/null 2>&1; }

start=$(date +%s)
for attempt in $(seq 1 50); do
  [ $(( $(date +%s) - start )) -gt $((MAXWAIT_H*3600)) ] && { log "MAXWAIT ${MAXWAIT_H}h reached -> abort"; break; }
  ck=$(cat "$TRACKER" 2>/dev/null || echo 0)
  log "=== runner attempt $attempt (auto-resume from checkpointed step ${ck}) ==="
  ( $COMPOSE --profile grpo run --rm -T grpo \
      training_pipeline/grpo_verl_runner.py --config "$CFG" --date "$DATE" >>"$LOG" 2>&1
    echo "RUNNER_EXIT=$?" >>"$LOG" ) &
  rpid=$!
  armed=0; stall=0
  while kill -0 "$rpid" 2>/dev/null; do
    sleep "$POLL"
    grep -q "RUNNER_EXIT=0" "$LOG" 2>/dev/null && break
    pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | cut -d. -f1); pw=${pw:-99}
    [ "${pw:-0}" -ge "$ARM_W" ] && armed=1            # training is live -> arm
    if [ "$armed" = 1 ] && [ "${pw:-99}" -lt "$WEDGE_W" ]; then stall=$((stall+1)); else stall=0; fi
    if [ "$stall" -ge "$WEDGE_STALL" ]; then
      log "WEDGE: power=${pw}W sustained ~$((WEDGE_STALL*POLL/60))min -> kill grpo + relaunch (auto-resume)"
      kill_grpo; kill "$rpid" 2>/dev/null; break
    fi
  done
  wait "$rpid" 2>/dev/null
  grep -q "RUNNER_EXIT=0" "$LOG" 2>/dev/null && { log "TRAINING COMPLETE (runner exit 0)"; break; }
  log "runner attempt $attempt ended without success (wedge/crash) -> retry"
done
log "SUPERVISOR EXIT (complete: $(grep -q 'RUNNER_EXIT=0' "$LOG" 2>/dev/null && echo yes || echo NO))"

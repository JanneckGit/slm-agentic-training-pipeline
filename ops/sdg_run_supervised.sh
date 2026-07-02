#!/usr/bin/env bash
# =============================================================================
# ops/sdg_run_supervised.sh — self-healing SDG trace_capture run
# -----------------------------------------------------------------------------
# Der Qwen3.6-35B-A3B-Teacher (Hybrid-Linear-Attention-MoE) deadlockt auf GB10
# intermittierend im Gated-DeltaNet-Recurrent-Decode-Kernel unter Last: GPU zeigt
# ~96% util bei nur ~12W, Generation-Throughput faellt auf 0, Engine eingefroren
# (nur /health antwortet noch). Ausloeser ist hohe Concurrency (grosser packed
# decode batch) -> conc 16 laeuft stabil, conc 24 haengt.
#
# Dieser Supervisor faehrt den Lauf unbeaufsichtigt zu Ende:
#   - frischer Start: alles down, alte Trace-Datei loeschen (saubere Auswertungsdaten)
#   - Teacher MIT cudagraphs (default, KEIN enforce-eager — das war nutzlos+langsamer).
#     NB: dieser Teacher-Hang ist der Gated-DeltaNet-Linear-Attn-Recurrent-Kernel (ANDERER Bug),
#     NICHT der FlashInfer-top-k/top-p-Sampler-Race der verl-Rollouts (vLLM #43885) — nicht verwechseln.
#     Beim Teacher hilft enforce-eager nicht; beim Rollout-Wedge hilft VLLM_USE_FLASHINFER_SAMPLER=0.
#   - trace_capture @ conc 16; Hang-Detektion (kein Fortschritt + Power<25W ueber 3min)
#     -> Teacher hart neustarten, ab den bereits geschriebenen Zeilen resumen
#   - fertig, wenn trace_capture sauber (rc=0) durchlaeuft
# =============================================================================
set -uo pipefail
cd /home/jaroeckelein/projects/SLM-Finetuning
DC="docker compose -f docker/docker-compose.yml"
F=data/generated/trace_distill.jsonl
CONC="${CONC:-16}"
GPU_UTIL="${GPU_UTIL:-0.7}"
MAXLEN="${MAXLEN:-8192}"
MAXTOK="${MAXTOK:-2560}"
MAXREGEN="${MAXREGEN:-0}"
THINKCHARS="${THINKCHARS:-6000}"
NSAMPLES="${NSAMPLES:-1110}"
MAX_ROUNDS="${MAX_ROUNDS:-25}"
HANG_MIN="${HANG_MIN:-3}"      # Minuten ohne Fortschritt bei Power<25W = Hang

ts(){ date +%H:%M:%S; }
clean_clients(){ docker ps -aq --filter name=docker-training-run | xargs -r docker rm -f >/dev/null 2>&1 || true; }

restart_teacher(){
  echo "[$(ts)] SUP restart teacher (cudagraphs ON, util $GPU_UTIL, len $MAXLEN)"
  clean_clients
  $DC --profile vllm down >/dev/null 2>&1 || true
  VLLM_MODEL=Qwen/Qwen3.6-35B-A3B VLLM_GPU_UTIL="$GPU_UTIL" VLLM_MAX_MODEL_LEN="$MAXLEN" \
    $DC --profile vllm up -d vllm >/dev/null 2>&1
  for i in $(seq 1 180); do
    curl -fs http://localhost:8000/health >/dev/null 2>&1 && { echo "[$(ts)] SUP teacher ready (~$((i*5))s)"; return 0; }
    sleep 5
  done
  echo "[$(ts)] SUP teacher NOT ready after 900s"; return 1
}

echo "============ SDG SUPERVISED RUN start ($(date)) ============"
clean_clients
if [ "${RESUME:-0}" = "1" ]; then
  echo "[$(ts)] SUP RESUME=1: behalte vorhandene Trace-Datei (kept=$(wc -l < "$F" 2>/dev/null))"
  if curl -fs http://localhost:8000/health >/dev/null 2>&1; then
    echo "[$(ts)] SUP teacher bereits healthy -> kein Restart"
  else
    restart_teacher || { echo "[$(ts)] SUP FATAL: teacher kommt nicht hoch"; exit 1; }
  fi
else
  echo "[$(ts)] SUP fresh start: down + alte Trace-Datei loeschen"
  $DC --profile vllm down >/dev/null 2>&1 || true
  $DC run --rm -T training bash -lc "rm -f $F" >/dev/null 2>&1 || true
  restart_teacher || { echo "[$(ts)] SUP FATAL: teacher kommt nicht hoch"; exit 1; }
fi

round=0
while [ "$round" -lt "$MAX_ROUNDS" ]; do
  round=$((round+1))
  kept=$(wc -l < "$F" 2>/dev/null || echo 0)
  echo "[$(ts)] SUP === round $round start, kept=$kept ==="
  curl -fs http://localhost:8000/health >/dev/null 2>&1 || restart_teacher
  killed=0
  $DC run --rm -T training python3 sdg_pipeline/trace_capture.py \
    --config config/pipeline_config.local.yaml --seed-file data/raw/seed_sdg_input.jsonl \
    --n-samples "$NSAMPLES" --concurrency "$CONC" --max-tokens "$MAXTOK" \
    --max-regen "$MAXREGEN" --max-thinking-chars "$THINKCHARS" --output "$F" --show 0 &
  TCPID=$!
  prev=$kept; stall=0
  while kill -0 "$TCPID" 2>/dev/null; do
    sleep 60
    cur=$(wc -l < "$F" 2>/dev/null || echo 0)
    pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | cut -d. -f1); [ -z "$pw" ] && pw=0
    if [ "$cur" -gt "$prev" ]; then
      echo "[$(ts)] SUP progress kept=$cur (+$((cur-prev))) pw=${pw}W"
      prev=$cur; stall=0
    else
      stall=$((stall+1))
      echo "[$(ts)] SUP no-progress ${stall}min kept=$cur pw=${pw}W"
      if [ "$stall" -ge "$HANG_MIN" ] && [ "$pw" -lt 25 ]; then
        echo "[$(ts)] SUP *** HANG erkannt (Gated-DeltaNet deadlock) -> kill+restart ***"
        killed=1
        kill "$TCPID" 2>/dev/null; wait "$TCPID" 2>/dev/null
        restart_teacher
        break
      fi
    fi
  done
  if [ "$killed" -eq 0 ]; then
    wait "$TCPID"; rc=$?
    kept=$(wc -l < "$F" 2>/dev/null || echo 0)
    echo "[$(ts)] SUP trace_capture exit rc=$rc kept=$kept"
    if [ "$rc" -eq 0 ]; then echo "[$(ts)] SUP *** ALL DONE kept=$kept ***"; break; fi
    echo "[$(ts)] SUP non-hang exit rc=$rc -> retry in 15s"
    sleep 15
    curl -fs http://localhost:8000/health >/dev/null 2>&1 || restart_teacher
  fi
done
echo "============ SDG SUPERVISED RUN end ($(date)) kept=$(wc -l < "$F" 2>/dev/null) ============"

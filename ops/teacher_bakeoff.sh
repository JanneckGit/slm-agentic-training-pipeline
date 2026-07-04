#!/usr/bin/env bash
# ops/teacher_bakeoff.sh — Phase-3 teacher bake-off (Plan B), SHORTENED protocol.
#
# Per candidate: (bg-download next) -> serve via compose vllm -> health-wait -> coherence pre-check
# -> 12 stratified bakeoff_dev tasks x k=1 through rollout.py + verifier (hard 25-min cap)
# -> teardown -> update docs/teacher-bakeoff.md.  Candidates that fail to serve/eval are logged and
# SKIPPED (agentic error policy: try, record, move on).  Resume-safe: completed candidates are skipped.
#
# Usage: bash ops/teacher_bakeoff.sh [start_index]
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
COMPOSE="docker compose -f docker/docker-compose.yml"
TAU2PY=${TAU2PY:-$REPO/.venv-tau2/bin/python}
N_TASKS=12; K=1; MAX_TURNS=8; MAXTOK=2048; EVAL_TIMEOUT=1500; HEALTH_TIMEOUT=900
mkdir -p logs data/generated

# short|model|extra vllm args (incl. --max-num-seqs)|gpu_util|max_len|omit_thinking_kwarg(0/1)|maxtok_per_turn
CANDIDATES=(
  "q36-35b-a3b|Qwen/Qwen3.6-35B-A3B|--max-num-seqs 4 --gdn-prefill-backend triton|0.85|8192|0|2048"
  "q36-27b|Qwen/Qwen3.6-27B|--max-num-seqs 4 --gdn-prefill-backend triton|0.85|8192|0|2048"
  "q3-30b-a3b-think|Qwen/Qwen3-30B-A3B-Thinking-2507|--max-num-seqs 4|0.85|8192|0|3072"
  "magistral-24b|mistralai/Magistral-Small-2509|--max-num-seqs 4 --tokenizer-mode mistral --config-format mistral --load-format mistral|0.85|8192|1|2048"
  "seed-oss-36b|ByteDance-Seed/Seed-OSS-36B-Instruct|--max-num-seqs 4|0.85|8192|0|3072"
  "nemotron-49b-nvfp4|nvidia/Llama-3_3-Nemotron-Super-49B-v1_5-NVFP4|--max-num-seqs 4 --trust-remote-code|0.85|8192|0|2048"
  "q3-next-80b-fp8|Qwen/Qwen3-Next-80B-A3B-Thinking-FP8|--max-num-seqs 4 --gdn-prefill-backend triton|0.85|8192|0|3072"
  "glm45-air-fp8|zai-org/GLM-4.5-Air-FP8|--max-num-seqs 2|0.90|6144|0|1024"
)

dl() {  # download via sdg container (hf_cache hub/ is root-owned)
  local model=$1
  $COMPOSE run --rm -T sdg python -c "
from huggingface_hub import snapshot_download
snapshot_download('$model')" >>"logs/bakeoff_download.log" 2>&1 \
    && echo "[dl] $model done" >>logs/bakeoff_download.log \
    || echo "[dl] $model FAILED" >>logs/bakeoff_download.log
}

teardown() { $COMPOSE --profile vllm down vllm >/dev/null 2>&1 || true; sleep 3; }

START=${1:-0}
for i in "${!CANDIDATES[@]}"; do
  [ "$i" -lt "$START" ] && continue
  IFS='|' read -r SHORT MODEL EXTRA GPU_UTIL MAXLEN OMIT CAND_MAXTOK <<<"${CANDIDATES[$i]}"
  CAND_MAXTOK=${CAND_MAXTOK:-$MAXTOK}
  TRACE="data/generated/db_traces_bakeoff_dev_${SHORT}.jsonl"
  LOG="logs/bakeoff_${SHORT}.log"

  # resume: skip candidates that already have all records
  if [ -f "$TRACE" ] && [ "$(wc -l <"$TRACE")" -ge $((N_TASKS * K)) ]; then
    echo "== [$i] $SHORT already complete, skipping"; continue
  fi

  # kick off background download of the NEXT candidate
  NEXTI=$((i + 1))
  if [ "$NEXTI" -lt "${#CANDIDATES[@]}" ]; then
    IFS='|' read -r _ NEXTMODEL _ _ _ _ _ <<<"${CANDIDATES[$NEXTI]}"
    ( dl "$NEXTMODEL" ) &
  fi

  echo "== [$i] $SHORT ($MODEL) — download/verify weights"
  dl "$MODEL"

  echo "== [$i] $SHORT — serve (util=$GPU_UTIL len=$MAXLEN extra='$EXTRA')"
  teardown
  VLLM_MODEL="$MODEL" VLLM_GPU_UTIL="$GPU_UTIL" VLLM_MAX_MODEL_LEN="$MAXLEN" \
    VLLM_EXTRA_ARGS="$EXTRA" \
    $COMPOSE --profile vllm up -d vllm >"$LOG" 2>&1

  ok=0
  for _ in $(seq $((HEALTH_TIMEOUT / 10))); do
    curl -sf localhost:8000/health >/dev/null 2>&1 && { ok=1; break; }
    docker ps --format '{{.Names}}' | grep -q text2sql_vllm_teacher || break  # container died
    sleep 10
  done
  if [ "$ok" != 1 ]; then
    echo "== [$i] $SHORT FAILED to serve (health timeout / container died) — skipping" | tee -a "$LOG"
    docker logs text2sql_vllm_teacher --tail 40 >>"$LOG" 2>&1 || true
    teardown; continue
  fi

  # coherence pre-check (NaN/garbage gate, base lesson)
  RESP=$(curl -sf localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d "{
    \"model\": \"$MODEL\", \"max_tokens\": 60,
    \"messages\": [{\"role\": \"user\", \"content\": \"Antworte kurz auf Deutsch: Was ist ein ICE?\"}]}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'][:120])" 2>/dev/null)
  echo "== [$i] $SHORT coherence: ${RESP:-EMPTY}" | tee -a "$LOG"
  if [ -z "${RESP:-}" ]; then
    echo "== [$i] $SHORT FAILED coherence — skipping" | tee -a "$LOG"; teardown; continue
  fi

  echo "== [$i] $SHORT — eval ($N_TASKS tasks x k=$K, cap $((EVAL_TIMEOUT / 60))min)"
  OMITFLAG=""; [ "$OMIT" = 1 ] && OMITFLAG="--omit-thinking-kwarg"
  timeout "$EVAL_TIMEOUT" env PYTHONPATH="$REPO" LOGURU_LEVEL=ERROR "$TAU2PY" \
    sdg_pipeline/db_bahn/rollout.py --config config/pipeline_config.yaml \
    --split bakeoff_dev --n-tasks "$N_TASKS" --stratify --k "$K" \
    --api-base http://localhost:8000/v1 --model "$MODEL" --teacher-name "$SHORT" \
    --max-turns "$MAX_TURNS" --max-tokens-per-turn "$CAND_MAXTOK" --max-regen 0 \
    --concurrency 4 --output "$TRACE" $OMITFLAG 2>&1 | tail -12 | tee -a "$LOG"

  teardown
  python3 sdg_pipeline/db_bahn/bakeoff_summary.py --write | tail -3
done

wait  # drain background downloads
echo "== BAKE-OFF DONE =="
python3 sdg_pipeline/db_bahn/bakeoff_summary.py --write

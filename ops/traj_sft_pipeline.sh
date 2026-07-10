#!/usr/bin/env bash
# ops/traj_sft_pipeline.sh — Phase 6 (Plan B): BEFORE-eval -> traj_sft train -> merge -> AFTER-eval.
# Sequential (GB10 has no MIG). Eval = rollout.py on the heldout_eval tasks through the same verifier
# (before = base Qwen3.5-4B, after = merged student). Idempotent-ish; logs to logs/traj_sft.log.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
COMPOSE="docker compose -f docker/docker-compose.yml"
TAU2PY=${TAU2PY:-$(pwd)/.venv-tau2/bin/python}
# host eval writes to the SAME physical store as the training container (../mlruns == /app/mlruns)
export MLFLOW_TRACKING_URI="file://$REPO/mlruns"
BASE="Qwen/Qwen3.5-4B"
MERGED_HOST="data/final/checkpoints/db_bahn_traj_merged"
MERGED_CTR="/app/data/final/checkpoints/db_bahn_traj_merged"
ADAPTER="data/final/checkpoints/db_bahn_traj_lora"

serve() {  # $1=model $2=served-name-model $3=gpu_util
  $COMPOSE --profile vllm down vllm >/dev/null 2>&1; sleep 3
  VLLM_MODEL="$1" VLLM_GPU_UTIL="${3:-0.85}" VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}" \
    VLLM_EXTRA_ARGS="--max-num-seqs 4 --gdn-prefill-backend triton" \
    $COMPOSE --profile vllm up -d vllm >>logs/traj_sft.log 2>&1
  for _ in $(seq 90); do curl -sf localhost:8000/health >/dev/null 2>&1 && return 0; \
    docker ps --format '{{.Names}}' | grep -q text2sql_vllm_teacher || return 1; sleep 10; done
  curl -sf localhost:8000/health >/dev/null 2>&1
}

eval_heldout() {  # $1=served-model $2=label $3=outfile
  timeout 1800 env PYTHONPATH="$REPO" LOGURU_LEVEL=ERROR "$TAU2PY" \
    sdg_pipeline/db_bahn/rollout.py --config config/pipeline_config.yaml \
    --split heldout_eval --k 1 --api-base http://localhost:8000/v1 --model "$1" --teacher-name "$2" \
    --max-turns 8 --max-tokens-per-turn 1536 --max-regen 1 --concurrency 4 \
    --mlflow --mlflow-run-name "$2" \
    --output "$3" 2>&1 | tail -6
}

echo "==== PHASE 6 START $(date) ===="

echo "== [1/4] BEFORE-eval: base $BASE on heldout_eval =="
rm -f data/generated/db_traces_heldout_before.jsonl
if serve "$BASE" "$BASE" 0.85; then
  eval_heldout "$BASE" "before_base_4b" "data/generated/db_traces_heldout_before.jsonl"
else echo "== BEFORE serve FAILED"; docker logs text2sql_vllm_teacher --tail 30 >>logs/traj_sft.log 2>&1; fi
$COMPOSE --profile vllm down vllm >/dev/null 2>&1; sleep 3

echo "== [2/4] TRAIN traj_sft (446 traces, 2 epochs, LoRA) =="
$COMPOSE run --rm -T training python3 training_pipeline/train_traj.py \
  --config config/pipeline_config.yaml --data data/final/db_traces_chat.jsonl \
  --model "$BASE" --out "$ADAPTER" --epochs 2 --max-seq-len 4096 2>&1 | grep -vE "Copyright|NVIDIA|reserved|GOVERNING|found at|PyTorch|Idiap|Google|Caffe|Facebook|Deepmind|NEC|NYU|Yangqing|Various|====" | tail -20

echo "== [3/4] MERGE adapter -> $MERGED_HOST (sharded) =="
$COMPOSE run --rm -T training python3 serving/merge_adapter.py \
  --adapter-path "$ADAPTER" --output-path "$MERGED_HOST" --config config/pipeline_config.yaml 2>&1 | tail -6

echo "== [4/4] AFTER-eval: merged student on heldout_eval =="
rm -f data/generated/db_traces_heldout_after.jsonl
if serve "$MERGED_CTR" "$MERGED_CTR" 0.85; then
  eval_heldout "$MERGED_CTR" "after_traj_sft" "data/generated/db_traces_heldout_after.jsonl"
else echo "== AFTER serve FAILED"; docker logs text2sql_vllm_teacher --tail 30 >>logs/traj_sft.log 2>&1; fi
$COMPOSE --profile vllm down vllm >/dev/null 2>&1

echo "==== PHASE 6 DONE $(date) ===="

#!/usr/bin/env bash
# =============================================================================
# ops/run_baseline_pipeline.sh  <SIZE: 0.8B|2B|4B|9B>  <VARIANT: thinking|nothink>
# -----------------------------------------------------------------------------
# Full per-model-per-variant baseline pipeline, end to end:
#   1 data-swap (variant)  2 train  3 LoRA->MM merge  4 preprocessor configs
#   5 vLLM serve  6 accuracy eval  7 efficiency eval  8 teardown
#
# UNIFORM recipe: everything comes from config/pipeline_config.local.yaml (2 ep,
# lr 2e-4, micro_batch 1, grad_accum 32, seq 8192, seed 42, LoRA r16/a32). Only
# the student model (--student-model-id) and the data variant change between
# runs, so the 6 baselines stay directly comparable. Run from the repo root.
# =============================================================================
set -euo pipefail

SIZE="${1:?usage: run_baseline_pipeline.sh <SIZE: 0.8B|2B|4B|9B|q3-4b|14B> <VARIANT: thinking|nothink>}"
VARIANT="${2:?usage: run_baseline_pipeline.sh <SIZE> <VARIANT: thinking|nothink>}"
[[ "$VARIANT" == "thinking" || "$VARIANT" == "nothink" ]] || { echo "bad VARIANT: $VARIANT"; exit 2; }

DC="docker compose -f docker/docker-compose.yml"
CFG_TRAIN="config/pipeline_config.local.yaml"     # student + uniform hyperparams
CFG_EVAL="config/pipeline_config.yaml"            # test_clean.jsonl path
# Accuracy-Eval-Concurrency: memory-bound Decode auf GB10 amortisiert die
# Gewicht-Reads ueber die Batch -> hoehere Concurrency = ~3-4x Durchsatz. MUSS
# ueber alle Vergleichslaeufe UNIFORM sein (greedy ist ueber Concurrency-Stufen
# nicht bit-deterministisch). vLLM self-limitiert bei KV-Druck (kein Crash).
EVAL_CONC="${EVAL_CONC:-16}"

# Modell-Familie -> Modell-ID, Kurzname (= _short_model_name), Merge-Pfad.
#   Qwen3.5 (0.8B/2B/4B/9B) = MULTIMODAL -> MM-Remap-Merge + Preprocessor-Configs.
#   Qwen3-14B               = TEXT-ONLY  -> einfacher LoRA-Merge, kein Preprocessor.
case "$SIZE" in
  0.8B|2B|4B|9B)
    MODEL="Qwen/Qwen3.5-$SIZE"
    SHORT="qwen35$(echo "$SIZE" | tr 'A-Z' 'a-z' | tr -d '.')"
    MERGE_MODE=mm ;   MERGED_NAME="${SHORT}_student_${VARIANT}_mm_merged" ;;
  q3-4b)
    MODEL="Qwen/Qwen3-4B"
    SHORT="qwen34b"
    MERGE_MODE=text ; MERGED_NAME="${SHORT}_student_${VARIANT}_merged" ;;
  14B)
    MODEL="Qwen/Qwen3-14B"
    SHORT="qwen314b"
    MERGE_MODE=text ; MERGED_NAME="${SHORT}_student_${VARIANT}_merged" ;;
  *) echo "FATAL: unbekannte SIZE '$SIZE' (erlaubt: 0.8B|2B|4B|9B|q3-4b|14B)"; exit 2 ;;
esac
MERGED_PATH="/data/hf_cache/${MERGED_NAME}"
REF_MERGE="/data/hf_cache/qwen354b_student_thinking_mm_merged"  # MM-Preprocessor-Donor
LOG="sweep_logs/baseline_${SHORT}_${VARIANT}.log"   # host-writable (logs/ is root-owned)
mkdir -p sweep_logs

if [[ "$VARIANT" == "thinking" ]]; then
  EVAL_THINK="--enable-thinking"; EFF_THINK="--enable-thinking"
  # thinking generiert ~3k (bis ~12k) Reasoning-Tokens -> grosszuegiges Budget,
  # sonst wird mitten im Reasoning abgeschnitten und die EX untertrieben.
  # Redo: saubere Traces sind max ~1150 Tok -> 4096 = 2.6x Headroom + klares Loop-Signal,
  # ~3x schnellere Eval. serve-len 8192 reicht (prompt ~1-2k + 4096 gen).
  ACC_MAXTOK=4096; SERVE_LEN=8192
else
  EVAL_THINK=""; EFF_THINK="--disable-thinking"
  ACC_MAXTOK=2048; SERVE_LEN=8192    # nothink = kurzes SQL, kein langes Budget noetig
fi

# Tier-1: gradient-checkpointing AUS für die kleinen Modelle (passen in 128GB,
# spart den Recompute -> schneller; ergebnis-invariant). 9B sicherheitshalber AN.
case "$SIZE" in 0.8B|2B) GC="off";; *) GC="on";; esac

say(){ echo "[$(date +%H:%M:%S)] $*"; }
say "================ PIPELINE  $MODEL / $VARIANT  (merged=$MERGED_NAME) ================"

# --- 1. data-swap: canonical train/eval files -> this variant ----------------
say "[1/8] data-swap -> $VARIANT"
GUARD_OUT="$($DC run --rm training bash -lc "set -e
  cp -f data/final/train_chat_${VARIANT}.jsonl data/final/train_chat.jsonl
  cp -f data/final/eval_chat_${VARIANT}.jsonl  data/final/eval_chat.jsonl
  python3 -c \"import json;print('GUARD_HT='+str(json.loads(open('data/final/train_chat.jsonl').readline())['_meta']['has_thinking']))\"
")"
HT="$(echo "$GUARD_OUT" | grep -aoE 'GUARD_HT=(True|False)' | tail -1 | cut -d= -f2)"
if { [[ "$VARIANT" == "thinking" && "$HT" != "True" ]] || [[ "$VARIANT" == "nothink" && "$HT" != "False" ]]; }; then
  echo "FATAL: variant guard failed (has_thinking=$HT, want $VARIANT)"; exit 1
fi
say "  guard ok: has_thinking=$HT"

# --- 2. train (uniform recipe, only student overridden) ----------------------
say "[2/8] train $MODEL"
# Resume: ein bereits FERTIG trainierter Checkpoint dieses Modells+Variante wird
# wiederverwendet (adapter_model.safetensors = Training abgeschlossen) -> ein
# späterer Stufen-Fehler kostet kein erneutes Training.
CKPT=""
for d in $(ls -dt data/final/checkpoints/*_s-${SHORT}_*_${VARIANT} 2>/dev/null); do
  [[ -f "$d/adapter_model.safetensors" ]] && { CKPT="$d"; break; }
done
if [[ -n "$CKPT" ]]; then
  say "  resume: reuse trained checkpoint $CKPT (skip train)"
else
  set -o pipefail
  $DC run --rm training python3 training_pipeline/train.py \
    --config "$CFG_TRAIN" --student-model-id "$MODEL" --grad-checkpointing "$GC" 2>&1 | tee "$LOG"
  RUN_NAME="$(grep -aoE 'Run name: [^[:space:]]+' "$LOG" | tail -1 | awk '{print $3}')"
  CKPT="data/final/checkpoints/${RUN_NAME}"
  [[ -d "$CKPT" ]] || CKPT="$(ls -dt data/final/checkpoints/*_${VARIANT} 2>/dev/null | head -1)"
fi
[[ -d "$CKPT" ]] || { echo "FATAL: checkpoint dir not found"; exit 1; }
[[ "$CKPT" == *_${VARIANT} ]] || { echo "FATAL: checkpoint '$CKPT' variant mismatch (want $VARIANT)"; exit 1; }
say "  checkpoint: $CKPT"

# --- 3. LoRA -> base model merge --------------------------------------------
# MM (Qwen3.5): Remap-Merge ins volle multimodale Modell. text (Qwen3-14B):
# einfacher PEFT-Merge. Resume akzeptiert single- ODER sharded-safetensors (14B).
say "[3/8] merge adapter into base ($MERGE_MODE)"
if [[ -f "$MERGED_PATH/model.safetensors" || -f "$MERGED_PATH/model.safetensors.index.json" ]]; then
  say "  resume: merged model exists (skip merge)"
elif [[ "$MERGE_MODE" == "mm" ]]; then
  $DC run --rm \
    -e MERGE_ADAPTER="$CKPT" -e MERGE_BASE="$MODEL" -e MERGE_OUT="$MERGED_PATH" \
    training python3 serving/merge_adapter_mm.py
else
  $DC run --rm training python3 serving/merge_adapter.py \
    --adapter-path "$CKPT" --output-path "$MERGED_PATH" --config "$CFG_EVAL"
fi

# --- 4. preprocessor configs (nur MM; merge_adapter_mm speichert nur model+tok) --
say "[4/8] copy preprocessor configs from base model snapshot"
# merge_adapter_mm speichert nur model+tok -> vLLM braucht aber den MM-Preprocessor.
# Quelle = der HF-Snapshot des Basismodells dieser Groesse (immer vorhanden, kein
# Donor-Zwang). Snapshot-Files sind Symlinks auf blobs -> cp -L dereferenziert.
if [[ "$MERGE_MODE" != "mm" ]]; then
  say "  text-only model -> kein Preprocessor noetig, skip"
else
  $DC run --rm -e BASE_ID="$MODEL" -e DST="$MERGED_PATH" training bash -lc '
    set -e
    SNAP=$(ls -d /data/hf_cache/hub/models--${BASE_ID//\//--}/snapshots/*/ 2>/dev/null | head -1)
    [ -n "$SNAP" ] || { echo "FATAL: kein base snapshot fuer $BASE_ID"; exit 1; }
    echo "  base snapshot: $SNAP"
    for f in preprocessor_config.json processor_config.json video_preprocessor_config.json; do
      if [ -e "$SNAP/$f" ]; then cp -fL "$SNAP/$f" "$DST/$f" && echo "  copied $f"; else echo "  (base hat kein $f, skip)"; fi
    done
    ls "$DST" | tr "\n" " "; echo
  '
fi

# --- 5. serve merged model on vLLM ------------------------------------------
say "[5/8] vLLM serve $MERGED_NAME"
VLLM_MODEL="$MERGED_PATH" VLLM_MAX_MODEL_LEN="$SERVE_LEN" VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.5}" \
  $DC --profile vllm up -d vllm
ready=0
for i in $(seq 1 90); do
  if curl -fs http://localhost:8000/health >/dev/null 2>&1; then ready=1; say "  vLLM ready (~$((i*5))s)"; break; fi
  sleep 5
done
[[ "$ready" == 1 ]] || { echo "FATAL: vLLM not healthy"; $DC --profile vllm down; exit 1; }

# --- 6. accuracy eval --------------------------------------------------------
say "[6/8] accuracy eval ($VARIANT)"
$DC run --rm training python3 evaluation/evaluate.py \
  --config "$CFG_EVAL" \
  --model-path "$MERGED_PATH" --api-base http://vllm:8000/v1 --api-model-name "$MERGED_PATH" \
  --concurrency "$EVAL_CONC" --n-samples 100 --max-tokens "$ACC_MAXTOK" $EVAL_THINK \
  --output "data/final/eval/${MERGED_NAME}/student_${VARIANT}.json"

# --- 7. efficiency eval ------------------------------------------------------
say "[7/8] efficiency eval ($VARIANT)"
$DC run --rm training python3 evaluation/efficiency_benchmark.py \
  --model-id "$MERGED_PATH" --target http://vllm:8000 $EFF_THINK --output-base data/final/eval

# --- 8. teardown -------------------------------------------------------------
say "[8/8] teardown vLLM"
$DC --profile vllm down

EX="$(python3 -c "import json;print(json.load(open('data/final/eval/${MERGED_NAME}/student_${VARIANT}.json'))['overall']['execution_accuracy'])" 2>/dev/null || echo '?')"
say "================ DONE  $MODEL / $VARIANT  ->  EX=$EX  (ckpt=$CKPT) ================"

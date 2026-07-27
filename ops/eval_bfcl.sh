#!/usr/bin/env bash
# ops/eval_bfcl.sh — BFCL-v4-Benchmark EINES Modells (Base, merged Student, beliebiger Checkpoint).
# Preflight -> serve -> bfcl generate -> bfcl evaluate -> Report -> MLflow -> Teardown.
#
#   bash ops/eval_bfcl.sh Qwen/Qwen3-4B qwen3-4b_base
#   bash ops/eval_bfcl.sh /app/data/final/checkpoints/db_bahn_traj_merged_qwen3-4b/ep3 qwen3-4b_sft-ep3
#   BFCL_CATEGORIES=non_live bash ops/eval_bfcl.sh Qwen/Qwen3-4B qwen3-4b_base   # gruppenweise
#
# MODEL ist der Pfad, wie ihn der vLLM-CONTAINER sieht (/app/data/... fuer lokale Checkpoints) oder
# eine HF-ID. LABEL benennt den Lauf: data/generated/eval/bfcl/<LABEL>/ (Modell zuerst, damit
# `ls` alle Laeufe eines Modells gruppiert: qwen3-4b_base, qwen3-4b_sft-ep3, qwen3-8b_base, ...).
#
# RESUME IST DER NORMALFALL: `bfcl generate` ueberspringt bereits vorhandene Results (--allow-overwrite
# ist per Default AUS). Ein abgebrochener 12-h-Lauf wird durch denselben Befehl fortgesetzt.
# Neu generieren erzwingt BFCL_ALLOW_OVERWRITE=1 — das VERWIRFT vorhandene Ergebnisse dieses Labels.
#
# Env-Overrides (Defaults in Klammern):
#   BFCL_CATEGORIES (full)      'full' = alle WERTENDEN Kategorien (5.017 Items, ohne web_search).
#                               'everything' nimmt format_sensitivity dazu (+5.200 Items, NICHT
#                               gewertet — nur eine Max-Delta-Kennzahl). Sonst: bfcl-Gruppe
#                               (non_live|live|multi_turn|memory|single_turn|...) oder Komma-Liste.
#   TEMPERATURE (0.6)           BFCLs CLI-Default waere 0.001 -> muss gesetzt werden.
#   NUM_THREADS (21)            Client-Concurrency; 21 = dieselbe Herleitung wie EVAL_CONC in
#                               ops/eval_heldout.sh (75 % der von vLLM gemeldeten Max concurrency).
#   VLLM_MAX_MODEL_LEN (40960)  = max_position_embeddings von Qwen3-4B, der Deckel des Modells.
#                               Zu klein = stille Nullergebnisse, zu gross = vLLM startet nicht;
#                               der Preflight prueft beide Richtungen. S. Kommentar am serve-Block.
#   VLLM_GPU_UTIL (0.85)   BFCL_ENABLE_WEB_SEARCH (0)   BFCL_ALLOW_OVERWRITE (0)
#   BFCL_RUN_IDS (leer)         Pfad zu einer test_case_ids_to_generate.json -> nur diese IDs
#                               (Smoke-/Probe-Modus, zieht automatisch --partial-eval nach).
#   MLFLOW_EXPERIMENT (bfcl_eval)   SKIP_MLFLOW (0)   BFCLPY (.venv-bfcl/bin/python)
set -uo pipefail   # KEIN -e: `compose down` scheitert legitim ohne Container, und `timeout` liefert 124
cd "$(dirname "$0")/.."
REPO=$(pwd)
COMPOSE="docker compose -f docker/docker-compose.yml"
BFCLPY=${BFCLPY:-$REPO/.venv-bfcl/bin/python}
BFCLBIN=$(dirname "$BFCLPY")/bfcl
TAU2PY=${TAU2PY:-$REPO/.venv-tau2/bin/python}   # traegt mlflow (das bfcl-venv nicht)

MODEL=${1:?usage: eval_bfcl.sh <MODEL> <LABEL> [CATEGORIES]}
LABEL=${2:?usage: eval_bfcl.sh <MODEL> <LABEL> [CATEGORIES]}
CATEGORIES=${3:-${BFCL_CATEGORIES:-full}}
TEMPERATURE=${TEMPERATURE:-0.6}
NUM_THREADS=${NUM_THREADS:-21}
MAX_LEN=${VLLM_MAX_MODEL_LEN:-40960}   # = max_position_embeddings von Qwen3-4B; mehr lehnt vLLM ab
ROOT="$REPO/data/generated/eval/bfcl/$LABEL"

mkdir -p "$ROOT/logs" logs
LOG="$ROOT/logs/run.log"

teardown() { $COMPOSE --profile vllm down vllm >/dev/null 2>&1; }
trap teardown EXIT

echo "==== BFCL $LABEL ($MODEL) — $(date '+%F %T') ====" | tee -a "$LOG"

# ---------------------------------------------------------------------------------------------
# 1) Preflight — CPU-only. Faellt hier etwas aus, ist nichts serviert und nichts geschrieben.
#    Prueft u. a. das Sampling: BFCL sendet NUR temperature; top_p/top_k kommen aus der
#    generation_config.json des servierten Modells. Weicht die ab, misst man still daneben.
# ---------------------------------------------------------------------------------------------
PF_ARGS=(--model "$MODEL" --label "$LABEL" --root "$ROOT" --categories "$CATEGORIES"
         --temperature "$TEMPERATURE" --max-model-len "$MAX_LEN" --num-threads "$NUM_THREADS")
[ "${BFCL_ENABLE_WEB_SEARCH:-0}" = "1" ] && PF_ARGS+=(--enable-web-search)
[ -n "${BFCL_RUN_IDS:-}" ] && PF_ARGS+=(--run-ids "$BFCL_RUN_IDS")
"$BFCLPY" evaluation/benchmarks/bfcl/preflight.py "${PF_ARGS[@]}" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -ne 0 ] && { echo "== PREFLIGHT FAILED — abort."; exit 1; }

CATS=$(cat "$ROOT/categories.txt")
MODEL_KEY=$("$TAU2PY" -c "import json;print(json.load(open('$ROOT/run_manifest.json'))['model_key'])")
ALIAS=$("$TAU2PY" -c "import json;print(json.load(open('$ROOT/run_manifest.json'))['model_alias'])")

# ---------------------------------------------------------------------------------------------
# 2) Serven.
#    --served-model-name $ALIAS ist ZWINGEND: bfcl-eval kennt keinen Registry-Key fuer das hybride
#    Qwen3-4B, der Handler sendet deshalb `model=<ALIAS>` an /v1/completions. Ohne Alias 404 auf
#    JEDEN Request — der Lauf liefe durch und schriebe lauter Nullen.
#    KEIN --reasoning-parser: der QwenFCHandler parst <think> SELBST aus dem Completion-Text
#    (qwen_fc.py). Ein serverseitiger Parser wuerde das Feld wegnehmen -> think% waere ueberall 0.
#    --max-model-len: BFCL fordert max_tokens = min(4096, ctx - input - 2) an, der Server muss also
#    input + 4096 tragen. Zu klein heisst NICHT "abgeschnitten", sondern der Request geht gar nicht
#    durch und landet als input_token_count 0 im Result — genau so sind im alten Quickrun 4 von 5
#    multi_turn_long_context-Episoden still als 0 % durchgelaufen.
# ---------------------------------------------------------------------------------------------
teardown; sleep 3
echo "-- serve: ctx=$MAX_LEN seqs=$NUM_THREADS alias=$ALIAS" | tee -a "$LOG"
VLLM_MODEL="$MODEL" VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}" VLLM_MAX_MODEL_LEN="$MAX_LEN" \
  VLLM_EXTRA_ARGS="--max-num-seqs $NUM_THREADS --served-model-name $ALIAS" \
  $COMPOSE --profile vllm up -d vllm >>"$LOG" 2>&1

for _ in $(seq 180); do
  curl -sf localhost:8000/health >/dev/null 2>&1 && break
  docker ps --format '{{.Names}}' | grep -q text2sql_vllm_teacher || break
  sleep 10
done
if ! curl -sf localhost:8000/health >/dev/null 2>&1; then
  echo "== SERVE FAILED — abort" | tee -a "$LOG"; docker logs text2sql_vllm_teacher --tail 40 2>&1 | tail -40; exit 1
fi

# Gate auf den ALIAS (nicht auf $MODEL): kommt VLLM_MODEL nicht an, greift der Compose-Fallback und
# der Server liefe klaglos mit dem FALSCHEN Modell unter richtigem Namen.
SERVED=$(curl -s localhost:8000/v1/models | "$TAU2PY" -c \
  'import json,sys; print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null)
if [ "$SERVED" != "$ALIAS" ]; then
  echo "== WRONG MODEL SERVED: erwartet Alias '$ALIAS', serviert '$SERVED' — abort." | tee -a "$LOG"; exit 1
fi
docker logs text2sql_vllm_teacher 2>&1 | grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' \
  | tail -1 | sed 's/^/   vLLM: /' | tee -a "$LOG"

# ---------------------------------------------------------------------------------------------
# 3) Generieren + Auswerten. BFCL_PROJECT_ROOT trennt die Modelle: bfcl legt result/ und score/
#    unter dem MODELL-KEY ab, der fuer alle unsere Laeufe identisch ist — zwei Laeufe im selben
#    Root wuerden sich gegenseitig ueberschreiben.
# ---------------------------------------------------------------------------------------------
export BFCL_PROJECT_ROOT="$ROOT" LOCAL_SERVER_PORT=8000

GEN_ARGS=(--model "$MODEL_KEY" --test-category "$CATS" --temperature "$TEMPERATURE"
          --num-threads "$NUM_THREADS" --skip-server-setup)
EVAL_ARGS=(--model "$MODEL_KEY" --test-category "$CATS")
if [ -n "${BFCL_RUN_IDS:-}" ]; then
  cp "$BFCL_RUN_IDS" "$ROOT/test_case_ids_to_generate.json"
  GEN_ARGS+=(--run-ids)          # liest die Datei aus BFCL_PROJECT_ROOT
  EVAL_ARGS+=(--partial-eval)    # ohne das bricht evaluate an den fehlenden IDs ab
  echo "-- ID-Modus: $(basename "$BFCL_RUN_IDS")" | tee -a "$LOG"
fi
[ "${BFCL_ALLOW_OVERWRITE:-0}" = "1" ] && GEN_ARGS+=(--allow-overwrite)

echo "-- generate: $(echo "$CATS" | tr ',' ' ' | wc -w) Kategorien, temp $TEMPERATURE, $NUM_THREADS Threads" | tee -a "$LOG"
"$BFCLBIN" generate "${GEN_ARGS[@]}" 2>&1 | tee -a "$LOG" | tail -5
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && echo "== GENERATE FAILED (exit $rc) — evaluate laeuft trotzdem ueber das Vorhandene" | tee -a "$LOG"

echo "-- evaluate" | tee -a "$LOG"
"$BFCLBIN" evaluate "${EVAL_ARGS[@]}" 2>&1 | tee -a "$LOG" | tail -5

# ---------------------------------------------------------------------------------------------
# 4) Report + MLflow. Der Report liefert Exit 1, wenn Items mit input_token_count 0 auftauchen —
#    kaputter Lauf, kein Modellergebnis.
# ---------------------------------------------------------------------------------------------
echo | tee -a "$LOG"
python3 evaluation/benchmarks/bfcl/bfcl_report.py --run "$ROOT" 2>&1 | tee -a "$LOG"
report_rc=${PIPESTATUS[0]}

if [ "${SKIP_MLFLOW:-0}" != "1" ]; then
  MLFLOW_TRACKING_URI="file://$REPO/mlruns" MLFLOW_ALLOW_FILE_STORE=true \
    "$TAU2PY" evaluation/benchmarks/bfcl/log_mlflow.py --run "$ROOT" \
      --experiment "${MLFLOW_EXPERIMENT:-bfcl_eval}" 2>&1 | tee -a "$LOG" | tail -2
fi

echo "==== BFCL $LABEL DONE $(date '+%F %T') -> $ROOT ====" | tee -a "$LOG"
exit "$report_rc"

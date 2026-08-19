#!/usr/bin/env bash
# ops/eval_bfcl.sh — BFCL-v4-Benchmark EINES Modells (Base, merged Student, beliebiger Checkpoint).
# Preflight -> serve -> bfcl generate -> bfcl evaluate -> Report -> MLflow -> Teardown.
#
#   bash ops/eval_bfcl.sh Qwen/Qwen3-4B qwen3-4b_base
#   bash ops/eval_bfcl.sh /app/data/final/checkpoints/db_bahn_traj_merged_qwen3-4b/ep3 qwen3-4b_sft-ep3
#   BFCL_CATEGORIES=non_live bash ops/eval_bfcl.sh Qwen/Qwen3-4B qwen3-4b_base   # gruppenweise
#
# MODEL ist der Pfad, wie ihn der vLLM-CONTAINER sieht (/app/data/... fuer lokale Checkpoints) oder
# eine HF-ID. LABEL benennt den Lauf (data/generated/eval/bfcl/<LABEL>/) nach der Grammatik
#
#     <modell>_<stand>[_<zweck>]
#
#   <modell>  Basis-Modell-Slug, >= 6 Zeichen, im Modellpfad/der HF-ID wiederauffindbar
#             (qwen3-4b, qwen3-8b, ...) — Modell ZUERST, damit `ls` die Laeufe gruppiert
#   <stand>   base | sft-ep3 | grpo-s2-ep1 | harness-validation | ...
#   <zweck>   optional: smoke-<name> | probe-<name> | r2 (bewusste Wiederholungsmessung)
#
# Semantik: GLEICHES Label = Resume DESSELBEN Laufs; eine bewusste Neumessung braucht ein neues
# Label (_r2). smoke-/probe-Label nach der Auswertung manuell loeschen — sie loggen nicht nach
# MLflow (s. BFCL_RUN_IDS), bleiben aber sonst liegen. Der Preflight WARNT bei Verstoessen
# (check_label), failt aber nicht. Beispiele: qwen3-4b_base, qwen3-4b_sft-ep3,
# qwen3-4b_sft-ep3_smoke-crossgroup, qwen3-4b_probe-conc48, qwen3-8b_harness-validation.
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
#   NUM_THREADS_FAST (48)       Client-Concurrency fuer non_live+live (kurze, unabhaengige
#                               Items). GEMESSEN 2026-08-18 (4B, GB10): Knie der t90-Kurve,
#                               1,7x schneller als 21; mit Qwen3-8B verifiziert.
#   NUM_THREADS_SLOW (21)       Concurrency fuer multi_turn+memory+Rest. multi_turn-Episoden
#                               sind SERIELLE Ketten — 48 ist dort 47 % LANGSAMER (per-Stream-
#                               Tempo sinkt, Wall haengt an der langsamsten Kette) und
#                               produzierte den einzigen infra-Fehler der Probe. Nicht blind
#                               hochdrehen; andere Modelle/Hardware -> neu messen.
#   NUM_THREADS (leer)          Explizit gesetzt = EINHEITLICH fuer alle Gruppen (altes
#                               Verhalten, schlaegt beide Defaults). Der Lauf macht dann
#                               weiterhin zwei Paesse, nur mit identischem Wert.
#   VLLM_MAX_MODEL_LEN (leer)   Default: der Preflight leitet max_position_embeddings des
#                               SERVIERTEN Modells ab (steht danach im Manifest). Zu klein =
#                               stille Nullergebnisse, zu gross = vLLM startet nicht; der
#                               Preflight prueft beide Richtungen. S. Kommentar am serve-Block.
#   BFCL_MODEL_KEY (leer)       ECHTER bfcl-Registry-Key -> keine Injektion (Harness-Validierung
#                               gegen offizielle Leaderboard-Zeilen, z. B. Qwen/Qwen3-8B-FC).
#                               Default: Key wird abgeleitet (local/<slug>-FC) und ein Eintrag
#                               injiziert, dessen model_name das servierte Modell IST — damit
#                               budgetiert bfcl max_tokens gegen das WAHRE Kontextfenster
#                               (s. registry_inject.py; vorher: 262.144-Falle des geliehenen
#                               Instruct-2507-Keys -> stille 0 % in multi_turn_long_context).
#   HF_HOME (/data/hf_cache)    Cache-Sicht des Harness-venv: bfcl laedt AutoConfig/AutoTokenizer
#                               des model_name auf dem HOST — ohne HF_HOME schaut es in ~/.cache,
#                               wo unsere Modelle fehlen.
#   VLLM_GPU_UTIL (0.85)   BFCL_ENABLE_WEB_SEARCH (0)   BFCL_ALLOW_OVERWRITE (0)
#   BFCL_RUN_IDS (leer)         Pfad zu einer test_case_ids_to_generate.json -> nur diese IDs
#                               (Smoke-/Probe-Modus, zieht automatisch --partial-eval nach und
#                               setzt SKIP_MLFLOW=1, damit Smokes mlruns nicht verschmutzen).
#   MLFLOW_EXPERIMENT (bfcl_eval)   SKIP_MLFLOW (0; 1 bei BFCL_RUN_IDS)   BFCLPY (.venv-bfcl/bin/python)
set -uo pipefail   # KEIN -e: `compose down` scheitert legitim ohne Container, und `timeout` liefert 124
cd "$(dirname "$0")/.."
REPO=$(pwd)
COMPOSE="docker compose -f docker/docker-compose.yml"
BFCLPY=${BFCLPY:-$REPO/.venv-bfcl/bin/python}   # bfcl laeuft via run_bfcl.py-Shim (Injektion)
TAU2PY=${TAU2PY:-$REPO/.venv-tau2/bin/python}   # traegt mlflow (das bfcl-venv nicht)

MODEL=${1:?usage: eval_bfcl.sh <MODEL> <LABEL> [CATEGORIES]}
LABEL=${2:?usage: eval_bfcl.sh <MODEL> <LABEL> [CATEGORIES]}
CATEGORIES=${3:-${BFCL_CATEGORIES:-full}}
TEMPERATURE=${TEMPERATURE:-0.6}
# Concurrency-Defaults (48/21) leben im Preflight — hier nur Overrides durchreichen.
ROOT="$REPO/data/generated/eval/bfcl/$LABEL"

# Cache-Sicht des Harness angleichen: bfcl laedt AutoConfig/AutoTokenizer des Registry-model_name
# im HOST-venv (immer, auch mit --skip-server-setup). Ohne HF_HOME schaut transformers in
# ~/.cache/huggingface — dort fehlen unsere Modelle, der Lauf stuerbe erst beim generate.
export HF_HOME=${HF_HOME:-/data/hf_cache}

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
         --temperature "$TEMPERATURE")
[ -n "${NUM_THREADS:-}" ] && PF_ARGS+=(--num-threads "$NUM_THREADS")
[ -n "${NUM_THREADS_FAST:-}" ] && PF_ARGS+=(--num-threads-fast "$NUM_THREADS_FAST")
[ -n "${NUM_THREADS_SLOW:-}" ] && PF_ARGS+=(--num-threads-slow "$NUM_THREADS_SLOW")
[ -n "${VLLM_MAX_MODEL_LEN:-}" ] && PF_ARGS+=(--max-model-len "$VLLM_MAX_MODEL_LEN")
[ -n "${BFCL_MODEL_KEY:-}" ] && PF_ARGS+=(--model-key "$BFCL_MODEL_KEY")
[ "${BFCL_ENABLE_WEB_SEARCH:-0}" = "1" ] && PF_ARGS+=(--enable-web-search)
[ -n "${BFCL_RUN_IDS:-}" ] && PF_ARGS+=(--run-ids "$BFCL_RUN_IDS")
"$BFCLPY" evaluation/benchmarks/bfcl/preflight.py "${PF_ARGS[@]}" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -ne 0 ] && { echo "== PREFLIGHT FAILED — abort."; exit 1; }

CATS=$(cat "$ROOT/categories.txt")
CATS_FAST=$(cat "$ROOT/categories_fast.txt")
CATS_SLOW=$(cat "$ROOT/categories_slow.txt")
read -r MODEL_KEY ALIAS MAX_LEN MAX_SEQS NT_FAST NT_SLOW <<<"$("$TAU2PY" -c "
import json; m = json.load(open('$ROOT/run_manifest.json')); s = m['serving']
print(m['model_key'], m['model_alias'], s['max_model_len'], s['max_num_seqs'],
      s['num_threads_fast'], s['num_threads_slow'])")"

# ---------------------------------------------------------------------------------------------
# 2) Serven.
#    --served-model-name $ALIAS ist ZWINGEND: der Handler sendet `model=<ALIAS>` an
#    /v1/completions (ALIAS = model_name des Registry-Eintrags; bei injiziertem Key ist das per
#    Konstruktion das servierte Modell selbst). Ohne Alias 404 auf JEDEN Request — der Lauf
#    liefe durch und schriebe lauter Nullen.
#    KEIN --reasoning-parser: der QwenFCHandler parst <think> SELBST aus dem Completion-Text
#    (qwen_fc.py). Ein serverseitiger Parser wuerde das Feld wegnehmen -> think% waere ueberall 0.
#    --max-model-len: BFCL fordert max_tokens = min(4096, ctx - input - 2) an — seit der
#    Registry-Injektion gegen das WAHRE Fenster (das min() klemmt Riesen-Prompts jetzt selbst).
#    Zu klein heisst trotzdem NICHT "abgeschnitten", sondern der Request geht gar nicht durch und
#    landet als input_token_count 0 im Result — so sind im alten Quickrun 4 von 5
#    multi_turn_long_context-Episoden still als 0 % durchgelaufen.
# ---------------------------------------------------------------------------------------------
teardown; sleep 3
echo "-- serve: ctx=$MAX_LEN seqs=$MAX_SEQS alias=$ALIAS" | tee -a "$LOG"
VLLM_MODEL="$MODEL" VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}" VLLM_MAX_MODEL_LEN="$MAX_LEN" \
  VLLM_EXTRA_ARGS="--max-num-seqs $MAX_SEQS --served-model-name $ALIAS" \
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
# 3) Generieren + Auswerten — via run_bfcl.py, das den im Manifest festgehaltenen Registry-
#    Eintrag injiziert, BEVOR es an die bfcl-CLI delegiert. Beide Schritte brauchen die
#    Injektion: `evaluate` laeuft ueber die result-Verzeichnisse und schlaegt jeden Key im
#    Mapping nach (KeyError ohne Injektion). BFCL_PROJECT_ROOT trennt die Laeufe zusaetzlich
#    per Label (und muss VOR jedem bfcl_eval-Import gesetzt sein — mkdir beim Import).
# ---------------------------------------------------------------------------------------------
export BFCL_PROJECT_ROOT="$ROOT" LOCAL_SERVER_PORT=8000
BFCL_SHIM=(evaluation/benchmarks/bfcl/run_bfcl.py --manifest "$ROOT/run_manifest.json" --)

# Zwei generate-Paesse mit gemessener Gruppen-Concurrency (s. Header): fast = non_live+live,
# slow = Rest. Der Preflight hat die Buckets partitioniert; leere Buckets werden uebersprungen.
GEN_COMMON=(--model "$MODEL_KEY" --temperature "$TEMPERATURE" --skip-server-setup)
EVAL_ARGS=(--model "$MODEL_KEY" --test-category "$CATS")
if [ -n "${BFCL_RUN_IDS:-}" ]; then
  cp "$BFCL_RUN_IDS" "$ROOT/test_case_ids_to_generate.json"
  GEN_COMMON+=(--run-ids)        # liest die Datei aus BFCL_PROJECT_ROOT
  EVAL_ARGS+=(--partial-eval)    # ohne das bricht evaluate an den fehlenden IDs ab
  echo "-- ID-Modus: $(basename "$BFCL_RUN_IDS")" | tee -a "$LOG"
fi
[ "${BFCL_ALLOW_OVERWRITE:-0}" = "1" ] && GEN_COMMON+=(--allow-overwrite)

gen_rc=0
for PASS in fast slow; do
  if [ "$PASS" = fast ]; then PCATS=$CATS_FAST; PNT=$NT_FAST; else PCATS=$CATS_SLOW; PNT=$NT_SLOW; fi
  [ -z "$PCATS" ] && continue
  echo "-- generate[$PASS]: $(echo "$PCATS" | tr ',' ' ' | wc -w) Kategorien, temp $TEMPERATURE, $PNT Threads" | tee -a "$LOG"
  "$BFCLPY" "${BFCL_SHIM[@]}" generate --test-category "$PCATS" --num-threads "$PNT" "${GEN_COMMON[@]}" 2>&1 | tee -a "$LOG" | tail -5
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { gen_rc=$rc; echo "== GENERATE[$PASS] FAILED (exit $rc) — weiter mit dem Rest" | tee -a "$LOG"; }
done
[ "$gen_rc" -ne 0 ] && echo "== GENERATE teilweise fehlgeschlagen — evaluate laeuft ueber das Vorhandene" | tee -a "$LOG"

echo "-- evaluate" | tee -a "$LOG"
"$BFCLPY" "${BFCL_SHIM[@]}" evaluate "${EVAL_ARGS[@]}" 2>&1 | tee -a "$LOG" | tail -5

# ---------------------------------------------------------------------------------------------
# 4) Report + MLflow. Der Report liefert Exit 1, wenn Items mit input_token_count 0 auftauchen —
#    kaputter Lauf, kein Modellergebnis.
# ---------------------------------------------------------------------------------------------
echo | tee -a "$LOG"
python3 evaluation/benchmarks/bfcl/bfcl_report.py --run "$ROOT" 2>&1 | tee -a "$LOG"
report_rc=${PIPESTATUS[0]}

# Smokes (ID-Modus) landen nicht in MLflow: ein 4-Item-Lauf neben echten 5.017-Item-Zeilen im
# selben Experiment waere Verschmutzung. Explizites SKIP_MLFLOW=0 uebersteuert.
[ -n "${BFCL_RUN_IDS:-}" ] && SKIP_MLFLOW=${SKIP_MLFLOW:-1}
if [ "${SKIP_MLFLOW:-0}" != "1" ]; then
  MLFLOW_TRACKING_URI="file://$REPO/mlruns" MLFLOW_ALLOW_FILE_STORE=true \
    "$TAU2PY" evaluation/benchmarks/bfcl/log_mlflow.py --run "$ROOT" \
      --experiment "${MLFLOW_EXPERIMENT:-bfcl_eval}" 2>&1 | tee -a "$LOG" | tail -2
fi

echo "==== BFCL $LABEL DONE $(date '+%F %T') -> $ROOT ====" | tee -a "$LOG"
exit "$report_rc"

#!/usr/bin/env bash
# ops/toolace_backfill.sh — Think-Backfill fuer den ToolACE-Leg.
# Serve (hybrider Qwen3.6-Teacher) -> backfill_toolace_think.py -> Teardown.
#
#   bash ops/toolace_backfill.sh                          # Volllauf ueber die ganze Vorauswahl
#   LIMIT=10 RAW=/tmp/x.jsonl OUT=/tmp/y.jsonl \
#     bash ops/toolace_backfill.sh                        # Smoke
#   PER_CLASS=25 CLASSES=multi,irrelevance,parallel2,parallel4plus \
#     RUN_NAME=pilot_100 bash ops/toolace_backfill.sh     # stratifizierter Pilot
#
# Das Sampling kommt BEWUSST aus config[trajectory] (= Qwen3.6-TEACHER-Rezept: temp 1.0, top_p 0.95,
# top_k 20, presence 1.5, thinking an). Anders als bei ops/eval_heldout.sh ist das hier der richtige
# Default — wir generieren mit dem Teacher, nicht mit dem Studenten. Das Skript druckt das aufgeloeste
# Rezept, damit eine stille Fehlkonfiguration auffaellt.
#
# Env-Overrides (Defaults in Klammern):
#   TEACHER (Qwen/Qwen3.6-35B-A3B)  CONCURRENCY (48)  LIMIT  PER_CLASS  CLASSES  SEED (42)
#   PRESELECT/RAW/OUT  RUN_NAME  MLFLOW_EXPERIMENT (toolace_think_backfill)  NO_MLFLOW (leer)
#   BACKFILL_TIMEOUT (43200)  VLLM_MAX_MODEL_LEN (20480)  VLLM_GPU_UTIL (0.85)
set -uo pipefail   # KEIN -e: `compose down` scheitert legitim ohne Container, `timeout` liefert 124
cd "$(dirname "$0")/.."
REPO=$(pwd)
COMPOSE="docker compose -f docker/docker-compose.yml"
TAU2PY=${TAU2PY:-$REPO/.venv-tau2/bin/python}

TEACHER=${TEACHER:-"Qwen/Qwen3.6-35B-A3B"}
CONCURRENCY=${CONCURRENCY:-48}          # Server --max-num-seqs folgt diesem Wert
PRESELECT=${PRESELECT:-data/generated/sdg/toolace_preselect.jsonl}
RAW=${RAW:-data/generated/sdg/toolace_backfill_raw.jsonl}
OUT=${OUT:-data/generated/legs/toolace_chat.jsonl}
SEED=${SEED:-42}

mkdir -p data/generated/legs data/generated/sdg logs
export MLFLOW_TRACKING_URI="file://$REPO/mlruns"   # derselbe Store wie Training-Container + UI

# Teardown gehoert diesem Skript: ohne ihn haelt vLLM 0.85 x 128 GB und jedes Folgetraining verhungert.
teardown() { $COMPOSE --profile vllm down vllm >/dev/null 2>&1; }
trap teardown EXIT

ARGS=(--config config/pipeline_config.yaml --preselect "$PRESELECT" --raw "$RAW" --out "$OUT"
      --seed "$SEED" --concurrency "$CONCURRENCY" --api-base http://localhost:8000/v1 --model "$TEACHER")
[ -n "${LIMIT:-}" ]     && ARGS+=(--limit "$LIMIT")
[ -n "${PER_CLASS:-}" ] && ARGS+=(--per-class "$PER_CLASS")
[ -n "${CLASSES:-}" ]   && ARGS+=(--classes "$CLASSES")
[ -z "${NO_MLFLOW:-}" ] && ARGS+=(--mlflow --mlflow-experiment "${MLFLOW_EXPERIMENT:-toolace_think_backfill}")
[ -n "${RUN_NAME:-}" ]  && ARGS+=(--mlflow-run-name "$RUN_NAME")

echo "==== TOOLACE BACKFILL — $(date '+%F %T') ===="
echo "   teacher : $TEACHER  conc=$CONCURRENCY"
echo "   in/out  : $PRESELECT -> $RAW -> $OUT"
teardown; sleep 3

# Hybrider GDN-Teacher -> --gdn-prefill-backend triton (wie ops/gen_traces.sh).
# KEIN --reasoning-parser: der schoebe das Think in ein eigenes Feld, waehrend wir es INLINE aus dem
# Content parsen (extract_think) — es kaeme sonst leer an und JEDER Record fiele als no_think durch.
VLLM_MODEL="$TEACHER" VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}" \
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-20480}" \
  VLLM_EXTRA_ARGS="--max-num-seqs $CONCURRENCY --gdn-prefill-backend triton ${VLLM_EXTRA:-}" \
  $COMPOSE --profile vllm up -d vllm >>logs/toolace_backfill.log 2>&1

for _ in $(seq 180); do
  curl -sf localhost:8000/health >/dev/null 2>&1 && break
  docker ps --format '{{.Names}}' | grep -q text2sql_vllm_teacher || break
  sleep 10
done
if ! curl -sf localhost:8000/health >/dev/null 2>&1; then
  echo "== SERVE FAILED — abort"; docker logs text2sql_vllm_teacher --tail 40 2>&1 | tail -40; exit 1
fi

# VLLM_MODEL reist nur als Inline-Env zu `compose up`. Kommt es nicht an, greift der Compose-Fallback,
# der Server laeuft klaglos mit dem FALSCHEN Modell und das Backfill schriebe lauter Fehlschlaege.
SERVED=$(curl -s localhost:8000/v1/models | "$TAU2PY" -c \
  'import json,sys; print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null)
if [ "$SERVED" != "$TEACHER" ]; then
  echo "== WRONG MODEL SERVED: angefordert '$TEACHER', serviert '$SERVED' — abort."; exit 1
fi
docker logs text2sql_vllm_teacher 2>&1 | grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' \
  | tail -1 | sed 's/^/   vLLM: /'

# KEIN rm -f auf $RAW: das Backfill ist resume-fest (append-only, Schluessel = _meta.id) — ein
# abgebrochener Lauf soll fortsetzen, nicht von vorn beginnen. Zum Neuanfang die Datei selbst loeschen.
timeout "${BACKFILL_TIMEOUT:-43200}" env PYTHONPATH="$REPO" LOGURU_LEVEL=ERROR "$TAU2PY" \
  data_pipeline/backfill_toolace_think.py "${ARGS[@]}" 2>&1 | tee -a logs/toolace_backfill.log | tail -25
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && echo "== BACKFILL FAILED (exit $rc; 3 = Circuit Breaker, 124 = BACKFILL_TIMEOUT) =="

echo "==== TOOLACE BACKFILL DONE $(date '+%F %T') -> $OUT ===="
# Exit-Code WEITERREICHEN: ohne dies endet das Skript mit dem Status des letzten echo, und ein
# gescheiterter Lauf saehe fuer einen Aufrufer (tmux-Wrapper, CI) wie ein erfolgreicher aus.
exit "$rc"

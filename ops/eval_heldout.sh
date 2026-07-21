#!/usr/bin/env bash
# ops/eval_heldout.sh — Heldout-Eval EINES Modells (Base, merged Student, beliebiger Checkpoint).
# Serve -> rollout.py auf dem Eval-Split -> Teardown -> Per-Template-Report.
#
#   bash ops/eval_heldout.sh Qwen/Qwen3-4B base_think
#   bash ops/eval_heldout.sh /app/data/final/checkpoints/db_bahn_traj_merged/ep2 after_ep2
#
# SINGLE-SHOT (--max-regen 0): genau EIN Versuch pro Aufgabe. Aeltere Heldout-Zahlen entstanden mit
# --max-regen 1 = bis zu 3 Versuche mit best-of-Auswahl; sie sind mit diesen NICHT vergleichbar.
#
# Env-Overrides (Defaults in Klammern):
#   SPLIT (heldout_eval)  EVAL_CONC (24)  ROLLOUT_TIMEOUT_S (1800)  EVAL_TIMEOUT (21600)
#   VLLM_MAX_MODEL_LEN (20480)  VLLM_GPU_UTIL (0.85)  TAU2PY  MLFLOW_EXPERIMENT (db_bahn_traj_eval)
set -uo pipefail   # KEIN -e: `compose down` scheitert legitim ohne Container, und `timeout` liefert 124
cd "$(dirname "$0")/.."
REPO=$(pwd)
COMPOSE="docker compose -f docker/docker-compose.yml"
TAU2PY=${TAU2PY:-$REPO/.venv-tau2/bin/python}

MODEL=${1:?usage: eval_heldout.sh <MODEL> <LABEL> [OUTFILE]}
LABEL=${2:?usage: eval_heldout.sh <MODEL> <LABEL> [OUTFILE]}
SPLIT=${SPLIT:-heldout_eval}
OUT=${3:-data/generated/db_traces_${SPLIT}_${LABEL}.jsonl}
# 75 % der von vLLM gemeldeten "Maximum concurrency", gemessen 2026-07-21 bei ctx 20480 / util 0.85:
#   Qwen3-4B  KV 663.424 Tok -> 32,39x -> 24     Qwen3-8B  KV 594.048 Tok -> 29,01x -> 21
# 21 = der bindende (kleinere) Fall, deckt damit jedes SLM bis ~10B ab, ohne ans Limit zu gehen.
# Neues Modell? Einmal servieren und `docker logs text2sql_vllm_teacher | grep 'Maximum concurrency'`.
EVAL_CONC=${EVAL_CONC:-21}
mkdir -p data/generated logs
export MLFLOW_TRACKING_URI="file://$REPO/mlruns"   # derselbe Store wie der Training-Container

# Dieses Skript ist EIGENTUEMER von Teardown und rm -f (beides lag frueher im Aufrufer):
#  - ohne Teardown bekaeme ein nachfolgendes Training keinen Speicher (vLLM haelt 0.85 x 128 GB)
#  - ohne rm -f wuerde rollout.py in eine vorhandene Datei RESUMEN (append + skip bekannter
#    (task_id, sample_idx)) und die Zahlen zweier Modelle vermischen
teardown() { $COMPOSE --profile vllm down vllm >/dev/null 2>&1; }
trap teardown EXIT

echo "==== EVAL $LABEL ($MODEL) auf $SPLIT — $(date '+%F %T') ===="
teardown; sleep 3

# NUR Standard-Transformer hier (Base 4B / merged Student) -> kein --gdn-prefill-backend,
# der gehoert zum hybriden Teacher in ops/gen_traces.sh.
VLLM_MODEL="$MODEL" VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}" \
  VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-20480}" \
  VLLM_EXTRA_ARGS="--max-num-seqs $EVAL_CONC" \
  $COMPOSE --profile vllm up -d vllm >>logs/eval_heldout.log 2>&1

for _ in $(seq 180); do
  curl -sf localhost:8000/health >/dev/null 2>&1 && break
  docker ps --format '{{.Names}}' | grep -q text2sql_vllm_teacher || break
  sleep 10
done
if ! curl -sf localhost:8000/health >/dev/null 2>&1; then
  echo "== SERVE FAILED — abort"; docker logs text2sql_vllm_teacher --tail 40 2>&1 | tail -40; exit 1
fi

# VLLM_MODEL reist nur als Inline-Env zu `compose up`; kommt es nicht an, greift der Compose-Fallback
# (woertlich Qwen/Qwen3-4B). Der Server liefe dann klaglos mit dem FALSCHEN Modell, rollout.py bekaeme
# 404 je Request und schriebe eine vollstaendige Datei aus lauter Nullen bei Exit 0.
SERVED=$(curl -s localhost:8000/v1/models | "$TAU2PY" -c \
  'import json,sys; print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null)
if [ "$SERVED" != "$MODEL" ]; then
  echo "== WRONG MODEL SERVED: angefordert '$MODEL', serviert '$SERVED' — abort."; exit 1
fi
docker logs text2sql_vllm_teacher 2>&1 | grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' \
  | tail -1 | sed 's/^/   vLLM: /'

rm -f "$OUT"
# Sampling-Flags WOERTLICH: der Config-Default traegt das Qwen3.6-TEACHER-Rezept (temp 1.0 /
# presence 1.5), nicht das Studenten-Rezept. Fehlt ein Flag, wird still falsch evaluiert.
# --max-regen 0 = genau ein Versuch (siehe rollout.py solve_task).
timeout "${EVAL_TIMEOUT:-21600}" env PYTHONPATH="$REPO" LOGURU_LEVEL=ERROR "$TAU2PY" \
  sdg_pipeline/db_bahn/rollout.py --config config/pipeline_config.yaml \
  --split "$SPLIT" --k 1 --api-base http://localhost:8000/v1 --model "$MODEL" --teacher-name "$LABEL" \
  --enable-thinking --temperature 0.6 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 0.0 \
  --max-turns 15 --max-tokens-per-turn 3072 --max-regen 0 \
  --concurrency "$EVAL_CONC" --rollout-timeout-s "${ROLLOUT_TIMEOUT_S:-1800}" \
  --mlflow --mlflow-experiment "${MLFLOW_EXPERIMENT:-db_bahn_traj_eval}" --mlflow-run-name "$LABEL" \
  --output "$OUT" 2>&1 | tee -a logs/eval_heldout.log | tail -12
rc=${PIPESTATUS[0]}
[ "$rc" -ne 0 ] && echo "== ROLLOUT FAILED (exit $rc; 124 = EVAL_TIMEOUT) =="

echo
"$TAU2PY" evaluation/eval_report.py --input "$OUT" | tee -a logs/eval_heldout.log
echo "==== EVAL $LABEL DONE $(date '+%F %T') -> $OUT ===="

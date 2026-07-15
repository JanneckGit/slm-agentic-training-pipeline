#!/usr/bin/env bash
# ops/gen_traces.sh — Welle-2 Trace-Generierung (Etappe 3, DATEN NUR — kein Training/Eval).
# Serve Sieger-Teacher -> rollout sft_train (k=1, branch-on-fail) -> k=2 Top-up NUR auf den in
# Pass 1 gescheiterten Tasks -> format_traj -> data/final/db_traces_chat.jsonl.
# Alles nach MLflow (Experiment db_bahn_traj_gen) + weiterhin ins Trace-JSONL (Quelle der Wahrheit).
#
# LÄUFT MEHRERE STUNDEN — detached starten, z. B.:
#   nohup ops/gen_traces.sh >logs/gen_traces.main.log 2>&1 &
#
# Env-Overrides (Defaults in Klammern):
#   TEACHER (Qwen/Qwen3.6-35B-A3B)  SHORT (q36-35b-a3b)  SPLIT (sft_train)
#   TOPUP (1 = k=2-Top-up an; 0 = nur k=1)   MAX_TURNS (12)   CONCURRENCY (4)
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
COMPOSE="docker compose -f docker/docker-compose.yml"
TAU2PY=${TAU2PY:-$REPO/.venv-tau2/bin/python}

TEACHER=${TEACHER:-"Qwen/Qwen3.6-35B-A3B"}       # bake-off winner: 92% yield, ~16s/rollout
SHORT=${SHORT:-"q36-35b-a3b"}
SPLIT=${SPLIT:-"sft_train"}
TOPUP=${TOPUP:-1}
MAX_TURNS=${MAX_TURNS:-12}                        # headroom for wave-2 multi-tool + replan chains
CONCURRENCY=${CONCURRENCY:-4}                     # GB10 vLLM guidance (--max-num-seqs 4)
EXP="db_bahn_traj_gen"

TRACE="data/generated/db_traces_${SPLIT}_${SHORT}.jsonl"
FAILED="data/generated/_topup_failed_ids.txt"
CHAT="data/final/db_traces_chat.jsonl"
SPLITF="data/raw/db_sandbox/split_tasks.json"
mkdir -p data/generated data/final logs
# host eval/gen writes to the SAME physical store the training container mounts (../mlruns)
export MLFLOW_TRACKING_URI="file://$REPO/mlruns"

serve() {  # winner's flags, from ops/teacher_bakeoff.sh
  $COMPOSE --profile vllm down vllm >/dev/null 2>&1; sleep 3
  VLLM_MODEL="$TEACHER" VLLM_GPU_UTIL="0.85" VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}" \
    VLLM_EXTRA_ARGS="--max-num-seqs 4 --gdn-prefill-backend triton" \
    $COMPOSE --profile vllm up -d vllm >>logs/gen_traces.log 2>&1
  for _ in $(seq 120); do curl -sf localhost:8000/health >/dev/null 2>&1 && return 0; \
    docker ps --format '{{.Names}}' | grep -q text2sql_vllm_teacher || return 1; sleep 10; done
  curl -sf localhost:8000/health >/dev/null 2>&1
}

roll() {  # $1 = mlflow run-name ; $2.. = extra rollout flags (e.g. --k / --task-ids-file)
  local RUN="$1"; shift
  # 24h cap (was 12h — killed the 9.2k-task wave-2.5 pass 1 at ~48% on 2026-07-12; back then the
  # run "completed" silently because roll's exit code was not checked -> now we HARD-FAIL here).
  timeout 86400 env PYTHONPATH="$REPO" LOGURU_LEVEL=ERROR "$TAU2PY" \
    sdg_pipeline/db_bahn/rollout.py --config config/pipeline_config.yaml \
    --split "$SPLIT" --model "$TEACHER" --teacher-name "$SHORT" \
    --api-base http://localhost:8000/v1 \
    --branch-on-fail --max-turns "$MAX_TURNS" --max-tokens-per-turn 1536 --concurrency "$CONCURRENCY" \
    --mlflow --mlflow-experiment "$EXP" --mlflow-run-name "$RUN" \
    --output "$TRACE" "$@" || {
      local rc=$?
      echo "== ROLLOUT '$RUN' FAILED (exit $rc; 124 = 24h timeout) — ABORT."
      echo "   Rollout is resume-safe: fix/wait, then rerun this script to continue where it stopped."
      $COMPOSE --profile vllm down vllm >/dev/null 2>&1
      exit $rc
    }
}

echo "==== GEN-TRACES START $(date) — teacher=$SHORT split=$SPLIT topup=$TOPUP ===="

echo "== [1/4] serve teacher $TEACHER =="
if ! serve; then
  echo "== SERVE FAILED — abort"; docker logs text2sql_vllm_teacher --tail 40 2>&1 | tail -40; exit 1
fi

echo "== [2/4] PASS 1: rollout $SPLIT k=1 (branch-on-fail) =="
roll "gen_k1_branch" --k 1

if [ "$TOPUP" = "1" ]; then
  N_FAIL=$("$TAU2PY" - <<PY
import json, collections
sft = set(json.load(open("$SPLITF"))["$SPLIT"])          # only top up tasks still in this split
best = collections.defaultdict(float)
for ln in open("$TRACE"):
    r = json.loads(ln); best[r["task_id"]] = max(best[r["task_id"]], r["score"]["score"])
failed = sorted(t for t, s in best.items() if s < 1.0 and t in sft)
open("$FAILED", "w").write("\n".join(failed))
print(len(failed))
PY
)
  echo "== [3/4] TOP-UP: $N_FAIL tasks failed pass 1 -> k=2 second sample on that subset =="
  if [ "${N_FAIL:-0}" -gt 0 ]; then
    roll "gen_topup_k2" --task-ids-file "$FAILED" --k 2
  else
    echo "   (nothing to top up — pass 1 verified everything)"
  fi
else
  echo "== [3/4] TOP-UP skipped (TOPUP=0) =="
fi

echo "== teardown vllm =="
$COMPOSE --profile vllm down vllm >/dev/null 2>&1

echo "== [4/4] format verified traces -> $CHAT =="
env PYTHONPATH="$REPO" "$TAU2PY" data_pipeline/format_traj_for_training.py --input "$TRACE" \
  --split-file "$SPLITF" --split "$SPLIT" --output "$CHAT" || {
    echo "== FORMAT FAILED (exit $?) — ABORT (traces in $TRACE are intact, rerun step 4 after fixing)."
    exit 1
  }
KEPT=$(wc -l < "$CHAT" 2>/dev/null || echo 0)

echo "==== GEN-TRACES DONE $(date) ===="
echo "  raw rollouts   : $(wc -l < "$TRACE") records -> $TRACE"
echo "  verified traces: $KEPT -> $CHAT   (this is the SFT training input)"
echo "  MLflow         : experiment '$EXP' (runs gen_k1_branch[, gen_topup_k2]) @ $MLFLOW_TRACKING_URI"

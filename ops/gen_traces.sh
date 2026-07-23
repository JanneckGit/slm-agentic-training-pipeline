#!/usr/bin/env bash
# ops/gen_traces.sh — Welle-2 Trace-Generierung (Etappe 3, DATEN NUR — kein Training/Eval).
# Serve Sieger-Teacher -> rollout sft_train (k=1, branch-on-fail) -> k=2 Top-up NUR auf den in
# Pass 1 gescheiterten Tasks -> format_traj -> data/generated/legs/db_traces_chat.jsonl.
# Alles nach MLflow (Experiment db_bahn_traj_gen) + weiterhin ins Trace-JSONL (Quelle der Wahrheit).
#
# LÄUFT MEHRERE STUNDEN — detached starten, z. B.:
#   nohup ops/gen_traces.sh >logs/gen_traces.main.log 2>&1 &
#
# Env-Overrides (Defaults in Klammern):
#   TEACHER (Qwen/Qwen3.6-35B-A3B)  SHORT (q36-35b-a3b)  SPLIT (sft_train)
#   TOPUP (1 = k=2-Top-up an; 0 = nur k=1)   MAX_TURNS (14)   CONCURRENCY (48)
#   ROLL_TIMEOUT (259200 = 72h Hard-Cap pro Rollout-Pass)
#   ROLLOUT_TIMEOUT_S (1800 = Pro-Rollout-Budget, zwischen Turns geprueft). MIT CONCURRENCY MITHEBEN:
#     hoehere Concurrency => hoehere Per-Turn-Latenz => sonst werden Rollouts als score=0 gekappt.
#     Genau das war der "conc 12 ist langsamer"-Befund aus Smoke S4 (2026-07-19 widerlegt, s. u.).
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
COMPOSE="docker compose -f docker/docker-compose.yml"
TAU2PY=${TAU2PY:-$REPO/.venv-tau2/bin/python}

TEACHER=${TEACHER:-"Qwen/Qwen3.6-35B-A3B"}       # bake-off winner: 92% yield, ~16s/rollout
# wave-3 default is a NEW trace file (-w3): the old q36-35b-a3b file holds wave-2.5 NO-THINK
# traces whose task_ids partially overlap — resuming into it would silently mix them.
SHORT=${SHORT:-"q36-35b-a3b-w3"}
SPLIT=${SPLIT:-"sft_train"}
TOPUP=${TOPUP:-1}
MAX_TURNS=${MAX_TURNS:-14}                        # wave-3: iteration chains + transient retries
# GB10-Messung 2026-07-19 (20-min-Fenster, Wave-3.5-Volllauf), Qualitaet in allen Faellen 100% verified:
#   conc  8 / timeout  300 = 147 Tasks/h   (alter "smoke-S4-Sieger" — war der 300s-Cap, nicht die HW)
#   conc 24 / timeout  900 = 288 Tasks/h   (Skalierungs-Effizienz 65%)
#   conc 48 / timeout 1800 = 486 Tasks/h   (84%) <- hier, danach queued der Server (Waiting>0)
# Nicht hoeher: der vLLM-KV-Pool ist zwar reserviert, seine Seiten materialisieren sich aber erst bei
# Benutzung (+3 GB Host-RAM je +12pp KV) — conc 64 landet bei ~117 von 119 GB. Vorher VLLM_GPU_UTIL senken.
CONCURRENCY=${CONCURRENCY:-48}                    # server --max-num-seqs folgt diesem Wert
EXP="db_bahn_traj_gen"

TRACE="data/generated/sdg/db_traces_${SPLIT}_${SHORT}.jsonl"
FAILED="data/generated/sdg/_topup_failed_ids.txt"
CHAT="data/generated/legs/db_traces_chat.jsonl"
# Split-Tasks, die es in KEINE Chat-Trace geschafft haben -> trainiert-ungesehen, damit Kandidaten
# fuer den Stage-2-GRPO-Pool (build_grpo_pool.py loest eine blanke task_id gegen tasks/answer_keys auf).
FAILED_RL="data/generated/sdg/db_failed-for-SFT_rl-candidates.jsonl"
SPLITF="data/raw/db_sandbox/split_tasks.json"
mkdir -p data/generated/legs data/generated/sdg data/final logs
# host eval/gen writes to the SAME physical store the training container mounts (../mlruns)
export MLFLOW_TRACKING_URI="file://$REPO/mlruns"

serve() {  # winner's flags, from ops/teacher_bakeoff.sh; wave-3: 20480 ctx (inline <think> grows
  # the dialog), --max-num-seqs coupled to CONCURRENCY (hybrid-GDN KV is ~20 KB/token -> cheap)
  $COMPOSE --profile vllm down vllm >/dev/null 2>&1; sleep 3
  # GPU_UTIL steuert die Groesse des vLLM-Pools. Dessen Seiten materialisieren sich erst bei Benutzung
  # (~3 GB Host-RAM je +12pp KV-Nutzung), d.h. er ist die Obergrenze fuer CONCURRENCY: wer hoeher als
  # 48 will, muss hier runter, sonst reicht der Host-RAM (119 GB) nicht. KV-Bedarf ist klein — bei
  # conc 48 nur ~6 GB von ~25 GB Pool —, ein kleinerer Pool kostet also praktisch nichts.
  VLLM_MODEL="$TEACHER" VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}" VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-20480}" \
    VLLM_EXTRA_ARGS="--max-num-seqs $CONCURRENCY --gdn-prefill-backend triton" \
    $COMPOSE --profile vllm up -d vllm >>logs/gen_traces.log 2>&1
  for _ in $(seq 120); do curl -sf localhost:8000/health >/dev/null 2>&1 && return 0; \
    docker ps --format '{{.Names}}' | grep -q text2sql_vllm_teacher || return 1; sleep 10; done
  curl -sf localhost:8000/health >/dev/null 2>&1
}

roll() {  # $1 = mlflow run-name ; $2.. = extra rollout flags (e.g. --k / --task-ids-file)
  local RUN="$1"; shift
  # 72h cap (12h killed the 9.2k-task wave-2.5 pass 1 at ~48% on 2026-07-12, and 24h would not
  # cover wave 3.5 either: 13.9k tasks at ~245 tasks/h with thinking on is a ~57h pass 1).
  # A cap firing is a HARD FAIL (roll's exit code was unchecked back then -> run "completed"
  # silently). The real hang detector is progress-based, not wall-clock — see ops/watch_gen.sh.
  timeout "${ROLL_TIMEOUT:-259200}" env PYTHONPATH="$REPO" LOGURU_LEVEL=ERROR "$TAU2PY" \
    sdg_pipeline/db_bahn/rollout.py --config config/pipeline_config.yaml \
    --split "$SPLIT" --model "$TEACHER" --teacher-name "$SHORT" \
    --api-base http://localhost:8000/v1 \
    --branch-on-fail --max-turns "$MAX_TURNS" --max-tokens-per-turn 3072 --concurrency "$CONCURRENCY" \
    --rollout-timeout-s "${ROLLOUT_TIMEOUT_S:-1800}" \
    --mlflow --mlflow-experiment "$EXP" --mlflow-run-name "$RUN" \
    --output "$TRACE" "$@" || {
      local rc=$?
      echo "== ROLLOUT '$RUN' FAILED (exit $rc; 124 = ROLL_TIMEOUT hit) — ABORT."
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
# wave-3 accept semantics (mirror of rollout.accepted): verified AND not token-capped AND not
# degenerate — a verified-but-degenerate trace must be re-rolled, format_traj would drop it.
best = collections.defaultdict(float)
for ln in open("$TRACE"):
    r = json.loads(ln)
    d = r.get("degen") or {}
    ok = (r["score"]["score"] == 1.0 and not r.get("truncated")
          and d.get("think_ngram_dup_ratio", 0.0) <= 0.5
          and d.get("max_think_chars", 0) <= 12000)
    best[r["task_id"]] = max(best[r["task_id"]], 1.0 if ok else 0.0)
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
  --split-file "$SPLITF" --split "$SPLIT" --output "$CHAT" --dropped-out "$FAILED_RL" || {
    echo "== FORMAT FAILED (exit $?) — ABORT (traces in $TRACE are intact, rerun step 4 after fixing)."
    exit 1
  }
KEPT=$(wc -l < "$CHAT" 2>/dev/null || echo 0)

echo "==== GEN-TRACES DONE $(date) ===="
echo "  raw rollouts   : $(wc -l < "$TRACE") records -> $TRACE"
echo "  verified traces: $KEPT -> $CHAT   (this is the SFT training input)"
echo "  RL candidates  : $(wc -l < "$FAILED_RL" 2>/dev/null || echo 0) -> $FAILED_RL   (split tasks in no chat trace)"
echo "  MLflow         : experiment '$EXP' (runs gen_k1_branch[, gen_topup_k2]) @ $MLFLOW_TRACKING_URI"

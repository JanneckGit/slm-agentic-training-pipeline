#!/usr/bin/env bash
# =============================================================================
# tau2-bench eval — Doppel-Serving (Agent + User-Sim) + Preflight + Report + MLflow
#
#   bash ops/eval_tau2.sh <MODEL> <LABEL> [DOMAINS]
#
#   MODEL    Agent-Modell, wie der vLLM-Container es sieht (HF-ID oder /app/data/…)
#   LABEL    Lauf-Verzeichnis + MLflow-Run-Name; Grammatik <modell>_<stand>[_<zweck>]
#   DOMAINS  'full' (Default) oder Kommaliste: telecom,airline,banking_knowledge,retail
#
# Beispiele:
#   bash ops/eval_tau2.sh Qwen/Qwen3-4B qwen3-4b_base                       # Volllauf
#   TAU2_NUM_TASKS=3 TAU2_TRIALS=1 bash ops/eval_tau2.sh Qwen/Qwen3-4B smoke-telecom telecom
#
# Env-Knoepfe:
#   TAU2_TRIALS / TAU2_MC          Trials / max-concurrency (Default aus benchmark_config.yaml)
#   TAU2_NUM_TASKS / TAU2_TASK_IDS Smoke-Teilmengen (erzwingen SKIP_MLFLOW=1)
#   TAU2_GT_GATE=0                 Oracle-Eichung (mock, llm_agent_gt) ueberspringen (Default: an)
#   TAU2_GT_ONLY=1                 NUR die Oracle-Eichung fahren, dann Ende
#   TAU2_AGENT_PARSER_ARGS / TAU2_USER_PARSER_ARGS   vLLM-Parser-Flags pro Instanz
#                                  (Defaults aus benchmark_config: agent=hermes, sim=qwen3_xml)
#   TAU2_BASELINE=<root>           Δ-Spalte im Report gegen einen Vergleichslauf
#   VLLM_GPU_UTIL / VLLM_USER_GPU_UTIL   GPU-Budgets (Defaults aus dem Manifest; Summe <= ~0.85!)
#   SKIP_MLFLOW=1                  kein MLflow-Log (bei Smokes automatisch)
#
# WICHTIG (Gegenteil der BFCL-Regel!): tau2 nutzt die Chat-API mit nativer tools-API
# und strippt <think> NIRGENDS -> beide vLLM-Instanzen laufen MIT
# --reasoning-parser + --tool-call-parser. Der Report failt bei <think> im content.
#
# Lange Laeufe IMMER in tmux (SSH-Abbruch toetet Hintergrund-Tasks der Agent-Session):
#   tmux new -s tau2 'bash ops/eval_tau2.sh Qwen/Qwen3-4B qwen3-4b_base 2>&1 | tee -a logs/tau2.log'
# Resume = derselbe Befehl (--auto-resume ist gesetzt; fertige Sims werden uebersprungen).
# =============================================================================
set -uo pipefail   # KEIN -e: compose down scheitert legitim ohne Container
cd "$(dirname "$0")/.."
REPO=$(pwd)
COMPOSE="docker compose -f docker/docker-compose.yml"
T2BPY=${T2BPY:-$REPO/.venv-tau2bench/bin/python}   # Harness-venv (tau2==1.0.1)
TAU2PY=${TAU2PY:-$REPO/.venv-tau2/bin/python}      # traegt mlflow (Harness-venv nicht)
# tau2s Loader bauen selbst DATA_DIR/tau2/domains/… -> auf …/data zeigen, NICHT …/data/tau2
export TAU2_DATA_DIR="$REPO/data/raw/tau2-bench/data"
export HF_HOME=${HF_HOME:-/data/hf_cache}          # Preflight loest gen-configs ueber HF_HOME auf

MODEL=${1:?usage: eval_tau2.sh <MODEL> <LABEL> [DOMAINS]}
LABEL=${2:?usage: eval_tau2.sh <MODEL> <LABEL> [DOMAINS]}
DOMAINS=${3:-${TAU2_DOMAINS:-full}}
ROOT="$REPO/data/generated/eval/tau2/$LABEL"
mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/run.log"
BENCH="evaluation/benchmarks/tau2"

teardown() {
  $COMPOSE --profile vllm down vllm >/dev/null 2>&1
  $COMPOSE --profile vllm-user down vllm-user >/dev/null 2>&1
}
trap teardown EXIT

echo "== tau2-bench | model=$MODEL | label=$LABEL | domains=$DOMAINS" | tee -a "$LOG"

# ---------------------------------------------------------------- Preflight --
PF_ARGS=(--model "$MODEL" --label "$LABEL" --root "$ROOT" --domains "$DOMAINS")
[ -n "${TAU2_TRIALS:-}" ]    && PF_ARGS+=(--num-trials "$TAU2_TRIALS")
[ -n "${TAU2_MC:-}" ]        && PF_ARGS+=(--max-concurrency "$TAU2_MC")
[ -n "${TAU2_TOTAL_CAP:-}" ] && PF_ARGS+=(--total-cap "$TAU2_TOTAL_CAP")
[ -n "${TAU2_MAX_STEPS:-}" ] && PF_ARGS+=(--max-steps "$TAU2_MAX_STEPS")
"$T2BPY" "$BENCH/preflight.py" "${PF_ARGS[@]}" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" -ne 0 ] && { echo "== PREFLIGHT FAILED — abort."; exit 1; }

MANIFEST="$ROOT/run_manifest.json"
mread() { "$TAU2PY" -c "import json; m=json.load(open('$MANIFEST')); print($1)"; }
ALIAS=$(mread "m['model_alias']")
AGENT_ARGS_JSON=$(mread "json.dumps(m['agent']['llm_args'])")
USER_LLM=$(mread "m['user_sim']['llm']")
USER_ARGS_JSON=$(mread "json.dumps(m['user_sim']['llm_args'])")
U_MODEL=$(mread "m['user_sim']['model']")
U_EXTRA=$(mread "m['user_sim']['extra_args']")
read -r A_MML A_SEQS A_UTIL U_MML U_SEQS U_UTIL TRIALS SEED MC MAXSTEPS MAXERR <<<"$(mread \
  "m['agent']['max_model_len'], m['agent']['max_num_seqs'], m['agent']['gpu_util'], \
   m['user_sim']['max_model_len'], m['user_sim']['max_num_seqs'], m['user_sim']['gpu_util'], \
   m['run']['num_trials'], m['run']['seed'], m['run']['max_concurrency'], \
   m['run']['max_steps'], m['run']['max_errors']" | tr -d ',()')"

# Parser sind PRO INSTANZ verschieden: Qwen3-4B/8B = hermes-JSON, Qwen3.6 = qwen3_xml (Smoke-Befund)
A_TCP=$(mread "m['agent']['tool_call_parser']"); A_RP=$(mread "m['agent']['reasoning_parser']")
U_TCP=$(mread "m['user_sim']['tool_call_parser']"); U_RP=$(mread "m['user_sim']['reasoning_parser']")
AGENT_PARSER_ARGS=${TAU2_AGENT_PARSER_ARGS:-"--enable-auto-tool-choice --tool-call-parser $A_TCP --reasoning-parser $A_RP"}
USER_PARSER_ARGS=${TAU2_USER_PARSER_ARGS:-"--enable-auto-tool-choice --tool-call-parser $U_TCP --reasoning-parser $U_RP"}

# ------------------------------------------------------------ Doppel-Serving --
wait_health() { # port container
  local port=$1 cname=$2
  for _ in $(seq 180); do
    curl -sf "localhost:$port/health" >/dev/null 2>&1 && return 0
    docker ps --format '{{.Names}}' | grep -q "$cname" || {
      echo "== $cname ist weg — EngineCore-Fehler + letzte Logs:" | tee -a "$LOG"
      docker logs "$cname" 2>&1 | grep -B3 -A12 "EngineCore failed\|ERROR " | tail -40 >> "$LOG" 2>/dev/null
      docker logs "$cname" 2>&1 | tail -80 >> "$LOG" 2>/dev/null
      return 1; }
    sleep 10
  done
  echo "== $cname Health-Timeout — letzte Logs:" | tee -a "$LOG"
  docker logs "$cname" 2>&1 | tail -80 >> "$LOG" 2>/dev/null
  return 1
}

# SEQUENZIELL serven (Agent gesund -> dann Sim): beide Instanzen profilieren beim Start
# den freien Speicher — paralleles Laden auf unified memory laesst eine mit negativem
# KV-Budget sterben (Smoke-Befund 2026-08-19). sleep 10: unified memory gibt eine
# 76-GB-Instanz nach compose down nicht in 3 s frei (Start-Race, Smoke 2026-08-20).
teardown; sleep 10
echo "== Serve Agent :8000 ($MODEL als '$ALIAS', util=${VLLM_GPU_UTIL:-$A_UTIL})" | tee -a "$LOG"
VLLM_MODEL="$MODEL" VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-$A_UTIL}" VLLM_MAX_MODEL_LEN="$A_MML" \
  VLLM_EXTRA_ARGS="--max-num-seqs $A_SEQS --served-model-name $ALIAS $AGENT_PARSER_ARGS" \
  $COMPOSE --profile vllm up -d vllm >>"$LOG" 2>&1
wait_health 8000 text2sql_vllm_teacher || { echo "== Agent-vLLM kam nicht hoch"; exit 1; }

echo "== Serve User-Sim :8001 ($U_MODEL, util=${VLLM_USER_GPU_UTIL:-$U_UTIL})" | tee -a "$LOG"
VLLM_USER_MODEL="$U_MODEL" VLLM_USER_GPU_UTIL="${VLLM_USER_GPU_UTIL:-$U_UTIL}" \
  VLLM_USER_MAX_MODEL_LEN="$U_MML" \
  VLLM_USER_EXTRA_ARGS="--max-num-seqs $U_SEQS $U_EXTRA $USER_PARSER_ARGS" \
  $COMPOSE --profile vllm-user up -d vllm-user >>"$LOG" 2>&1
wait_health 8001 text2sql_vllm_user    || { echo "== User-vLLM kam nicht hoch"; exit 1; }

# Identity-Gates: es muss der ALIAS bzw. das Sim-Modell antworten, nicht ein Fallback.
curl -sf localhost:8000/v1/models | grep -qF "\"$ALIAS\"" || {
  echo "== /v1/models:8000 kennt '$ALIAS' nicht — falsches Modell serviert?"; exit 1; }
curl -sf localhost:8001/v1/models | grep -qF "\"$U_MODEL\"" || {
  echo "== /v1/models:8001 kennt '$U_MODEL' nicht — falsches Sim-Modell?"; exit 1; }
docker logs text2sql_vllm_teacher 2>&1 | grep -m1 "Maximum concurrency" | tee -a "$LOG" || true
docker logs text2sql_vllm_user    2>&1 | grep -m1 "Maximum concurrency" | tee -a "$LOG" || true

SHIM=("$T2BPY" "$BENCH/run_tau2.py" --manifest "$MANIFEST" --)
LLM_ARGS=(--agent-llm "openai/$ALIAS" --agent-llm-args "$AGENT_ARGS_JSON"
          --user-llm "$USER_LLM" --user-llm-args "$USER_ARGS_JSON")

# ------------------------------------------------- Oracle-Eichung (GT-Agent) --
if [ "${TAU2_GT_GATE:-1}" = "1" ]; then
  echo "== GT-Oracle (mock, llm_agent_gt): Harness-Eichung" | tee -a "$LOG"
  "${SHIM[@]}" run --domain mock --agent llm_agent_gt "${LLM_ARGS[@]}" \
    --num-trials 1 --seed "$SEED" --max-steps "$MAXSTEPS" --max-errors "$MAXERR" \
    --max-concurrency "$MC" --save-to "$ROOT/oracle_mock" --auto-resume \
    2>&1 | tee -a "$LOG" | tail -5
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "== GT-Lauf selbst schlug fehl"; exit 1; }
  "$T2BPY" "$BENCH/tau2_report.py" --gt-check "$ROOT/oracle_mock" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" -ne 0 ] && { echo "== GT-Oracle FAILED — Harness pruefen, kein Modellbefund."; exit 1; }
  [ "${TAU2_GT_ONLY:-0}" = "1" ] && { echo "== TAU2_GT_ONLY — Ende nach Eichung."; exit 0; }
fi

# ------------------------------------------------------------- Domaenen-Loop --
gen_rc=0
for DOM in $(cat "$ROOT/domains.txt"); do
  read -r TSET TSPLIT RETR <<<"$(mread \
    "m['domains']['$DOM']['task_set'], m['domains']['$DOM'].get('task_split') or '-', \
     m['domains']['$DOM'].get('retrieval_config') or '-'" | tr -d ',()')"
  RUN_ARGS=(run --domain "$DOM" --task-set-name "$TSET" "${LLM_ARGS[@]}"
            --num-trials "$TRIALS" --seed "$SEED" --max-steps "$MAXSTEPS"
            --max-errors "$MAXERR" --max-concurrency "$MC"
            --save-to "$ROOT/$DOM" --auto-resume)
  [ "$TSPLIT" != "-" ] && RUN_ARGS+=(--task-split-name "$TSPLIT")
  [ "$RETR"  != "-" ] && RUN_ARGS+=(--retrieval-config "$RETR")
  [ -n "${TAU2_NUM_TASKS:-}" ] && RUN_ARGS+=(--num-tasks "$TAU2_NUM_TASKS")
  # shellcheck disable=SC2086
  [ -n "${TAU2_TASK_IDS:-}" ] && RUN_ARGS+=(--task-ids $TAU2_TASK_IDS)
  echo "== RUN $DOM (set=$TSET split=$TSPLIT trials=$TRIALS conc=$MC)" | tee -a "$LOG"
  "${SHIM[@]}" "${RUN_ARGS[@]}" 2>&1 | tee -a "$LOG" | tail -8
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { gen_rc=$rc; echo "== RUN[$DOM] FAILED (exit $rc) — weiter mit dem Rest" | tee -a "$LOG"; }
done

teardown

# ---------------------------------------------------------- Report + MLflow --
"$T2BPY" "$BENCH/tau2_report.py" --run "$ROOT" \
  ${TAU2_BASELINE:+--baseline "$TAU2_BASELINE"} 2>&1 | tee -a "$LOG"
report_rc=${PIPESTATUS[0]}

case "$LABEL" in smoke*) SKIP_MLFLOW=${SKIP_MLFLOW:-1};; esac
[ -n "${TAU2_NUM_TASKS:-}${TAU2_TASK_IDS:-}" ] && SKIP_MLFLOW=${SKIP_MLFLOW:-1}
if [ "${SKIP_MLFLOW:-0}" != "1" ]; then
  MLFLOW_TRACKING_URI="file://$REPO/mlruns" MLFLOW_ALLOW_FILE_STORE=true \
    "$TAU2PY" "$BENCH/log_mlflow.py" --run "$ROOT" \
      --experiment "${MLFLOW_EXPERIMENT:-tau2_eval}" 2>&1 | tee -a "$LOG" | tail -2
fi

echo "== done: $ROOT | report_rc=$report_rc run_rc=$gen_rc" | tee -a "$LOG"
[ "$gen_rc" -ne 0 ] && exit "$gen_rc"
exit "$report_rc"

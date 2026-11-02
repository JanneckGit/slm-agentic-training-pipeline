#!/usr/bin/env bash
# =============================================================================
# After a Stage-2 GRPO run: export + K=8 heldout eval of fixed candidate checkpoints, each compared to a baseline.
#
#   bash ops/grpo_post_eval.sh <run-name> <step> [<step> ...]     # in tmux, may start while the run still trains
#
# 1) Waits until ops/grpo_run.sh of this run has ended and goes on only after "done rc=0"
#    (ALLOW_TRAIN_RC=1: any rc, e.g. for a run stopped by hand).
# 2) Prints the run's report; checkpoints from a divergence onset on are skipped (the rule of grpo_metrics.py --select).
# 3) Per candidate, after a GPU free-check: ops/grpo_export.sh, ops/eval_heldout.sh, evaluation/k8_class_compare.py.
# 4) Prints a summary. Nothing is selected: that stays a decision (grpo_metrics.py <run> --pick-step N --note ...).
#
# eval_heldout.sh exits 0 even when its rollouts failed and deletes existing traces unless RESUME=1, so this script
# checks the row count itself, continues a partial file with RESUME=1 (one retry) and never re-runs a complete one.
# Starting it again after an interruption is therefore safe. A failed candidate does not stop the next one.
#
# Env: BASE (traces to compare against, default r2 step 24), MIN_AVAIL_GB (100), ALLOW_TRAIN_RC (0), TAU2PY,
#      POLL_S (60, waiting for the run), GPU_POLL_S (300, waiting for the GPU), FREE_GAP_S (60, between the two probes).
# Output: logs/<run>.post_eval.log, traces data/generated/eval/db_traces_heldout_eval_after_<run>_step<N>_k8.jsonl,
#         comparisons data/generated/eval/grpo_post_eval/<run>_step<N>_vs_base.json
# =============================================================================
set -uo pipefail   # no -e: a failed candidate must not end the chain
cd "$(dirname "$0")/.."
REPO=$(pwd)
USAGE="usage: grpo_post_eval.sh <run-name> <step> [<step> ...]"
RUN=${1:?$USAGE}; shift
[ $# -gt 0 ] || { echo "$USAGE"; exit 2; }
for s in "$@"; do case "$s" in ''|*[!0-9]*) echo "not a step number: '$s'"; exit 2 ;; esac; done

K=8
TAU2PY=${TAU2PY:-$REPO/.venv-tau2/bin/python}
BASE=${BASE:-data/generated/eval/db_traces_heldout_eval_after_grpo_db_bahn_r2_step24_k8.jsonl}
MIN_AVAIL_GB=${MIN_AVAIL_GB:-100}
ALLOW_TRAIN_RC=${ALLOW_TRAIN_RC:-0}
POLL_S=${POLL_S:-60}; GPU_POLL_S=${GPU_POLL_S:-300}; FREE_GAP_S=${FREE_GAP_S:-60}
TRAIN_LOG=logs/${RUN}.log
RESULT_DIR=data/generated/eval/grpo_post_eval
mkdir -p logs "$RESULT_DIR"
exec > >(tee -a "logs/${RUN}.post_eval.log") 2>&1

rows() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }
# exact argv match: a pattern search would also hit any shell whose command line merely mentions the script
runner_alive() { ps -eo args | awk -v run="$RUN" '$1=="bash" && $2=="ops/grpo_run.sh" && $4==run {f=1} END {exit !f}'; }

# empty = free; otherwise what is in the way. Anything unreadable counts as not free.
gpu_problem() {
  local apps avail names
  runner_alive && { echo "ops/grpo_run.sh of $RUN is running again"; return; }
  apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>&1) || { echo "nvidia-smi failed"; return; }
  [ -z "$apps" ] || { echo "compute apps: $(echo "$apps" | tr '\n' ';')"; return; }
  avail=$(free -g | awk '/^Speicher|^Mem/{print $7}')
  case "$avail" in ''|*[!0-9]*) echo "available memory unreadable: '$avail'"; return ;; esac
  [ "$avail" -ge "$MIN_AVAIL_GB" ] || { echo "only $avail GB available (need $MIN_AVAIL_GB)"; return; }
  names=$(docker ps --format '{{.Names}}') || { echo "docker ps failed"; return; }
  # eval_heldout.sh tears this container down before it serves
  echo "$names" | grep -qx text2sql_vllm_teacher && echo "container text2sql_vllm_teacher is running"
}

wait_gpu_free() {
  local p last=""
  while :; do
    p=$(gpu_problem)
    if [ -z "$p" ]; then
      sleep "$FREE_GAP_S"
      p=$(gpu_problem)
      [ -z "$p" ] && { echo "== $(date '+%F %T') GPU free (2 probes, ${FREE_GAP_S} s apart)"; return; }
    fi
    [ "$p" != "$last" ] && echo "== $(date '+%F %T') waiting for the GPU: $p"
    last=$p
    sleep "$GPU_POLL_S"
  done
}

# --- everything checkable now fails now, not after the training ----------------------------------------
EXPECTED=$(( $(python3 -c 'import json;print(len(json.load(open("data/raw/db_sandbox/split_tasks.json"))["heldout_eval"]))') * K ))
[ "$EXPECTED" -gt 0 ] || { echo "== cannot read the heldout split; abort"; exit 1; }
for f in ops/grpo_export.sh ops/eval_heldout.sh evaluation/k8_class_compare.py training_pipeline/grpo_metrics.py "$TAU2PY" "$BASE"; do
  [ -e "$f" ] || { echo "== missing: $f; abort"; exit 1; }
done
[ "$(rows "$BASE")" -eq "$EXPECTED" ] || { echo "== $BASE has $(rows "$BASE") rows, expected $EXPECTED; abort"; exit 1; }

echo "==== post-eval $RUN steps: $* (K=$K, base $(basename "$BASE")) — $(date '+%F %T') ===="

# --- 1) wait for the training ---------------------------------------------------------------------------
if runner_alive; then echo "== waiting for ops/grpo_run.sh of $RUN to end"; fi
while runner_alive; do sleep "$POLL_S"; done
DONE_LINE=$(grep -a "^==== GRPO .* run=$RUN done rc=" "$TRAIN_LOG" 2>/dev/null | tail -1)
[ -n "$DONE_LINE" ] || { echo "== no ops/grpo_run.sh of $RUN running and no 'done rc=' line in $TRAIN_LOG; abort"; exit 1; }
echo "== $DONE_LINE"
TRAIN_RC=$(echo "$DONE_LINE" | sed -E 's/.* done rc=([0-9]+).*/\1/')
if [ "$TRAIN_RC" != 0 ] && [ "$ALLOW_TRAIN_RC" != 1 ]; then
  echo "== training ended with rc=$TRAIN_RC: nothing evaluated (ALLOW_TRAIN_RC=1 evaluates the existing checkpoints)"
  exit 1
fi

# --- 2) report + divergence -----------------------------------------------------------------------------
eval "$(python3 training_pipeline/grpo_config.py --run-env "$RUN")" || { echo "== cannot read the run's resolved config; abort"; exit 1; }
DIV_JSON=$RESULT_DIR/${RUN}.divergence.json
rm -f "$DIV_JSON"
python3 training_pipeline/grpo_metrics.py "$RUN" --select --dry --divergence-json "$DIV_JSON" \
  || echo "== WARNING: grpo_metrics.py failed"
ONSET=""
if [ -f "$DIV_JSON" ]; then
  ONSET=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["onset"] if d.get("fired") else "")' "$DIV_JSON")
else
  echo "== WARNING: divergence state unknown, no candidate is skipped"
fi

# --- 3) candidates --------------------------------------------------------------------------------------
EVALUATED=(); SKIPPED=(); FAILED=()
for N in "$@"; do
  if [ -n "$ONSET" ] && [ "$N" -ge "$ONSET" ]; then
    echo "== step $N skipped: at or after the divergence onset (step $ONSET)"; SKIPPED+=("$N"); continue
  fi
  if [ ! -d "${CKPT_DIR#/app/}/global_step_${N}/actor" ]; then
    echo "== step $N: no checkpoint in ${CKPT_DIR#/app/}"; FAILED+=("$N"); continue
  fi
  LABEL=after_${RUN}_step${N}
  MODEL=/app/data/final/checkpoints/db_bahn_grpo_merged_qwen3-4b/${RUN}/step_${N}
  OUT=data/generated/eval/db_traces_heldout_eval_${LABEL}_k${K}.jsonl
  attempt=0
  while [ "$(rows "$OUT")" -ne "$EXPECTED" ] && [ "$attempt" -lt 2 ]; do
    attempt=$((attempt + 1))
    wait_gpu_free
    echo "==== step $N, attempt $attempt — $(date '+%F %T') ===="
    if [ ! -f "${MODEL#/app/}/model.safetensors.index.json" ]; then
      bash ops/grpo_export.sh "$RUN" "$N" || { echo "== step $N: export failed"; continue; }
    fi
    RESUME=0
    if [ -s "$OUT" ] && [ -f "${OUT}.meta.json" ]; then RESUME=1; fi
    K=$K RESUME=$RESUME bash ops/eval_heldout.sh "$MODEL" "$LABEL" "$OUT"
  done
  if [ "$(rows "$OUT")" -ne "$EXPECTED" ]; then
    echo "== step $N: traces incomplete ($(rows "$OUT") of $EXPECTED rows)"; FAILED+=("$N"); continue
  fi
  if "$TAU2PY" evaluation/k8_class_compare.py --base "$BASE" --cand "$OUT" --json "$RESULT_DIR/${RUN}_step${N}_vs_base.json"; then
    EVALUATED+=("$N")
  else
    echo "== step $N: comparison failed"; FAILED+=("$N")
  fi
done

# --- 4) summary -----------------------------------------------------------------------------------------
echo
echo "==== post-eval $RUN summary — $(date '+%F %T') ===="
PYTHONPATH="$REPO" "$TAU2PY" - "$RUN" "$BASE" "$RESULT_DIR" "${EVALUATED[@]}" <<'PY'
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
from evaluation.eval_report import accepted

run, base, result_dir, *steps = sys.argv[1:]
files = [("ep3", "data/generated/eval/db_traces_heldout_eval_after_qwen3-4b_ep3_k8.jsonl"), (f"base {Path(base).stem.split('after_')[-1]}", base)]
files += [(f"step {s}", f"data/generated/eval/db_traces_heldout_eval_after_{run}_step{s}_k8.jsonl") for s in steps]
print(f"  {'model':34s} {'pass^8':>12s} {'mean':>7s}  finish != final_answer")
for name, path in files:
    if not Path(path).exists():
        print(f"  {name:34s} (no traces)")
        continue
    per_task, finish = defaultdict(list), Counter()
    for line in open(path):
        r = json.loads(line)
        per_task[r["task_id"]].append(accepted(r))
        finish[str(r.get("finish_reason", "?")).split(":")[0]] += 1
    n, flat = len(per_task), [a for v in per_task.values() for a in v]
    solved = sum(all(v) for v in per_task.values())
    off = {k: v for k, v in finish.items() if k != "final_answer"}
    print(f"  {name:34s} {solved:4d}/{n} {100 * solved / n:5.1f}% {100 * sum(flat) / len(flat):6.2f}%  {off or '-'}")
print("\n  failed rollouts vs base: diff [95 % CI], negative = fewer failures than base")
for s in steps:
    c = json.loads((Path(result_dir) / f"{run}_step{s}_vs_base.json").read_text())["classes"]
    print(f"  step {s}: " + " | ".join(f"{k} {c[k]['diff']:+d} [{c[k]['ci95'][0]:+d}, {c[k]['ci95'][1]:+d}]" for k in c))
PY
[ $? -eq 0 ] || echo "== WARNING: summary failed (the traces and comparisons are complete)"
echo
echo "   evaluated: ${EVALUATED[*]:-none} | skipped (divergence): ${SKIPPED[*]:-none} | FAILED: ${FAILED[*]:-none}"
echo "   next: python3 training_pipeline/grpo_metrics.py $RUN --pick-step N --note '...'   (selection is manual)"
rc=0; [ "${#FAILED[@]}" -eq 0 ] || rc=1
echo "==== post-eval $RUN done rc=$rc — $(date '+%F %T') ===="
exit "$rc"

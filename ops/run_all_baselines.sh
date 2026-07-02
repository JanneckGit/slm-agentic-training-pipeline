#!/usr/bin/env bash
# =============================================================================
# ops/run_all_baselines.sh
# -----------------------------------------------------------------------------
# Drives the full student sweep SEQUENTIALLY (single GPU), smallest to largest
# so any pipeline bug surfaces cheaply before the expensive large models:
#     Qwen3.5 0.8B/2B/4B/9B (multimodal) + Qwen3-14B (text)  x  {thinking, nothink}
# Each (model,variant) is one run_baseline_pipeline.sh call. With the checkpoints
# + merged models already present, the pipeline resumes (skips train+merge) and
# only re-evaluates. Continue-on-error; already-finished runs (eval JSON present)
# are skipped, so the sweep is resumable. Run from the repo root.
# =============================================================================
set -uo pipefail

PAIRS=(
  "0.8B thinking" "0.8B nothink"
  "2B thinking"   "2B nothink"
  "4B thinking"   "4B nothink"
  "9B thinking"   "9B nothink"
  "q3-4b thinking" "q3-4b nothink"
  "14B thinking"  "14B nothink"
)

# merged dir name — MUST mirror run_baseline_pipeline.sh (MM vs text naming).
merged_name(){
  local s="$1" v="$2"
  case "$s" in
    0.8B|2B|4B|9B) echo "qwen35$(echo "$s" | tr 'A-Z' 'a-z' | tr -d '.')_student_${v}_mm_merged" ;;
    q3-4b)         echo "qwen34b_student_${v}_merged" ;;
    14B)           echo "qwen314b_student_${v}_merged" ;;
    *)             echo "UNKNOWN_${s}_${v}" ;;
  esac
}

declare -a RESULTS
say(){ echo "[$(date '+%F %H:%M:%S')] $*"; }

for pair in "${PAIRS[@]}"; do
  set -- $pair; SIZE="$1"; VARIANT="$2"
  OUT="data/final/eval/$(merged_name "$SIZE" "$VARIANT")/student_${VARIANT}.json"

  if [[ -f "$OUT" ]]; then
    EX="$(python3 -c "import json;print(json.load(open('$OUT'))['overall']['execution_accuracy'])" 2>/dev/null || echo '?')"
    say "SKIP  $SIZE/$VARIANT (already done, EX=$EX)"
    RESULTS+=("$SIZE/$VARIANT  SKIP(EX=$EX)")
    continue
  fi

  say ">>>>>>>>>>>>>>>> START  $SIZE / $VARIANT >>>>>>>>>>>>>>>>"
  if bash ops/run_baseline_pipeline.sh "$SIZE" "$VARIANT"; then
    EX="$(python3 -c "import json;print(json.load(open('$OUT'))['overall']['execution_accuracy'])" 2>/dev/null || echo '?')"
    say "<<<<<<<<<<<<<<<< OK     $SIZE / $VARIANT  EX=$EX"
    RESULTS+=("$SIZE/$VARIANT  OK(EX=$EX)")
  else
    rc=$?
    say "!!!!!!!!!!!!!!!! FAIL   $SIZE / $VARIANT  (rc=$rc) — continuing"
    RESULTS+=("$SIZE/$VARIANT  FAIL(rc=$rc)")
    docker compose -f docker/docker-compose.yml --profile vllm down >/dev/null 2>&1 || true
  fi
done

echo
echo "=================== BASELINE SWEEP SUMMARY ==================="
for r in "${RESULTS[@]}"; do echo "  $r"; done
echo "============================================================="

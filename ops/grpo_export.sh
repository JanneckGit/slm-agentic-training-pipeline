#!/usr/bin/env bash
# =============================================================================
# Export a Stage-2 GRPO checkpoint: verl FSDP checkpoint -> PEFT LoRA adapter -> merged HF model.
#
#   bash ops/grpo_export.sh <profile> <global_step> [RUN_NAME]
#   bash ops/grpo_export.sh r1 selected            # the step behind runs/<run>/selection.json
#
# 1) verl's own merger (grpo container) reads global_step_N/actor (FSDP shard + lora_train_meta.json)
#    and writes  db_bahn_grpo_lora_qwen3-4b/<run>/step_N/lora_adapter/  (adapter_config + safetensors).
#    It also writes a copy of the BASE weights next to it — that copy is deleted (8 GB, redundant).
# 2) adapter_config.json gets base_model_name_or_path = the run's init model (verl leaves it empty;
#    serving/merge_adapter.py would otherwise fall back to the pipeline default Qwen/Qwen3-4B).
# 3) serving/merge_adapter.py (training container, PEFT merge_and_unload) ->
#    db_bahn_grpo_merged_qwen3-4b/<run>/step_N/  — servable like ep3; eval with ops/eval_heldout.sh.
# The SFT checkpoints (db_bahn_traj_*) are never touched.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
PROFILE=${1:?usage: grpo_export.sh <profile> <global_step|selected> [RUN_NAME]}
STEP=${2:?usage: grpo_export.sh <profile> <global_step|selected> [RUN_NAME]}
RUN=${3:-}
RUN_ARG=(); [ -n "$RUN" ] && RUN_ARG=(--run "$RUN")
COMPOSE="docker compose -f docker/docker-compose.yml"
eval "$(python3 training_pipeline/grpo_config.py "$PROFILE" "${RUN_ARG[@]}" --env)"
export GRPO_IMAGE="$IMAGE"
INIT=$(python3 training_pipeline/grpo_config.py "$PROFILE" "${RUN_ARG[@]}" --paths | python3 -c 'import json,sys; print(json.load(sys.stdin)["init_model"])')

if [ "$STEP" = "selected" ]; then
  SEL="${RUN_DIR#/app/}/selection.json"
  STEP=$(python3 -c "import json,sys; d=json.load(open('$SEL')); s=d.get('selected'); print(s['step'] if s else '')")
  [ -z "$STEP" ] && { echo "== no selected checkpoint in $SEL (RL brought no gain?)"; exit 1; }
fi
ACTOR="$CKPT_DIR/global_step_${STEP}/actor"
[ -d "${ACTOR#/app/}" ] || { echo "== checkpoint not found: ${ACTOR#/app/}"; exit 1; }
LORA_OUT="/app/data/final/checkpoints/db_bahn_grpo_lora_qwen3-4b/${RUN}/step_${STEP}"
MERGED_OUT="/app/data/final/checkpoints/db_bahn_grpo_merged_qwen3-4b/${RUN}/step_${STEP}"

echo "==== export $RUN step $STEP — $(date '+%F %T') ===="
echo "   actor : ${ACTOR#/app/}"
echo "   lora  : ${LORA_OUT#/app/}/lora_adapter"
echo "   merged: ${MERGED_OUT#/app/}"

# 1) verl merger -> adapter (+ redundant base copy)
$COMPOSE --profile grpo run --rm -T grpo -m verl.model_merger merge --backend fsdp \
    --local_dir "$ACTOR" --target_dir "$LORA_OUT" 2>&1 | grep -v "tau2\.\|Registering\|^\s*\"\|^\s*[\]\[}]" | tail -8
docker run --rm --entrypoint bash -v "$REPO":/app "$GRPO_IMAGE" -c "chown -R $(id -u):$(id -g) /app/data/final/checkpoints/db_bahn_grpo_lora_qwen3-4b 2>/dev/null; true"
LORA_HOST="${LORA_OUT#/app/}"
[ -f "$LORA_HOST/lora_adapter/adapter_config.json" ] || { echo "== merger produced no lora_adapter"; exit 1; }
find "$LORA_HOST" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.safetensors.index.json' \) -delete   # base copy
python3 - "$LORA_HOST/lora_adapter/adapter_config.json" "$INIT" "$STEP" "$RUN" <<'PY'
import json, sys
p, base, step, run = sys.argv[1:]
c = json.load(open(p)); c["base_model_name_or_path"] = base
json.dump(c, open(p, "w"), indent=2)
json.dump({"run": run, "global_step": int(step), "base_model": base}, open(p.replace("adapter_config.json", "export_meta.json"), "w"), indent=2)
print("adapter_config: base_model_name_or_path =", base, "| r =", c.get("r"), "alpha =", c.get("lora_alpha"), "targets =", sorted(c.get("target_modules", [])))
PY
du -sh "$LORA_HOST/lora_adapter" | awk '{print "   adapter size " $1}'

# 2) PEFT merge into the init model (training image: transformers/peft stack of the SFT merges).
#    verl's merger wrote the adapter_config with the grpo image's newer PEFT (0.19: peft_version, use_qalora, …);
#    the training image's PEFT (0.13) rejects unknown keys -> keep only the fields its LoraConfig knows.
$COMPOSE run --rm -T training python3 - "$LORA_OUT/lora_adapter/adapter_config.json" <<'PY' 2>&1 | grep -v "^\s*$\|Warning" | tail -2
import dataclasses, json, sys, peft
p = sys.argv[1]; c = json.load(open(p))
keep = {f.name for f in dataclasses.fields(peft.LoraConfig)} | {"peft_type"}
dropped = sorted(k for k in c if k not in keep)
json.dump({k: v for k, v in c.items() if k in keep}, open(p, "w"), indent=2)
print(f"adapter_config sanitized for peft {peft.__version__}: dropped {dropped}")
PY
$COMPOSE run --rm -T training python3 serving/merge_adapter.py \
    --adapter-path "$LORA_OUT/lora_adapter" --output-path "$MERGED_OUT" 2>&1 | grep -v "^\s*$" | tail -6
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "== merge_adapter.py failed"; exit 1; }
docker run --rm --entrypoint bash -v "$REPO":/app "$GRPO_IMAGE" -c "chown -R $(id -u):$(id -g) /app/data/final/checkpoints 2>/dev/null; true"
MERGED_HOST="${MERGED_OUT#/app/}"
ls "$MERGED_HOST"/*.safetensors >/dev/null 2>&1 || { echo "== merge produced no safetensors"; exit 1; }
du -sh "$MERGED_HOST" | awk '{print "   merged size " $1}'
echo "==== export done — serve/eval: bash ops/eval_heldout.sh $MERGED_OUT after_${RUN}_step${STEP}"

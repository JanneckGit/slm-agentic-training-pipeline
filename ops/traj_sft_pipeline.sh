#!/usr/bin/env bash
# ops/traj_sft_pipeline.sh — Phase 6 (Plan B): traj_sft train -> merge -> AFTER-eval.
# Der BEFORE-Eval liegt NICHT mehr hier: ops/build_sft_data.sh erzeugt ihn (skip-if-exists) und
# nutzt ihn doppelt — als Vergleichsbasis UND als Quelle der db_bahn-Template-Caps im Mix.
# Sequential (GB10 has no MIG). Eval = rollout.py on the heldout_eval tasks through the same verifier
# (before = Basis aus build_sft_data.sh, after = merged student). Logs to logs/traj_sft.log.
# BASE wird HART gegen das Mix-Manifest gegatet (sft_mix_manifest.json traegt den vollen
# Modellstring aus build_sft_data.sh) — Training auf dem Mix eines anderen Modells bricht ab.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
COMPOSE="docker compose -f docker/docker-compose.yml"
export TAU2PY=${TAU2PY:-$REPO/.venv-tau2/bin/python}   # exportiert: ops/eval_heldout.sh erbt den Override
# host eval writes to the SAME physical store as the training container (../mlruns == /app/mlruns)
export MLFLOW_TRACKING_URI="file://$REPO/mlruns"
BASE="Qwen/Qwen3-4B"   # dense, text-only, thinking student (verl-GRPO-proven; NOT the MM hybrid 3.5)
MERGED_HOST="data/final/checkpoints/db_bahn_traj_merged"
MERGED_CTR="/app/data/final/checkpoints/db_bahn_traj_merged"
ADAPTER="data/final/checkpoints/db_bahn_traj_lora"

# Serve + Eval + Report liegen in ops/eval_heldout.sh — EINE Quelle fuer BEFORE (via
# build_sft_data.sh) und AFTER, damit die Zahlen per Konstruktion vergleichbar sind. Dieses Skript
# ruft es nur fuer AFTER auf; Teardown und `rm -f` der Ausgabedatei gehoeren dort hin (sonst faende
# das Training keinen freien Speicher, bzw. ein zweiter Lauf wuerde hinein-resumen).

echo "==== PHASE 6 START $(date) ===="

DATA=${DATA:-data/final/sft_mix_chat.jsonl}     # 3-leg SFT mix (db_bahn gekappt + AReaL + ToolACE)
VAL=${VAL:-data/final/sft_mix_val.jsonl}         # held-out val split (eval_loss, never in gradient)
BEFORE=data/generated/eval/db_traces_heldout_before.jsonl
EXPECTED=$(python3 -c 'import json;print(len(json.load(open("data/raw/db_sandbox/split_tasks.json"))["heldout_eval"]))')
if ! { [ -f "$BEFORE" ] && [ "$(wc -l < "$BEFORE")" -ge "$EXPECTED" ] && [ -s "$DATA" ] && [ -s "$VAL" ]; }; then
  echo "ABORT: Vorbedingung fehlt ($BEFORE >=$EXPECTED Zeilen + $DATA + $VAL) — erst: bash ops/build_sft_data.sh"
  exit 1
fi
# Modell-Gate: der Mix traegt seine Identitaet im Manifest — BASE muss exakt dazu passen
MANIFEST=data/final/sft_mix_manifest.json
MIX_MODEL=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("model") or "")' "$MANIFEST" 2>/dev/null || true)
if [ "$MIX_MODEL" != "$BASE" ]; then
  echo "ABORT: Mix-Manifest-Modell '${MIX_MODEL:-<fehlt>}' != BASE '$BASE' — Mix fuer dieses Modell bauen: bash ops/build_sft_data.sh"
  exit 1
fi
MIX_LABEL=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("label") or "?")' "$MANIFEST")
echo "   BEFORE-Basis: $(readlink -f "$BEFORE")"
echo "   Mix: $(readlink -f "$DATA") (Modell $MIX_MODEL, Label $MIX_LABEL)"

echo "== [1/3] TRAIN traj_sft ($(wc -l < "$DATA" 2>/dev/null || echo '?') traces, 2 epochs, LoRA @12288) =="
# --save-epoch-adapters -> ${ADAPTER}_ep1 + _ep2 for checkpoint selection; --neftune 5 = noisy-embedding reg.
$COMPOSE run --rm -T training python3 training_pipeline/train_traj.py \
  --config config/pipeline_config.yaml --data "$DATA" --val-file "$VAL" \
  --model "$BASE" --out "$ADAPTER" --epochs 2 --max-seq-len 12288 \
  --attn flash_attention_2 --liger --neftune 5 --save-epoch-adapters --eval-steps 300 2>&1 | grep -vE "Copyright|NVIDIA|reserved|GOVERNING|found at|PyTorch|Idiap|Google|Caffe|Facebook|Deepmind|NEC|NYU|Yangqing|Various|====" | tail -20

echo "== [2/3] MERGE both epoch adapters -> ${MERGED_HOST}/ep{1,2} (sharded) =="
# one folder per run: <adapter>/ep{1,2} -> <merged>/ep{1,2}; the winner gets a 'selected' symlink below.
# merge_adapter.py hard-aborts on a no-op merge or a save mismatch -> a green merge here is verified.
for EP in 1 2; do
  echo "-- merge epoch $EP --"
  $COMPOSE run --rm -T training python3 serving/merge_adapter.py \
    --adapter-path "${ADAPTER}/ep${EP}" --output-path "${MERGED_HOST}/ep${EP}" \
    --config config/pipeline_config.yaml 2>&1 | tail -4
done

echo "== [3/3] AFTER-eval: eval BOTH epochs on heldout_eval, keep the higher verified_yield =="
for EP in 1 2; do
  bash ops/eval_heldout.sh "${MERGED_CTR}/ep${EP}" "after_ep${EP}" \
    "data/generated/eval/db_traces_heldout_after_ep${EP}.jsonl" \
    || echo "== AFTER-eval ep${EP} FAILED"
done

# checkpoint selection: same accept gate as the eval report (score==1.0 AND not truncated AND not
# degenerate) — a bare score==1.0 would prefer the checkpoint that produces MORE think-loops, and with
# single-shot evals those gates finally fire. Tie -> ep2.
WINNER=$(python3 - "data/generated/eval/db_traces_heldout_after_ep1.jsonl" "data/generated/eval/db_traces_heldout_after_ep2.jsonl" <<'PY'
import json, sys
def vyield(p):
    n = y = 0
    try:
        for line in open(p):
            line = line.strip()
            if not line: continue
            r = json.loads(line); n += 1
            d = r.get("degen") or {}
            if ((r.get("score") or {}).get("score") == 1.0 and not r.get("truncated")
                    and d.get("think_ngram_dup_ratio", 0.0) <= 0.5
                    and d.get("max_think_chars", 0) <= 12000): y += 1
    except FileNotFoundError:
        return -1.0, 0, 0
    return (y / n if n else 0.0), y, n
y1, c1, n1 = vyield(sys.argv[1]); y2, c2, n2 = vyield(sys.argv[2])
sys.stderr.write(f"ep1 verified_yield={y1:.3f} ({c1}/{n1})  |  ep2 verified_yield={y2:.3f} ({c2}/{n2})\n")
print(2 if y2 >= y1 else 1)
PY
)
WINNER=${WINNER:-2}
echo "== checkpoint selection: EPOCH $WINNER wins -> 'selected' symlinks =="
# 'selected' means "the candidate the eval picked" — in BOTH folders: the servable model and the adapter.
# in-container (root) -> handles root-owned dirs; relative targets resolve inside the mounted tree.
$COMPOSE run --rm -T training bash -c "
  ln -sfn 'ep${WINNER}' '${MERGED_CTR}/selected' &&
  ln -sfn 'ep${WINNER}' '/app/${ADAPTER}/selected' && echo 'selected -> ep${WINNER} (merged + adapter)'" 2>&1 | tail -1
cp -f "data/generated/eval/db_traces_heldout_after_ep${WINNER}.jsonl" data/generated/eval/db_traces_heldout_after.jsonl
echo "   servable model: ${MERGED_HOST}/selected   |   adapter: ${ADAPTER}/selected"

echo "==== PHASE 6 DONE $(date) ===="

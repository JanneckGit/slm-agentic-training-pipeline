#!/usr/bin/env bash
# ops/build_sft_data.sh — Daten-Phase vor dem SFT-Training: Basis-Eval (skip-if-exists) -> Caps -> Mix.
#
#   bash ops/build_sft_data.sh [MODEL] [LABEL]
#   bash ops/build_sft_data.sh                              # Qwen/Qwen3-4B, base_qwen3-4b (Eval liegt vor -> kein GPU)
#   bash ops/build_sft_data.sh Qwen/Qwen3-8B base_qwen3-8b  # Studentenwechsel: Rest leitet sich ab
#
# Fehlt die Eval-Datei fuer LABEL, laeuft ops/eval_heldout.sh (vLLM-Serve; 4B: ~40 min Wanduhr /
# ~12 h Compute) -> dann in tmux starten. Sampling-Overrides fremder Modellfamilien (TEMPERATURE,
# PRESENCE_PENALTY, VLLM_EXTRA, ...) als Env mitgeben — eval_heldout.sh erbt sie.
#
# Die Skip-Logik lebt bewusst HIER: eval_heldout.sh selbst macht rm -f auf seine Ausgabe
# (Anti-Resume) und liefert auch bei "ROLLOUT FAILED" Exit 0 — deshalb wird nach einem Lauf erneut
# auf Vollstaendigkeit geprueft und hart abgebrochen. Soll-Zeilen kommen aus split_tasks.json,
# nicht hartkodiert (Welle-2-Templates vergroessern den Split, das Gate zieht mit).
#
# Die abgeleiteten Caps sind modellspezifisch (Bandzuteilung haengt am Basis-Yield); die Ableitung
# traegt nur fuer faehige Basismodelle (>=4B) — bei kleinen misst der Yield Formatbeherrschung,
# nicht Aufgabenverstaendnis (siehe docs/heldout-eval-baselines.md), die Kuerzung liefe leer.
# Artefakte sind LABEL-suffigiert (mehrere Modelle liegen nebeneinander); die kanonischen Namen
# (sft_mix_chat.jsonl, db_bahn_caps.json, ...) sind Symlinks, die erst NACH grünem Build in einem
# Rutsch umgezeigt werden — ein Abbruch mittendrin lässt die alten Links intakt. Konsumenten
# (traj_sft_pipeline.sh, train_traj-Defaults) lesen weiter die kanonischen Namen; das Manifest
# traegt den vollen Modellstring, gegen den traj_sft_pipeline.sh sein BASE hart gatet.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:-${MODEL:-Qwen/Qwen3-4B}}
LABEL=${2:-${LABEL:-base_qwen3-4b}}
EVAL_FILE="data/generated/eval/db_traces_heldout_eval_${LABEL}.jsonl"   # == OUT-Default von eval_heldout.sh
CAPS="data/generated/eval/db_bahn_caps_${LABEL}.json"
MIX_CHAT="data/final/sft_mix_chat_${LABEL}.jsonl"
MIX_VAL="data/final/sft_mix_val_${LABEL}.jsonl"
MANIFEST="data/final/sft_mix_${LABEL}.manifest.json"
BEFORE="data/generated/eval/db_traces_heldout_before.jsonl"
EXPECTED=$(python3 -c 'import json;print(len(json.load(open("data/raw/db_sandbox/split_tasks.json"))["heldout_eval"]))')
mkdir -p logs

eval_ok() {  # vorhanden + >= EXPECTED Zeilen + jede Zeile parst
  [ -f "$EVAL_FILE" ] || return 1
  [ "$(wc -l < "$EVAL_FILE")" -ge "$EXPECTED" ] || return 1
  python3 -c 'import json,sys
for l in open(sys.argv[1]):
    l.strip() and json.loads(l)' "$EVAL_FILE" >/dev/null || return 1
}

if eval_ok; then
  echo "== [1/5] Basis-Eval vorhanden ($EVAL_FILE, >=$EXPECTED Zeilen) — skip =="
else
  echo "== [1/5] Basis-Eval fehlt/unvollstaendig -> ops/eval_heldout.sh $MODEL $LABEL (GPU!) =="
  bash ops/eval_heldout.sh "$MODEL" "$LABEL"
  eval_ok || { echo "ABORT: Eval nach Lauf unvollstaendig ($EVAL_FILE) — logs/eval_heldout.log pruefen"; exit 1; }
fi

echo "== [2/5] kanonischer BEFORE-Name -> $(basename "$EVAL_FILE") =="
# relativ + gleiche Ebene: haelt auf dem Host UND in jedem Container, der data/ mountet
ln -sfn "$(basename "$EVAL_FILE")" "$BEFORE"

echo "== [3/5] Caps ableiten =="
PYTHONPATH=. python3 data_pipeline/derive_db_caps.py \
  --eval "$EVAL_FILE" --model "$MODEL" --label "$LABEL" --out "$CAPS"

echo "== [4/5] SFT-Mix bauen =="
PYTHONPATH=. python3 data_pipeline/build_sft_mix.py --db-caps "$CAPS" \
  --out-train "$MIX_CHAT" --out-val "$MIX_VAL" --manifest "$MANIFEST" \
  | tee "logs/build_sft_mix_${LABEL}.log"

echo "== [5/5] kanonische Namen -> ${LABEL} =="
ln -sfn "$(basename "$CAPS")" data/generated/eval/db_bahn_caps.json
ln -sfn "$(basename "$MIX_CHAT")" data/final/sft_mix_chat.jsonl
ln -sfn "$(basename "$MIX_VAL")" data/final/sft_mix_val.jsonl
ln -sfn "$(basename "$MANIFEST")" data/final/sft_mix_manifest.json

echo "== FERTIG: sft_mix_{chat,val} -> ${LABEL} (Manifest: $MANIFEST) — Training separat (ops/traj_sft_pipeline.sh) =="

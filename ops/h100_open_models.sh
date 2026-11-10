#!/usr/bin/env bash
# =============================================================================
# ops/h100_open_models.sh — DB-Heldout (K=8) fuer offene Modelle auf einem Rechner OHNE Docker (Runde 1).
#
# Je Modell:  Download (das naechste laedt im Hintergrund vor)
#          -> Smoke: bakeoff_dev, K=1, plus Gate (evaluation/heldout_gate.py smoke: Fehler, truncated, Think, Tool-Calls)
#          -> K=8 auf heldout_eval (ops/eval_heldout.sh SERVE=venv MLFLOW=0; Protokoll wie die K=8-Serie)
#          -> Ergebnis-Tar + sha256 nach $OUTDIR -> Gewichte loeschen.
# Eintrag 0 = Eichlauf Qwen3-4B: statt Smoke ein Vergleich mit der GB10-Referenz (heldout_gate.py calib);
# faellt die Eichung durch, bricht die Kette ab (dann stimmt etwas an Rechner/Setup nicht).
# Ein Modell, dessen Smoke durchfaellt, wird uebersprungen (Gewichte bleiben liegen); die Kette laeuft weiter.
# Resume-fest: fertige Laeufe (Zeilenzahl = Soll) werden uebersprungen, unvollstaendige mit RESUME=1 fortgesetzt.
# MLflow wird spaeter auf dem Heimrechner nachgetragen (evaluation/log_heldout_mlflow.py).
#
# Start (tmux!):  tmux new -s h100-open
#                 bash ops/h100_open_models.sh                       # alle Eintraege
#                 ONLY=open_kolibri-1 bash ops/h100_open_models.sh   # nur einer
# Pfade per Env (Defaults unter H100_ROOT=/data/slm): VLLM_VENV, VLLM_BIN, TAU2PY, OUTDIR, REFERENCE.
# Stoppen: ZUERST die Kette  kill -KILL $(pgrep -f 'h100_open_models[.]sh'),  dann  kill <PID von eval_heldout.sh>
#          (SIGTERM): dessen Teardown stoppt erst rollout.py, dann den vLLM-Server, und sichert das Server-Log.
#          Neustart mit demselben Befehl setzt fort.
# NIE editieren, solange sie laeuft.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)

H100_ROOT=${H100_ROOT:-/data/slm}
VLLM_VENV=${VLLM_VENV:-$H100_ROOT/venv-vllm}
export VLLM_BIN=${VLLM_BIN:-$VLLM_VENV/bin/vllm}
export TAU2PY=${TAU2PY:-$H100_ROOT/venv-harness/bin/python}
HF_CLI=${HF_CLI:-$VLLM_VENV/bin/hf}
OUTDIR=${OUTDIR:-$H100_ROOT/out}
REFERENCE=${REFERENCE:-data/generated/eval/db_traces_heldout_eval_base_qwen3-4b_k8.jsonl}
ONLY=${ONLY:-}
SPACE_RESERVE_GB=${SPACE_RESERVE_GB:-30}
export SERVE_PORT=${SERVE_PORT:-8000}   # an eval_heldout.sh durchgereicht (vLLM nur auf 127.0.0.1)

# Alle Caches auf die grosse Platte (/home und /tmp sind dort klein); localhost nie ueber den Proxy.
C=$H100_ROOT/cache
export HF_HOME=$C/hf XDG_CACHE_HOME=$C/xdg PIP_CACHE_DIR=$C/pip TMPDIR=$C/tmp VLLM_CACHE_ROOT=$C/vllm \
       TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor CUDA_CACHE_PATH=$C/nv \
       FLASHINFER_WORKSPACE_BASE=$C/flashinfer
export no_proxy="localhost,127.0.0.1${no_proxy:+,$no_proxy}"; export NO_PROXY=$no_proxy
# huggingface_hub >= 2 legt Xet-Dateien in einen cacheweiten Speicher (hub/blobs/<xx>/<hash>), models--*/blobs
# sind dann nur Symlinks -> delete_weights (rm -rf models--*) gaebe keinen Platz frei. Klassisches Layout erzwingen.
export HF_HUB_DISABLE_SHARED_BLOBS=1
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$FLASHINFER_WORKSPACE_BASE" "$OUTDIR" logs data/generated/eval

# label|hf-repo|vllm-extra|gpu-util|temp|top_p|top_k|min_p|presence|chat-template-kwargs (JSON)|Download-GB
# Protokoll fest (eval_heldout.sh): 15 Zuege, 3.072 Token/Zug, Kontext 20.480, Single-Shot, Thinking an.
# Sampling + Template-Schalter je Modell laut Model-Card; top_k -1 = aus (Card nennt keins).
MODELS=(
  "base_qwen3-4b_h100|Qwen/Qwen3-4B||0.85|0.6|0.95|20|0|0.0||9"
  "open_qwen3.5-122b-a10b|Qwen/Qwen3.5-122B-A10B-GPTQ-Int4|--language-model-only --quantization moe_wna16|0.95|0.6|0.95|20|0|0.0||80"
  "open_nemotron-3.5-super-vl|nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4|--trust-remote-code|0.95|1.0|0.95|-1|0|0.0||76"
  "open_kolibri-1|Aleph-Alpha/Kolibri-1|--kv-cache-dtype fp8|0.95|1.0|0.97|128|0|0.0||80"
  'open_nanbeige4.3-33b|Nanbeige/Nanbeige4.3-33B||0.90|1.0|0.95|-1|0|0.0|{"preserve_thinking": true}|67'
  "open_granite-4.2-30b|ibm-granite/granite-4.2-30b||0.90|1.0|0.95|-1|0|0.0||60"
)
CALIB_LABEL=base_qwen3-4b_h100

CLOG=logs/h100_chain.log
log() { echo "== [h100] $(date '+%F %T') $*" | tee -a "$CLOG"; }
rows() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
split_n() { python3 -c 'import json,sys;print(len(json.load(open("data/raw/db_sandbox/split_tasks.json"))[sys.argv[1]]))' "$1"; }
free_gb() { df --output=avail -BG "$H100_ROOT" | tail -1 | tr -dc '0-9'; }
cache_dir() { echo "$HF_HOME/hub/models--${1//\//--}"; }

download() {   # <repo> -> 0 ok; drei Versuche (Proxy)
  local t
  for t in 1 2 3; do
    HF_HUB_DISABLE_PROGRESS_BARS=1 "$HF_CLI" download "$1" >>logs/h100_download.log 2>&1 && return 0
    echo "download $1 Versuch $t fehlgeschlagen" >>logs/h100_download.log; sleep 60
  done
  return 1
}

delete_weights() {   # <repo>
  local d; d=$(cache_dir "$1")
  [ -d "$d" ] && rm -rf "${d:?}" && log "Gewichte geloescht: $1"
}

pack() {   # <label> -> $OUTDIR/h100-results-<label>.tar.gz (+ .sha256)
  local t="$OUTDIR/h100-results-$1.tar.gz"
  # shellcheck disable=SC2046
  tar czf "$t" $(ls data/generated/eval/*_"$1"[._]* 2>/dev/null) "$CLOG" logs/h100_env.txt 2>/dev/null \
    && (cd "$OUTDIR" && sha256sum "$(basename "$t")" > "$(basename "$t").sha256") \
    && log "Paket: $t ($(du -h "$t" | cut -f1))"
}

max_conc() {   # <smoke-out> -> "Maximum concurrency"-Wert aus dem gesicherten Server-Log (leer, wenn keiner)
  zcat "$1".vllm-*.log.gz 2>/dev/null | grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' \
    | tail -1 | grep -oE '[0-9.]+x$' | tr -d x
}

# ---- Vorab-Checks -----------------------------------------------------------------------------------------
for f in "$VLLM_BIN" "$TAU2PY" "$HF_CLI"; do [ -x "$f" ] || { log "ABBRUCH: $f fehlt"; exit 1; }; done
for f in db.json tasks.json split_tasks.json answer_keys.json; do
  [ -f "data/raw/db_sandbox/$f" ] || { log "ABBRUCH: data/raw/db_sandbox/$f fehlt (Eingabe-Paket entpackt?)"; exit 1; }
done
[ -f "$REFERENCE" ] || { log "ABBRUCH: Eich-Referenz $REFERENCE fehlt"; exit 1; }
if (exec 3<>"/dev/tcp/127.0.0.1/$SERVE_PORT") 2>/dev/null; then log "ABBRUCH: Port $SERVE_PORT belegt"; exit 1; fi
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
[ "${USED:-0}" -lt 1024 ] || { log "ABBRUCH: GPU belegt (${USED} MiB) — laeuft noch ein Server?"; exit 1; }
{ date '+%F %T'; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  "$VLLM_BIN" --version 2>/dev/null | tail -1 | sed 's/^/vllm /'; git -C "$REPO" rev-parse HEAD 2>/dev/null || echo "kein git"; } > logs/h100_env.txt
log "Kette gestartet ($(tr '\n' ' ' < logs/h100_env.txt))"

N_SMOKE=$(split_n bakeoff_dev); N_K8=$(( $(split_n heldout_eval) * 8 ))
BG_PID=""; BG_REPO=""

for i in "${!MODELS[@]}"; do
  IFS='|' read -r LABEL REPO_ID EXTRA UTIL TEMP TOPP TOPK MINP PRES CTK GB <<<"${MODELS[$i]}"
  [ -n "$ONLY" ] && [ "$ONLY" != "$LABEL" ] && continue
  K8_OUT=data/generated/eval/db_traces_heldout_eval_${LABEL}_k8.jsonl
  SMOKE_OUT=data/generated/eval/db_traces_bakeoff_dev_${LABEL}.jsonl
  if [ "$(rows "$K8_OUT")" -ge "$N_K8" ]; then
    if [ "$LABEL" = "$CALIB_LABEL" ]; then   # Eichung fertig: nur das Gate erneut pruefen, kein Download
      "$TAU2PY" evaluation/heldout_gate.py calib --input "$K8_OUT" --reference "$REFERENCE" 2>&1 | tee -a "$CLOG" \
        || { log "ABBRUCH: Eichung durchgefallen — erst klaeren, dann neu starten"; exit 1; }
    fi
    log "$LABEL: K=8 schon vollstaendig — uebersprungen"; continue
  fi
  log "==== $LABEL ($REPO_ID) ===="

  # -- Download (ein laufender Hintergrund-Download fuer DIESES Modell wird abgewartet) --
  if [ -n "$BG_PID" ] && [ "$BG_REPO" = "$REPO_ID" ]; then wait "$BG_PID"; BG_PID=""; fi
  if [ ! -d "$(cache_dir "$REPO_ID")" ] && [ "$(free_gb)" -lt $((GB + SPACE_RESERVE_GB)) ]; then
    log "ABBRUCH: zu wenig Platz fuer $REPO_ID (${GB} GB + ${SPACE_RESERVE_GB} GB Reserve, frei $(free_gb) GB)"; exit 1
  fi
  download "$REPO_ID" || { log "$LABEL: Download fehlgeschlagen — uebersprungen (logs/h100_download.log)"; continue; }
  # naechstes Modell im Hintergrund vorladen, wenn Platz ist
  for j in $(seq $((i + 1)) $((${#MODELS[@]} - 1))); do
    IFS='|' read -r NL NR _ _ _ _ _ _ _ _ NGB <<<"${MODELS[$j]}"
    [ -n "$ONLY" ] && break
    [ "$(rows "data/generated/eval/db_traces_heldout_eval_${NL}_k8.jsonl")" -ge "$N_K8" ] && continue
    if [ -z "$BG_PID" ] && [ "$(free_gb)" -ge $((NGB + SPACE_RESERVE_GB)) ]; then
      download "$NR" & BG_PID=$!; BG_REPO=$NR; log "Hintergrund-Download: $NR (PID $BG_PID)"
    fi
    break
  done

  RUN_ENV=(SERVE=venv MLFLOW=0 SERVE_WAIT_S=3600 VLLM_MAX_MODEL_LEN=20480 VLLM_GPU_UTIL="$UTIL" VLLM_EXTRA="$EXTRA"
           TEMPERATURE="$TEMP" TOP_P="$TOPP" TOP_K="$TOPK" MIN_P="$MINP" PRESENCE_PENALTY="$PRES"
           CHAT_TEMPLATE_KWARGS="$CTK")

  # -- Smoke + Gate (nicht fuer den Eichlauf) --
  CONC=21
  if [ "$LABEL" != "$CALIB_LABEL" ]; then
    if [ "$(rows "$SMOKE_OUT")" -lt "$N_SMOKE" ]; then
      log "$LABEL: Smoke (bakeoff_dev, K=1)"
      env "${RUN_ENV[@]}" SPLIT=bakeoff_dev K=1 EVAL_CONC=8 bash ops/eval_heldout.sh "$REPO_ID" "$LABEL" \
        >>"logs/h100_${LABEL}.log" 2>&1
    fi
    if ! "$TAU2PY" evaluation/heldout_gate.py smoke --input "$SMOKE_OUT" 2>&1 | tee -a "$CLOG"; then
      log "$LABEL: SMOKE DURCHGEFALLEN — kein K=8, Modell uebersprungen (Gewichte bleiben)"; pack "$LABEL"; continue
    fi
    MC=$(max_conc "$SMOKE_OUT")
    if [ -n "$MC" ]; then
      CONC=$(python3 -c 'import sys; print(min(32, max(4, int(0.75 * float(sys.argv[1])))))' "$MC")
    else
      CONC=12
    fi
    log "$LABEL: Smoke ok — vLLM Maximum concurrency ${MC:-?}x -> EVAL_CONC $CONC"
  fi

  # -- K=8 (frisch oder Fortsetzung) --
  if [ "$(rows "$K8_OUT")" -lt "$N_K8" ]; then
    RES=0; [ -f "$K8_OUT" ] && [ -f "$K8_OUT.meta.json" ] && [ "$(rows "$K8_OUT")" -gt 0 ] && RES=1
    log "$LABEL: K=8 auf heldout_eval (EVAL_CONC $CONC, RESUME=$RES)"
    env "${RUN_ENV[@]}" K=8 EVAL_CONC="$CONC" EVAL_TIMEOUT=259200 RESUME=$RES \
      bash ops/eval_heldout.sh "$REPO_ID" "$LABEL" >>"logs/h100_${LABEL}.log" 2>&1
  fi
  N=$(rows "$K8_OUT")
  if [ "$N" -lt "$N_K8" ]; then
    log "$LABEL: K=8 UNVOLLSTAENDIG ($N/$N_K8) — Neustart der Kette setzt fort (logs/h100_${LABEL}.log)"
    pack "$LABEL"; continue
  fi
  log "$LABEL: K=8 vollstaendig ($N Zeilen)"

  # -- Eichung: Vergleich mit der GB10-Referenz; durchgefallen = ganze Kette stoppen --
  if [ "$LABEL" = "$CALIB_LABEL" ]; then
    if ! "$TAU2PY" evaluation/heldout_gate.py calib --input "$K8_OUT" --reference "$REFERENCE" 2>&1 | tee -a "$CLOG"; then
      pack "$LABEL"; log "ABBRUCH: Eichung durchgefallen — erst klaeren, dann neu starten"; exit 1
    fi
  fi
  pack "$LABEL"
  delete_weights "$REPO_ID"
done

[ -n "$BG_PID" ] && wait "$BG_PID"
log "Kette fertig. Pakete in $OUTDIR:"; ls -1 "$OUTDIR" | tee -a "$CLOG"

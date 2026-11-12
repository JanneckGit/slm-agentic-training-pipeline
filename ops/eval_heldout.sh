#!/usr/bin/env bash
# ops/eval_heldout.sh — Heldout-Eval EINES Modells (Base, merged Student, beliebiger Checkpoint).
# Serve -> rollout.py auf dem Eval-Split -> Teardown -> Per-Template-Report.
#
#   bash ops/eval_heldout.sh Qwen/Qwen3-4B base_think
#   bash ops/eval_heldout.sh /app/data/final/checkpoints/db_bahn_traj_merged/ep2 after_ep2
#
# SINGLE-SHOT (--max-regen 0): genau EIN Versuch pro Rollout. Aeltere Heldout-Zahlen entstanden mit
# --max-regen 1 = bis zu 3 Versuche mit best-of-Auswahl; sie sind mit diesen NICHT vergleichbar.
#
# K ROLLOUTS PRO TASK (Default 8): eval_report.py liefert daraus pass^k (Task zaehlt
# nur, wenn ALLE k Samples bestehen) UND den Mittelwert ueber k (= rauschaermerer pass@1, vergleichbar
# mit den alten k=1-Dateien). Bei K != 1 haengt der Dateiname `_k<K>` an, damit der rm -f unten
# keine k=1-Datei desselben Labels ueberschreibt. K=1 reproduziert das alte Verhalten 1:1.
# Korridor-Messung fuer den RL-Pool = derselbe Aufruf mit RL-Sampling (temp 1.0) auf dem RL-Split:
#   SPLIT=rl_train K=8 TEMPERATURE=1.0 MLFLOW_EXPERIMENT=db_bahn_rl_corridor \
#     bash ops/eval_heldout.sh <ep3-merged> ep3_corridor      # ~5 h; danach corridor_from_traces.py
#
# Env-Overrides (Defaults in Klammern):
#   K (8)  SPLIT (heldout_eval)  EVAL_CONC (21)  ROLLOUT_TIMEOUT_S (1800)
#   EVAL_TIMEOUT (5400 * K = 1,5 h je Rollout-Runde, bei K=8 also 12 h)
# Das HTTP-Lese-Timeout je Zug (rollout.py --request-timeout-s, Default 600 s) wird hier auf das Episoden-Budget
# ROLLOUT_TIMEOUT_S gesetzt: bei hoher Server-Parallelitaet dauert ein einzelner langer Zug sonst laenger als 600 s
# und wird als teacher_error (Score 0) gezaehlt — ein Serving-Artefakt, kein Modellfehler (2026-09-07: Teacher 4,4 %,
# Qwen3.8-27B 30 % der Rollouts). Die Modellgrenzen (Zuege, Token je Zug, Episoden-Budget) bleiben fuer alle gleich.
#   VLLM_MAX_MODEL_LEN (20480)  VLLM_GPU_UTIL (0.85)  VLLM_EXTRA  TAU2PY
#   MLFLOW_EXPERIMENT (db_bahn_traj_eval)
#   TEMPERATURE (0.6)  TOP_P (0.95)  TOP_K (20)  MIN_P (0)  PRESENCE_PENALTY (0.0)
#   RESUME (0)  1 = eine unterbrochene Ausgabe fortsetzen: rollout.py ueberspringt vorhandene (task, sample)-Paare;
#               erlaubt nur, wenn die Sidecar <OUT>.meta.json (Modell, Label, Split, K, Sampling, Serve-Flags, Timeouts)
#               exakt zu diesem Aufruf passt — sonst Abbruch VOR dem Serve. Eine halb geschriebene letzte Zeile wird verworfen.
#   WRITE_META_ONLY (0)  1 = nur die Sidecar fuer eine vorhandene (Legacy-)Ausgabe schreiben, kein Serve, kein Rollout.
#   VLLM_IMAGE  vLLM-Image (Default v0.21.0 per Digest im Compose); was wirklich serviert wurde (Version, Image,
#               Util, "verfuegbar") steht je Start in der Sidecar <OUT>.serve.json (evaluation/serve_info.py).
#   CHAT_TEMPLATE_KWARGS  zusaetzliche Template-Schalter als JSON-Objekt (z. B. '{"preserve_thinking": true}'), ueber
#               enable_thinking gemergt; steht nur dann in der Sidecar-Identitaet (alte Sidecars bleiben resume-faehig).
#   MLFLOW (1)  0 = kein MLflow-Log (z. B. auf einem Fremdrechner; spaeter per evaluation/log_heldout_mlflow.py nachtragen).
#   TOOL_MODE (prompt)  prompt = Tool-Schemas + <tool_call>-Format im System-Prompt, rollout.py parst den Text (K=8-Serie).
#               native = `tools` an den Server, dessen Parser zerlegen Calls + Denken (rollout.py --native-tools) — fuer
#               Modelle mit fremdem Heimformat (gpt-oss, Gemma 4, Muse). Braucht in VLLM_EXTRA
#               --enable-auto-tool-choice --tool-call-parser X --reasoning-parser Y. Steht nur bei native in der
#               Sidecar-Identitaet (alte Sidecars bleiben resume-faehig).
# Das vollstaendige vLLM-Server-Log landet beim Teardown (auch bei Abbruch) als <OUT>.vllm-<UTC>.log.gz daneben.
#
# SERVE — wer den Server stellt (Default compose = Verhalten wie bisher):
#   compose  vLLM-Container per docker compose (GB10).
#   venv     `$VLLM_BIN serve` aus einer venv, gleiche Flags wie der Compose-Befehl, nur 127.0.0.1:$SERVE_PORT (Default 8000;
#            Rechner ohne Docker, z. B. die H100). Ist der Port belegt, bricht das Skript ab — es beendet nie fremde Prozesse.
#   none     kein Start, kein Teardown: ein laufender Endpunkt unter API_BASE (Default http://localhost:8000/v1) wird
#            benutzt; Modellpruefung = "Modell ist in /v1/models gelistet". Baustein fuer spaetere API-Laeufe.
#
# BACKEND — wer antwortet (Default vllm = alles oben):
#   vllm       OpenAI-kompatibler Endpunkt (vLLM), Verhalten wie bisher.
#   anthropic  Claude-API ueber das offizielle SDK (rollout.py --backend anthropic), ohne GPU. Setzt SERVE=none und
#              TOOL_MODE=native (gleiches natives Protokoll wie Runde 2). KEINE Sampling-Parameter (die API lehnt sie ab:
#              TEMPERATURE/TOP_P/TOP_K/MIN_P/PRESENCE_PENALTY/CHAT_TEMPLATE_KWARGS/VLLM_EXTRA gesetzt = Abbruch), Denken
#              adaptiv + nur als Zusammenfassung sichtbar. Pflicht: EFFORT (low|medium|high|xhigh|max) — der Modell-Default,
#              aber EXPLIZIT (Sonnet 5.5 high, Haiku 5.5 medium). Key: ANTHROPIC_API_KEY, sonst aus ANTHROPIC_ENV
#              (Default ~/.config/anthropic/api.env, Zeile ANTHROPIC_API_KEY=...; wird nie ausgegeben). Modellpruefung =
#              models.retrieve (kostenlos). Account-Fehler (Key, Guthaben, Modell) stoppen den Lauf ohne Fehlerzeilen
#              (rollout.py Exit 3) -> RESUME=1 setzt fort. Nach dem Report: Token + Kosten (evaluation/api_usage.py).
#     BACKEND=anthropic EFFORT=high SPLIT=bakeoff_dev K=1 EVAL_CONC=8 MLFLOW=0 \
#       bash ops/eval_heldout.sh claude-sonnet-5-5 api_claude-sonnet-5-5
#   ANTHROPIC_API_BASE  anderer API-Endpunkt (nur Tests/Stub; leer = SDK-Default).
#   ORDER (task; bei BACKEND=anthropic round)  round = Runde fuer Runde (Sample 0 fuer ALLE Tasks, dann Sample 1, ...):
#               bricht ein Lauf ab (Guthaben leer, Fehlerserie), bleiben r vollstaendige Runden ueber alle Tasks ->
#               sauberes pass^r per `eval_report.py --max-sample r`; RESUME=1 ergaenzt spaeter genau den Rest. Steht in
#               der Sidecar unter info (nicht identity: aendert nur die Planung, nicht die Daten).
set -uo pipefail   # KEIN -e: `compose down` scheitert legitim ohne Container, und `timeout` liefert 124
cd "$(dirname "$0")/.."
REPO=$(pwd)
COMPOSE="docker compose -f docker/docker-compose.yml"
TAU2PY=${TAU2PY:-$REPO/.venv-tau2/bin/python}
BACKEND=${BACKEND:-vllm}
EFFORT=${EFFORT:-}
case "$BACKEND" in vllm|anthropic) ;; *) echo "BACKEND muss vllm oder anthropic sein, ist '$BACKEND'"; exit 2;; esac
if [ "$BACKEND" = anthropic ]; then
  # VOR den Defaults unten geprueft: nur was der Aufrufer selbst gesetzt hat, zaehlt als Widerspruch
  [ "${SERVE:-none}" = none ] || { echo "BACKEND=anthropic: SERVE muss none sein (kein Server), ist '$SERVE'"; exit 2; }
  [ "${TOOL_MODE:-native}" = native ] || { echo "BACKEND=anthropic: TOOL_MODE muss native sein, ist '$TOOL_MODE'"; exit 2; }
  SERVE=none; TOOL_MODE=native
  case "$EFFORT" in low|medium|high|xhigh|max) ;; *)
    echo "BACKEND=anthropic braucht EFFORT (low|medium|high|xhigh|max; Modell-Default explizit), ist '$EFFORT'"; exit 2;; esac
  for v in TEMPERATURE TOP_P TOP_K MIN_P PRESENCE_PENALTY CHAT_TEMPLATE_KWARGS VLLM_EXTRA; do
    [ -n "${!v:-}" ] && { echo "BACKEND=anthropic: $v wird nicht unterstuetzt (API ohne Sampling-/vLLM-Parameter)"; exit 2; }
  done
  ANTHROPIC_ENV=${ANTHROPIC_ENV:-$HOME/.config/anthropic/api.env}
  if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -f "$ANTHROPIC_ENV" ]; then set -a; . "$ANTHROPIC_ENV"; set +a; fi
  [ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "BACKEND=anthropic: ANTHROPIC_API_KEY fehlt (weder Env noch $ANTHROPIC_ENV)"; exit 2; }
  export ANTHROPIC_API_KEY
  SDK_VERSION=$("$TAU2PY" -c 'import anthropic; print(anthropic.__version__)' 2>/dev/null) \
    || { echo "BACKEND=anthropic: Paket anthropic fehlt in $TAU2PY"; exit 2; }
  API_BASE=${ANTHROPIC_API_BASE:-}   # leer = SDK-Default (api.anthropic.com)
elif [ -n "$EFFORT" ]; then
  echo "EFFORT gilt nur mit BACKEND=anthropic"; exit 2
fi
if [ "$BACKEND" = anthropic ]; then ORDER=${ORDER:-round}; else ORDER=${ORDER:-task}; fi
case "$ORDER" in task|round) ;; *) echo "ORDER muss task oder round sein, ist '$ORDER'"; exit 2;; esac
SERVE=${SERVE:-compose}
case "$SERVE" in compose|venv|none) ;; *) echo "SERVE muss compose, venv oder none sein, ist '$SERVE'"; exit 2;; esac
[ "$BACKEND" = anthropic ] || API_BASE=${API_BASE:-http://localhost:8000/v1}
SERVE_PORT=${SERVE_PORT:-8000}
[ "$SERVE" = compose ] && [ "$SERVE_PORT" != 8000 ] && { echo "SERVE=compose: der Port ist im Compose fest 8000"; exit 2; }
VLLM_BIN=${VLLM_BIN:-vllm}
MLFLOW=${MLFLOW:-1}
CHAT_TEMPLATE_KWARGS=${CHAT_TEMPLATE_KWARGS:-}
TOOL_MODE=${TOOL_MODE:-prompt}
case "$TOOL_MODE" in prompt|native) ;; *) echo "TOOL_MODE muss prompt oder native sein, ist '$TOOL_MODE'"; exit 2;; esac
if [ "$SERVE" = venv ] && ! command -v "$VLLM_BIN" >/dev/null 2>&1; then
  echo "SERVE=venv: VLLM_BIN '$VLLM_BIN' nicht gefunden"; exit 2
fi
if [ -n "$CHAT_TEMPLATE_KWARGS" ] && ! python3 -c 'import json,sys; assert isinstance(json.loads(sys.argv[1]), dict)' \
     "$CHAT_TEMPLATE_KWARGS" 2>/dev/null; then
  echo "CHAT_TEMPLATE_KWARGS ist kein JSON-Objekt: $CHAT_TEMPLATE_KWARGS"; exit 2
fi

MODEL=${1:?usage: eval_heldout.sh <MODEL> <LABEL> [OUTFILE]}
LABEL=${2:?usage: eval_heldout.sh <MODEL> <LABEL> [OUTFILE]}
SPLIT=${SPLIT:-heldout_eval}
K=${K:-8}
case "$K" in ''|*[!0-9]*|0) echo "K muss eine positive Ganzzahl sein, ist '$K'"; exit 2;; esac
KSUF=""; [ "$K" -ne 1 ] && KSUF="_k${K}"
OUT=${3:-data/generated/eval/db_traces_${SPLIT}_${LABEL}${KSUF}.jsonl}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-$((5400 * K))}
# 75 % der von vLLM gemeldeten "Maximum concurrency", gemessen 2026-07-21 bei ctx 20480 / util 0.85:
#   Qwen3-4B  KV 663.424 Tok -> 32,39x -> 24     Qwen3-8B  KV 594.048 Tok -> 29,01x -> 21
# 21 = der bindende (kleinere) Fall, deckt damit jedes SLM bis ~10B ab, ohne ans Limit zu gehen.
# Neues Modell? Einmal servieren und `docker logs text2sql_vllm_teacher | grep 'Maximum concurrency'`.
EVAL_CONC=${EVAL_CONC:-21}

# Sampling — Default ist das Qwen3-STUDENTEN-Rezept (Model-Card Qwen3-4B). Qwen3.5 faehrt nach dem
# bakeoff_dev-A/B dasselbe Rezept, braucht aber
# VLLM_EXTRA=--language-model-only. Andere Familien ueberschreiben per Env, z. B. der SDG-Teacher
# Qwen3.6 mit dem Thinking-Preset seiner Model-Card:
#   VLLM_EXTRA="--gdn-prefill-backend triton" TEMPERATURE=1.0 PRESENCE_PENALTY=1.5 \
#     bash ops/eval_heldout.sh Qwen/Qwen3.6-35B-A3B teacher_qwen3.6-35b-a3b
# Wer hier nichts setzt, misst mit dem Qwen3-Rezept — bei einer fremden Familie also potenziell
# die Fehlkonfiguration statt der Faehigkeit.
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.95}
TOP_K=${TOP_K:-20}
MIN_P=${MIN_P:-0}
PRESENCE_PENALTY=${PRESENCE_PENALTY:-0.0}
# zusaetzliche vLLM-Flags, z. B. --language-model-only bei multimodalen Modellen (Qwen3.5): laedt
# den Vision-Encoder nicht mit und gibt KV-Pool frei. Im Prompt-Modus KEIN --reasoning-parser (die K=8-Serie lief
# ohne; rollout.py parst das Think INLINE aus dem Content). Im nativen Modus (TOOL_MODE=native) ist er Pflicht:
# rollout.py liest das eigene Denkfeld und speichert es wieder inline.
VLLM_EXTRA=${VLLM_EXTRA:-}
RESUME=${RESUME:-0}; WRITE_META_ONLY=${WRITE_META_ONLY:-0}; RESUME_SUF=""
META="${OUT}.meta.json"
EXPECTED_ROWS=$(( $(python3 -c 'import json,sys;print(len(json.load(open("data/raw/db_sandbox/split_tasks.json"))[sys.argv[1]]))' "$SPLIT") * K ))

# Sidecar: die Identitaet eines Laufs. "identity" wird beim Resume verglichen, "info" nicht (Parallelitaet ist nur Tempo).
meta_json() {
  MODEL="$MODEL" LABEL="$LABEL" SPLIT="$SPLIT" K="$K" TEMPERATURE="$TEMPERATURE" TOP_P="$TOP_P" TOP_K="$TOP_K" MIN_P="$MIN_P" \
  PRESENCE_PENALTY="$PRESENCE_PENALTY" MAXLEN="${VLLM_MAX_MODEL_LEN:-20480}" GPU_UTIL="${VLLM_GPU_UTIL:-0.85}" VLLM_EXTRA="$VLLM_EXTRA" \
  RTO="${ROLLOUT_TIMEOUT_S:-1800}" EVAL_CONC="$EVAL_CONC" CTK="$CHAT_TEMPLATE_KWARGS" SERVE="$SERVE" API_BASE="$API_BASE" \
  TOOL_MODE="$TOOL_MODE" BACKEND="$BACKEND" EFFORT="$EFFORT" SDK_VERSION="${SDK_VERSION:-}" ORDER="$ORDER" \
  python3 - <<'PY'
import datetime, json, os, socket
e = os.environ
if e["BACKEND"] == "anthropic":   # API: keine Sampling-/vLLM-Felder (gibt es auf diesem Weg nicht), dafuer effort + Denkmodus
    ident = {"model": e["MODEL"], "label": e["LABEL"], "split": e["SPLIT"], "k": int(e["K"]), "backend": "anthropic",
             "effort": e["EFFORT"], "thinking": "adaptive/summarized", "enable_thinking": True, "max_turns": 15,
             "max_tokens_per_turn": 3072, "max_regen": 0, "tool_mode": "native",
             "rollout_timeout_s": float(e["RTO"]), "request_timeout_s": float(e["RTO"])}
else:
    ident = {"model": e["MODEL"], "label": e["LABEL"], "split": e["SPLIT"], "k": int(e["K"]),
             "temperature": float(e["TEMPERATURE"]), "top_p": float(e["TOP_P"]), "top_k": int(e["TOP_K"]), "min_p": float(e["MIN_P"]),
             "presence_penalty": float(e["PRESENCE_PENALTY"]), "enable_thinking": True, "max_turns": 15, "max_tokens_per_turn": 3072,
             "max_regen": 0, "max_model_len": int(e["MAXLEN"]), "gpu_util": float(e["GPU_UTIL"]), "vllm_extra": " ".join(e["VLLM_EXTRA"].split()),
             "rollout_timeout_s": float(e["RTO"]), "request_timeout_s": float(e["RTO"])}
    if e["CTK"]:   # nur wenn gesetzt -> Sidecars ohne Zusatzschalter bleiben identisch (Resume alter Laeufe)
        ident["chat_template_kwargs"] = json.loads(e["CTK"])
    if e["TOOL_MODE"] == "native":   # dito: nur der neue Modus steht drin
        ident["tool_mode"] = "native"
info = {"eval_conc": int(e["EVAL_CONC"]), "created": datetime.datetime.now().isoformat(timespec="seconds"), "host": socket.gethostname(),
        "serve": e["SERVE"]}
if e["SERVE"] == "none":
    info["api_base"] = e["API_BASE"] or "SDK-Default (api.anthropic.com)"
if e["BACKEND"] == "anthropic":
    info["sdk"] = f"anthropic {e['SDK_VERSION']}"
if e["ORDER"] == "round":   # nur wenn gesetzt -> vLLM-Sidecars bleiben unveraendert
    info["order"] = "round"
print(json.dumps({"identity": ident, "info": info}, indent=1, ensure_ascii=False))
PY
}

mkdir -p data/generated/eval logs
export MLFLOW_TRACKING_URI="file://$REPO/mlruns"   # derselbe Store wie der Training-Container

# Dieses Skript ist EIGENTUEMER von Teardown und der Ausgabedatei (beides lag frueher im Aufrufer):
#  - ohne Teardown bekaeme ein nachfolgendes Training keinen Speicher (vLLM haelt 0.85 x 128 GB)
#  - rollout.py RESUMT in eine vorhandene Datei (append + skip bekannter (task_id, sample_idx)); damit sich nie
#    zwei Modelle unter einem Label vermischen, wird die Datei entweder frisch angelegt (Default) oder nur mit
#    passender Sidecar fortgesetzt (RESUME=1) — geprueft VOR dem Serve.
# vLLM-Server-Log VOR compose down sichern (down loescht den Container samt Log): nur wenn DIESES Skript den
# Container gestartet hat (Merker), je Start eine Datei ${OUT}.vllm-<UTC>.log.gz neben der Ausgabe; danach
# Merker zurueck. Bei kill -KILL laeuft kein trap -> von Hand.
VLLM_UP=0
# SERVE=venv: der Server laeuft per setsid in einer eigenen Prozessgruppe; sein Log liegt bis zum Teardown unkomprimiert
# neben der Ausgabe und wird dann wie bei compose zu ${OUT}.vllm-<UTC>.log.gz.
VLLM_PID=""; VLLM_PGID=""
SERVE_LOG="${OUT}.vllm-serve.log"
SERVE_WAIT_S=${SERVE_WAIT_S:-1800}
save_serve_log() {   # <quelle: compose|venv>
  local f; f="${OUT}.vllm-$(date -u +%Y%m%dT%H%M%SZ).log.gz"
  if [ "$1" = compose ]; then
    if docker logs text2sql_vllm_teacher 2>&1 | gzip > "$f"; then echo "== vLLM-Log gesichert: $f"
    else rm -f "$f"; echo "== vLLM-Log NICHT gesichert (Container weg?)"; fi
  else
    if [ -f "$SERVE_LOG" ] && gzip -c "$SERVE_LOG" > "$f"; then rm -f "$SERVE_LOG"; echo "== vLLM-Log gesichert: $f"
    else rm -f "$f"; echo "== vLLM-Log NICHT gesichert ($SERVE_LOG fehlt)"; fi
  fi
}
ROLLOUT_PIDF="${OUT}.rollout.pid"
stop_rollout() {   # ein noch laufender Rollout endet VOR dem Server (siehe Aufruf unten)
  local p kids
  [ -f "$ROLLOUT_PIDF" ] || return 0
  p=$(cat "$ROLLOUT_PIDF" 2>/dev/null)
  # nur DIESEN Lauf treffen: die PID muss noch unser timeout mit genau dieser Ausgabedatei sein (alte Datei, PID neu vergeben)
  if [ -n "$p" ] && ps -o args= -p "$p" 2>/dev/null | grep -qF -- "--output $OUT"; then
    echo "== Rollout (PID $p) wird vor dem Server gestoppt"
    kids=$(pgrep -P "$p" 2>/dev/null)
    kill -TERM "$p" 2>/dev/null   # timeout reicht das an rollout.py weiter
    for _ in $(seq 30); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    # shellcheck disable=SC2086
    kill -0 "$p" 2>/dev/null && kill -KILL "$p" $kids 2>/dev/null
  fi
  rm -f "$ROLLOUT_PIDF"
}
teardown() {
  stop_rollout
  case "$SERVE" in
    compose)
      if [ "$VLLM_UP" = 1 ]; then save_serve_log compose; VLLM_UP=0; fi
      $COMPOSE --profile vllm down vllm >/dev/null 2>&1
      ;;
    venv)
      if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill -TERM -- "-${VLLM_PGID:-$VLLM_PID}" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null
        for _ in $(seq 60); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
        if kill -0 "$VLLM_PID" 2>/dev/null; then
          kill -KILL -- "-${VLLM_PGID:-$VLLM_PID}" 2>/dev/null || kill -KILL "$VLLM_PID" 2>/dev/null
        fi
      fi
      VLLM_PID=""
      if [ "$VLLM_UP" = 1 ]; then save_serve_log venv; VLLM_UP=0; fi
      ;;
    none) : ;;   # fremder Endpunkt: nichts zu stoppen
  esac
}
port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$SERVE_PORT") 2>/dev/null; }

echo "==== EVAL $LABEL ($MODEL) auf $SPLIT — k=$K — $(date '+%F %T') ===="
if [ "$BACKEND" = anthropic ]; then
  echo "   sampling: API-Default (keine Sampling-Parameter) | effort=$EFFORT | Denken adaptiv, Zusammenfassung"
  echo "   tools   : $TOOL_MODE"
  echo "   serving : Claude-API (SDK anthropic $SDK_VERSION, ${API_BASE:-api.anthropic.com}) conc=$EVAL_CONC"
else
  echo "   sampling: temp=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K min_p=$MIN_P presence=$PRESENCE_PENALTY${CHAT_TEMPLATE_KWARGS:+ template=$CHAT_TEMPLATE_KWARGS}"
  echo "   tools   : $TOOL_MODE"
  echo "   serving : $SERVE conc=$EVAL_CONC ctx=${VLLM_MAX_MODEL_LEN:-20480} extra='${VLLM_EXTRA:-–}'$([ "$SERVE" = none ] && echo " api=$API_BASE")"
fi
echo "   limits  : eval_timeout=${EVAL_TIMEOUT}s rollout_timeout=${ROLLOUT_TIMEOUT_S:-1800}s -> $OUT"
[ "$ORDER" = round ] && echo "   order   : round (Runde fuer Runde — Abbruch laesst vollstaendige Runden ueber alle Tasks)"

if [ "$WRITE_META_ONLY" = 1 ]; then
  [ -f "$OUT" ] || { echo "== WRITE_META_ONLY: $OUT fehlt — abort"; exit 1; }
  meta_json > "$META" && echo "== Sidecar geschrieben: $META ($(wc -l < "$OUT") Zeilen vorhanden)"; exit 0
fi
if [ "$RESUME" = 1 ]; then
  [ -f "$OUT" ] || { echo "== RESUME: $OUT fehlt — nichts fortzusetzen (RESUME=0 fuer einen frischen Lauf) — abort"; exit 1; }
  [ -f "$META" ] || { echo "== RESUME: Sidecar $META fehlt (Legacy-Datei) — mit den Original-Envs WRITE_META_ONLY=1 anlegen — abort"; exit 1; }
  DIFF=$(meta_json | python3 -c '
import json, sys
new = json.load(sys.stdin)["identity"]; old = json.load(open(sys.argv[1]))["identity"]
print(" ".join(sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))))' "$META")
  [ -n "$DIFF" ] && { echo "== RESUME: Sidecar passt nicht zu diesem Aufruf — abweichende Felder: $DIFF — abort"; exit 1; }
  REPAIR=$(python3 - "$OUT" <<'PY'
import json, sys
p = sys.argv[1]; lines = open(p).read().split("\n")
if lines and lines[-1] == "": lines.pop()
def ok(l):
    try: json.loads(l); return True
    except Exception: return False
bad = [i for i, l in enumerate(lines) if not ok(l)]
if bad and bad != [len(lines) - 1]: print(f"INVALID_MID Zeilen {[i + 1 for i in bad[:5]]}"); sys.exit(2)
if bad: open(p, "w").write("\n".join(lines[:-1]) + "\n"); print("TRIMMED: 1 ungueltige Endzeile verworfen")
else: print("OK")
PY
) || { echo "== RESUME: Ausgabe beschaedigt ($REPAIR) — abort"; exit 1; }
  [ "$REPAIR" != OK ] && echo "   repair  : $REPAIR"
  N_HAVE=$(wc -l < "$OUT"); RESUME_SUF="_resume"
  echo "   resume  : $N_HAVE vorhanden, $((EXPECTED_ROWS - N_HAVE)) fehlen (Soll $EXPECTED_ROWS) — Sidecar passt"
else
  rm -f "$OUT" "$META" "${OUT}.serve.json" "${OUT}".vllm-*.log.gz "$SERVE_LOG"
  meta_json > "$META"
fi
trap teardown EXIT
if [ "$SERVE" = compose ]; then
  teardown; sleep 3
elif [ "$SERVE" = venv ] && port_busy; then
  # nie fremde Prozesse beenden: ein liegengebliebener Server wird von Hand gestoppt
  echo "== SERVE=venv: Port $SERVE_PORT ist schon belegt — abort (alten Server per PID stoppen)."; exit 1
fi

if [ "$SERVE" = compose ]; then
  # NUR Standard-Transformer hier (Base 4B / merged Student) -> kein --gdn-prefill-backend,
  # der gehoert zum hybriden Teacher in ops/gen_traces.sh.
  VLLM_MODEL="$MODEL" VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}" \
    VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-20480}" \
    VLLM_EXTRA_ARGS="--max-num-seqs $EVAL_CONC $VLLM_EXTRA" \
    $COMPOSE --profile vllm up -d vllm >>logs/eval_heldout.log 2>&1
  VLLM_UP=1
elif [ "$SERVE" = venv ]; then
  # dieselben Flags und Env wie der vllm-Service in docker/docker-compose.yml, nur lokal gebunden
  # shellcheck disable=SC2086  # VLLM_EXTRA ist bewusst wortgetrennt
  VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_SAMPLER=0 \
    setsid "$VLLM_BIN" serve "$MODEL" --dtype bfloat16 \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-20480}" --gpu-memory-utilization "${VLLM_GPU_UTIL:-0.85}" \
    --served-model-name "$MODEL" --max-num-seqs "$EVAL_CONC" $VLLM_EXTRA \
    --host 127.0.0.1 --port "$SERVE_PORT" >"$SERVE_LOG" 2>&1 < /dev/null &
  VLLM_PID=$!
  VLLM_PGID=$(ps -o pgid= -p "$VLLM_PID" 2>/dev/null | tr -d ' ')
  [ "$VLLM_PGID" = "$VLLM_PID" ] || VLLM_PGID=""   # nur eine EIGENE Gruppe als Ganzes beenden
  VLLM_UP=1
  echo "   vLLM-PID: $VLLM_PID (Log $SERVE_LOG)"
fi

if [ "$SERVE" != none ]; then
  for _ in $(seq $((SERVE_WAIT_S / 10))); do
    curl -sf "localhost:$SERVE_PORT/health" >/dev/null 2>&1 && break
    if [ "$SERVE" = compose ]; then
      docker ps --format '{{.Names}}' | grep -q text2sql_vllm_teacher || break
    else
      kill -0 "$VLLM_PID" 2>/dev/null || break
    fi
    sleep 10
  done
  if ! curl -sf "localhost:$SERVE_PORT/health" >/dev/null 2>&1; then
    echo "== SERVE FAILED — abort"
    if [ "$SERVE" = compose ]; then docker logs text2sql_vllm_teacher --tail 40 2>&1 | tail -40
    else tail -40 "$SERVE_LOG" 2>/dev/null; fi
    exit 1
  fi
fi

# VLLM_MODEL reist nur als Inline-Env zu `compose up`; kommt es nicht an, greift der Compose-Fallback
# (woertlich Qwen/Qwen3-4B). Der Server liefe dann klaglos mit dem FALSCHEN Modell, rollout.py bekaeme
# 404 je Request und schriebe eine vollstaendige Datei aus lauter Nullen bei Exit 0.
# SERVE=none: ein fremder Endpunkt listet ggf. mehrere Modelle -> dort genuegt "ist gelistet".
[ "$SERVE" = none ] || API_BASE="http://localhost:$SERVE_PORT/v1"
if [ "$BACKEND" = anthropic ]; then
  # kostenlos: models.retrieve — scheitert bei falschem Key/Modell, bevor ein einziger Rollout Geld kostet
  CHECK=$(AB="$API_BASE" "$TAU2PY" -c '
import os, sys, anthropic
try:
    m = anthropic.Anthropic(base_url=os.environ.get("AB") or None, max_retries=3).models.retrieve(sys.argv[1])
    print(f"{m.id} ({m.display_name})")
except Exception as e:
    print(f"{type(e).__name__}: {str(e)[:200]}"); sys.exit(1)' "$MODEL" 2>&1) \
    || { echo "== MODEL NOT AVAILABLE: '$MODEL' — $CHECK — abort."; exit 1; }
  echo "   api     : $CHECK"
else
SERVED=$(curl -s "$API_BASE/models" | "$TAU2PY" -c \
  'import json,sys; print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))' 2>/dev/null)
if [ "$SERVE" = none ]; then
  case ",$SERVED," in *",$MODEL,"*) ;; *)
    echo "== MODEL NOT LISTED: '$MODEL' nicht in $API_BASE/models ('$SERVED') — abort."; exit 1;; esac
elif [ "$SERVED" != "$MODEL" ]; then
  echo "== WRONG MODEL SERVED: angefordert '$MODEL', serviert '$SERVED' — abort."; exit 1
fi
fi
if [ "$SERVE" = compose ]; then
  docker logs text2sql_vllm_teacher 2>&1 | grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' \
    | tail -1 | sed 's/^/   vLLM: /'
elif [ "$SERVE" = venv ]; then
  grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' "$SERVE_LOG" 2>/dev/null \
    | tail -1 | sed 's/^/   vLLM: /'
fi
# Was WIRKLICH serviert wurde (vLLM-Version/Image/Util) -> Sidecar ${OUT}.serve.json. Ohne Gate: rollout.py
# macht Prompt-and-Parse (kein tools-Parameter), die vLLM-Version wirkt hier nicht (anders als bei tau2).
# SERVE=venv: kein Container -> image_id bleibt leer, die Version kommt aus /version. SERVE=none: nichts messbar.
if [ "$SERVE" != none ]; then
  CONTAINER_ARGS=(); [ "$SERVE" = compose ] && CONTAINER_ARGS=(--container text2sql_vllm_teacher)
  "$TAU2PY" evaluation/serve_info.py record --file "${OUT}.serve.json" --role server --port "$SERVE_PORT" \
    "${CONTAINER_ARGS[@]}" --gpu-util "${VLLM_GPU_UTIL:-0.85}" 2>&1 | sed 's/^/   /'
fi
MLFLOW_ARGS=()
[ "$MLFLOW" = 1 ] && MLFLOW_ARGS=(--mlflow --mlflow-experiment "${MLFLOW_EXPERIMENT:-db_bahn_traj_eval}"
                                  --mlflow-run-name "${LABEL}${KSUF}${RESUME_SUF}")
CTK_ARGS=()
[ -n "$CHAT_TEMPLATE_KWARGS" ] && CTK_ARGS=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
TOOL_ARGS=()
[ "$TOOL_MODE" = native ] && TOOL_ARGS=(--native-tools)
[ "$ORDER" = round ] && TOOL_ARGS+=(--order round)   # nur wenn gesetzt -> vLLM-Aufruf byte-gleich
# Generierungs-Flags je Backend: vLLM = Sampling WOERTLICH (siehe unten); API = effort, keine Sampling-Parameter
if [ "$BACKEND" = anthropic ]; then
  GEN_ARGS=(--backend anthropic --effort "$EFFORT" --enable-thinking)
  [ -n "$API_BASE" ] && GEN_ARGS+=(--api-base "$API_BASE")
else
  GEN_ARGS=(--api-base "$API_BASE" --enable-thinking --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K"
            --min-p "$MIN_P" --presence-penalty "$PRESENCE_PENALTY" "${CTK_ARGS[@]}")
fi

# Ausgabe: frisch angelegt oder per RESUME fortgesetzt (siehe oben).
# Sampling-Flags WOERTLICH: der Config-Default traegt das Qwen3.6-TEACHER-Rezept (temp 1.0 /
# presence 1.5), nicht das Studenten-Rezept. Fehlt ein Flag, wird still falsch evaluiert.
# --max-regen 0 = genau ein Versuch (siehe rollout.py solve_task).
# Die Subshell schreibt ihre PID und wird per exec zum timeout-Prozess: der Teardown stoppt damit ZUERST den
# Rollout (timeout reicht SIGTERM an rollout.py weiter) und erst dann den Server — sonst liefe rollout.py nach
# einem Abbruch gegen den toten Server weiter und schriebe Fehlerzeilen, die ein Resume als erledigt zaehlt.
( echo "$BASHPID" > "$ROLLOUT_PIDF"
  exec timeout "$EVAL_TIMEOUT" env PYTHONPATH="$REPO" LOGURU_LEVEL=ERROR "$TAU2PY" \
  sdg_pipeline/db_bahn/rollout.py --config config/pipeline_config.yaml \
  --split "$SPLIT" --k "$K" --model "$MODEL" --teacher-name "$LABEL" \
  "${GEN_ARGS[@]}" "${TOOL_ARGS[@]}" \
  --max-turns 15 --max-tokens-per-turn 3072 --max-regen 0 \
  --concurrency "$EVAL_CONC" --rollout-timeout-s "${ROLLOUT_TIMEOUT_S:-1800}" --request-timeout-s "${ROLLOUT_TIMEOUT_S:-1800}" \
  "${MLFLOW_ARGS[@]}" \
  --output "$OUT" ) 2>&1 | tee -a logs/eval_heldout.log | tail -12
rc=${PIPESTATUS[0]}
rm -f "$ROLLOUT_PIDF"
if [ "$rc" -eq 3 ]; then
  echo "== API-ABBRUCH (exit 3: Guthaben/Key/Fehlerserie) — Daten sauber (keine Fehlerzeilen), RESUME=1 setzt fort =="
elif [ "$rc" -ne 0 ]; then
  echo "== ROLLOUT FAILED (exit $rc; 124 = EVAL_TIMEOUT) =="
fi

echo
"$TAU2PY" evaluation/eval_report.py --input "$OUT" | tee -a logs/eval_heldout.log
if [ "$BACKEND" = anthropic ]; then   # Token + Kosten (Listenpreise) des Laufs
  "$TAU2PY" evaluation/api_usage.py --input "$OUT" | tee -a logs/eval_heldout.log
fi
echo "==== EVAL $LABEL DONE $(date '+%F %T') -> $OUT ===="

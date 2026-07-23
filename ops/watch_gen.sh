#!/usr/bin/env bash
# ops/watch_gen.sh — Ops-Werkzeug (kein Pipeline-Schritt): beobachtet einen laufenden
# ops/gen_traces.sh und meldet Fortschritt/ETA/Health. GREIFT NICHT EIN, meldet nur.
#
# Ersetzt den blinden Wanduhr-Cap als Hang-Detektor: prüft, ob die Trace-JSONL WÄCHST.
# Der bekannte GB10-Failure-Mode (FlashInfer-Sampler-Race, vLLM #43885) zeigt sich als
# gepeggte GPU bei Durchsatz null — wall-clock sieht das nicht, ein Zeilen-Delta schon.
#
#   tmux-Pane:  ops/watch_gen.sh
#   detached :  nohup ops/watch_gen.sh >/dev/null 2>&1 &   (schreibt logs/gen_watch.log)
#
# Env-Overrides (Defaults in Klammern):
#   INTERVAL (600 s)  STALL_TICKS (3 = 30 min ohne Zuwachs -> STALL)  TARGET (13948)
#   SHORT (q36-35b-a3b-w3)  SPLIT (sft_train)  MAINLOG (logs/gen_traces.w35.log)
set -uo pipefail
cd "$(dirname "$0")/.."

INTERVAL=${INTERVAL:-600}
STALL_TICKS=${STALL_TICKS:-3}
TARGET=${TARGET:-13948}
SHORT=${SHORT:-"q36-35b-a3b-w3"}
SPLIT=${SPLIT:-"sft_train"}
TRACE="data/generated/sdg/db_traces_${SPLIT}_${SHORT}.jsonl"
MAINLOG=${MAINLOG:-"logs/gen_traces.w35.log"}
WATCHLOG="logs/gen_watch.log"
mkdir -p logs

say() { echo "$*" | tee -a "$WATCHLOG"; }

say "==== WATCH START $(date '+%F %T') — trace=$TRACE target=$TARGET interval=${INTERVAL}s ===="

lines() { [ -f "$1" ] && wc -l < "$1" || echo 0; }   # fehlende Datei = 0, kein stderr-Rauschen
# awk: NIE "cond ? a/b : c" — der Lexer liest das '/' nach '?' als Regex-Start (mawk). if/else.
per_h() { awk -v dn="$1" -v dt="$2" 'BEGIN{ if (dt>0) printf "%.0f", 3600*dn/dt; else print 0 }'; }

prev=-1; stall=0; t_prev=$(date +%s)
# Rollierendes Fenster (letzte WINDOW Ticks) statt Anker seit Start: ein Seit-Start-Mittel
# schleppt Modell-Ladezeit und Restart-Pausen ewig mit und zeigt eine ~2x zu pessimistische ETA.
WINDOW=${WINDOW:-6}
hist_t=(); hist_n=()

while :; do
  now=$(date +%s)
  n=$(lines "$TRACE")

  # --- Fortschritt / Rate / ETA ---------------------------------------------
  pct=$(awk -v n="$n" -v t="$TARGET" 'BEGIN{ if (t>0) printf "%.1f", 100*n/t; else print 0 }')
  dt=$((now - t_prev)); dn=$(( prev < 0 ? 0 : n - prev )); [ "$dn" -lt 0 ] && dn=0
  rate_now=$(per_h "$dn" "$dt")
  hist_t+=("$now"); hist_n+=("$n")
  while [ "${#hist_t[@]}" -gt $((WINDOW + 1)) ]; do hist_t=("${hist_t[@]:1}"); hist_n=("${hist_n[@]:1}"); done
  rate_avg=$(per_h "$((n - hist_n[0]))" "$((now - hist_t[0]))")
  if [ "${rate_avg:-0}" -gt 0 ] && [ "$n" -lt "$TARGET" ]; then
    eta=$(awk -v r="$rate_avg" -v rem="$((TARGET - n))" \
      'BEGIN{h=rem/r; printf "%dh%02dm", int(h), int((h-int(h))*60)}')
  else
    eta="-"
  fi

  # --- Health ---------------------------------------------------------------
  curl -sf localhost:8000/health >/dev/null 2>&1 && vllm="up" || vllm="DOWN"
  docker ps --format '{{.Names}}' 2>/dev/null | grep -q text2sql_vllm_teacher || vllm="$vllm/no-container"
  pgrep -f 'sdg_pipeline/db_bahn/rollout.py' >/dev/null 2>&1 && proc="alive" || proc="GONE"
  ram=$(free -g | awk '/^Speicher|^Mem/{print $3"/"$2"G"}')
  # verified-yield: letzte "  N/TODO  verified-yield=X%"-Zeile, die rollout.py alle 10 Rollouts druckt
  vy=$(grep -o 'verified-yield=[0-9]*%' "$MAINLOG" 2>/dev/null | tail -1)
  vy=${vy:-"verified-yield=?"}

  say "[$(date '+%F %T')] $n/$TARGET (${pct}%)  rate=${rate_now}/h avg=${rate_avg}/h  eta=$eta  ${vy}  vllm=$vllm proc=$proc ram=$ram"

  # --- Stall-Detektor -------------------------------------------------------
  if [ "$prev" -ge 0 ] && [ "$n" -le "$prev" ]; then
    stall=$((stall + 1))
    if [ "$stall" -ge "$STALL_TICKS" ]; then
      mins=$((stall * INTERVAL / 60))
      say "!!!! STALL — kein Zuwachs seit ${mins} min (vllm=$vllm proc=$proc). Pruefen:"
      say "!!!!   docker logs text2sql_vllm_teacher --tail 60"
      say "!!!!   tail -30 $MAINLOG"
      tmux rename-window -t sdg:0 "SDG-STALL" 2>/dev/null || true
    fi
  else
    [ "$stall" -gt 0 ] && say "     (Stall aufgeloest, Zuwachs wieder da)"
    stall=0
  fi

  # --- Ende erkennen --------------------------------------------------------
  if [ "$proc" = "GONE" ] && grep -q 'GEN-TRACES DONE' "$MAINLOG" 2>/dev/null; then
    say "==== GEN-TRACES DONE erkannt — Watch endet $(date '+%F %T') ===="
    tail -8 "$MAINLOG" | tee -a "$WATCHLOG"
    exit 0
  fi
  if [ "$proc" = "GONE" ] && grep -qE 'ROLLOUT .* FAILED|SERVE FAILED|FORMAT FAILED' "$MAINLOG" 2>/dev/null; then
    say "!!!! GEN-TRACES ABGEBROCHEN — Watch endet $(date '+%F %T') ===="
    tail -15 "$MAINLOG" | tee -a "$WATCHLOG"
    exit 1
  fi

  prev=$n; t_prev=$now
  sleep "$INTERVAL"
done

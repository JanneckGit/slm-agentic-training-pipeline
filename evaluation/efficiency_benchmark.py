"""
evaluation/efficiency_benchmark.py
==================================
Serving-EFFIZIENZ-Benchmark als Ergänzung zur Accuracy-Eval (evaluate.py).

Misst mit GuideLLM (vllm-project/guidellm, v0.6.0) die Serving-Performance eines
Modells, das auf dem vLLM-Service (docker-compose: ``vllm``) ausgeliefert wird,
und loggt die Effizienz-Metriken in DENSELBEN MLflow-Run wie die Accuracy-Eval
(Run-Namens-Konvention ``baseline_<model.lower()>`` — single source of truth via
``evaluate.baseline_run_name``).

Warum ein separates, ISOLIERTES guidellm?
  guidellm zieht eigenes torch/transformers/datasets nach. Es liegt deshalb in
  einem dedizierten venv unter ``/opt/guidellm`` (siehe docker/Dockerfile.training)
  und wird hier NUR per subprocess über das Binary ``/opt/guidellm/bin/guidellm``
  aufgerufen — die Trainings-Env (transformers 5.6.2, torch 2.10 nv25.11) bleibt
  unangetastet. Dieses Skript selbst läuft in der Trainings-Env (nutzt deren mlflow).

FIXE Vergleichs-Config (Defaults, für JEDES Modell IDENTISCH → Cross-Modell-
Vergleichbarkeit; NICHT pro Modell ändern):
  --profile concurrent              Last-Profil: feste Anzahl gleichzeitiger Requests
  --rate 16                         => 16 concurrent requests
  --max-requests 100                100 gemessene Requests
  --warmup 0.1                      erste 10 % als Warmup (nicht gemessen)
  --cooldown 0.1                    letzte 10 % als Cooldown (nicht gemessen)
  --request-type chat_completions   wie evaluate.py (Endpoint /v1/chat/completions)

Output: ``data/final/eval/<model>/efficiency/<mode>/{benchmarks.json,benchmarks.html}``
  (<model> = Basename der HF-ID, identisch zur ``eval/<model>/``-Konvention von
  evaluate.py; <mode> = think|nothink|default je nach Thinking-Flag, damit sich
  Thinking- und Non-Thinking-Lauf desselben Modells nicht überschreiben.)
  benchmarks.html wird als MLflow-Artefakt (artifact_path efficiency/<mode>/) angehängt.

Geloggte MLflow-Metriken (Modus-Präfix ``eff_nothink_`` bzw. ``eff_think_``, Tag
``efficiency=true``; Thinking zusätzlich ``efficiency_thinking=true``):
  <p>output_toks_per_sec, <p>total_toks_per_sec, <p>req_per_sec, <p>concurrency,
  <p>ttft_{mean,p50,p95,p99}_ms, <p>itl_{mean,p50,p95,p99}_ms,
  <p>tpot_{mean,p50,p95,p99}_ms, <p>req_latency_{mean,p95}_ms,
  <p>total_requests, <p>successful_requests, <p>errored_requests, <p>total_duration_s
  (<p> = eff_nothink_ | eff_think_).

Thinking-Modus (``--enable-thinking``, analog evaluate.py): hängt
``--backend-args '{"extras":{"body":{"chat_template_kwargs":{"enable_thinking":true}}}}'``
an guidellm an (in 0.6.0 am Quelltext + echten Lauf verifiziert). ACHTUNG: ein
top-level ``{"extra_body":...}`` wirft in 0.6.0 TypeError (--backend-args sind
Konstruktor-kwargs des OpenAI-Backends, das nur ``extras`` kennt; dessen body wird
per deep_update in den Request-Body gemergt, messages bleiben erhalten). Der
<think>-Block bleibt im content → alle Reasoning-Tokens werden mitgezählt. Lange
Outputs → höherer Request-Timeout: GUIDELLM__REQUEST_TIMEOUT wird im Thinking-Modus
auf 600 s gesetzt (per ``--request-timeout`` überschreibbar). Beide Modi koexistieren
im selben baseline_<model>-Run; eff_think_* − eff_nothink_* = die "Thinking-Steuer".
``--disable-thinking`` erzwingt enable_thinking=false (ECHTE Non-Thinking-Baseline bei
Modellen, die per Default denken, z.B. Qwen3.5-4B: 3060 vs 455 Output-Tokens); ohne
Flag bleibt es beim Template-Default (treu zu evaluate.py). Beide Non-Thinking-Wege
loggen nach eff_nothink_*; der Param eff_thinking_kwarg (true|false|default) hält fest,
was tatsächlich gesendet wurde.
Fallback (falls eine künftige guidellm-Version auch ``extras`` ablehnt): vLLM
serverseitig mit ``--default-chat-template-kwargs '{"enable_thinking": true}'`` und
OHNE ``--reasoning-parser`` starten (dann bleibt <think> im content) und hier OHNE
--enable-thinking messen.

Einheiten (aus guidellm v0.6.0 am echten benchmarks.json verifiziert): die
*_ms-Metriken (TTFT/ITL/TPOT) sind MILLISEKUNDEN; ``request_latency`` und
``duration`` sind SEKUNDEN — die Request-Latenz wird daher für die _ms-Metriken
nach ms umgerechnet.

Status-Auswahl (analog zu guidellms eigenen Konsolen-Tabellen):
  Latenzen (TTFT/ITL/TPOT/Request-Latency) -> "successful" (Completed Requests)
  Durchsatz/Rate/Concurrency               -> "total"      (All Requests)

JSON-Struktur (guidellm 0.6.0):
  report["benchmarks"][i]["metrics"][<name>][<status>]["mean" | "percentiles"][pXX]
  Ausnahme: report[...]["metrics"]["request_totals"][<status>] sind reine ints.

Usage (im Trainings-Container; vLLM muss das Modell ausliefern):
  docker compose --profile vllm up -d vllm          # vLLM mit --served-model-name <ID>
  docker compose run --rm training \\
    python evaluation/efficiency_benchmark.py --model-id Qwen/Qwen2.5-7B-Instruct
  # Thinking-Messung (gleicher Run, eff_think_*):
  docker compose run --rm training \\
    python evaluation/efficiency_benchmark.py --model-id Qwen/Qwen3.5-4B --enable-thinking

Hinweis (bewusst NICHT umgesetzt): guidellm v0.6.0 hat KEINEN direkten Output-Token-
Cap auf ``benchmark run``; ein Cap (z.B. um die Thinking-Latenz zu deckeln) ginge nur
über synthetische Daten (``--data '{"output_tokens":N}'``) oder einen
``output_tokens_count_column`` im ``--data-column-mapper`` — hier nicht verdrahtet,
weil die volle Output-Länge gerade die zu messende Thinking-Steuer IST.
"""

import argparse
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path

# Run-Namens-Konvention + Experiment-Name aus evaluate.py beziehen (single source
# of truth, wie rescore.py). evaluate.py hat KEINE top-level torch/transformers-
# Imports → der Import ist billig und berührt die Trainings-Env nicht.
from evaluation.evaluate import MLFLOW_EXPERIMENT, baseline_run_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import mlflow
except ImportError:
    mlflow = None
    logger.warning("mlflow nicht installiert – MLflow-Tracking deaktiviert")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
GUIDELLM_BIN_DEFAULT = os.environ.get("GUIDELLM_BIN", "/opt/guidellm/bin/guidellm")
# evaluate.py loggt nach file:///app/mlruns (Container-Mount). Effizienz läuft im
# selben Container → gleiche Default-URI; per --tracking-uri / $MLFLOW_TRACKING_URI
# überschreibbar.
DEFAULT_TRACKING_URI = "file:///app/mlruns"

# FIXE Vergleichs-Config (siehe Modul-Docstring).
FIXED_PROFILE = "concurrent"
FIXED_RATE = 16
FIXED_MAX_REQUESTS = 100
FIXED_WARMUP = 0.1
FIXED_COOLDOWN = 0.1

# Welche StatusBreakdown-Verteilung pro Metrik-Familie gilt.
LAT_STATUS = "successful"   # Latenzen: nur erfolgreich abgeschlossene Requests
THR_STATUS = "total"        # Durchsatz/Rate/Concurrency: alle Requests

# Default-Request-Timeout (s) im Thinking-Modus: lange Reasoning-Outputs sonst
# ggf. von guidellm gecancelt (env GUIDELLM__REQUEST_TIMEOUT).
THINKING_REQUEST_TIMEOUT = 600.0


def metric_prefix(thinking: bool) -> str:
    """Modus-Präfix für die eff_-Metriken: eff_think_ bzw. eff_nothink_. So
    koexistieren beide Messungen im selben baseline_<model>-Run; ihre Differenz
    ist die 'Thinking-Steuer'."""
    return "eff_think_" if thinking else "eff_nothink_"


def mode_slug(enable_thinking: bool, disable_thinking: bool) -> str:
    """Modus-Slug für den Output-Unterordner: think | nothink | default — die
    GLEICHE 3-Wege-Bestimmung, die in build_cmd als enable_thinking true/false/None
    in den Request-Body geht. Trennt benchmarks.{json,html} von Thinking- und Non-
    Thinking-Lauf desselben Modells auf der Platte (eigener Unterordner je Modus),
    statt sie sich gegenseitig überschreiben zu lassen. Unabhängig vom Metrik-
    Präfix (metric_prefix): 'default' loggt – wie 'nothink' – weiter unter eff_nothink_*."""
    return "think" if enable_thinking else "nothink" if disable_thinking else "default"


# ---------------------------------------------------------------------------
# guidellm-Aufruf
# ---------------------------------------------------------------------------

def build_cmd(bin_path: str, args: argparse.Namespace, out_dir: Path) -> list[str]:
    """Baut den ``guidellm benchmark run``-Aufruf. ``--target`` ist die Basis-URL
    OHNE /v1 (guidellm hängt die Route je nach --request-type selbst an)."""
    cmd = [
        bin_path, "benchmark", "run",
        "--target", args.target,
        "--request-type", args.request_type,
        "--model", args.model_id,            # served-model-name am vLLM-Endpoint
        "--processor", args.model_id,        # Tokenizer für die Token-Zählung
        "--data", args.data,
        # FLACHES JSON in v0.6.0 (KEIN {"column_mappings": ...}-Wrapper!). "question"
        # läge ohnehin in guidellms Auto-Detect-Liste – explizit für Determinismus.
        "--data-column-mapper", json.dumps({"text_column": args.text_column}),
        "--profile", args.profile,
        "--rate", str(args.rate),
        "--max-requests", str(args.max_requests),
        "--warmup", str(args.warmup),
        "--cooldown", str(args.cooldown),
        "--output-dir", str(out_dir),
        "--outputs", "json,html",            # HTML ist in v0.6.0 NICHT Default
        "--disable-console-interactive",     # saubere Logs (finale Tabellen bleiben)
    ]
    # Thinking-Toggle in den Request-BODY durchreichen. In guidellm 0.6.0 über
    # --backend-args extras.body (am echten Lauf + Quelltext verifiziert): der
    # OpenAI-Backend-Konstruktor nimmt KEIN top-level "extra_body" (→ TypeError),
    # wohl aber "extras" (GenerationRequestArguments), dessen body per
    # model_combine/deep_update in den Body gemergt wird (messages bleiben erhalten).
    #   True  → enable_thinking=true:  <think>-Block bleibt im content, alle Tokens zählen.
    #   False → enable_thinking=false: erzwingt Non-Thinking (Modelle, die per Default denken).
    #   None  → nichts senden:         Template-Default des Modells (treu zu evaluate.py).
    enable = True if args.enable_thinking else False if args.disable_thinking else None
    if enable is not None:
        cmd += ["--backend-args",
                json.dumps({"extras": {"body": {"chat_template_kwargs": {"enable_thinking": enable}}}})]
    return cmd


def preflight_target(target: str, model_id: str) -> None:
    """Prüft VOR dem guidellm-Start, dass der vLLM-Endpoint erreichbar ist und das
    Modell ausliefert. Sonst bricht guidellm nur mit einem kryptischen httpx-
    Traceback ab ("Temporary failure in name resolution" / connection refused),
    wenn der vLLM-Service nicht läuft, noch lädt oder im falschen Netz hängt."""
    import urllib.request
    from urllib.parse import urlsplit

    url = target.rstrip("/") + "/v1/models"
    host = urlsplit(target).hostname or target
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = resp.read().decode("utf-8", "replace")
    except Exception as e:
        raise SystemExit(
            f"\nvLLM-Endpoint nicht erreichbar: {url}\n"
            f"  Fehler: {e}\n"
            f"  Checkliste:\n"
            f"   1) vLLM-Service läuft?              docker compose -f docker/docker-compose.yml ps\n"
            f"   2) vLLM gesund (Modell geladen)?    docker logs text2sql_vllm_teacher --tail 50\n"
            f"   3) Hostname '{host}' ist NUR aus dem gleichen compose-Netz auflösbar →\n"
            f"      dieses Skript via 'docker compose -f docker/docker-compose.yml run --rm training ...'\n"
            f"      starten (nicht vom Host); bei Host-Ausführung --target http://localhost:8000.\n")
    if model_id and model_id not in body:
        logger.warning(f"Modell '{model_id}' nicht in {url} gelistet – stimmt der "
                       f"--served-model-name am vLLM mit --model-id überein?")
    else:
        logger.info(f"vLLM erreichbar, Modell ausgeliefert: {url}")


def run_guidellm(cmd: list[str], env: dict | None = None) -> None:
    """Führt guidellm aus und streamt dessen Ausgabe auf unsere Konsole. ``env``
    überschreibt die Umgebung (z.B. GUIDELLM__REQUEST_TIMEOUT im Thinking-Modus)."""
    logger.info("guidellm:\n  " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"guidellm beendete sich mit Code {proc.returncode}")


# ---------------------------------------------------------------------------
# benchmarks.json parsen  (Struktur am echten v0.6.0-Output verifiziert)
# ---------------------------------------------------------------------------

def _dist(metric, status: str) -> dict:
    """DistributionSummary einer Status-Verteilung – schlägt LAUT fehl (mit
    sichtbarem Pfad), falls die erwartete Verschachtelung fehlt (Versions-Drift)."""
    if not isinstance(metric, dict) or status not in metric:
        have = list(metric) if isinstance(metric, dict) else type(metric).__name__
        raise KeyError(f"Status '{status}' fehlt in Metrik – vorhanden: {have}. "
                       f"benchmarks.json-Schema von guidellm hat sich evtl. geändert.")
    return metric[status]


def parse_report(report_path: Path) -> dict:
    """Liest benchmarks.json und extrahiert die Effizienz-Metriken des (einzigen,
    bei concurrent single-rate) Benchmarks. Gibt {'metrics': {...}, 'meta': {...}}."""
    with open(report_path) as f:
        report = json.load(f)

    benches = report.get("benchmarks") or []
    if not benches:
        raise ValueError(f"Keine 'benchmarks' in {report_path}")
    if len(benches) > 1:
        logger.warning(f"{len(benches)} Benchmarks im Report – logge nur den LETZTEN "
                       f"(die fixe Config ist single-rate concurrent → unerwartet).")
    b = benches[-1]
    m = b["metrics"]

    def lat_mean(name: str) -> float:
        return _dist(m[name], LAT_STATUS)["mean"]

    def lat_pct(name: str, p: str) -> float:
        return _dist(m[name], LAT_STATUS)["percentiles"][p]

    def thr_mean(name: str) -> float:
        return _dist(m[name], THR_STATUS)["mean"]

    totals = m["request_totals"]  # reine ints: {successful, errored, incomplete, total}

    # SHORT keys (ohne Präfix). Das Modus-Präfix (eff_think_ / eff_nothink_) wird
    # erst beim Loggen/Anzeigen angehängt, damit Thinking- und Non-Thinking-Lauf
    # im selben MLflow-Run koexistieren statt sich zu überschreiben.
    eff = {
        # Durchsatz / Rate / Concurrency (alle Requests)
        "output_toks_per_sec": thr_mean("output_tokens_per_second"),
        "total_toks_per_sec": thr_mean("tokens_per_second"),
        "req_per_sec": thr_mean("requests_per_second"),
        "concurrency": thr_mean("request_concurrency"),
        # TTFT (ms)
        "ttft_mean_ms": lat_mean("time_to_first_token_ms"),
        "ttft_p50_ms": lat_pct("time_to_first_token_ms", "p50"),
        "ttft_p95_ms": lat_pct("time_to_first_token_ms", "p95"),
        "ttft_p99_ms": lat_pct("time_to_first_token_ms", "p99"),
        # ITL (ms)
        "itl_mean_ms": lat_mean("inter_token_latency_ms"),
        "itl_p50_ms": lat_pct("inter_token_latency_ms", "p50"),
        "itl_p95_ms": lat_pct("inter_token_latency_ms", "p95"),
        "itl_p99_ms": lat_pct("inter_token_latency_ms", "p99"),
        # TPOT (ms)
        "tpot_mean_ms": lat_mean("time_per_output_token_ms"),
        "tpot_p50_ms": lat_pct("time_per_output_token_ms", "p50"),
        "tpot_p95_ms": lat_pct("time_per_output_token_ms", "p95"),
        "tpot_p99_ms": lat_pct("time_per_output_token_ms", "p99"),
        # Request-Latenz: in SEKUNDEN gespeichert → ×1000 für die _ms-Metriken
        "req_latency_mean_ms": lat_mean("request_latency") * 1000.0,
        "req_latency_p95_ms": lat_pct("request_latency", "p95") * 1000.0,
        # Counts + Dauer
        "total_requests": float(totals["total"]),
        "successful_requests": float(totals["successful"]),
        "errored_requests": float(totals["errored"]),
        "total_duration_s": float(b["duration"]),
    }
    meta = {
        "guidellm_version": (report.get("metadata") or {}).get("guidellm_version"),
        "args": report.get("args") or {},
    }
    return {"metrics": eff, "meta": meta}


# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

def resolve_tracking_uri(arg: str | None) -> str:
    """--tracking-uri > $MLFLOW_TRACKING_URI > file:///app/mlruns (Container-Mount)."""
    if arg:
        return arg
    return os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI


def log_to_mlflow(run_name: str, tracking_uri: str, model_id: str,
                  eff: dict, meta: dict, params: dict,
                  html_path: Path, json_path: Path, thinking: bool = False,
                  mode: str = "default") -> None:
    """Findet den baseline_<model>-Run (bei Duplikaten den zuletzt gestarteten),
    setzt ihn per start_run(run_id=...) fort und loggt die Metriken mit dem
    Modus-Präfix (eff_think_ / eff_nothink_); legt sonst einen neuen Run mit der
    Konvention an (+ Warnung). Hängt benchmarks.html (+ .json) als Artefakt unter
    efficiency/<mode>/ an (modus-getrennt, sonst überschreiben sich die Läufe auch
    im Artifact-Store) und taggt efficiency=true (Thinking zusätzlich efficiency_thinking=true)."""
    if mlflow is None:
        logger.warning("mlflow nicht verfügbar – überspringe MLflow-Logging.")
        return
    from mlflow.tracking import MlflowClient

    # Run-Auflösung. Wie rescore.py/evaluate.py NICHT crashen, wenn der Store nicht
    # erreichbar/ungültig ist: der Benchmark ist fertig, benchmarks.json/.html liegen
    # lokal vor und print_summary lief bereits → nur warnen und aussteigen.
    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
        runs = client.search_runs([exp.experiment_id],
                                  order_by=["attributes.start_time DESC"],
                                  max_results=2000) if exp is not None else []
    except Exception as e:
        logger.warning(f"[mlflow] Store @ {tracking_uri} nicht erreichbar ({e}); "
                       f"überspringe MLflow-Logging (benchmarks.json/.html liegen lokal vor).")
        return

    run_id = None
    # newest-first → erster Namens-Treffer ist der zuletzt gestartete Run.
    for run in runs:
        if run.data.tags.get("mlflow.runName", "") == run_name:
            run_id = run.info.run_id
            break

    created = run_id is None
    try:
        if created:
            logger.warning(f"[mlflow] kein Run '{run_name}' in '{MLFLOW_EXPERIMENT}' "
                           f"@ {tracking_uri} – lege NEUEN (Effizienz-only) Run an.")
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            ctx = mlflow.start_run(run_name=run_name)
        else:
            logger.info(f"[mlflow] setze Run '{run_name}' ({run_id}) @ {tracking_uri} fort.")
            ctx = mlflow.start_run(run_id=run_id)

        # `with ctx:` beendet den Run auch bei einer Exception im Block (kein
        # dangling active run), bevor das äußere except greift.
        prefix = metric_prefix(thinking)
        with ctx:
            for k, v in eff.items():
                if v is not None:
                    mlflow.log_metric(prefix + k, float(v))
            mlflow.set_tag("efficiency", "true")
            if thinking:
                mlflow.set_tag("efficiency_thinking", "true")
            if created:
                mlflow.log_param("model_id", model_id)
                mlflow.set_tag("efficiency_created_run", "true")
            # Provenance-Params (eff_-Präfix → kein Clash mit eval-Params; gleicher
            # Wert bei erneutem Lauf ist idempotent).
            if meta.get("guidellm_version"):
                params = {**params, "eff_guidellm_version": meta["guidellm_version"]}
            for k, v in params.items():
                try:
                    mlflow.log_param(k, v)
                except Exception as e:  # bereits mit anderem Wert gesetzt o.ä.
                    logger.warning(f"[mlflow] log_param {k}={v} übersprungen ({e})")
            # Artefakte modus-getrennt ablegen (efficiency/<mode>/), analog zum
            # Disk-Output: sonst überschreibt der eine Modus die benchmarks.{json,html}
            # des anderen auch im MLflow-Artifact-Store (genau das hatte den früheren
            # Thinking-Report vernichtet – die Metriken via eff_*-Präfix blieben erhalten).
            for p in (html_path, json_path):
                if p and Path(p).exists():
                    mlflow.log_artifact(str(p), artifact_path=f"efficiency/{mode}")
    except Exception as e:
        logger.warning(f"[mlflow] Logging fehlgeschlagen ({e}); "
                       f"benchmarks.json/.html liegen lokal vor.")
        return

    where = "NEU angelegt" if created else f"fortgesetzt ({run_id})"
    logger.info(f"[mlflow] {metric_prefix(thinking)}*-Metriken geloggt; Run {where}.")


# ---------------------------------------------------------------------------
# Konsolen-Zusammenfassung
# ---------------------------------------------------------------------------

def print_summary(model_id: str, eff: dict, out_dir: Path, thinking: bool, mode_label: str) -> None:
    logger.info("\n" + "=" * 64)
    logger.info(f"EFFICIENCY BENCHMARK ({mode_label})  —  {model_id}")
    logger.info(f"  MLflow-Präfix: {metric_prefix(thinking)}*")
    logger.info("=" * 64)
    logger.info(f"  Output throughput : {eff['output_toks_per_sec']:.1f} tok/s")
    logger.info(f"  Total throughput  : {eff['total_toks_per_sec']:.1f} tok/s")
    logger.info(f"  Requests/sec      : {eff['req_per_sec']:.2f}")
    logger.info(f"  Concurrency (mean): {eff['concurrency']:.2f}")
    logger.info(f"  TTFT  ms  mean/p95/p99 : {eff['ttft_mean_ms']:.1f} / "
                f"{eff['ttft_p95_ms']:.1f} / {eff['ttft_p99_ms']:.1f}")
    logger.info(f"  ITL   ms  mean/p95     : {eff['itl_mean_ms']:.2f} / "
                f"{eff['itl_p95_ms']:.2f}")
    logger.info(f"  TPOT  ms  mean/p95     : {eff['tpot_mean_ms']:.2f} / "
                f"{eff['tpot_p95_ms']:.2f}")
    logger.info(f"  Req latency ms mean/p95: {eff['req_latency_mean_ms']:.1f} / "
                f"{eff['req_latency_p95_ms']:.1f}")
    logger.info(f"  Requests (ok/err/total): {int(eff['successful_requests'])} / "
                f"{int(eff['errored_requests'])} / {int(eff['total_requests'])}")
    logger.info(f"  Duration          : {eff['total_duration_s']:.1f} s")
    logger.info("=" * 64)
    logger.info(f"  Report: {out_dir}/benchmarks.json (+ .html)")
    logger.info("=" * 64)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GuideLLM-Effizienz-Benchmark gegen den vLLM-Endpoint, geloggt "
                    "in den MLflow-Run des Modells (eff_-Metriken).")
    parser.add_argument("--model-id", required=True,
                        help="HF-ID = served-model-name am vLLM-Endpoint UND "
                             "--processor für die Token-Zählung (z.B. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--target", default="http://vllm:8000",
                        help="Basis-URL des Backends OHNE /v1 (guidellm hängt die "
                             "Route selbst an). Default http://vllm:8000")
    parser.add_argument("--request-type", default="chat_completions",
                        help="guidellm request-format (wie evaluate.py: chat_completions "
                             "→ /v1/chat/completions). Default chat_completions")
    parser.add_argument("--data", default="data/final/test_clean.jsonl",
                        help="Prompt-Datensatz (jsonl). Default data/final/test_clean.jsonl")
    parser.add_argument("--text-column", default="question",
                        help="Spalte mit dem Prompt-Text. Default 'question'")
    parser.add_argument("--mlflow-run-name", default=None,
                        help="MLflow-Run-Name. Default-Konvention baseline_<model.lower()> "
                             "(wie evaluate.py/rescore.py).")
    parser.add_argument("--tracking-uri", default=None,
                        help="MLflow Tracking-URI. Default $MLFLOW_TRACKING_URI oder "
                             f"{DEFAULT_TRACKING_URI}")
    parser.add_argument("--output-base", default="data/final/eval",
                        help="Basisverzeichnis; Output landet unter "
                             "<base>/<model>/efficiency/<think|nothink|default>/. "
                             "Default data/final/eval")
    parser.add_argument("--guidellm-bin", default=GUIDELLM_BIN_DEFAULT,
                        help=f"Pfad zum isolierten guidellm-Binary. Default {GUIDELLM_BIN_DEFAULT}")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Nur Report erzeugen, kein MLflow-Logging.")
    # Thinking-Steuerung (gegenseitig exklusiv). Ohne Flag: Template-Default des
    # Modells (treu zu evaluate.py); Metriken dann ebenfalls eff_nothink_*.
    think_group = parser.add_mutually_exclusive_group()
    think_group.add_argument("--enable-thinking", action="store_true",
                        help="Misst im Thinking-Modus (analog evaluate.py --enable-thinking): sendet "
                             "chat_template_kwargs.enable_thinking=true. Metriken → eff_think_*.")
    think_group.add_argument("--disable-thinking", action="store_true",
                        help="Misst explizit OHNE Thinking (chat_template_kwargs.enable_thinking=false). "
                             "Nötig für eine ECHTE Non-Thinking-Baseline bei Modellen, die per Default "
                             "denken (z.B. Qwen3.5-4B: 3060 vs 455 Output-Tokens). Metriken → eff_nothink_*. "
                             "Ohne dieses Flag: Template-Default (ebenfalls eff_nothink_).")
    parser.add_argument("--request-timeout", type=float, default=None,
                        help="guidellm Request-Timeout (s) via GUIDELLM__REQUEST_TIMEOUT. Default: "
                             f"{int(THINKING_REQUEST_TIMEOUT)} im Thinking-Modus (lange Reasoning-"
                             "Outputs sonst gecancelt), sonst guidellm-Default.")
    # FIXE Vergleichs-Config – überschreibbar, aber für Cross-Modell-Vergleiche
    # IDENTISCH halten (Abweichung wird gewarnt).
    parser.add_argument("--profile", default=FIXED_PROFILE)
    parser.add_argument("--rate", default=FIXED_RATE)
    parser.add_argument("--max-requests", type=int, default=FIXED_MAX_REQUESTS)
    parser.add_argument("--warmup", type=float, default=FIXED_WARMUP)
    parser.add_argument("--cooldown", type=float, default=FIXED_COOLDOWN)
    args = parser.parse_args()

    # Comparability-Guard: warnen, falls die fixe Last-Config verändert wurde.
    changed = []
    if str(args.profile) != FIXED_PROFILE: changed.append(f"profile={args.profile}")
    if str(args.rate) != str(FIXED_RATE): changed.append(f"rate={args.rate}")
    if args.max_requests != FIXED_MAX_REQUESTS: changed.append(f"max_requests={args.max_requests}")
    if args.warmup != FIXED_WARMUP: changed.append(f"warmup={args.warmup}")
    if args.cooldown != FIXED_COOLDOWN: changed.append(f"cooldown={args.cooldown}")
    if changed:
        logger.warning("FIXE Vergleichs-Config verändert (" + ", ".join(changed) +
                       ") – Cross-Modell-Vergleichbarkeit nicht mehr garantiert.")

    if not Path(args.guidellm_bin).exists():
        raise SystemExit(
            f"guidellm-Binary nicht gefunden: {args.guidellm_bin}. Es liegt im "
            f"isolierten venv aus docker/Dockerfile.training (RUN python3 -m venv "
            f"/opt/guidellm ...). Image neu bauen oder --guidellm-bin setzen.")
    if not Path(args.data).exists():
        raise SystemExit(f"Datensatz nicht gefunden: {args.data}")

    model_name = Path(args.model_id).name  # "Qwen/Qwen2.5-7B-Instruct" -> "Qwen2.5-7B-Instruct"
    # Output je Modus in einen eigenen Unterordner (think|nothink|default), sonst
    # überschreiben sich benchmarks.{json,html} von Thinking- und Non-Thinking-Lauf
    # desselben Modells gegenseitig (MLflow trennt sie via eff_think_/eff_nothink_,
    # die Dateien auf Platte gingen sonst verloren). Derselbe Modus-Slug steuert
    # auch das mode_label – eine einzige Bestimmung, keine parallele Logik.
    mode = mode_slug(args.enable_thinking, args.disable_thinking)
    out_dir = Path(args.output_base) / model_name / "efficiency" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.mlflow_run_name or baseline_run_name(model_name)

    mode_label = {
        "think": "THINKING (enable_thinking=true)",
        "nothink": "non-thinking (enable_thinking=false)",
        "default": "non-thinking (Template-Default)",
    }[mode]
    logger.info(f"Modell: {args.model_id} | target: {args.target} | run: {run_name} "
                f"| Modus: {mode_label} ({metric_prefix(args.enable_thinking)}*)")
    logger.info(f"Last-Config: profile={args.profile} rate={args.rate} "
                f"max_requests={args.max_requests} warmup={args.warmup} cooldown={args.cooldown}")

    # Request-Timeout: im Thinking-Modus hochsetzen, sonst werden lange Reasoning-
    # Requests von guidellm gecancelt (→ errored). Per env an den Subprozess.
    timeout = args.request_timeout
    if timeout is None and args.enable_thinking:
        timeout = THINKING_REQUEST_TIMEOUT
    env = None
    if timeout is not None:
        env = {**os.environ, "GUIDELLM__REQUEST_TIMEOUT": str(timeout)}
        logger.info(f"GUIDELLM__REQUEST_TIMEOUT={timeout}")

    # 0. Pre-Flight: klare Fehlermeldung statt guidellm-httpx-Traceback, wenn
    #    der vLLM-Endpoint nicht erreichbar ist (Service down/lädt noch/falsches Netz).
    preflight_target(args.target, args.model_id)

    # 1. Benchmark
    run_guidellm(build_cmd(args.guidellm_bin, args, out_dir), env=env)

    json_path = out_dir / "benchmarks.json"
    html_path = out_dir / "benchmarks.html"
    if not json_path.exists():
        raise SystemExit(f"guidellm lief, aber {json_path} fehlt – Output-Flags prüfen.")

    # 2. Parsen
    parsed = parse_report(json_path)
    eff, meta = parsed["metrics"], parsed["meta"]

    # 3. Zusammenfassung
    print_summary(args.model_id, eff, out_dir, args.enable_thinking, mode_label)

    # 4. MLflow
    if not args.no_mlflow:
        # Shared Load-Config-Params (für beide Modi identisch → idempotent). Den
        # Timeout nur protokollieren, wenn gesetzt.
        params = {
            "eff_profile": args.profile,
            "eff_rate": args.rate,
            "eff_max_requests": args.max_requests,
            "eff_warmup": args.warmup,
            "eff_cooldown": args.cooldown,
            "eff_request_type": args.request_type,
            "eff_target": args.target,
            "eff_data": Path(args.data).name,
        }
        if timeout is not None:
            params["eff_request_timeout"] = timeout
        # Festhalten, WAS gesendet wurde (true/false = explizit, default = Template).
        params["eff_thinking_kwarg"] = ("true" if args.enable_thinking
                                        else "false" if args.disable_thinking else "default")
        log_to_mlflow(run_name, resolve_tracking_uri(args.tracking_uri), args.model_id,
                      eff, meta, params, html_path, json_path, thinking=args.enable_thinking,
                      mode=mode)
    else:
        logger.info("--no-mlflow gesetzt – kein MLflow-Logging.")


if __name__ == "__main__":
    main()

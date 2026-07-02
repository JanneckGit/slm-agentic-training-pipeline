"""
sdg_pipeline/trace_capture.py
=============================
Trace-distillation capture harness (verification + production).

Calls the teacher model (vLLM, OpenAI-compatible endpoint) with thinking ENABLED
on seed examples, then splits each raw teacher output into two fields:

  - ``thinking``: the reasoning trace, i.e. everything BEFORE the last ``</think>``
  - ``sql``:      the final SQL query, extracted via the v2 extractor
                  (``evaluation.evaluate.extract_sql``)

This is exactly the plumbing the large SDG trace-distillation run scales up. It
deliberately REUSES the eval-time SQL extractor and prompt construction so that
teacher generation, training targets, and evaluation all agree on how the
``<think>...</think>`` / SQL boundary is drawn.

Usage (inside the training container, on the compose network so ``vllm`` resolves):
    # full 750-example distillation run (concurrent, resumable, append-only)
    python sdg_pipeline/trace_capture.py \
        --config config/pipeline_config.local.yaml \
        --n-samples 750 --concurrency 16 \
        --output data/generated/trace_distill.jsonl

The teacher endpoint + served-model-name are read from the active vllm_local
teacher config (api_base, model), matching run_sdg.py's resolution.

Production robustness:
  - Generation cap: --max-tokens (default 6144) bounds runaway traces; any
    example that hits the cap (finish_reason="length") or comes back without a
    clean closing ``</think>`` is a TRUNCATED trace and is DROPPED, not trained.
  - Resumable: output is appended per kept example and flushed immediately; on
    restart, examples already in --output are skipped, so an interrupted
    over-night run loses nothing and a re-run continues where it stopped.
"""

import argparse
import json
import logging
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx
import yaml

# Reuse the v2 SQL extractor, prompt construction, and the SQLite execution
# harness from the eval code — single source of truth, no reimplementation.
from evaluation.evaluate import extract_sql, build_messages, execute_sql_on_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("trace_capture")

try:
    import mlflow
except ImportError:
    mlflow = None
    logger.warning("mlflow not installed — SDG run will not be tracked")

# Own experiment for SDG runs — kept separate from evaluate.py's
# "text2sql-slm-finetuning" baseline_* runs so distillation provenance does not
# pollute the eval leaderboard.
MLFLOW_EXPERIMENT = "sdg"
MLFLOW_TRACKING_URI = "file:///app/mlruns"


def _git_commit() -> str | None:
    """Best-effort short git SHA for provenance; never raises."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _pct(values: list[float], p: float) -> float:
    """p-th percentile (0-100) via nearest-rank; safe on empty/short lists."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def split_thinking_sql(raw_output: str) -> tuple[str, str]:
    """
    Splits a raw thinking-teacher output into (thinking, sql).

    The boundary is the LAST ``</think>`` — identical to the rule the v2
    extractor (extract_sql) uses, so trace and SQL are exact complements:
      - thinking = text before the last ``</think>`` (leading ``<think>`` removed)
      - sql      = extract_sql(raw_output)  (strips reasoning, unwraps ``` fences)
    """
    if "</think>" in raw_output:
        thinking = raw_output.rsplit("</think>", 1)[0]
        # Qwen3.x usually emits an opening <think>; drop it if present so the
        # stored trace is the reasoning text only.
        thinking = thinking.split("<think>", 1)[-1].strip()
    else:
        thinking = ""
    sql = extract_sql(raw_output)
    return thinking, sql


# ---------------------------------------------------------------------------
# Trace-quality control (the 2026-06-11 redo fix)
# ---------------------------------------------------------------------------
# The first distill run captured the thinking-teacher's RAW <think> verbatim —
# verbose, self-doubting ("Wait… Actually… Or maybe…"). Students distilled that
# style and looped. Fix at the SOURCE, not at inference:
#   (1) NUDGE the teacher toward brief, committed reasoning (system prompt),
#   (2) FILTER out the still-degenerate traces (too long / hedge-heavy / repeating),
#   (3) REGENERATE a filtered example a few times before dropping it.
# Eval stays pure greedy so the </think>-close-rate is the honest test that this
# worked.

TRACE_SYSTEM_PROMPT = (
    "You are an expert SQL query writer. Reason through the problem BRIEFLY and "
    "DIRECTLY, then write the final SQL.\n"
    "- In a few short steps: identify the needed tables, the joins, the filters, "
    "and any aggregation/ordering.\n"
    "- COMMIT to your answer. Do NOT second-guess, do NOT reconsider, do NOT "
    "explore alternatives. Never write 'wait', 'actually', 'hmm', 'or maybe', "
    "'alternatively', or 'let me reconsider'.\n"
    "- Keep the reasoning to a handful of sentences, then output the SQL and STOP "
    "— do not verify, confirm, restate, or re-check it afterwards."
)

# Hedge markers whose density signals the verbose self-doubt that becomes a loop.
HEDGE_MARKERS = ("wait", "actually", "hmm", "or maybe", "alternatively",
                 "let me reconsider", "on second thought", "or perhaps")


def build_trace_messages(question: str, schema: str) -> list[dict]:
    """Prompt for trace GENERATION — uses the nudge system prompt (not the terse
    eval one) so the teacher's reasoning is brief and committed."""
    user = f"Database schema:\n{schema}\n\nQuestion: {question}"
    return [{"role": "system", "content": TRACE_SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def trace_quality_reason(thinking: str, max_chars: int, max_hedge: int,
                         min_uniqword: float) -> str | None:
    """Returns a drop-reason if the (clean-closed) trace is degenerate, else None.
    Loose by design — only kills the broken ones; the nudge handles brevity."""
    if len(thinking) > max_chars:
        return "drop_long"
    low = thinking.lower()
    if sum(low.count(m) for m in HEDGE_MARKERS) > max_hedge:
        return "drop_hedge"
    words = thinking.split()
    if words and (len(set(w.lower() for w in words)) / len(words)) < min_uniqword:
        return "drop_repetition"
    return None


def make_teacher_call(api_base: str, model: str, api_key: str,
                      max_tokens: int, temperature: float,
                      timeout: float = 1200.0, max_retries: int = 0):
    """
    Returns call(question, schema) -> (raw_output, finish_reason) for a thinking
    teacher.

    Mirrors evaluate.make_endpoint_predict_fn: same build_messages, same
    chat_template_kwargs={"enable_thinking": True} toggle passed through vLLM's
    OpenAI-compatible endpoint to the chat template.

    finish_reason is the OpenAI field: "stop" (model emitted EOS normally),
    "length" (hit the max_tokens cap → the trace is TRUNCATED), or "error" when
    the request ultimately failed. The caller uses it to drop truncated traces.

    timeout default 1200s + max_retries=0 (1 attempt): the MoE teacher is fast,
    so 1200s is generous margin and a genuine hang fails quickly (~20 min) rather
    than blocking a worker for ~90 min across 3 attempts. A failed request is
    isolated and dropped by the caller — it never aborts the run.
    """
    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    client = httpx.Client(timeout=timeout)

    def call(question: str, schema: str) -> tuple[str, str]:
        payload = {
            "model": model,
            "messages": build_trace_messages(question, schema),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
        }
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                choice = resp.json()["choices"][0]
                raw = choice["message"]["content"] or ""
                finish_reason = choice.get("finish_reason") or "stop"
                return raw, finish_reason
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
        logger.warning(f"Teacher request failed after {max_retries + 1} attempts: {last_err}")
        return f"<API_ERROR: {last_err}>", "error"

    return call


def resolve_teacher(config: dict) -> dict:
    """Active teacher backend config (only vllm_local is meaningful here)."""
    backend = config["teacher"]["backend"]
    cfg = dict(config["teacher"][backend])
    cfg["backend"] = backend
    return cfg


def _log_mlflow_run(args, model: str, counters: dict, stats: dict,
                    wall_clock: float, n_total: int):
    """
    Logs this SDG run to MLflow (experiment "sdg") for provenance + comparability.
    Entirely best-effort: any failure is swallowed so tracking never crashes the
    run. The JSONL output is the product; MLflow is a side-channel.
    """
    if mlflow is None:
        return
    try:
        n_kept = counters["kept"]
        n_dropped = sum(v for k, v in counters.items() if k != "kept")
        lat = stats["latency"]
        th = stats["thinking_chars"]
        sq = stats["sql_chars"]

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"sdg_{model}_{ts}".replace("/", "_")
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "teacher_model_id": model,
                "n_samples": args.n_samples,
                "concurrency": args.concurrency,
                "max_tokens": args.max_tokens,
                "timeout_s": args.timeout,
                "temperature": args.temperature,
                "enable_thinking": True,
                "nudge_prompt": True,
                "max_thinking_chars": args.max_thinking_chars,
                "max_hedge": args.max_hedge,
                "min_uniqword": args.min_uniqword,
                "max_regen": args.max_regen,
                "seed_file": args.seed_file or "",
                "output_path": args.output,
                "git_commit": _git_commit() or "unknown",
            })
            metrics = {
                "n_total": n_total,
                "n_kept": n_kept,
                "n_dropped": n_dropped,
                "drop_length": counters["drop_length"],
                "drop_no_think": counters["drop_no_think"],
                "drop_error": counters["drop_error"],
                "drop_long": counters["drop_long"],
                "drop_hedge": counters["drop_hedge"],
                "drop_repetition": counters["drop_repetition"],
                "keep_rate": (n_kept / n_total) if n_total else 0.0,
                "total_wall_clock_s": round(wall_clock, 1),
                "throughput_ex_per_s": (n_total / wall_clock) if wall_clock > 0 else 0.0,
                "thinking_chars_median": statistics.median(th) if th else 0,
                "thinking_chars_max": max(th) if th else 0,
                "sql_chars_median": statistics.median(sq) if sq else 0,
                "sql_chars_max": max(sq) if sq else 0,
                "latency_median_s": round(statistics.median(lat), 1) if lat else 0.0,
                "latency_p95_s": round(_pct(lat, 95), 1),
            }
            mlflow.log_metrics(metrics)

            # Optional: log the final JSONL as an artifact (training-data provenance).
            out_path = Path(args.output)
            if out_path.exists():
                try:
                    mlflow.log_artifact(str(out_path), artifact_path="trace_distill")
                except Exception as e:
                    logger.warning(f"MLflow artifact log skipped: {e}")
        logger.info(f"MLflow run logged: experiment='{MLFLOW_EXPERIMENT}' run='{run_name}'")
    except Exception as e:
        logger.warning(f"MLflow logging failed (run unaffected): {e}")


def main():
    p = argparse.ArgumentParser(description="Capture teacher thinking traces for distillation")
    p.add_argument("--config", default="config/pipeline_config.yaml")
    p.add_argument("--seed-file", default=None, help="Override seed JSONL path")
    p.add_argument("--n-samples", type=int, default=8, help="How many seed examples to process")
    p.add_argument("--output", default="data/generated/trace_verify.jsonl")
    p.add_argument("--max-tokens", type=int, default=6144,
                   help="Generation budget. Thinking traces are ~3000+ tokens, "
                        "so keep this well above that to avoid truncation.")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--show", type=int, default=3, help="Print full diagnostics for the first N kept")
    p.add_argument("--concurrency", type=int, default=16,
                   help="Parallel teacher requests. Uses vLLM continuous batching "
                        "(big speedup on the GB10).")
    p.add_argument("--timeout", type=float, default=1200.0,
                   help="Per-request HTTP timeout (s). The MoE teacher is fast; 1200s "
                        "is margin and fails a genuine hang quickly (1 attempt, no retry).")
    # Trace-quality filter + regenerate (the redo fix). Loose defaults — only kill
    # degenerate traces; the nudge prompt handles brevity. Tune at Gate 2b.
    p.add_argument("--max-thinking-chars", type=int, default=4000,
                   help="Drop a trace longer than this (~1000 tok). Runaway guard — "
                        "loose; the nudge handles brevity, this only kills extremes.")
    p.add_argument("--max-hedge", type=int, default=4,
                   help="Drop a trace with more than this many hedge markers "
                        "(wait/actually/or maybe/...). Loop-style guard.")
    p.add_argument("--min-uniqword", type=float, default=0.3,
                   help="Drop a trace with unique-word ratio below this (repetition loops).")
    p.add_argument("--max-regen", type=int, default=2,
                   help="Re-sample a dropped trace up to N extra times before giving up.")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    teacher = resolve_teacher(config)
    api_base = teacher["api_base"]
    model = teacher.get("model") or teacher.get("deployment_name")
    api_key = teacher.get("api_key", "token-local")
    logger.info(f"Teacher endpoint: {api_base} | served-model-name: {model}")
    logger.info(f"Thinking: ON | max_tokens={args.max_tokens} | temperature={args.temperature}")

    seed_path = Path(args.seed_file) if args.seed_file else \
        Path(config["data"]["raw_dir"]) / "seed_sample.jsonl"
    if not seed_path.exists():
        logger.error(f"Seed file not found: {seed_path}")
        sys.exit(1)

    examples = []
    with open(seed_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
            if len(examples) >= args.n_samples:
                break
    logger.info(f"Loaded {len(examples)} seed examples from {seed_path}")

    # -- Resume: skip examples already present in the output file. Keyed on
    #    (question, schema) since the seed has no stable id. A crashed over-night
    #    run can be re-launched with the same args and picks up where it left off.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def key(ex: dict) -> tuple[str, str]:
        return (ex.get("question", "").strip(), ex.get("schema", "").strip())

    done_keys = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done_keys.add(key(json.loads(line)))
                    except json.JSONDecodeError:
                        continue
        logger.info(f"Resume: {len(done_keys)} examples already in {out_path} → skipping them")

    todo = [ex for ex in examples if key(ex) not in done_keys]
    logger.info(f"To process this run: {len(todo)} (skipped {len(examples) - len(todo)} already done)")
    if not todo:
        logger.info("Nothing to do — output already complete for this seed slice.")
        return

    call = make_teacher_call(api_base, model, api_key, args.max_tokens,
                             args.temperature, timeout=args.timeout)

    # Incremental, append-only writes guarded by a lock so concurrent workers
    # never interleave a line. Each kept example is durably on disk immediately.
    import threading
    write_lock = threading.Lock()
    out_f = open(out_path, "a")
    kept_records = []        # in-memory, kept only, for end-of-run diagnostics
    counters = {"kept": 0, "drop_length": 0, "drop_no_think": 0, "drop_error": 0,
                "drop_long": 0, "drop_hedge": 0, "drop_repetition": 0}
    stats = {"latency": [], "thinking_chars": [], "sql_chars": []}  # for MLflow metrics

    def _process_one(i: int, ex: dict):
        question = ex.get("question", "")
        schema = ex.get("schema", "")
        t0 = time.time()
        # -- Truncation + quality filter, with REGENERATE. A trace is dropped if
        #    it was cut mid-thought (finish=length / no </think>) OR degenerate
        #    (too long / hedge-heavy / repeating). Because temperature>0, we
        #    re-sample a failed example up to max_regen extra times before
        #    giving up — a question the teacher rambled on first often comes back
        #    clean on a retry, so we keep hard examples instead of dropping them.
        raw, finish_reason, thinking, sql, drop_reason, attempt = "", "stop", "", "", None, 0
        for attempt in range(args.max_regen + 1):
            raw, finish_reason = call(question, schema)
            thinking, sql = split_thinking_sql(raw)
            if finish_reason == "error":
                drop_reason = "drop_error"
            elif finish_reason == "length":
                drop_reason = "drop_length"        # hit max_tokens → truncated
            elif "</think>" not in raw:
                drop_reason = "drop_no_think"       # no clean closing tag → truncated/malformed
            else:
                drop_reason = trace_quality_reason(
                    thinking, args.max_thinking_chars, args.max_hedge, args.min_uniqword)
            if drop_reason is None:
                break                                # clean + good → keep this attempt
        dt = time.time() - t0

        if drop_reason:
            with write_lock:
                counters[drop_reason] += 1
                stats["latency"].append(dt)
            logger.warning(f"[{i+1}/{len(todo)}] DROP ({drop_reason}) after "
                           f"{attempt+1} attempt(s) finish={finish_reason} {dt:.1f}s "
                           f"thinking={len(thinking)}c | {question[:60]}")
            return

        record = {
            "question": question,
            "schema": schema,
            "sql": sql,                 # teacher's final SQL (distillation target)
            "thinking": thinking,       # teacher reasoning trace
            "gold_sql": ex.get("sql", ""),  # seed reference, for QA only
            "complexity": ex.get("complexity", "unknown"),
            "domain": ex.get("domain", "unknown"),
            "task_type": ex.get("task_type", "unknown"),
            "source": "teacher_trace_distill",
        }
        with write_lock:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            counters["kept"] += 1
            kept_records.append({**record, "_raw": raw})
            stats["latency"].append(dt)
            stats["thinking_chars"].append(len(thinking))
            stats["sql_chars"].append(len(sql))
        logger.info(f"[{i+1}/{len(todo)}] kept {dt:.1f}s attempt={attempt+1} "
                    f"finish={finish_reason} thinking={len(thinking)}c sql={len(sql)}c")

    def process(i: int, ex: dict):
        # Per-request error isolation (the pipeline fix): a timeout, a malformed
        # response, or ANY unexpected exception is caught here, counted as a drop,
        # and the loop continues with the next examples. One bad request never
        # aborts the run or tears down the ThreadPool.
        try:
            _process_one(i, ex)
        except Exception as e:
            with write_lock:
                counters["drop_error"] += 1
            logger.warning(f"[{i+1}/{len(todo)}] DROP (drop_error) exception "
                           f"{type(e).__name__}: {e} | {ex.get('question', '')[:60]}")

    from concurrent.futures import ThreadPoolExecutor
    t_run0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            # process() never raises, so a single failure cannot break the map.
            list(pool.map(lambda p: process(*p), list(enumerate(todo))))
    finally:
        out_f.close()
    wall_clock = time.time() - t_run0

    logger.info("Run complete. kept=%d  dropped: length=%d no_think=%d error=%d "
                "long=%d hedge=%d repetition=%d → %s" % (
                    counters["kept"], counters["drop_length"], counters["drop_no_think"],
                    counters["drop_error"], counters["drop_long"], counters["drop_hedge"],
                    counters["drop_repetition"], out_path))

    # -- MLflow provenance + comparability. Never let a tracking error crash the
    #    run: the JSONL output is the product, MLflow is best-effort. --
    _log_mlflow_run(args, model, counters, stats, wall_clock, len(todo))

    records = kept_records   # diagnostics below operate on kept-only records

    # -- Per-example diagnostics for the first --show kept records --
    for i, r in enumerate(records[:args.show]):
        print("\n" + "=" * 80)
        print(f"EXAMPLE {i+1}")
        print(f"  QUESTION: {r['question'][:200]}")
        raw = r["_raw"]
        print(f"  RAW TEACHER OUTPUT ({len(raw)} chars, truncated):")
        print("    " + raw[:400].replace("\n", "\n    "))
        if len(raw) > 800:
            print("    ...")
            print("    " + raw[-400:].replace("\n", "\n    "))
        th = r["thinking"]
        th_lines = th.splitlines()
        print(f"  EXTRACTED thinking: {len(th)} chars, {len(th_lines)} lines")
        if th_lines:
            print(f"    first line: {th_lines[0][:150]}")
            print(f"    last line : {th_lines[-1][:150]}")
        print(f"  EXTRACTED sql:\n    {r['sql'].replace(chr(10), chr(10)+'    ')}")

    # -- Aggregate verification checks --
    print("\n" + "=" * 80)
    print("VERIFICATION CHECKS")
    n = len(records)
    th_ok = sum(1 for r in records if len(r["thinking"].strip()) > 20)

    residue_markers = ("<think>", "</think>", "okay,", "okay so", "first,", "thinking process")

    def has_residue(sql: str) -> bool:
        low = sql.lower()
        return any(m in low for m in residue_markers)

    sql_clean = sum(1 for r in records if r["sql"].strip() and not has_residue(r["sql"]))
    exec_ok = 0
    exec_fail = []
    for r in records:
        ok, res = execute_sql_on_schema(r["sql"], r["schema"])
        if ok:
            exec_ok += 1
        else:
            exec_fail.append((r["question"][:60], str(res)[:80]))

    print(f"  thinking filled (>20 chars):     {th_ok}/{n}")
    print(f"  sql clean (no reasoning residue):{sql_clean}/{n}")
    print(f"  sql executes in SQLite:          {exec_ok}/{n}")
    if exec_fail:
        print("  execution failures:")
        for q, err in exec_fail:
            print(f"    - [{q}] {err}")
    print("=" * 80)


if __name__ == "__main__":
    main()

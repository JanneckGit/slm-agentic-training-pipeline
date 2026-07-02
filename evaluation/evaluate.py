"""
evaluation/evaluate.py
=======================
Evaluates a finetuned Text-to-SQL model using Execution Accuracy (EX).

Execution Accuracy = fraction of test examples where the model's predicted
SQL produces the SAME result set as the ground-truth SQL when executed
against the actual database.

This is the gold standard metric for Text-to-SQL (used in Spider & BIRD benchmarks).

We also report:
  - Exact Match (EM): predicted SQL == gold SQL (normalized)
  - Per-complexity accuracy breakdown

Usage:
    python evaluation/evaluate.py --config config/pipeline_config.yaml --model-path ./data/final/checkpoints/final_merged
    python evaluation/evaluate.py --model-path ./data/final/checkpoints/lora --use-adapter
    python evaluation/evaluate.py --model-path ./data/final/checkpoints/lora --use-adapter --n-samples 100
"""

import argparse
import json
import logging
import random
import re
import sqlite3
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

try:
    import mlflow
except ImportError:
    mlflow = None
    logger.warning("mlflow nicht installiert – MLflow-Tracking deaktiviert")


# MLflow conventions, shared with evaluation/rescore.py so the offline rescore
# targets the exact runs evaluate.py creates (single source of truth).
MLFLOW_EXPERIMENT = "text2sql-slm-finetuning"


def baseline_run_name(model_name: str) -> str:
    """Run-name convention for baseline eval runs: baseline_<modelname.lower()>."""
    return "baseline_" + model_name.lower()


# ---------------------------------------------------------------------------
# SQL normalization for exact-match comparison
# ---------------------------------------------------------------------------

def normalize_sql(sql: str) -> str:
    """
    Normalizes a SQL string for comparison:
    - Lowercase
    - Collapse whitespace
    - Remove trailing semicolon
    """
    sql = sql.strip().lower()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.rstrip(";")
    return sql


# ---------------------------------------------------------------------------
# SQL execution for Execution Accuracy
# ---------------------------------------------------------------------------

def execute_sql_on_schema(sql: str, schema_ddl: str) -> tuple[bool, any]:
    """
    Executes a SQL query against an in-memory SQLite database
    created from the given schema DDL.

    Returns:
        (success: bool, result: any)
        On success: result is the list of raw result rows (tuples)
        On failure: result is the error message
    """
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        # Execute the DDL statements (CREATE TABLE, INSERT INTO, etc.)
        # Split on ";" to handle multiple statements
        for stmt in schema_ddl.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    cursor.execute(stmt)
                except sqlite3.Error:
                    pass  # Some DDL statements may fail (e.g. schema differences) – that's ok

        conn.commit()

        # Execute the query
        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()

        # Return raw rows; canonicalization for strict/loose comparison happens
        # in execution_match (it needs the raw tuples for both variants).
        return True, rows

    except sqlite3.Error as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)


def execution_match(
    pred_sql: str, gold_sql: str, schema_ddl: str
) -> tuple[bool, bool, str | None, bool]:
    """
    Compares pred_sql and gold_sql by executing both against the schema.

    Returns (match_loose, match_strict, error, gold_failed):
      - match_strict: result rows are equal with row order ignored but COLUMN
        order preserved (``sorted(str(row) ...)`` — the previous behavior).
      - match_loose: result rows are equal with BOTH row and column order
        ignored. A differing number of columns still counts as a mismatch, so
        under-complete answers stay wrong. This is the canonical metric.
      - error: the execution error message if gold (or pred) failed, else None.
        A pure result-set mismatch (both executed) is not an error.
      - gold_failed: True if the gold query itself failed to execute.

    By construction strict ⇒ loose: a strict match is also a loose match, while
    loose additionally forgives pure column reordering (e.g. "count(*), type"
    vs "type, count(*)").
    """
    gold_ok, gold_result = execute_sql_on_schema(gold_sql, schema_ddl)
    if not gold_ok:
        return False, False, f"gold execution failed: {gold_result}", True

    pred_ok, pred_result = execute_sql_on_schema(pred_sql, schema_ddl)
    if not pred_ok:
        return False, False, str(pred_result), False

    # strict: row order ignored, column order preserved.
    gold_strict = sorted(str(row) for row in gold_result)
    pred_strict = sorted(str(row) for row in pred_result)
    match_strict = gold_strict == pred_strict

    # loose: row AND column order ignored; differing column count stays a mismatch.
    gold_loose = sorted(tuple(sorted(str(v) for v in row)) for row in gold_result)
    pred_loose = sorted(tuple(sorted(str(v) for v in row)) for row in pred_result)
    match_loose = gold_loose == pred_loose

    return match_loose, match_strict, None, False


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def load_model(model_path: str, use_adapter: bool = False):
    """
    Loads the finetuned model for inference.
    Supports:
      - Merged model (no adapter)
      - LoRA adapter over base model
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if use_adapter:
        from peft import PeftModel
        # Find the base model ID from adapter config
        adapter_cfg_path = Path(model_path) / "adapter_config.json"
        if adapter_cfg_path.exists():
            with open(adapter_cfg_path) as f:
                adapter_cfg = json.load(f)
            base_model_id = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
        else:
            base_model_id = "Qwen/Qwen2.5-7B-Instruct"
            logger.warning(f"adapter_config.json not found, assuming base: {base_model_id}")

        logger.info(f"Loading base model: {base_model_id}")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        logger.info(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    logger.info("Model loaded successfully")
    return model, tokenizer


def build_messages(question: str, schema: str) -> list[dict]:
    """
    Builds the chat messages for SQL prediction. Shared by the local and the
    endpoint mode so both use identical prompt construction.
    """
    system = "You are an expert SQL query writer. Given a natural language question and a database schema, write the correct SQL query. Output ONLY the SQL query."
    user_content = f"Database schema:\n{schema}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


# Well-formed Markdown code fence: optional language tag on the opening line,
# then the body up to the next closing fence. Group 1 = language, group 2 = body.
_FENCE_RE = re.compile(
    r"```[ \t]*([A-Za-z0-9_+#.-]*)[ \t]*\n(.*?)```",
    re.DOTALL,
)
_SQL_KEYWORDS = ("select", "insert", "update", "delete", "with",
                 "create", "drop", "alter", "replace", "pragma")
# Recover a query that leaked into plain reasoning text (no fence, no tags):
# inline `...` spans, and the last line that begins with a SQL statement.
_INLINE_BACKTICK_RE = re.compile(r"`([^`]+)`")
_SQL_LINE_START_RE = re.compile(r"(?im)^\s*(?:select|with|insert|update|delete)\b")


def extract_sql(raw_output: str) -> str:
    """
    Extracts the SQL from a raw model output. Shared by both modes so the
    extraction is guaranteed identical.

    Steps: (1) strip any <think>...</think> reasoning block, then (2) if the
    remainder is wrapped in a Markdown code fence (```sql ... ``` or a bare
    ``` ... ```), return only the fenced SQL. Thinking models (Qwen3/3.5/3.6)
    routinely emit the final query inside such a fence; left in place the
    backticks make SQLite raise a syntax error and the model scores a false 0%.
    (3) If there is no fence and the text is not already a bare query, the
    reasoning leaked as PLAIN TEXT (some Qwen3.6 variants emit "Thinking
    Process: ..." with no tags and put the final query only in inline
    backticks) — recover the query so the prose does not land in the SQL
    (which otherwise yields "near 'Thinking': syntax error").
    """
    # 1. Strip the reasoning block. Some thinking models (Qwen3.5) inject the
    #    opening <think> into the prompt, so the generated output carries only
    #    reasoning text + a closing </think> + SQL — no opening tag. Keying off
    #    the LAST </think> covers both cases (with/without opening tag) and
    #    leaves non-thinking outputs untouched.
    if "</think>" in raw_output:
        text = raw_output.rsplit("</think>", 1)[-1].strip()
    else:
        text = raw_output.strip()

    # 2. Unwrap a Markdown code fence if present. With multiple fences, prefer
    #    one explicitly tagged ```sql, else the first whose body looks like SQL,
    #    else the first block.
    blocks = _FENCE_RE.findall(text)
    if blocks:
        sql_tagged = [body for lang, body in blocks if lang.lower() == "sql"]
        looks_sql = [body for _, body in blocks
                     if body.lstrip().lower().startswith(_SQL_KEYWORDS)]
        chosen = (sql_tagged or looks_sql or [blocks[0][1]])[0]
        return chosen.strip()

    # 3. No fence. Only intervene when the text is not already a bare query, so
    #    clean outputs are untouched. Prefer the LAST inline-backtick span that
    #    looks like SQL; otherwise slice from the LAST line that starts a query.
    if not text.lower().startswith(_SQL_KEYWORDS):
        inline_sql = [m.group(1).strip()
                      for m in _INLINE_BACKTICK_RE.finditer(text)
                      if m.group(1).strip().lower().startswith(_SQL_KEYWORDS)]
        if inline_sql:
            return inline_sql[-1].strip()
        line_starts = list(_SQL_LINE_START_RE.finditer(text))
        if line_starts:
            return text[line_starts[-1].start():].strip()

    return text


def predict_sql(
    model,
    tokenizer,
    question: str,
    schema: str,
    max_new_tokens: int = 256,
    enable_thinking: bool = False,
) -> tuple[str, str]:
    """
    Runs local (HF Transformers) inference to predict SQL from a question + schema.
    Returns (raw_output, extracted_sql): the full decoded model output and
    the same output with any <think>...</think> block stripped.
    """
    import torch

    messages = build_messages(question, schema)

    # Apply chat template (Qwen2.5 has a built-in one)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,      # Low temp for deterministic SQL
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (not the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return raw_output, extract_sql(raw_output)


def make_endpoint_predict_fn(
    api_base: str,
    api_model_name: str,
    api_key: str = "token-local",
    max_new_tokens: int = 256,
    max_retries: int = 2,
    enable_thinking: bool = False,
):
    """
    Builds a predict_fn(question, schema) -> (raw_output, extracted_sql) that
    talks to an OpenAI-compatible chat/completions endpoint (e.g. a vLLM server).

    Uses the same prompt construction (build_messages) and SQL extraction
    (extract_sql) as the local mode. Greedy decoding (temperature=0).
    """
    import httpx

    url = api_base.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    # Grosszuegiges Timeout: grosse Modelle (14B) bei hoher Concurrency auf
    # bandbreiten-limitierter Hardware (GB10) generieren lange thinking-Traces
    # langsam; 1200s schnitt ~40% der 14B-thinking-Requests ab -> leer -> falsch.
    client = httpx.Client(timeout=3600.0)

    def predict(question: str, schema: str) -> tuple[str, str]:
        payload = {
            "model": api_model_name,
            "messages": build_messages(question, schema),
            "temperature": 0,
            "max_tokens": max_new_tokens,
            # Thinking-Toggle IMMER explizit ans Chat-Template durchreichen –
            # in BEIDE Richtungen. Wird er weggelassen, greift der Template-Default
            # (Qwen3.5: <think> offen = Thinking AN), wodurch ein non-thinking-
            # Student trotzdem reasont und – bei knappem max_tokens – abgeschnitten
            # wird, bevor das SQL kommt. enable_thinking=False prefüllt stattdessen
            # <think>\n\n</think>\n\n, exakt das Trainings-Target dieses Students.
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                raw_output = resp.json()["choices"][0]["message"]["content"] or ""
                return raw_output, extract_sql(raw_output)
            except Exception as e:  # network / HTTP / payload errors
                last_err = e
                if attempt < max_retries:
                    time.sleep(1.0 * (attempt + 1))
        logger.warning(f"Endpoint request failed after {max_retries + 1} attempts: {last_err}")
        sentinel = f"<API_ERROR: {last_err}>"
        return sentinel, ""

    return predict


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate(
    predict_fn,
    test_examples: list[dict],
    n_samples: int | None = None,
    concurrency: int = 1,
) -> dict:
    """
    Runs evaluation on test examples.
    predict_fn(question, schema) -> (raw_output, extracted_sql) abstracts over
    the local (HF) and the endpoint (vLLM API) inference modes.
    Returns a results dict with per-example and aggregate metrics.

    Bei concurrency > 1 werden die predict_fn-Aufrufe über einen
    ThreadPoolExecutor parallel gefeuert. Das nutzt vLLMs Continuous Batching
    (großer Speedup im Endpoint-Modus). Determinismus bleibt erhalten: jeder
    Request ist unabhängig + greedy, und die Ergebnisse werden am Ende wieder
    nach Original-Index sortiert.
    """
    if n_samples:
        test_examples = random.sample(test_examples, min(n_samples, len(test_examples)))

    logger.info(f"Evaluating on {len(test_examples)} examples "
                f"(concurrency={concurrency})...")

    def run_one(i: int, ex: dict) -> dict:
        question = ex.get("question", "")
        schema = ex.get("schema", "")
        gold_sql = ex.get("sql", "")
        complexity = ex.get("complexity", "unknown")

        # Predict
        raw_output, pred_sql = predict_fn(question, schema)

        # Exact Match
        em = normalize_sql(pred_sql) == normalize_sql(gold_sql)

        # Execution Accuracy (loose = canonical; strict kept alongside).
        ex_loose, ex_strict, exec_error, gold_failed = execution_match(
            pred_sql, gold_sql, schema)

        return {
            "index": i,
            "complexity": complexity,
            "question": question,
            "gold_sql": gold_sql,
            "raw_output": raw_output,
            "extracted_sql": pred_sql,
            "exact_match": em,
            "execution_match": ex_loose,
            "execution_match_strict": ex_strict,
            "gold_failed": gold_failed,
            "error": exec_error,
        }

    if concurrency > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        done = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(run_one, i, ex): i
                       for i, ex in enumerate(test_examples)}
            for fut in as_completed(futures):
                results.append(fut.result())
                done += 1
                if done % 25 == 0:
                    logger.info(f"  Progress: {done}/{len(test_examples)}")
        # Reihenfolge + Determinismus: nach Original-Index sortieren.
        results.sort(key=lambda r: r["index"])
    else:
        results = []
        for i, ex in enumerate(test_examples):
            if i % 25 == 0:
                logger.info(f"  Progress: {i}/{len(test_examples)}")
            results.append(run_one(i, ex))

    # Aggregate by complexity (aus der sortierten Ergebnisliste).
    by_complexity = defaultdict(lambda: {"em": 0, "ex": 0, "ex_strict": 0, "total": 0})
    for r in results:
        by_complexity[r["complexity"]]["total"] += 1
        if r["exact_match"]:
            by_complexity[r["complexity"]]["em"] += 1
        if r["execution_match"]:
            by_complexity[r["complexity"]]["ex"] += 1
        if r["execution_match_strict"]:
            by_complexity[r["complexity"]]["ex_strict"] += 1

    # Overall metrics. execution_accuracy is the LOOSE variant (canonical);
    # execution_accuracy_strict keeps the previous column-order-sensitive value.
    n = len(results)
    overall_em = sum(r["exact_match"] for r in results) / n
    overall_ex = sum(r["execution_match"] for r in results) / n
    overall_ex_strict = sum(r["execution_match_strict"] for r in results) / n

    return {
        "overall": {
            "n": n,
            "exact_match": round(overall_em, 4),
            "execution_accuracy": round(overall_ex, 4),
            "execution_accuracy_strict": round(overall_ex_strict, 4),
        },
        "by_complexity": {
            level: {
                "n": v["total"],
                "exact_match": round(v["em"] / v["total"], 4) if v["total"] else 0,
                "execution_accuracy": round(v["ex"] / v["total"], 4) if v["total"] else 0,
                "execution_accuracy_strict": round(v["ex_strict"] / v["total"], 4) if v["total"] else 0,
            }
            for level, v in by_complexity.items()
        },
        "examples": results,
    }


def print_results(results: dict, order: list[str] | None = None):
    """Pretty-prints evaluation results. `order` = Anzeige-Reihenfolge der
    Komplexitätsklassen (i.d.R. config['complexity_classes'])."""
    overall = results["overall"]
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Total examples:      {overall['n']}")
    logger.info(f"  Exact Match (EM):    {overall['exact_match']:.1%}")
    logger.info(f"  Execution Acc (EX):  {overall['execution_accuracy']:.1%}  (loose)")
    logger.info(f"  Execution Acc strict:{overall.get('execution_accuracy_strict', 0):.1%}")
    logger.info("")
    logger.info("  Per-complexity breakdown:")

    order = order or []
    for level in order + [k for k in results["by_complexity"] if k not in order]:
        if level not in results["by_complexity"]:
            continue
        row = results["by_complexity"][level]
        bar = "█" * int(row["execution_accuracy"] * 20)
        logger.info(f"  {level:30s}: EX={row['execution_accuracy']:.1%}  {bar}  (n={row['n']})")

    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate finetuned Text-to-SQL model")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--model-path", required=True,
                        help="Path to merged model or LoRA adapter")
    parser.add_argument("--use-adapter", action="store_true",
                        help="Load as LoRA adapter (not merged model)")
    parser.add_argument("--test-file", default=None,
                        help="Override test data path")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Evaluate on a random subset")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed für das Subsample (überschreibt Config; Default 42)")
    parser.add_argument("--output", default=None,
                        help="Output-Dateiname (nur Basename; landet immer in "
                             "data/final/eval/<model_name>/). Default: results.json")
    parser.add_argument("--api-base", default=None,
                        help="OpenAI-kompatibler Endpoint (z.B. http://vllm:8000/v1). "
                             "Wenn gesetzt → Endpoint-Modus statt lokalem HF-Loading.")
    parser.add_argument("--api-model-name", default=None,
                        help="served-model-name am vLLM-Endpoint (Pflicht im Endpoint-Modus)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max. generierte Tokens pro Beispiel. Default 256 für "
                             "Non-Thinking. Für Thinking-Modelle 2048-4096 empfohlen, "
                             "sonst wird das Reasoning abgeschnitten.")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Erzwingt Thinking-Modus. Nötig für Qwen3.5-Small (denken "
                             "per Default nicht). Harmlos für Non-Thinking-Modelle "
                             "(Kwarg wird dort vom Template ignoriert).")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Parallele Requests im Endpoint-Modus. >1 nutzt vLLMs "
                             "Batching (großer Speedup auf dem GB10). Nur bei --api-base "
                             "wirksam; lokaler HF-Modus läuft immer sequenziell.")
    args = parser.parse_args()

    if args.api_base and not args.api_model_name:
        parser.error("--api-base gesetzt, aber --api-model-name fehlt "
                     "(served-model-name ist im Endpoint-Modus Pflicht)")

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    if args.seed is None and "seed" not in config:
        logger.warning(f"Kein 'seed' in CLI/Config – nutze Default {seed}")
    logger.info(f"Globaler Seed: {seed}")
    random.seed(seed)

    # Load test data
    # Default-Test-Set aus Config (ausführbarkeits-gefiltert, 7 Klassen);
    # die mix-produzierte test.jsonl wird fürs finale Eval NICHT genutzt.
    test_path = Path(args.test_file) if args.test_file else \
                Path(config["data"]["final_dir"]) / config["data"].get("eval_test_file", "test_clean.jsonl")

    if not test_path.exists():
        logger.error(f"Test file not found: {test_path}")
        return

    test_examples = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                # Handle both raw format and chat format
                if "messages" in ex:
                    # Skip chat-format – use raw test.jsonl instead
                    continue
                test_examples.append(ex)

    logger.info(f"Test examples: {len(test_examples)}")

    # Build predict_fn for the selected mode.
    # Beide Modi beziehen das Token-Budget aus genau einer Quelle: args.max_tokens.
    logger.info(f"Thinking-Modus: {'AN' if args.enable_thinking else 'AUS'}")
    if args.api_base:
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "token-local")
        concurrency = args.concurrency
        logger.info(f"Endpoint-Modus: {args.api_base} | model: {args.api_model_name} "
                    f"| max_tokens={args.max_tokens} | concurrency={concurrency}")
        predict_fn = make_endpoint_predict_fn(args.api_base, args.api_model_name, api_key,
                                              max_new_tokens=args.max_tokens,
                                              enable_thinking=args.enable_thinking)
    else:
        # Lokaler HF-Modus: Single-GPU, model.generate ist nicht thread-safe →
        # Parallelisierung verboten.
        concurrency = 1
        if args.concurrency > 1:
            logger.warning(f"--concurrency={args.concurrency} wird im lokalen HF-Modus "
                           f"ignoriert (Single-GPU, model.generate nicht thread-safe) "
                           f"– laufe sequenziell.")
        model, tokenizer = load_model(args.model_path, use_adapter=args.use_adapter)
        predict_fn = lambda question, schema: predict_sql(
            model, tokenizer, question, schema, max_new_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking)

    mlflow_active = False
    if mlflow is not None:
        run_id_path = Path(args.model_path) / "mlflow_run_id.txt"
        if run_id_path.exists():
            try:
                run_id = run_id_path.read_text().strip()
                mlflow.set_tracking_uri("file:///app/mlruns")
                mlflow.start_run(run_id=run_id)
                mlflow_active = True
                logger.info(f"MLflow run resumed: {run_id}")
            except Exception as e:
                logger.warning(f"MLflow-Resume fehlgeschlagen ({e}) – Eval läuft ohne MLflow")
        else:
            try:
                mlflow.set_tracking_uri("file:///app/mlruns")
                mlflow.set_experiment(MLFLOW_EXPERIMENT)
                run_name = baseline_run_name(Path(args.model_path).name)
                mlflow.start_run(run_name=run_name)
                mlflow.log_param("model_path", args.model_path)
                mlflow.log_param("type", "baseline_untrained")
                mlflow_active = True
                logger.info(f"MLflow Baseline-Run gestartet: {run_name}")
            except Exception as e:
                logger.warning(f"MLflow-Baseline-Run fehlgeschlagen ({e}) – Eval läuft ohne MLflow")

    # Run evaluation
    results = evaluate(predict_fn, test_examples, n_samples=args.n_samples,
                       concurrency=concurrency)

    # Print results
    print_results(results, config.get("complexity_classes"))

    if mlflow_active and mlflow is not None:
        try:
            overall = results["overall"]
            mlflow.log_metric("eval_em", overall["exact_match"])
            mlflow.log_metric("eval_ex", overall["execution_accuracy"])
            mlflow.log_metric("eval_ex_strict", overall["execution_accuracy_strict"])
            for klasse, v in results["by_complexity"].items():
                klasse_safe = klasse.replace(" ", "_")
                mlflow.log_metric(f"eval_ex_{klasse_safe}", v["execution_accuracy"])
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"MLflow-Logging fehlgeschlagen: {e}")

    # Save results
    # Eval-Ergebnisse in data/final/eval/<modellname>/<output> speichern.
    # --output wird nur als Dateiname genutzt (Verzeichnisanteil wird ignoriert),
    # damit Ergebnis + Predictions-Sidecar immer unter eval/<model_name>/ gruppiert
    # landen statt flach in eval/.
    model_name = Path(args.model_path).name
    eval_dir = Path(config["data"]["final_dir"]) / "eval" / model_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_name = Path(args.output).name if args.output else "results.json"
    output_path = str(eval_dir / output_name)
    with open(output_path, "w") as f:
        # Don't serialize all examples by default (can be very large)
        summary = {k: v for k, v in results.items() if k != "examples"}
        json.dump(summary, f, indent=2)
    logger.info(f"\nResults saved: {output_path}")

    # Per-sample diagnostics alongside results.json, sharing its stem so
    # runs with a custom --output don't collide on a fixed filename.
    output_p = Path(output_path)
    predictions_path = output_p.with_name(output_p.stem + "_predictions.jsonl")
    with open(predictions_path, "w") as f:
        for ex in results["examples"]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Predictions saved: {predictions_path}")


def _selftest_extract_sql() -> int:
    """
    Asserts that extract_sql unwraps <think> blocks, Markdown SQL fences, and
    reasoning that leaked as plain text. Run with:
        python evaluation/evaluate.py --selftest-extract-sql
    Returns the number of failing cases (0 = all good) for use as exit code.
    """
    cases = [
        ("pure SQL",
         "SELECT * FROM users WHERE id = 1;",
         "SELECT * FROM users WHERE id = 1;"),
        ("<think> + ```sql fences",
         "<think>The question asks for names of high earners.</think>\n"
         "```sql\n"
         "SELECT name FROM employees WHERE salary > 50000;\n"
         "```",
         "SELECT name FROM employees WHERE salary > 50000;"),
        ("bare ``` fences (no lang)",
         "```\n"
         "SELECT COUNT(*) FROM orders;\n"
         "```",
         "SELECT COUNT(*) FROM orders;"),
        # (i) closing </think> only, no opening tag (Qwen3.5/3.6 prompt-injected <think>).
        ("closing </think> only (no opening tag)",
         "Okay, reasoning...\n</think>\n\nSELECT 1;",
         "SELECT 1;"),
        # (ii) reasoning leaked as plain text, final query in inline backticks.
        ("plaintext reasoning + inline-backtick query",
         "Thinking Process:\nFirst I find the relevant table, then count rows.\n"
         "Final query: `SELECT 1;`",
         "SELECT 1;"),
    ]
    n_fail = 0
    for label, raw, expected in cases:
        got = extract_sql(raw)
        ok = got == expected
        n_fail += int(not ok)
        print(f"[{'ok' if ok else 'FAIL'}] {label}")
        if not ok:
            print(f"    raw:      {raw!r}")
            print(f"    expected: {expected!r}")
            print(f"    got:      {got!r}")
    print(f"\n{len(cases) - n_fail}/{len(cases)} cases passed")
    return n_fail


if __name__ == "__main__":
    import sys
    if "--selftest-extract-sql" in sys.argv:
        sys.exit(_selftest_extract_sql())
    else:
        main()

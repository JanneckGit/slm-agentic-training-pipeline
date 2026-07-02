"""
data_pipeline/prepare_data.py
=======================
Downloads the gretelai/synthetic_text_to_sql seed dataset from HuggingFace,
applies basic quality filtering, and saves a clean JSONL to data/raw/.

This is Step 1 of the pipeline – run this once before the SDG step.

Usage:
    python data_pipeline/prepare_data.py --config config/pipeline_config.yaml
    python data_pipeline/prepare_data.py --n-samples 2000  # quick test
"""

import argparse
import json
import logging
import random
from pathlib import Path

import yaml
from datasets import load_dataset

from data_pipeline.complexity_taxonomy import load_complexity_classes, normalize_complexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema of the gretelai/synthetic_text_to_sql dataset
# Columns we care about:
#   sql_prompt        – natural language question
#   sql_context       – CREATE TABLE statements (schema)
#   sql               – ground-truth SQL query
#   sql_explanation   – explanation of what the SQL does
#   domain            – e.g. "finance", "healthcare", ...
#   sql_complexity    – "basic SQL", "aggregation", "window functions", ...
#   sql_task_type     – "analytics and reporting", "data manipulation", ...
# ---------------------------------------------------------------------------

# Die gültigen Komplexitätsklassen kommen aus config['complexity_classes']
# (Single Source of Truth, siehe data_pipeline/complexity_taxonomy.py) – nicht mehr
# hier hardcodiert, damit alle Stages dieselbe Liste teilen.

# We filter to keep only reasonable-length examples
MAX_CONTEXT_LEN = 1500   # chars – very long schemas hurt training
MIN_SQL_LEN = 20         # chars – too-short queries aren't interesting
MAX_SQL_LEN = 600        # chars – very long queries are rare edge cases


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def quality_filter(example: dict) -> bool:
    """
    Returns True if the example passes quality checks.
    We want clean, non-trivial, well-formed examples.
    """
    sql = example.get("sql", "")
    context = example.get("sql_context", "")
    prompt = example.get("sql_prompt", "")

    # Length filters
    if not MIN_SQL_LEN <= len(sql) <= MAX_SQL_LEN:
        return False
    if len(context) > MAX_CONTEXT_LEN:
        return False
    if len(prompt) < 10:
        return False

    # Must have a CREATE TABLE statement in context
    if "CREATE TABLE" not in context.upper():
        return False

    # Skip examples with common data quality issues
    if "TODO" in sql or "FIXME" in sql:
        return False

    return True


def normalize_example(example: dict) -> dict:
    """
    Normalizes a raw dataset example into a clean, consistent dict.
    This is the canonical format used throughout the pipeline.
    """
    return {
        # Core fields
        "question": example["sql_prompt"].strip(),
        "schema": example["sql_context"].strip(),
        "sql": example["sql"].strip(),
        "explanation": example.get("sql_explanation", "").strip(),
        # Metadata (useful for stratified sampling in SDG)
        "domain": example.get("domain", "unknown"),
        "complexity": example.get("sql_complexity", "unknown"),
        "task_type": example.get("sql_task_type", "analytics and reporting"),
        # Source tracking
        "source": "gretelai/synthetic_text_to_sql",
        "split": "seed",
    }


def stratified_sample(examples: list[dict], n: int) -> list[dict]:
    """
    Sample n examples with stratification over complexity levels,
    so we get a balanced distribution of easy/hard SQL.
    """
    by_complexity = {}
    for ex in examples:
        level = ex["complexity"]
        by_complexity.setdefault(level, []).append(ex)

    n_per_level = max(1, n // len(by_complexity))
    sampled = []
    for level, items in by_complexity.items():
        k = min(n_per_level, len(items))
        sampled.extend(random.sample(items, k))
        logger.info(f"  complexity='{level}': sampled {k}/{len(items)}")

    # If we're under budget, top up randomly
    remaining = n - len(sampled)
    if remaining > 0:
        pool = [ex for ex in examples if ex not in sampled]
        sampled.extend(random.sample(pool, min(remaining, len(pool))))

    random.shuffle(sampled)
    return sampled[:n]


def main():
    parser = argparse.ArgumentParser(description="Prepare seed dataset for Text-to-SQL pipeline")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Override seed_sample_size from config")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load config
    config = load_config(args.config)
    data_cfg = config["data"]
    # SSoT der gültigen Klassen (Hard-Fail mit Key + Config-Datei wenn fehlt)
    complexity_classes = load_complexity_classes(config, args.config)
    allowed_classes = set(complexity_classes)
    raw_dir = Path(data_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    n_samples = args.n_samples or data_cfg["seed_sample_size"]
    dataset_name = data_cfg["seed_dataset"]
    split = data_cfg["seed_split"]

    logger.info(f"Loading dataset: {dataset_name} (split={split})")
    dataset = load_dataset(dataset_name, split=split)
    logger.info(f"Raw dataset size: {len(dataset)}")

    # Apply quality filter
    logger.info("Applying quality filter...")
    filtered = [normalize_example(ex) for ex in dataset if quality_filter(ex)]
    logger.info(f"After filtering: {len(filtered)} examples ({len(filtered)/len(dataset):.1%} kept)")

    # Complexity-Filter (frühester Punkt): nur die gültigen Klassen behalten.
    # CTEs (und sonstige Off-Class-Labels) fliegen hier raus, damit sie gar
    # nicht erst in SDG/Mix/Training gelangen.
    before_cx = len(filtered)
    filtered = [ex for ex in filtered
                if normalize_complexity(ex["complexity"]) in allowed_classes]
    dropped_cx = before_cx - len(filtered)
    logger.info(f"Complexity-Filter ({len(allowed_classes)} gültige Klassen): "
                f"{before_cx} → {len(filtered)} (Off-Class/CTEs entfernt: {dropped_cx})")

    # Stratified sample
    logger.info(f"Stratified sampling {n_samples} examples...")
    sampled = stratified_sample(filtered, n_samples)
    logger.info(f"Final sample size: {len(sampled)}")

    # Save full filtered set (for reference)
    full_out = raw_dir / "seed_full.jsonl"
    with open(full_out, "w") as f:
        for ex in filtered:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Full filtered set saved: {full_out} ({len(filtered)} examples)")

    # Save sampled set (this is what SDG uses)
    sample_out = raw_dir / "seed_sample.jsonl"
    with open(sample_out, "w") as f:
        for ex in sampled:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Seed sample saved: {sample_out} ({len(sampled)} examples)")

    # Print complexity distribution
    logger.info("\n=== Complexity distribution of sample ===")
    from collections import Counter
    counts = Counter(normalize_complexity(ex["complexity"]) for ex in sampled)
    for level in complexity_classes:
        n = counts.get(level, 0)
        bar = "█" * (n // 5)
        logger.info(f"  {level:30s}: {n:4d}  {bar}")

    logger.info("\n✅ Data preparation complete. Next: run sdg_pipeline/run_sdg.py")


if __name__ == "__main__":
    main()

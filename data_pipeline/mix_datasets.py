"""
data_pipeline/mix_datasets.py
=======================
Mixes seed data and SDG-generated synthetic data into a final training dataset.

Steps:
1. Load seed examples (data/raw/seed_sample.jsonl)
2. Load synthetic examples (data/generated/generated_enriched.jsonl)
3. Expand synthetic examples: each row in the generated set can yield
   multiple training examples (original, upgraded, variant, with reasoning)
4. Mix according to the configured ratio
5. Save train/eval/test splits to data/final/

Usage:
    python data_pipeline/mix_datasets.py --config config/pipeline_config.yaml
"""

import argparse
import json
import logging
import random
from collections import Counter
from pathlib import Path

import yaml

from data_pipeline.complexity_taxonomy import load_complexity_classes, normalize_complexity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Complexity-Label-Normalisierung lebt jetzt zentral in
# data_pipeline/complexity_taxonomy.py (Single Source of Truth), damit prepare_data,
# mix und evaluate dieselbe Taxonomie/gültige Klassenliste teilen.


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def save_jsonl(examples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(examples)} examples → {path}")


def expand_synthetic_examples(generated: list[dict]) -> list[dict]:
    """
    Expands each generated row into multiple training examples.

    Each SDG-generated row contains:
    - Original seed example (question, schema, sql)
    - Upgraded version (upgraded_question, upgraded_sql)
    - Schema variant (variant_schema, variant_question, variant_sql)
    - Reasoning trace (reasoning_trace)

    For each, separate training examples with appropriate
    "thinking" (reasoning trace) fields (where available) are applied.
    """
    expanded = []

    for row in generated:
        quality = row.get("quality_score", 1.0)
        if quality < 0:   # Filtered out by quality scorer
            continue

        # 1. Original with reasoning trace
        if row.get("question") and row.get("sql"):
            expanded.append({
                "question": row["question"],
                "schema": row["schema"],
                "sql": row["sql"],
                "thinking": row.get("reasoning_trace", ""),
                "complexity": row.get("complexity", "unknown"),
                "domain": row.get("domain", "unknown"),
                "source": "seed_with_reasoning",
                "quality_score": quality,
            })

        # 2. Upgraded version (harder SQL) with reasoning trace
        if row.get("upgraded_question") and row.get("upgraded_sql"):
            if not row["upgraded_sql"].startswith("[MOCK]"):
                expanded.append({
                    "question": row["upgraded_question"],
                    "schema": row["schema"],   # Same schema as original
                    "sql": row["upgraded_sql"],
                    "thinking": "",
                    "complexity": row.get("upgraded_complexity", "unknown"),
                    "domain": row.get("domain", "unknown"),
                    "source": "synthetic_upgraded",
                    "quality_score": quality,
                })

        # 3. Schema variant (new schema, new question)
        if row.get("variant_question") and row.get("variant_sql") and row.get("variant_schema"):
            if not row["variant_sql"].startswith("[MOCK]"):
                expanded.append({
                    "question": row["variant_question"],
                    "schema": row["variant_schema"],
                    "sql": row["variant_sql"],
                    "thinking": "",   # No reasoning trace for variants
                    "complexity": row.get("complexity", "unknown"),
                    "domain": row.get("domain", "unknown"),
                    "source": "synthetic_variant",
                    "quality_score": quality,
                })

    return expanded


def format_training_example(ex: dict) -> dict:
    """
    Final formatting: ensures every example has the right fields
    for the training formatter downstream.
    """
    return {
        "question": ex.get("question", "").strip(),
        "schema": ex.get("schema", "").strip(),
        "sql": ex.get("sql", "").strip(),
        "thinking": ex.get("thinking", "").strip(),
        # Single Funnel: alle Examples (Seed + expandierte SDG-Varianten)
        # laufen hier durch, also reicht ein Aufruf zum Normalisieren.
        "complexity": normalize_complexity(ex.get("complexity")),
        "domain": ex.get("domain", "unknown"),
        "source": ex.get("source", "unknown"),
        "quality_score": ex.get("quality_score", 1.0),
    }


def split_dataset(examples: list[dict], train_ratio: float = 0.9, eval_ratio: float = 0.05) -> tuple:
    """Splits into train/eval/test. Returns (train, eval, test)."""
    n = len(examples)
    n_train = int(n * train_ratio)
    n_eval = int(n * eval_ratio)
    train = examples[:n_train]
    eval_ = examples[n_train:n_train + n_eval]
    test = examples[n_train + n_eval:]
    return train, eval_, test


def main():
    parser = argparse.ArgumentParser(description="Mix seed and synthetic datasets")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed für Sampling/Shuffle (überschreibt Config; Default 42)")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    if args.seed is None and "seed" not in config:
        logger.warning(f"Kein 'seed' in CLI/Config – nutze Default {seed}")
    logger.info(f"Globaler Seed: {seed}")
    random.seed(seed)

    raw_dir = Path(data_cfg["raw_dir"])
    generated_dir = Path(data_cfg["generated_dir"])
    final_dir = Path(data_cfg["final_dir"])

    mix_ratio = data_cfg.get("mix_ratio", {"seed": 0.4, "synthetic": 0.6})

    # -----------------------------------------------------------------------
    # 1. Load datasets
    # -----------------------------------------------------------------------
    seed_path = raw_dir / "seed_sample.jsonl"
    generated_path = generated_dir / "generated_enriched.jsonl"

    if not seed_path.exists():
        raise FileNotFoundError(f"Seed file not found: {seed_path}. Run prepare_data.py first.")

    seed_examples = load_jsonl(seed_path)
    logger.info(f"Loaded {len(seed_examples)} seed examples")

    synthetic_examples = []
    if generated_path.exists():
        raw_generated = load_jsonl(generated_path)
        synthetic_examples = expand_synthetic_examples(raw_generated)
        logger.info(f"Loaded {len(raw_generated)} generated rows → expanded to {len(synthetic_examples)} examples")
    else:
        logger.warning(f"No generated data found at {generated_path}")
        logger.warning("Proceeding with seed data only (mix_ratio ignored)")

    # -----------------------------------------------------------------------
    # 2. Format all examples (Single Funnel) – Seed UND expandierte SDG-Quellen
    #    laufen durch dieselbe Normalisierung (insb. normalize_complexity),
    #    sonst tragen die SDG-Examples weiter Roh-Labels (z.B. "multiple joins"
    #    mit Space, Compound-Labels wie "subqueries and CTEs") ins train/test.
    # -----------------------------------------------------------------------
    seed_formatted = [format_training_example(ex) for ex in seed_examples]
    synthetic_examples = [format_training_example(ex) for ex in synthetic_examples]

    # -----------------------------------------------------------------------
    # 2a. Complexity-Gate (defensives Endgate): nur gültige Klassen behalten.
    #     SSoT = config['complexity_classes'] (Hard-Fail mit Key + Datei wenn
    #     fehlt). Fängt alles ab, was am prepare_data-Filter vorbeikam (insb.
    #     vom SDG-Upgrade erzeugte Off-Class-Labels).
    # -----------------------------------------------------------------------
    allowed_classes = set(load_complexity_classes(config, args.config))

    def complexity_gate(lst, name):
        before = len(lst)
        kept = [ex for ex in lst if ex["complexity"] in allowed_classes]
        dropped = before - len(kept)
        if dropped:
            logger.info(f"Complexity-Gate ({name}): {before} → {len(kept)} "
                        f"(off-class entfernt: {dropped})")
        return kept

    seed_formatted = complexity_gate(seed_formatted, "seed")
    synthetic_examples = complexity_gate(synthetic_examples, "synthetic")

    # -----------------------------------------------------------------------
    # 2b. Test-Overlap-Ausschluss (Leakage-Schutz): das finale Eval-Test-Set
    #     (data/final/test_clean.jsonl) wurde aus DEMSELBEN Seed-Pool gesampelt.
    #     Diese Beispiele MÜSSEN aus train UND in-training-eval raus. Dedup über
    #     (question, sql), da kein ID-Feld existiert.
    # -----------------------------------------------------------------------
    test_clean_path = final_dir / data_cfg.get("eval_test_file", "test_clean.jsonl")
    if test_clean_path.exists():
        test_keys = {
            (ex.get("question", "").strip(), ex.get("sql", "").strip())
            for ex in load_jsonl(test_clean_path)
        }

        def exclude_test_overlap(lst, name):
            before = len(lst)
            kept = [ex for ex in lst
                    if (ex.get("question", "").strip(), ex.get("sql", "").strip()) not in test_keys]
            removed = before - len(kept)
            logger.info(f"Test-Overlap-Ausschluss ({name}): {removed} Beispiele wegen "
                        f"Overlap mit {test_clean_path.name} entfernt ({before} → {len(kept)})")
            return kept

        seed_formatted = exclude_test_overlap(seed_formatted, "seed")
        synthetic_examples = exclude_test_overlap(synthetic_examples, "synthetic")
    else:
        raise FileNotFoundError(
            f"Eval-Test-Set {test_clean_path} nicht gefunden – der Test-Overlap-Ausschluss "
            f"(Leakage-Guard) kann nicht laufen. Erst test_clean.jsonl erzeugen oder "
            f"data.eval_test_file korrigieren. (Hartes Fail statt stiller Leakage.)")

    # -----------------------------------------------------------------------
    # 3. Mix according to ratio
    # -----------------------------------------------------------------------
    if synthetic_examples:
        ratio_seed = mix_ratio.get("seed", 0.4)
        ratio_synth = mix_ratio.get("synthetic", 0.6)

        total_target = len(seed_formatted) + len(synthetic_examples)
        n_seed = int(total_target * ratio_seed)
        n_synth = int(total_target * ratio_synth)

        # Sample (or use all if we don't have enough)
        seed_sampled = random.sample(seed_formatted, min(n_seed, len(seed_formatted)))
        synth_sampled = random.sample(synthetic_examples, min(n_synth, len(synthetic_examples)))

        mixed = seed_sampled + synth_sampled
        logger.info(f"Mixed dataset: {len(seed_sampled)} seed + {len(synth_sampled)} synthetic = {len(mixed)} total")
    else:
        mixed = seed_formatted
        logger.info(f"Using seed-only dataset: {len(mixed)} examples")

    random.shuffle(mixed)

    # -----------------------------------------------------------------------
    # 4. Split and save
    # -----------------------------------------------------------------------
    train, eval_, test = split_dataset(mixed)
    logger.info(f"Splits: train={len(train)}, eval={len(eval_)}, test={len(test)}")

    save_jsonl(train, final_dir / "train.jsonl")
    save_jsonl(eval_, final_dir / "eval.jsonl")
    save_jsonl(test, final_dir / "test.jsonl")

    # -----------------------------------------------------------------------
    # 5. Print dataset statistics
    # -----------------------------------------------------------------------
    logger.info("\n=== Final Dataset Statistics ===")
    sources = Counter(ex["source"] for ex in train)
    logger.info("Source distribution (train):")
    for src, cnt in sources.most_common():
        pct = cnt / len(train) * 100
        logger.info(f"  {src:30s}: {cnt:5d} ({pct:.1f}%)")

    complexities = Counter(ex["complexity"] for ex in train)
    logger.info("Complexity distribution (train):")
    for level, cnt in complexities.most_common():
        bar = "█" * (cnt // 20)
        logger.info(f"  {level:30s}: {cnt:5d} {bar}")

    has_thinking = sum(1 for ex in train if ex.get("thinking"))
    logger.info(f"\nExamples with reasoning trace: {has_thinking}/{len(train)} ({has_thinking/len(train):.1%})")

    logger.info(f"\n✅ Dataset mixing complete!")
    logger.info(f"   Final data in: {final_dir}")
    logger.info(f"   Next step: python training_pipeline/train.py")


if __name__ == "__main__":
    main()

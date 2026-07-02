"""
sdg_pipeline/run_sdg.py
=======================
Main entrypoint for the SDG Hub synthetic data generation pipeline.

This script:
1. Loads the seed dataset from data/raw/seed_sample.jsonl
2. Configures the teacher model backend (modular – API or local)
3. Runs the SDG Hub enrichment flow
4. Filters low-quality examples
5. Saves the generated data to data/generated/

Usage:
    python sdg_pipeline/run_sdg.py --config config/pipeline_config.yaml
    python sdg_pipeline/run_sdg.py --config config/pipeline_config.yaml --n-samples 100 --dry-run

The teacher backend is controlled by config.teacher.backend:
    "anthropic"    → Anthropic Claude API
    "openai"       → OpenAI API
    "vllm_local"   → Local vLLM server (GPU-hosted)
    "ollama_local" → Local Ollama server
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import time
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("sdg_pipeline")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_teacher_config(config: dict) -> dict:
    """
    Returns the active teacher backend config dict,
    merging common fields with backend-specific ones.
    """
    backend = config["teacher"]["backend"]
    teacher_cfg = config["teacher"][backend]
    teacher_cfg["backend"] = backend
    logger.info(f"Teacher backend: {backend} | model: {teacher_cfg.get('model')}")
    return teacher_cfg


# ---------------------------------------------------------------------------
# Teacher model connector (modular)
# ---------------------------------------------------------------------------

class TeacherModelConnector:
    """
    Abstraction layer for different teacher model backends.
    SDG Hub uses OpenAI-compatible APIs, so we translate all backends
    into a unified (api_base, api_key, model) triple that SDG Hub can consume.

    This is the modularity layer: swap backends without touching the flow.
    """

    def __init__(self, teacher_cfg: dict):
        self.backend = teacher_cfg["backend"]
        # "model" kann auch "deployment_name" heißen (Azure/custom endpoints)
        self.model = teacher_cfg.get("model") or teacher_cfg.get("deployment_name")
        if not self.model:
            raise KeyError("Config muss entweder 'model' oder 'deployment_name' enthalten")
        self.max_tokens = teacher_cfg.get("max_tokens", 2048)
        self.temperature = teacher_cfg.get("temperature", 0.7)

        # Resolve api_base and api_key per backend
        if self.backend == "anthropic":
            # Use LiteLLM proxy format: SDG Hub supports "anthropic/" prefix
            self.api_base = None   # LiteLLM handles this internally
            self.api_key = teacher_cfg["api_key"]
            # SDG Hub model string for Anthropic via LiteLLM
            self.sdg_model_str = f"anthropic/{self.model}"
            os.environ["ANTHROPIC_API_KEY"] = self.api_key

        elif self.backend == "openai":
            self.api_base = teacher_cfg.get("api_base", "https://api.openai.com/v1")
            self.api_key = teacher_cfg["api_key"]
            self.api_version = teacher_cfg.get("api_version", "2024-02-01")

            # Detect Azure endpoint automatically by URL pattern
            is_azure = "openai.azure.com" in self.api_base
            if is_azure:
                # LiteLLM requires "azure/<deployment>" for Azure OpenAI
                self.sdg_model_str = f"azure/{self.model}"
                os.environ["AZURE_API_KEY"] = self.api_key
                os.environ["AZURE_API_BASE"] = self.api_base
                os.environ["AZURE_API_VERSION"] = self.api_version
                logger.info(f"  Detected Azure endpoint → using model string: {self.sdg_model_str}")
            else:
                self.sdg_model_str = f"openai/{self.model}"
                os.environ["OPENAI_API_KEY"] = self.api_key
                os.environ["OPENAI_API_BASE"] = self.api_base
                logger.info(f"  Standard OpenAI endpoint → using model string: {self.sdg_model_str}")

        elif self.backend == "azure":
            # Azure OpenAI via LiteLLM
            # LiteLLM model string format: "azure/<deployment_name>"
            self.deployment_name = teacher_cfg["deployment_name"]
            self.api_base = teacher_cfg["api_base"].rstrip("/")  # no trailing slash
            self.api_key = teacher_cfg["api_key"]
            self.api_version = teacher_cfg.get("api_version", "2024-02-01")
            self.sdg_model_str = f"azure/{self.deployment_name}"
            # LiteLLM reads these from environment
            os.environ["AZURE_API_KEY"] = self.api_key
            os.environ["AZURE_API_BASE"] = self.api_base
            os.environ["AZURE_API_VERSION"] = self.api_version
            logger.info(f"  Azure endpoint: {self.api_base}")
            logger.info(f"  Azure deployment: {self.deployment_name}")
            logger.info(f"  Azure API version: {self.api_version}")

        elif self.backend == "vllm_local":
            self.api_base = teacher_cfg["api_base"]
            self.api_key = teacher_cfg.get("api_key", "token-local")
            # vLLM exposes OpenAI-compatible API
            self.sdg_model_str = f"hosted_vllm/{self.model}"

        elif self.backend == "ollama_local":
            self.api_base = teacher_cfg["api_base"]
            self.api_key = teacher_cfg.get("api_key", "ollama")
            self.sdg_model_str = f"ollama_chat/{self.model}"

        else:
            raise ValueError(f"Unknown teacher backend: {self.backend}")

        logger.info(f"Teacher connector initialized: {self.sdg_model_str}")
        if self.api_base:
            logger.info(f"  API base: {self.api_base}")

    def get_sdg_hub_model_config(self) -> dict:
        """Returns the dict to pass to flow.set_model_config()"""
        cfg = {
            "model": self.sdg_model_str,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.api_base:
            cfg["api_base"] = self.api_base
        if self.api_key:
            cfg["api_key"] = self.api_key
        # Azure needs api_version passed explicitly
        # (also when Azure is auto-detected via openai backend)
        if hasattr(self, "api_version") and self.api_version:
            cfg["api_version"] = self.api_version
        return cfg


# ---------------------------------------------------------------------------
# SDG Hub pipeline runner
# ---------------------------------------------------------------------------

def run_sdg_pipeline(
    config: dict,
    seed_path: Path,
    output_dir: Path,
    n_samples: int | None = None,
    dry_run: bool = False,
) -> Path:
    """
    Runs the full SDG Hub enrichment pipeline.

    Returns the path to the generated output file.
    """
    try:
        from datasets import Dataset
        from sdg_hub import FlowRegistry, Flow
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Install with: pip install sdg-hub datasets")
        sys.exit(1)

    logger.info("Initializing SDG pipeline...")

    # Load seed data
    logger.info(f"Loading seed data from: {seed_path}")
    examples = []
    with open(seed_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    import random
    n_samples = n_samples or config["data"].get("sdg_seed_input_size") or config["sdg"].get("sdg_seed_input_size")
    if n_samples:
        random.shuffle(examples)
        examples = examples[:n_samples]
    logger.info(f"Seed examples to process: {len(examples)}")

    if dry_run:
        logger.info("DRY RUN mode – skipping LLM calls, saving mock output")
        _save_mock_output(examples[:5], output_dir)
        return output_dir / "generated_dry_run.jsonl"

    # Configure teacher model
    teacher_cfg = resolve_teacher_config(config)
    connector = TeacherModelConnector(teacher_cfg)

    # Discover and load flow
    logger.info("Discovering SDG Hub flows...")
    FlowRegistry.discover_flows()

    flow_path = Path(__file__).parent / "flows" / "text2sql_enrichment.yaml"
    if not flow_path.exists():
        raise FileNotFoundError(f"Flow not found: {flow_path}")

    logger.info(f"Loading flow: {flow_path}")
    flow = Flow.from_yaml(str(flow_path))

    # Set teacher model config (this is where modularity lives)
    model_config = connector.get_sdg_hub_model_config()
    logger.info(f"Setting model config: {model_config}")
    flow.set_model_config(**model_config)

    # Convert to HuggingFace Dataset (SDG Hub's native format)
    dataset = Dataset.from_list(examples)
    logger.info(f"Starting SDG generation on {len(dataset)} examples...")
    logger.info(f"Async batch size: {config['sdg'].get('async_batch_size', 32)}")

    start = time.time()
    # Rate Limit Strategie:
    # 1. max_concurrency: max parallele Requests pro Batch
    # 2. Wir splitten das Dataset in kleine Batches mit Pause dazwischen
    max_concurrency = config["sdg"].get("max_concurrency", 20)
    batch_size = config["sdg"].get("batch_size", 20)
    pause_seconds = config["sdg"].get("pause_between_batches_seconds", 2)

    logger.info(f"Rate limit Strategie: max_concurrency={max_concurrency}, batch_size={batch_size}, pause={pause_seconds}s")

    # Dataset in Batches aufteilen
    from datasets import Dataset as HFDataset

    all_rows = [dataset[i] for i in range(len(dataset))]
    batches = [all_rows[i:i+batch_size] for i in range(0, len(all_rows), batch_size)]
    logger.info(f"Verarbeite {len(batches)} Batches à {batch_size} Beispiele...")

    result_rows = []
    for batch_idx, batch in enumerate(batches):
        logger.info(f"Batch {batch_idx+1}/{len(batches)} ({len(batch)} Beispiele)...")
        batch_dataset = HFDataset.from_list(batch)
        batch_result = flow.generate(batch_dataset, max_concurrency=max_concurrency)
        result_rows.extend([batch_result[i] for i in range(len(batch_result))])

        if batch_idx < len(batches) - 1:
            logger.info(f"Pause {pause_seconds}s (Rate Limit Schutz)...")
            time.sleep(pause_seconds)

    from datasets import Dataset as HFDataset2
    result_dataset = HFDataset2.from_list(result_rows)
    elapsed = time.time() - start
    logger.info(f"Generation complete in {elapsed:.1f}s ({len(result_dataset)} examples)")

    # Quality filter: keep rows where key fields were successfully parsed
    before = len(result_dataset)
    def has_required_fields(row):
        return (
            bool(row.get("upgraded_sql", "").strip()) and
            bool(row.get("upgraded_question", "").strip()) and
            bool(row.get("reasoning_trace", "").strip())
        )
    result_dataset = result_dataset.filter(has_required_fields)
    after = len(result_dataset)
    logger.info(f"Quality filter: {before} → {after} examples kept ({after/before:.1%})")

    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "generated_enriched.jsonl"

    with open(output_path, "w") as f:
        for ex in result_dataset:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(f"✅ Generated data saved: {output_path}")

    # Print summary statistics
    _print_generation_summary(result_dataset)

    return output_path


def _save_mock_output(examples: list[dict], output_dir: Path):
    """Saves mock output for dry-run testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mock_examples = []
    for ex in examples:
        mock = dict(ex)
        mock["upgraded_question"] = f"[MOCK] Harder version of: {ex['question']}"
        mock["upgraded_sql"] = f"SELECT t1.*, COUNT(t2.id) FROM ({ex['sql']}) t1 LEFT JOIN orders t2 ON t1.id = t2.user_id GROUP BY t1.id"
        mock["upgraded_complexity"] = "multiple joins"
        mock["variant_schema"] = "CREATE TABLE mock_table (id INT, value TEXT);"
        mock["variant_question"] = "[MOCK] How many records are in mock_table?"
        mock["variant_sql"] = "SELECT COUNT(*) FROM mock_table;"
        mock["reasoning_trace"] = "[MOCK] Step 1: Identify table. Step 2: Use COUNT. Step 3: Done."
        mock["quality_score"] = 0.85
        mock_examples.append(mock)

    out = output_dir / "generated_dry_run.jsonl"
    with open(out, "w") as f:
        for ex in mock_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Mock output saved: {out}")


def _print_generation_summary(dataset: Any):
    """Prints a summary of the generated dataset."""
    from collections import Counter
    complexities = Counter()
    domains = Counter()
    for ex in dataset:
        complexities[ex.get("complexity", "unknown")] += 1
        domains[ex.get("domain", "unknown")] += 1

    logger.info("\n=== Generation Summary ===")
    logger.info(f"Total examples: {len(dataset)}")
    logger.info("Top domains:")
    for domain, count in domains.most_common(5):
        logger.info(f"  {domain}: {count}")
    logger.info("Complexity distribution:")
    for level, count in complexities.most_common():
        logger.info(f"  {level}: {count}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run SDG Hub pipeline for Text-to-SQL synthetic data generation"
    )
    parser.add_argument("--config", default="config/pipeline_config.yaml",
                        help="Path to pipeline config YAML")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Override number of seed examples to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls, save mock output (for testing)")
    parser.add_argument("--seed-file", default=None,
                        help="Override path to seed JSONL file")
    args = parser.parse_args()

    config = load_config(args.config)

    seed_path = Path(args.seed_file) if args.seed_file else \
                Path(config["data"]["raw_dir"]) / "seed_sample.jsonl"

    if not seed_path.exists():
        logger.error(f"Seed file not found: {seed_path}")
        logger.error("Run: python data_pipeline/prepare_data.py first")
        sys.exit(1)

    output_dir = Path(config["data"]["generated_dir"])

    output_path = run_sdg_pipeline(
        config=config,
        seed_path=seed_path,
        output_dir=output_dir,
        n_samples=args.n_samples,
        dry_run=args.dry_run,
    )

    logger.info(f"\n✅ SDG pipeline complete!")
    logger.info(f"   Output: {output_path}")
    logger.info(f"   Next step: python data_pipeline/mix_datasets.py")


if __name__ == "__main__":
    main()

"""
serving/merge_adapter.py
========================
Merged LoRA-Adapter in das Basismodell.
Das gemergte Modell kann direkt mit vLLM oder HuggingFace geladen werden.

Usage:
    python3 serving/merge_adapter.py \
        --adapter-path data/final/checkpoints/qwen0.5b_text2sql_v1_50seeds_3epochs \
        --output-path data/final/checkpoints/qwen0.5b_text2sql_v1_50seeds_3epochs_merged \
        --config config/pipeline_config.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    output_path = Path(args.output_path)

    # Base model aus adapter_config.json lesen
    adapter_cfg_file = adapter_path / "adapter_config.json"
    if adapter_cfg_file.exists():
        with open(adapter_cfg_file) as f:
            adapter_cfg = json.load(f)
        base_model_id = adapter_cfg.get("base_model_name_or_path")
        logger.info(f"Base model aus adapter_config: {base_model_id}")
    else:
        # Fallback auf pipeline_config
        with open(args.config) as f:
            config = yaml.safe_load(f)
        base_model_id = config["student"].get("model_path") or config["student"]["model_id"]
        logger.info(f"Base model aus pipeline_config: {base_model_id}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info(f"Lade Basismodell: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    ).to(torch.bfloat16).cuda()

    logger.info(f"Lade LoRA Adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    logger.info("Merge Adapter in Basismodell...")
    model = model.merge_and_unload()

    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Speichere gemergtes Modell (sharded): {output_path}")
    # SHARDED save (max_shard_size) — single-file safetensors stalls verl's 0.23-nightly vLLM loader.
    model.save_pretrained(str(output_path), safe_serialization=True, max_shard_size="5GB")
    # carry the chat template (Qwen3 thinking) if the base/adapter shipped one as a sidecar
    import shutil
    for src in (adapter_path / "chat_template.jinja",):
        if src.exists():
            shutil.copy(src, output_path / "chat_template.jinja")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(str(output_path))

    logger.info(f"✅ Merge abgeschlossen: {output_path}")


if __name__ == "__main__":
    main()

"""
serving/merge_adapter.py
========================
Merged LoRA-Adapter in das Basismodell. Das gemergte Modell kann direkt mit vLLM oder HuggingFace geladen
werden. Modell-agnostisch: das Basismodell kommt aus adapter_config.json.

Zwei HARTE Asserts (sys.exit(1)) — geerbt von der Disziplin des früheren merge_adapter_mm.py:
  1. NO-OP-Guard: greifen die Adapter-Keys nicht, lädt PEFT still 0 Gewichte und merge_and_unload() tut
     nichts -> man deployt das BASISMODELL im Glauben, es sei trainiert. Jede nachgelagerte Eval liest das
     als "das Training hat nichts gebracht" — ein False Negative, das wie ein Ergebnis aussieht.
  2. SAVE-Guard: das Gewicht wird aus dem GESPEICHERTEN Artefakt zurückgelesen und gegen den gemergten
     In-Memory-Wert geprüft (der Welle-1-Crash war genau ein Save-/Config-Bug).

Usage (im Training-Container ist PYTHONPATH=/app gebacken; auf dem Host PYTHONPATH=. voranstellen):
    python3 serving/merge_adapter.py \
        --adapter-path data/final/checkpoints/db_bahn_traj_lora/ep2 \
        --output-path data/final/checkpoints/db_bahn_traj_merged/ep2 \
        --config config/pipeline_config.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from data_pipeline.common import load_config

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
        config = load_config(args.config)
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

    # Probe-Gewicht VOR dem Merge sichern (ein LoRA-Ziel-Modul) -> Grundlage beider Asserts unten.
    probe_key = next((n for n, _ in base_model.named_parameters()
                      if n.endswith("layers.0.self_attn.q_proj.weight")), None)
    if probe_key is None:
        logger.error("Kein Probe-Gewicht gefunden (erwartet q_proj in Layer 0) -> Merge nicht verifizierbar.")
        sys.exit(1)
    probe_before = base_model.state_dict()[probe_key].detach().clone()

    logger.info(f"Lade LoRA Adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))

    logger.info("Merge Adapter in Basismodell...")
    model = model.merge_and_unload()

    # --- ASSERT 1: hat der Merge die Gewichte ueberhaupt veraendert? ---------------------------------
    probe_after = model.state_dict()[probe_key].detach()
    if torch.equal(probe_before.cpu(), probe_after.cpu()):
        logger.error(f"MERGE NO-OP: {probe_key} unveraendert -> der Adapter hat NICHT gegriffen "
                     f"(Key-Mismatch?). Das waere das Basismodell im Trainings-Kostuem. ABBRUCH.")
        sys.exit(1)
    if not torch.isfinite(probe_after).all():
        logger.error("MERGE erzeugte nicht-finite Gewichte (NaN/Inf). ABBRUCH.")
        sys.exit(1)
    delta = (probe_after.float() - probe_before.float().to(probe_after.device)).abs().max().item()
    logger.info(f"✓ Merge verifiziert: {probe_key} veraendert (max|Δ| = {delta:.3e})")

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

    # --- ASSERT 2: steht auf der Platte wirklich das, was wir gemergt haben? -------------------------
    from safetensors import safe_open
    idx_file = output_path / "model.safetensors.index.json"
    shard = (json.load(open(idx_file))["weight_map"][probe_key] if idx_file.exists()
             else "model.safetensors")
    with safe_open(str(output_path / shard), framework="pt") as f:
        saved = f.get_tensor(probe_key)
    if not torch.equal(saved.cpu(), probe_after.cpu()):
        logger.error(f"SAVE-MISMATCH: {probe_key} auf der Platte != gemergtes Gewicht "
                     f"(Shard {shard}). Das Artefakt ist unbrauchbar. ABBRUCH.")
        sys.exit(1)
    logger.info(f"✓ Save verifiziert: {probe_key} auf Platte == gemergtes Gewicht (Shard {shard})")

    logger.info(f"✅ Merge abgeschlossen: {output_path}")


if __name__ == "__main__":
    main()

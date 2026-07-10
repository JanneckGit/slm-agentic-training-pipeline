"""
training_pipeline/train_traj.py
===============================
Phase 6 of Plan (B): traj_sft — LoRA SFT of the ~4B student on the verified multi-turn DB traces, with
ASSISTANT-ONLY loss masking (training_pipeline/collator_multiturn.py). Standalone (does not touch the
SQL run_lora_sft path); reuses config[training].lora + the TRL/PEFT stack in the training image.

Pre-tokenizes each trace with build_masked_labels (labels=-100 off assistant spans), drops over-length,
trains with a plain HF Trainer + TrajSFTCollator, saves the LoRA adapter (+ tokenizer + chat template).

Usage (training container):
    python3 training_pipeline/train_traj.py --config config/pipeline_config.yaml \
        --data data/final/db_traces_chat.jsonl --model Qwen/Qwen3.5-4B \
        --out data/final/checkpoints/db_bahn_traj_lora --epochs 2 --max-seq-len 4096
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from training_pipeline.collator_multiturn import TrajSFTCollator, build_masked_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline_config.yaml")
    ap.add_argument("--data", default="data/final/db_traces_chat.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--out", default="data/final/checkpoints/db_bahn_traj_lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--max-samples", type=int, default=None, help="smoke: cap #examples")
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke: cap steps")
    ap.add_argument("--run-name", default=None, help="MLflow run name (default: traj_sft_<model>_<epochs>ep)")
    ap.add_argument("--no-mlflow", action="store_true", help="disable MLflow tracking (report_to=[])")
    args = ap.parse_args()

    # MLflow (HF-native via report_to): the training container sets MLFLOW_TRACKING_URI=file:///app/mlruns;
    # default the experiment here so it also works outside compose. Metrics/params are auto-logged by HF's
    # MLflowCallback every logging_steps. File-based logs (console/logs/*.log) stay the source of truth.
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "db_bahn_traj_sft")
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # mlflow>=3.14 gates the file backend
    report_to = [] if args.no_mlflow else ["mlflow"]
    run_name = args.run_name or f"traj_sft_{Path(args.model).name}_{args.epochs:g}ep"

    config = yaml.safe_load(open(args.config)) if Path(args.config).exists() else {}
    t_cfg = config.get("training", {})
    lora_cfg = t_cfg.get("lora", {})
    seed = config.get("seed", 42)
    set_seed(seed)
    torch.manual_seed(seed)

    from peft import LoraConfig, TaskType, get_peft_model

    logger.info(f"Tokenizer/Model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token, tok.pad_token_id = tok.eos_token, tok.eos_token_id

    # pre-tokenize with assistant-only masks
    examples = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.max_samples:
        examples = examples[:args.max_samples]
    feats, dropped, mask_frac = [], 0, []
    for ex in examples:
        ids, labels = build_masked_labels(tok, ex["messages"], max_len=args.max_seq_len)
        if len(ids) >= args.max_seq_len:  # truncated -> mask likely broken, drop
            dropped += 1
            continue
        if not any(l != -100 for l in labels):  # nothing to learn
            dropped += 1
            continue
        feats.append({"input_ids": ids, "labels": labels})
        mask_frac.append(sum(1 for l in labels if l != -100) / len(labels))
    logger.info(f"examples {len(examples)} -> train {len(feats)} (dropped {dropped}); "
                f"mean assistant-token frac {sum(mask_frac) / max(1, len(mask_frac)):.0%}")
    ds = Dataset.from_list(feats)

    logger.info("Loading model (bf16, cuda)...")
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(torch.bfloat16).cuda()
    model.enable_input_require_grads()
    model.config.use_cache = False
    lora = LoraConfig(r=lora_cfg.get("r", 16), lora_alpha=lora_cfg.get("alpha", 32),
                      lora_dropout=lora_cfg.get("dropout", 0.05),
                      target_modules=lora_cfg.get("target_modules",
                                                  ["q_proj", "k_proj", "v_proj", "o_proj",
                                                   "gate_proj", "up_proj", "down_proj"]),
                      bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ta = TrainingArguments(
        output_dir=args.out + "_ckpt", num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=t_cfg.get("micro_batch_size", 2),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(t_cfg.get("learning_rate", 2.0e-4)),
        lr_scheduler_type=t_cfg.get("lr_scheduler", "cosine"), warmup_ratio=0.03,
        bf16=True, gradient_checkpointing=True, logging_steps=5, save_strategy="no",
        report_to=report_to, run_name=run_name, seed=seed)
    trainer = Trainer(model=model, args=ta, train_dataset=ds, data_collator=TrajSFTCollator(tok))
    logger.info(f"Training: {len(feats)} traces, {args.epochs} epochs, "
                f"eff.batch {ta.per_device_train_batch_size * ta.gradient_accumulation_steps}")
    trainer.train()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    logger.info(f"✅ adapter saved -> {out}")


if __name__ == "__main__":
    main()

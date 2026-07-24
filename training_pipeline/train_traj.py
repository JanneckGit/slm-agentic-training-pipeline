"""
training_pipeline/train_traj.py
===============================
Phase 6 of Plan (B): traj_sft — LoRA SFT of the ~4B student on the verified multi-turn DB traces, with
ASSISTANT-ONLY loss masking (training_pipeline/collator_multiturn.py). Standalone (does not touch the
SQL run_lora_sft path); reuses config[training].lora + the TRL/PEFT stack in the training image.

Pre-tokenizes each trace with build_masked_labels (labels=-100 off assistant spans), drops over-length,
trains with a plain HF Trainer + TrajSFTCollator, saves the LoRA adapter (+ tokenizer + chat template).

Assistant turns BEFORE the last user message are masked too (the template renders them think-less) —
opt out with --train-context-turns. The per-leg line in the startup log shows what that costs each leg.

Usage (training container):
    python3 training_pipeline/train_traj.py --config config/pipeline_config.yaml \
        --data data/final/sft_mix_chat.jsonl --model Qwen/Qwen3-4B \
        --out data/final/checkpoints/db_bahn_traj_lora --epochs 2 --max-seq-len 12288
"""

import argparse
import json
import logging
import os
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback,
                          TrainingArguments, set_seed)

from data_pipeline.common import STUDENT_MODEL_DEFAULT, load_config
from training_pipeline.collator_multiturn import TrajSFTCollator, build_masked_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class EpochAdapterSaver(TrainerCallback):
    """Save the LoRA adapter (+ tokenizer/chat_template) at each epoch end -> <out>/ep{N}, so the pipeline can
    eval epoch-1 vs epoch-2 (rollout verified_yield) and keep the better one (checkpoint selection).

    Layout: ONE folder per run instead of sibling _ep1/_ep2/_ckpt dirs —
        <out>/ep1/ · <out>/ep2/ · <out>/selected -> ep{winner}   (symlink, set after the eval picks)
    """

    def __init__(self, model, tok, out):
        self.model, self.tok, self.out = model, tok, out

    def on_epoch_end(self, args, state, control, **kwargs):
        ep = int(round(state.epoch))  # 1.0 -> 1, 2.0 -> 2 (fires only on full epochs, not smoke max_steps)
        dst = Path(self.out) / f"ep{ep}"
        dst.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(dst))
        self.tok.save_pretrained(str(dst))
        logger.info(f"✅ epoch-{ep} adapter saved -> {dst}")


class MlflowExtraParams(TrainerCallback):
    """Log LoRA config + run provenance that HF's MLflowCallback misses (LoraConfig lives in the PEFT model,
    not TrainingArguments, so r/alpha/targets are otherwise untracked). Runs after MLflowCallback started the run."""

    def __init__(self, params):
        self.params = params

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_params(self.params)
                logger.info(f"[mlflow] extra params logged: {self.params}")
        except Exception as e:  # never let tracking break training
            logger.warning(f"[mlflow] extra-params log skipped: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline_config.yaml")
    ap.add_argument("--data", default="data/final/sft_mix_chat.jsonl")  # the 3-leg Stage-1 mix
    ap.add_argument("--model", default=STUDENT_MODEL_DEFAULT)  # dense, text-only, thinking (NOT the MM hybrid 3.5)
    ap.add_argument("--out", default="data/final/checkpoints/db_bahn_traj_lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-seq-len", type=int, default=12288)  # 3-leg mix: db_bahn<=5.9k, AReaL trimmed@12288
    ap.add_argument("--val-file", default=None, help="held-out val JSONL -> eval_loss (never in gradient)")
    ap.add_argument("--eval-steps", type=int, default=300)  # overfit-diag only (no early-stop) -> few eval points suffice
    ap.add_argument("--attn", default="flash_attention_2",
                    help="attention impl; falls back to sdpa if it fails to load (e.g. FA2 unbuilt on sm_121)")
    ap.add_argument("--liger", action="store_true",
                    help="fused linear cross-entropy (Liger) -> no full-seq logit tensor; unlocks micro_batch>4 @12k")
    ap.add_argument("--grad-checkpointing", default="on", choices=["on", "off"],
                    help="recompute activations in the backward pass to save memory (default on). "
                         "off -> ~20-30%% faster if it fits (128GB unified) — throughput knob, no quality effect.")
    ap.add_argument("--micro-batch", type=int, default=None, help="smoke: override config micro_batch_size")
    ap.add_argument("--grad-accum", type=int, default=None, help="smoke: override config gradient_accumulation_steps")
    ap.add_argument("--neftune", type=float, default=None,
                    help="NEFTune noise alpha (noisy input embeddings, train-time only, no serving cost); None=off")
    ap.add_argument("--train-context-turns", action="store_true",
                    help="opt-out: also take gradient on assistant turns BEFORE the last user message. "
                         "Those render think-less (Qwen3 template) — only for A/B measurements.")
    ap.add_argument("--save-epoch-adapters", action="store_true",
                    help="save adapter+tok to <out>_ep{N} at each epoch end (for epoch-1-vs-2 checkpoint selection)")
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

    config = load_config(args.config) if Path(args.config).exists() else {}
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

    # pre-tokenize with assistant-only masks (shared by train + val)
    final_turns_only = not args.train_context_turns
    logger.info(f"context-turn masking: {'ON' if final_turns_only else 'OFF (--train-context-turns)'} — "
                f"gradient only on assistant turns after the last user message")

    def pretokenize(path, cap=None):
        exs = [json.loads(l) for l in open(path) if l.strip()]
        if cap:
            exs = exs[:cap]
        # per-leg accounting: without it the effect of the masking is invisible (a single global mean
        # hides that db_bahn is untouched while AReaL loses ~78% of its trained tokens).
        feats, dropped, mask_frac, srcs = [], 0, [], Counter()
        legs = {}
        for ex in exs:
            ids, labels = build_masked_labels(tok, ex["messages"], max_len=args.max_seq_len,
                                              final_turns_only=final_turns_only)
            if len(ids) >= args.max_seq_len:      # truncated -> mask likely broken, drop
                dropped += 1
                continue
            if not any(l != -100 for l in labels):  # nothing to learn
                dropped += 1
                continue
            feats.append({"input_ids": ids, "labels": labels, "length": len(ids)})  # length -> group_by_length
            src = (ex.get("_meta") or {}).get("mix_source", "unknown")
            srcs[src] += 1
            trained = sum(1 for l in labels if l != -100)
            mask_frac.append(trained / len(labels))
            st = legs.setdefault(src, Counter())
            st["n"] += 1
            st["fwd"] += len(ids)
            st["trained"] += trained
            # DIAGNOSTIC ONLY (no behaviour change): a last assistant turn without its own <think> gets
            # Qwen3's canonical empty "<think>\n\n</think>" wrapper (template `loop.last` branch).
            tail = ex["messages"][-1]
            if tail.get("role") == "assistant" and "<think>" not in (tail.get("content") or ""):
                st["empty_think_wrapper"] += 1
        return feats, dropped, (sum(mask_frac) / max(1, len(mask_frac))), srcs, legs

    def log_legs(tag, legs):
        tot = max(1, sum(s["trained"] for s in legs.values()))
        for src, s in sorted(legs.items()):
            logger.info(f"  [{tag}] {src:8s} n={s['n']:6d} fwd={s['fwd']:10d} trained={s['trained']:9d} "
                        f"({100*s['trained']/s['fwd']:4.1f}% of own fwd, {100*s['trained']/tot:4.1f}% of gradient) "
                        f"empty-think-wrapper={s['empty_think_wrapper']}")

    feats, dropped, frac, mix_counts, legs = pretokenize(args.data, args.max_samples)
    logger.info(f"train examples -> {len(feats)} (dropped {dropped}); mean assistant-token frac {frac:.0%}; "
                f"mix {dict(mix_counts)}")
    log_legs("train", legs)
    ds = Dataset.from_list(feats)
    eval_ds, n_val = None, 0
    if args.val_file and Path(args.val_file).exists():
        vfeats, vdrop, vfrac, _, _ = pretokenize(args.val_file, args.max_samples)
        eval_ds, n_val = Dataset.from_list(vfeats), len(vfeats)
        logger.info(f"val examples -> {len(vfeats)} (dropped {vdrop}); frac {vfrac:.0%}")

    logger.info(f"Loading model (bf16, cuda, attn={args.attn})...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, attn_implementation=args.attn,
            torch_dtype=torch.bfloat16).cuda()
    except Exception as e:  # FA2 may be unbuilt/incompatible on sm_121 -> graceful fallback
        logger.warning(f"attn={args.attn} load failed ({type(e).__name__}: {str(e)[:80]}); falling back to sdpa")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, attn_implementation="sdpa",
            torch_dtype=torch.bfloat16).cuda()
    logger.info(f"attn_implementation in use: {getattr(model.config, '_attn_implementation', '?')}")
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
        # HF requires an output_dir even with save_strategy="no" (it would create an empty <out>_ckpt next to
        # the results). Park it in a scratch dir — stays usable once we add save_steps/resume for preemption.
        output_dir="data/final/checkpoints/_trainer_scratch",
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.micro_batch or t_cfg.get("micro_batch_size", 2),
        gradient_accumulation_steps=args.grad_accum or t_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(t_cfg.get("learning_rate", 2.0e-4)),
        lr_scheduler_type=t_cfg.get("lr_scheduler", "cosine"), warmup_ratio=0.03,
        bf16=True, gradient_checkpointing=(args.grad_checkpointing == "on"), logging_steps=5,
        save_strategy="no",  # logging_steps is a print only — no runtime cost; eval/save are off
        # transformers 5.x: the group_by_length bool became train_sampling_strategy. Homogeneous-length
        # batches -> micro_batch>1 doesn't pad to 12k on mixed batches (LengthGroupedSampler + "length" col).
        train_sampling_strategy="group_by_length", length_column_name="length",
        eval_strategy=("steps" if eval_ds is not None else "no"), eval_steps=args.eval_steps,
        per_device_eval_batch_size=1, use_liger_kernel=args.liger, neftune_noise_alpha=args.neftune,
        report_to=report_to, run_name=run_name, seed=seed)
    # echo the EFFECTIVE knobs (read back from ta, not args) — proves what actually reached the trainer
    logger.info(f"[cfg] gradient_checkpointing={ta.gradient_checkpointing} "
                f"micro_batch={ta.per_device_train_batch_size} grad_accum={ta.gradient_accumulation_steps} "
                f"(eff {ta.per_device_train_batch_size * ta.gradient_accumulation_steps}) attn={args.attn} "
                f"liger={ta.use_liger_kernel} neftune={ta.neftune_noise_alpha} max_steps={ta.max_steps}")
    callbacks = []
    if args.save_epoch_adapters:
        callbacks.append(EpochAdapterSaver(model, tok, args.out))
    if not args.no_mlflow:  # log the LoRA/provenance params HF's MLflowCallback misses
        extra = {"lora_r": lora_cfg.get("r", 16), "lora_alpha": lora_cfg.get("alpha", 32),
                 "lora_targets": len(lora.target_modules), "base_model": args.model,
                 "data_file": args.data, "train_n": len(feats), "val_n": n_val,
                 "max_seq_len": args.max_seq_len, "hardware": os.environ.get("HW_TAG", "GB10/sm_121"),
                 "final_turns_only": final_turns_only}
        extra.update({f"mix_{k}": v for k, v in mix_counts.items()})  # per-source composition of the mix
        extra.update({f"trained_tok_{k}": s["trained"] for k, s in legs.items()})  # gradient share per leg
        callbacks.append(MlflowExtraParams(extra))
    trainer = Trainer(model=model, args=ta, train_dataset=ds, eval_dataset=eval_ds,
                      data_collator=TrajSFTCollator(tok), callbacks=callbacks)
    logger.info(f"Training: {len(feats)} traces, {args.epochs} epochs, "
                f"eff.batch {ta.per_device_train_batch_size * ta.gradient_accumulation_steps}")
    trainer.train()

    if args.save_epoch_adapters:
        # the per-epoch candidates in <out>/ep{N} ARE the result; a final save here would just duplicate the
        # last epoch (263 MB) and pre-empt the eval's pick. The winner gets an <out>/selected symlink instead.
        logger.info(f"✅ epoch candidates saved -> {args.out}/ep* (selected symlink is set by the eval)")
    else:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(out))
        tok.save_pretrained(str(out))
        logger.info(f"✅ adapter saved -> {out}")


if __name__ == "__main__":
    main()

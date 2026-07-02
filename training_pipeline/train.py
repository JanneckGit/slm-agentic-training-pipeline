"""
training_pipeline/train.py
==========================
Haupt-Trainingsskript für das Text-to-SQL Finetuning.

Nutzt direkt HuggingFace TRL + PEFT (LoRA) – getestet auf GB10 (ARM64/SM_121).

Getestete Versionskombination:
  - PyTorch 2.10 nv25.11 (NGC, SM_121)
  - peft 0.13.2 (--no-deps, kein torchao)
  - accelerate >=1.3.0
  - transformers 5.6.2 (NGC nativ)
  - trl 0.12.0 (NGC nativ)

Unterstützte Algorithmen (--algorithm):
  - lora_sft  → LoRA fine-tuning (Standard, empfohlen)

Usage:
    python3 training_pipeline/train.py --config config/pipeline_config.yaml
    python3 training_pipeline/train.py --config config/pipeline_config.yaml --algorithm lora_sft
    python3 training_pipeline/train.py --config config/pipeline_config.yaml --dry-run
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("training_pipeline")

try:
    import mlflow
except ImportError:
    mlflow = None
    logger.warning("mlflow nicht installiert – MLflow-Tracking deaktiviert")


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_model_path(config: dict) -> str:
    student_cfg = config["student"]
    local_path = student_cfg.get("model_path")
    if local_path and Path(local_path).exists():
        logger.info(f"Nutze lokales Modell: {local_path}")
        return local_path
    model_id = student_cfg["model_id"]
    logger.info(f"Nutze HuggingFace Modell: {model_id}")
    return model_id


def _short_model_name(model_id: str) -> str:
    """e.g. 'Qwen/Qwen2.5-0.5B-Instruct' -> 'qwen05b'"""
    import re
    name = model_id.split("/")[-1].lower()
    name = re.sub(r"-instruct$", "", name)
    name = re.sub(r"qwen2[.]5-", "qwen", name)
    name = re.sub(r"(\d+)\.(\d+)b", r"\1\2b", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name or "unknown"


def detect_data_variant(data_path: Path) -> str:
    """
    Determines whether the training data is the thinking or non-thinking
    variant by inspecting the first record's _meta.has_thinking (set by
    format_for_training.py). Falls back to the raw `thinking` field, else
    defaults to "thinking". Drives the MLflow run name / data_variant tag so
    the two comparison runs are labelled correctly regardless of which file
    is currently placed at train_chat.jsonl.
    """
    try:
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                meta = rec.get("_meta", {})
                if "has_thinking" in meta:
                    return "thinking" if meta["has_thinking"] else "nothink"
                return "thinking" if rec.get("thinking", "").strip() else "nothink"
    except Exception:
        pass
    return "thinking"


def generate_run_name(config: dict, variant: str | None = None) -> str:
    """
    Builds a run name encoding teacher, student, SDG-seed count, epochs, seed
    and timestamp (plus the data variant, if given). Example:
        t-qwen14b-bf16_s-qwen05b_sdg750_3ep_seed42_20260526_1918_thinking
    """
    from datetime import datetime

    teacher_cfg = config.get("teacher", {})
    backend = teacher_cfg.get("backend")
    if backend and backend in teacher_cfg:
        backend_block = teacher_cfg[backend]
        teacher_model = backend_block.get("model", "")
        teacher_short = _short_model_name(teacher_model) if teacher_model else "unknown"
        teacher_dtype = backend_block.get("dtype")
        if not teacher_dtype:
            logger.warning(
                f"Kein 'dtype' im teacher.{backend}-Block – nutze Default 'bf16' für Run-Name"
            )
            teacher_dtype = "bf16"
    else:
        logger.warning(
            f"Teacher-Backend '{backend}' nicht in Config auflösbar – nutze 'unknown-bf16' für Run-Name"
        )
        teacher_short = "unknown"
        teacher_dtype = "bf16"

    student_id = config["student"].get("model_path") or config["student"].get("model_id", "")
    student_short = _short_model_name(student_id) if student_id else "unknown"

    seeds = config["data"].get("sdg_seed_input_size")
    if seeds is None:
        logger.warning("Kein 'data.sdg_seed_input_size' in Config – nutze 0 für Run-Name")
        seeds = 0

    epochs = config["training"].get("num_epochs")
    if epochs is None:
        logger.warning("Kein 'training.num_epochs' in Config – nutze 3 für Run-Name")
        epochs = 3

    seed = config.get("seed")
    if seed is None:
        logger.warning("Kein 'seed' in Config – nutze 42 für Run-Name")
        seed = 42

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    return (
        f"t-{teacher_short}-{teacher_dtype}"
        f"_s-{student_short}"
        f"_sdg{seeds}"
        f"_{epochs}ep"
        f"_seed{seed}"
        f"_{timestamp}"
        + (f"_{variant}" if variant else "")
    )


def resolve_data_path(config: dict) -> Path:
    final_dir = Path(config["data"]["final_dir"])
    chat_file = final_dir / "train_chat.jsonl"
    raw_file = final_dir / "train.jsonl"
    if chat_file.exists():
        logger.info(f"Nutze Chat-Format Daten: {chat_file}")
        return chat_file
    elif raw_file.exists():
        logger.warning(f"train_chat.jsonl nicht gefunden, nutze: {raw_file}")
        logger.warning("Für beste Ergebnisse: python3 data_pipeline/format_for_training.py")
        return raw_file
    else:
        raise FileNotFoundError(
            f"Keine Trainingsdaten in {final_dir}. "
            "Bitte zuerst prepare_data.py, run_sdg.py, mix_datasets.py und "
            "format_for_training.py ausführen."
        )


def run_lora_sft(config: dict, model_path: str, data_path: Path, dry_run: bool = False,
                 max_steps: int | None = None, max_train_samples: int | None = None):
    """
    LoRA Fine-Tuning mit HuggingFace TRL + PEFT.
    Getestet auf GB10 (ARM64/SM_121) mit peft==0.13.2 + accelerate>=1.3.0.

    Smoke-Test (max_steps / max_train_samples gesetzt): begrenzt Schritte und
    Trainings-Samples für einen winzigen Validierungslauf. In diesem Modus wird
    MLflow übersprungen (kein Pollution der Leaderboard-Runs).
    """
    smoke = max_steps is not None or max_train_samples is not None
    t_cfg = config["training"]
    lora_cfg = t_cfg.get("lora", {})
    variant = detect_data_variant(data_path)   # "thinking" | "nothink"
    run_name = generate_run_name(config, variant)
    ckpt_dir = Path(t_cfg["checkpoint_dir"]) / run_name
    logger.info(f"Run name: {run_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = ckpt_dir / "data_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    final_dir = Path(config["data"]["final_dir"])
    for fname in ("train_chat.jsonl", "eval_chat.jsonl", "test_chat.jsonl"):
        src = final_dir / fname
        if src.exists():
            shutil.copy(src, snapshot_dir / fname)
        else:
            logger.warning(f"Snapshot-Quelldatei fehlt, übersprungen: {src}")
    logger.info(f"Daten-Snapshot gespeichert: {snapshot_dir}")

    # -----------------------------------------------------------------------
    # MLflow: eigenes Experiment "sft_distill" (getrennt von den eval
    # baseline_*-Runs in "text2sql-slm-finetuning" und von "sdg"), damit beide
    # Distill-Läufe (thinking/nothink) + ihre eval_loss-Kurven sauber
    # vergleichbar sind. Der TRL/transformers MLflowCallback (report_to=
    # "mlflow") loggt loss/eval_loss in den hier von uns gestarteten aktiven
    # Run; er liest MLFLOW_TRACKING_URI aus dem Env, daher setzen wir es
    # explizit auf denselben file-Store wie der Rest der Pipeline.
    MLFLOW_URI = "file:///app/mlruns"
    MLFLOW_EXPERIMENT = "sft_distill"
    eff_batch = t_cfg.get("micro_batch_size", 4) * t_cfg.get("gradient_accumulation_steps", 4)
    base_model = config["student"].get("model_path") or config["student"].get("model_id", "unknown")

    mlflow_active = False
    if mlflow is not None and not smoke:
        try:
            teacher_cfg_block = config.get("teacher", {})
            backend = teacher_cfg_block.get("backend")
            backend_block = teacher_cfg_block.get(backend, {}) if backend else {}

            os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URI
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            # Run-Name unterscheidet Modell UND Variante explizit, aus dem
            # tatsächlichen Student-Modell abgeleitet (Multi-Modell-Baselines),
            #   z.B. sft_qwen3.5-9b_thinking  vs  sft_qwen3.5-0.8b_nothink
            student_name = config["student"].get("model_path") or config["student"].get("model_id", "unknown")
            model_pretty = student_name.split("/")[-1].lower()
            mlflow_run_name = f"sft_{model_pretty}_{variant}"
            mlflow.start_run(run_name=mlflow_run_name)
            mlflow.set_tags({
                "data_variant": variant,
                "stage": "sft_distill",
                "run_name_full": run_name,
            })
            mlflow.log_params({
                "data_variant": variant,
                "base_model": base_model,
                "teacher_model": backend_block.get("model", "unknown"),
                "teacher_dtype": backend_block.get("dtype", "bf16"),
                "student_model": base_model,
                "sdg_seeds": config["data"].get("sdg_seed_input_size", 0),
                "epochs": t_cfg.get("num_epochs", 3),
                "learning_rate": t_cfg.get("learning_rate", 2e-4),
                "lora_r": lora_cfg.get("r", 16),
                "lora_alpha": lora_cfg.get("alpha", 32),
                "micro_batch_size": t_cfg.get("micro_batch_size", 4),
                "grad_accum_steps": t_cfg.get("gradient_accumulation_steps", 4),
                "effective_batch": eff_batch,
                "max_seq_len": t_cfg.get("max_seq_len", 2048),
                "seed": config.get("seed", 42),
            })
            run_id = mlflow.active_run().info.run_id
            (ckpt_dir / "mlflow_run_id.txt").write_text(run_id)
            mlflow_active = True
            logger.info(f"MLflow aktiv: exp='{MLFLOW_EXPERIMENT}' run='{mlflow_run_name}' "
                        f"variant={variant} uri={MLFLOW_URI} run_id={run_id}")
        except Exception as e:
            logger.warning(f"MLflow-Setup fehlgeschlagen ({e}) – Training läuft ohne MLflow weiter")

    logger.info("=" * 60)
    logger.info("LoRA Fine-Tuning (PEFT + TRL)")
    logger.info(f"  Modell:  {model_path}")
    logger.info(f"  Daten:   {data_path}")
    logger.info(f"  Output:  {ckpt_dir}")
    logger.info(f"  LoRA r={lora_cfg.get('r', 16)}, alpha={lora_cfg.get('alpha', 32)}")
    logger.info(f"  Epochen: {t_cfg.get('num_epochs', 3)}")
    logger.info(f"  LR:      {t_cfg.get('learning_rate', 2e-4)}")
    logger.info("=" * 60)

    if dry_run:
        logger.info("DRY RUN – kein echtes Training")
        return

    import torch
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA verfügbar: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    seed = config.get("seed", 42)
    if "seed" not in config:
        logger.warning(f"Kein 'seed' in Config – nutze Default {seed}")
    logger.info(f"Globaler Seed: {seed}")
    set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    # Tokenizer laden
    logger.info("Lade Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Modell laden – direkt auf GPU schieben (kein device_map, kein accelerate nötig)
    logger.info("Lade Modell...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
    ).to(torch.bfloat16).cuda()
    model.enable_input_require_grads()

    # LoRA konfigurieren
    logger.info("Konfiguriere LoRA...")
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset laden
    logger.info("Lade Dataset...")
    examples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    logger.info(f"Geladene Beispiele: {len(examples)}")

    if max_train_samples is not None and max_train_samples < len(examples):
        examples = examples[:max_train_samples]
        logger.info(f"SMOKE: Trainings-Samples auf {len(examples)} begrenzt")

    # Chat-Template anwenden
    def format_example(ex):
        if "messages" in ex:
            return {
                "text": tokenizer.apply_chat_template(
                    ex["messages"],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            }
        # Fallback für Raw-Format
        return {
            "text": (
                f"### Schema:\n{ex.get('schema', '')}\n\n"
                f"### Frage:\n{ex.get('question', '')}\n\n"
                f"### SQL:\n{ex.get('sql', '')}"
            )
        }

    formatted = [format_example(ex) for ex in examples]
    dataset = Dataset.from_list(formatted)
    logger.info(f"Dataset formatiert: {len(dataset)} Beispiele")

    # -----------------------------------------------------------------------
    # Eval-Dataset (für periodische Validation-Loss-Logs während des Trainings)
    # -----------------------------------------------------------------------
    # Eval-Pfad analog zum Train-Pfad ableiten: gleiches Verzeichnis, gleiche
    # Endung, nur "train" → "eval" im Dateinamen (deckt train.jsonl und
    # train_chat.jsonl gleichermaßen ab). Fehlt die Datei, läuft das Training
    # ohne Eval weiter – kein Fail, nur Warning.
    eval_path = data_path.parent / data_path.name.replace("train", "eval", 1)
    eval_dataset = None
    if eval_path.exists():
        eval_examples = []
        with open(eval_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_examples.append(json.loads(line))
        # Gleiche Formatierungs-Funktion wie für Train – Schema bleibt identisch
        eval_formatted = [format_example(ex) for ex in eval_examples]
        eval_dataset = Dataset.from_list(eval_formatted)
        logger.info(f"Eval-Dataset geladen: {len(eval_dataset)} Beispiele ({eval_path})")
    else:
        logger.warning(
            f"Keine Eval-Daten gefunden bei {eval_path} – "
            "Training läuft ohne periodische Evaluation (eval_loss wird nicht geloggt)"
        )

    # -----------------------------------------------------------------------
    # eval_steps gegen die voraussichtliche Run-Länge abgleichen
    # -----------------------------------------------------------------------
    # Eval-/Save-Strategie: Overfit-Watch
    # -----------------------------------------------------------------------
    # Pro Epoche evaluieren UND speichern, am Ende den besten Checkpoint nach
    # eval_loss laden (3 Epochen → 3 Eval-Punkte + bester ckpt statt eines
    # willkürlichen eval_steps-Werts). load_best_model_at_end verlangt, dass
    # eval_strategy == save_strategy.
    #
    # Im Smoke-Lauf (max_steps) ist Eval AUS: ein ~10-Step-Lauf erreicht keine
    # Epochengrenze, und load_best_model_at_end ohne Eval würde fehlschlagen.
    # Der Completion-only-Collator (unten) ist im Smoke trotzdem aktiv – genau
    # der wird dort getestet.
    effective_batch = t_cfg.get("micro_batch_size", 4) * t_cfg.get("gradient_accumulation_steps", 4)
    do_eval = eval_dataset is not None and not smoke
    if do_eval:
        strategy_kwargs = dict(
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
    else:
        strategy_kwargs = dict(
            eval_strategy="no",
            save_strategy="steps",
            save_steps=t_cfg.get("save_steps", 200),
        )
    logger.info(f"Eval-/Save-Strategie: {'epoch (+load_best eval_loss)' if do_eval else 'smoke/no-eval'} "
                f"| eff_batch={effective_batch}")

    # -----------------------------------------------------------------------
    # Training konfigurieren
    # -----------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=str(ckpt_dir),
        num_train_epochs=t_cfg.get("num_epochs", 3),
        per_device_train_batch_size=t_cfg.get("micro_batch_size", 4),
        # Eval-Batch-Size = Train-Batch-Size; eval ist nur Forward-Pass,
        # passt also mindestens genauso gut in den VRAM
        per_device_eval_batch_size=t_cfg.get("micro_batch_size", 4),
        gradient_accumulation_steps=t_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=t_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=t_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=t_cfg.get("warmup_ratio", 0.03),
        max_seq_length=t_cfg.get("max_seq_len", 2048),
        logging_steps=1 if smoke else t_cfg.get("logging_steps", 10),
        # Smoke: harte Schrittbegrenzung (überschreibt num_train_epochs).
        **({"max_steps": max_steps} if max_steps is not None else {}),
        bf16=True,
        fp16=False,
        dataloader_num_workers=0,
        report_to=["mlflow"] if mlflow_active else ["tensorboard"],
        dataset_text_field="text",
        # Gradient-Checkpointing spart Aktivierungsspeicher, kostet aber einen
        # zweiten Forward (Recompute). Ergebnis-invariant -> kann zur
        # Beschleunigung abgeschaltet werden, wenn der Speicher reicht (128GB
        # unified mem -> für die kleinen Modelle problemlos aus). Default True
        # = unverändert/vergleichbar zu den bereits trainierten 4B-Baselines.
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        **strategy_kwargs,
    )

    # -----------------------------------------------------------------------
    # Completion-only Loss: nur auf dem Assistant-Turn trainieren
    # -----------------------------------------------------------------------
    # Ohne Maskierung läuft der Loss über die ganze Sequenz (System+User+
    # Assistant). Beim no-thinking-Student (kurzes SQL-Target) dominiert dann
    # der Prompt den Loss und der Student bekommt kaum SQL-Gradient. Der
    # DataCollatorForCompletionOnlyLM maskiert alles bis einschließlich des
    # Assistant-Headers und trainiert nur auf der Antwort
    # (<think>...</think>+sql beim thinking, sql beim no-think).
    #
    # KLASSISCHER FEHLERFALL: matcht das Response-Template nicht exakt auf die
    # tokenisierte Sequenz, maskiert der Collator ALLES → Loss 0/NaN. Für den
    # Qwen-ChatML-Header ist <|im_start|> ein einzelnes Special-Token, das
    # Template tokenisiert daher kontextstabil und matcht exakt (verifiziert).
    from trl import DataCollatorForCompletionOnlyLM
    response_template = "<|im_start|>assistant\n"
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # Bei None überspringt der Trainer Eval automatisch
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starte Training...")
    trainer.train()

    logger.info(f"Speichere Modell nach {ckpt_dir}...")
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    if mlflow_active and mlflow is not None:
        try:
            mlflow.end_run()
        except Exception as e:
            logger.warning(f"mlflow.end_run() fehlgeschlagen: {e}")

    logger.info("✅ Training abgeschlossen!")
    logger.info(f"   Checkpoints: {ckpt_dir}")
    logger.info("   Nächster Schritt: python3 evaluation/evaluate.py")


# Algorithmus-Registry – erweiterbar für spätere Algorithmen
ALGORITHM_RUNNERS = {
    "lora_sft": run_lora_sft,
}


def main():
    parser = argparse.ArgumentParser(description="Text-to-SQL SLM Training")
    parser.add_argument("--config", default="config/pipeline_config.yaml",
                        help="Pfad zur Pipeline-Config")
    parser.add_argument("--algorithm", choices=list(ALGORITHM_RUNNERS.keys()),
                        default="lora_sft",
                        help="Trainingsalgorithmus")
    parser.add_argument("--dry-run", action="store_true",
                        help="Konfiguration prüfen ohne Training")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Smoke-Test: harte Begrenzung der Trainingsschritte "
                             "(überschreibt num_epochs, deaktiviert MLflow + Eval)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Smoke-Test: nur die ersten N Trainings-Samples nutzen")
    parser.add_argument("--micro-batch-size", type=int, default=None,
                        help="Override für training.micro_batch_size (Memory-Tuning/Smoke)")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Override für training.max_seq_len (Memory-Tuning/Smoke)")
    parser.add_argument("--grad-accum-steps", type=int, default=None,
                        help="Override für training.gradient_accumulation_steps (Smoke)")
    parser.add_argument("--student-model-id", default=None,
                        help="Override für student.model_id (Multi-Modell-Baselines, "
                             "z.B. Qwen/Qwen3.5-9B) – ohne den Config-File zu mutieren")
    parser.add_argument("--grad-checkpointing", choices=["on", "off"], default=None,
                        help="Override für training.gradient_checkpointing. 'off' = schneller "
                             "(kein Recompute), nur wenn der Speicher reicht; ergebnis-invariant.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.student_model_id is not None:
        config["student"]["model_id"] = args.student_model_id
        # Stale lokalen Pfad entfernen, damit der Override greift (resolve_model_path
        # bevorzugt sonst student.model_path).
        config["student"].pop("model_path", None)
        logger.info(f"Student-Modell überschrieben: {args.student_model_id}")
    if args.micro_batch_size is not None:
        config["training"]["micro_batch_size"] = args.micro_batch_size
    if args.max_seq_len is not None:
        config["training"]["max_seq_len"] = args.max_seq_len
    if args.grad_accum_steps is not None:
        config["training"]["gradient_accumulation_steps"] = args.grad_accum_steps
    if args.grad_checkpointing is not None:
        config["training"]["gradient_checkpointing"] = (args.grad_checkpointing == "on")
        logger.info(f"gradient_checkpointing überschrieben: {args.grad_checkpointing}")

    logger.info(f"Algorithmus: {args.algorithm}")

    model_path = resolve_model_path(config)
    data_path = resolve_data_path(config)

    runner = ALGORITHM_RUNNERS[args.algorithm]
    runner(config, model_path, data_path, dry_run=args.dry_run,
           max_steps=args.max_steps, max_train_samples=args.max_train_samples)


if __name__ == "__main__":
    main()

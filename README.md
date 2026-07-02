# Text-to-SQL SLM Finetuning Pipeline

End-to-end pipeline for finetuning a Small Language Model (SLM) on Text-to-SQL, built around:

- **SDG Hub** (Red Hat AI Innovation Team) – synthetic data generation
- **Training Hub** (Red Hat AI Innovation Team) – LoRA / SFT / OSFT finetuning
- **NVIDIA GB10 (DGX Spark)** – 128 GB unified VRAM, runs via Docker over SSH
- **vLLM** – OpenAI-compatible serving of the teacher (SDG) and the student (eval / efficiency)
- **MLflow** – experiment tracking (accuracy + serving-efficiency metrics in one run)
- **verl** (Training Hub) – optional **GRPO / RLVR** (LoRA, async server-mode) to lift the weak SQL categories after SFT
- **llm-compressor** – optional **FP8 (W8A8)** quantization of the deploy students

The pipeline is reproducible: a single global `seed: 42` in the config is shared by data mixing,
LoRA init and the eval subsample.

## Architecture Overview

```
gretelai/synthetic_text_to_sql (seed data)
        │
        ▼
┌─────────────────────────────┐
│   SDG Pipeline              │  ← sdg_pipeline/
│   (SDG Hub + Teacher)       │    Teacher backend is modular (config.teacher.backend):
│                             │    Azure / OpenAI / Anthropic API  ·  local vLLM  ·  Ollama
│                             │    production: local Qwen3.6-35B-A3B (thinking MoE) for clean traces
└────────────┬────────────────┘
             │  enriched JSONL  (data/generated/)
             ▼
┌─────────────────────────────┐
│   Data Prep & Mixing        │  ← data_pipeline/prepare_data.py · data_pipeline/mix_datasets.py
│   + chat formatting         │    data_pipeline/format_for_training.py
└────────────┬────────────────┘
             │  training_data.jsonl  (data/final/)
             ▼
┌─────────────────────────────┐
│   Training Pipeline         │  ← training_pipeline/train.py
│   (Training Hub LoRA/SFT)   │    Student: Qwen3.5-4B (primary; + Qwen3-4B/14B · think/nothink)
└────────────┬────────────────┘
             │  LoRA adapters  (data/final/checkpoints/)
             ▼
┌─────────────────────────────┐        ┌──────────────────────────────┐
│   Accuracy Evaluation       │        │   Serving-Efficiency          │
│   evaluation/evaluate.py    │  ───▶  │   evaluation/efficiency_*.py  │
│   EX (loose/strict) + EM    │        │   GuideLLM throughput/latency │
│   on data/final/test_clean  │        │   (same vLLM endpoint)        │
└────────────┬────────────────┘        └───────────────┬──────────────┘
             │                                          │
             └──────────────►  MLflow run  ◄────────────┘
                       (baseline_<model> · single source of truth)

                 │  (optional) RL refinement on the SFT-distilled thinking student
                 ▼
┌──────────────────────────────────────────────────────────────┐
│   GRPO / RLVR  (verl · LoRA · async server-mode)             │  ← training_pipeline/grpo_verl_runner.py
│   reward = execution accuracy (loose-EX, evaluation/reward.py)│    lifts window functions / set
│   on reachability-filtered weak categories                   │    operations / subqueries
└──────────────────────────────────────────────────────────────┘    MLflow experiment: grpo_distill
```

## Project Structure

```
SLM-Finetuning/
├── config/
│   ├── pipeline_config.yaml          # Central config (committed, no secrets)
│   └── pipeline_config.local.yaml    # Your secrets/overrides (gitignored)
├── sdg_pipeline/
│   ├── blocks/
│   │   └── sql_complexity_filter.py  # Custom SDG Hub block
│   ├── flows/
│   │   ├── text2sql_enrichment.yaml  # SDG Hub flow definition
│   │   └── prompts/
│   │       ├── complexity_upgrade.yaml   # Prompt: make SQL harder
│   │       ├── schema_variant.yaml       # Prompt: generate schema variants
│   │       └── reasoning_trace.yaml      # Prompt: add CoT reasoning
│   ├── run_sdg.py                    # Main SDG entrypoint
│   └── trace_capture.py              # Clean thinking-trace distillation (nudge + filter)
├── data_pipeline/                        # data-prep stage (seed + synthetic + thinking traces)
│   ├── prepare_data.py               # Download & filter the seed dataset
│   ├── prepare_sdg_input.py          # Thinking path: leakage-free + SQLite-exec SDG input (pre-SDG)
│   ├── mix_datasets.py               # Mix seed + synthetic into the final set
│   ├── clean_traces.py               # Thinking path: strip ritual tails / drop non-exec SQL from traces
│   ├── build_train_clean.py          # Thinking path: exec-gate + leakage-guard + train/val split (post-SDG)
│   ├── complexity_taxonomy.py        # Single source of truth for SQL complexity classes
│   └── format_for_training.py        # Data formatter (chat JSONL)
├── training_pipeline/
│   ├── train.py                      # Main SFT training entrypoint (Training Hub)
│   ├── grpo_verl_runner.py           # GRPO / RLVR runner (verl, LoRA) — builds the main_ppo command
│   ├── build_weak_pool.py            # GRPO: build weak-category pool (exec+leakage gate, carve held-out)
│   └── reachability_probe.py         # GRPO: reachability filter (k=8, 50%-biased, variance gate)
├── evaluation/
│   ├── evaluate.py                   # Accuracy eval: EX (loose/strict) + EM, MLflow
│   ├── reward.py                     # EX-reward for GRPO (reuses evaluate.extract_sql + loose-EX)
│   ├── rescore.py                    # Offline re-scoring of saved predictions (no inference)
│   └── efficiency_benchmark.py       # GuideLLM serving-efficiency benchmark
├── serving/                              # model merge + vLLM serving
│   ├── merge_adapter.py              # Merge a LoRA adapter into the base model (sharded save)
│   ├── merge_adapter_mm.py           # Remap a text-only LoRA into the full multimodal model
│   ├── query_model.py                # Ad-hoc Text-to-SQL query against a vLLM server
│   └── deploy_vllm.sh                # Merge adapter + start a vLLM server
├── ops/                                  # operational shell glue (GB10 runbook-as-code)
│   ├── run_baseline_pipeline.sh      # Per-model train→merge→serve→eval pipeline
│   ├── run_all_baselines.sh          # Sweep run_baseline_pipeline over all students × variants
│   ├── sdg_run_supervised.sh         # SDG teacher wedge watchdog (restart + resume)
│   ├── grpo_pilot_supervised.sh      # GRPO wedge watchdog (restart + verl auto-resume)
│   └── setup_remote.sh               # One-time GB10 (DGX Spark) host setup
├── tools/                                # diagnostics / one-off utilities
│   ├── close_rate_probe.py           # Thinking-termination probe (</think>-close / loop / trigram-rep)
│   ├── quantize_fp8.py               # FP8 (W8A8) quantization via the isolated llm-compressor venv
│   └── bench_deltanet.py             # Kernel fast-path microbenchmark (ms/step + loss)
├── docker/
│   ├── Dockerfile.sdg                # Container for the SDG pipeline
│   ├── Dockerfile.training           # Container for training + eval (+ isolated GuideLLM)
│   ├── Dockerfile.grpo               # Container for GRPO (verl 0.8.0 + nightly vLLM, isolated)
│   ├── docker-compose.yml            # sdg · training · grpo · vllm · mlflow services
│   └── requirements_sdg.txt          # Extra Python deps for the sdg image (training/grpo install inline)
├── data/
│   ├── raw/                          # Downloaded seed dataset
│   ├── generated/                    # SDG Hub output
│   └── final/                        # Mixed/formatted training data, test_clean.jsonl, eval/, grpo/ (RL pools)
├── docs/                                 # experiment logs & design notes
│   ├── experiments.md                    #   Full experiment log
│   ├── experiments-short.md              #   One-table-per-run summary
│   ├── experiments-baselines.md          #   Untrained clean-baseline control group (14 models)
│   ├── experiments-limits.md             #   Structural weak spots + thinking-cost caveats
│   ├── experiments-hardware.md           #   GB10 Blackwell training speedup (DeltaNet kernels)
│   ├── experiments-compressor.md         #   FP8 quantization pilot (llm-compressor)
│   └── experiments-verl_RL_lora-grpo.md  #   GRPO / RLVR pilot (verl) — recipe + root-cause chronicle
└── README.md
```

## Quick Start

### 1. Prerequisites (on the remote GB10 machine)

```bash
git clone https://github.com/JanneckGit/SLM-Finetuning.git
cd SLM-Finetuning

# Put secrets/overrides in a local config (gitignored, never committed):
cp config/pipeline_config.yaml config/pipeline_config.local.yaml
# then edit config/pipeline_config.local.yaml:
#   teacher.<backend>.api_key / api_base   (the SDG teacher)
# and export HF_TOKEN in your shell / docker/.env for gated model downloads.
```

### 2. Download and prepare seed data

```bash
docker compose -f docker/docker-compose.yml run --rm sdg \
  python data_pipeline/prepare_data.py \
  --config config/pipeline_config.local.yaml
```

### 3. Run synthetic data generation

```bash
docker compose -f docker/docker-compose.yml run --rm sdg \
  python sdg_pipeline/run_sdg.py \
  --config config/pipeline_config.local.yaml
```

> **Thinking traces (clean-distill path).** The flow-based `reasoning_trace` produced verbose teacher
> traces that made students loop. The production **thinking** students use a separate, hand-run trace
> pipeline (distinct from the seed+synthetic SQL-label path above):
>
> ```
> prepare_data.py → prepare_sdg_input.py → SDG (trace_capture.py) → clean_traces.py
>   → build_train_clean.py → format_for_training.py → train.py
> ```
>
> `data_pipeline/prepare_sdg_input.py` builds the leakage-free, SQLite-executable SDG input (pre-SDG);
> `sdg_pipeline/trace_capture.py` (nudge+filter) generates the traces; `data_pipeline/clean_traces.py`
> (strip ritual tails) then `data_pipeline/build_train_clean.py` (exec-gate + leakage-guard vs
> `test_clean.jsonl` + train/val split) produce the training splits. See
> [experiments.md](docs/experiments.md).

### 4. Mix and format datasets

```bash
docker compose -f docker/docker-compose.yml run --rm sdg \
  python data_pipeline/mix_datasets.py \
  --config config/pipeline_config.local.yaml

# QWEN Instruct models require chat-formatted training data
docker compose -f docker/docker-compose.yml run --rm sdg \
  python data_pipeline/format_for_training.py \
  --config config/pipeline_config.local.yaml
```

### 5. Run finetuning

```bash
docker compose -f docker/docker-compose.yml build training

docker compose -f docker/docker-compose.yml run --rm training \
  python3 training_pipeline/train.py \
  --config config/pipeline_config.local.yaml \
  --algorithm lora_sft
```

### 6. Evaluate (accuracy)

See [Evaluation](#evaluation) for what EM / loose-EX / strict-EX mean and the available flags.

```bash
# Local HF mode (loads the adapter directly, single GPU, sequential):
docker compose -f docker/docker-compose.yml run --rm training \
  python3 evaluation/evaluate.py \
  --config config/pipeline_config.local.yaml \
  --model-path data/final/checkpoints/lora \
  --use-adapter \
  --n-samples 100
```

## Serving with vLLM

The `vllm` service exposes an OpenAI-compatible endpoint and currently serves the chosen student
**Qwen3.5-4B** (`docker/docker-compose.yml`: bf16; defaults `--max-model-len 16384`, `--gpu-memory-utilization 0.85`, overridable via the `VLLM_MAX_MODEL_LEN` / `VLLM_GPU_UTIL` / `VLLM_MODEL` env vars).
Swap the `--model` / `--served-model-name` to serve the teacher or a merged adapter instead.

```bash
# Start the endpoint (only runs with the vllm profile):
docker compose -f docker/docker-compose.yml --profile vllm up -d vllm
# health: curl http://localhost:8000/health
```

Evaluating against the endpoint is much faster than local HF mode because it uses vLLM's continuous
batching via `--concurrency`:

```bash
docker compose -f docker/docker-compose.yml run --rm training \
  python3 evaluation/evaluate.py \
  --config config/pipeline_config.local.yaml \
  --model-path Qwen/Qwen3.5-4B \
  --api-base http://vllm:8000/v1 \
  --api-model-name Qwen/Qwen3.5-4B \
  --concurrency 16 \
  --enable-thinking         # only for thinking models (Qwen3/3.5/3.6); harmless otherwise
```

The same endpoint also serves the merged **thinking** student for GRPO **reachability probing** and
**close-rate** checks (see [Reinforcement Learning (GRPO)](#reinforcement-learning-grpo) and the
close-rate probe below).

## Evaluation

`evaluation/evaluate.py` scores predictions on `data/final/test_clean.jsonl` (7 complexity classes,
every gold executable, leakage-free) and reports per-complexity and overall metrics:

- **EM** – Exact Match on normalized SQL strings.
- **EX (loose)** – Execution Accuracy with **row and column order ignored** (a differing column
  *count* still counts as wrong). This is the **canonical headline metric**.
- **EX (strict)** – Execution Accuracy with row order ignored but **column order preserved** (the
  previous behavior). Reported alongside loose; by construction `strict ⇒ loose`.

**Extractor-v2** (`extract_sql`) robustly recovers the final query from model output: it strips
`<think>…</think>` blocks (including a closing `</think>` with no opening tag, as emitted by
prompt-injected thinking models), unwraps Markdown ` ```sql ` fences, and recovers a query that
leaked into plain reasoning text (inline backticks / a line starting with a SQL keyword). This
fixed several earlier false `0%` scores. Self-test:

```bash
python3 evaluation/evaluate.py --selftest-extract-sql   # exits non-zero on any failing case
```

Key flags: `--api-base`/`--api-model-name` (endpoint vs. local HF mode), `--concurrency` (endpoint
batching; ignored in local mode), `--enable-thinking`, `--max-tokens` (default 256; use 2048–4096
for thinking models), `--n-samples`, `--seed`, `--test-file`, `--output`.

Results are written to `data/final/eval/<model>/` and, when MLflow is available, logged to a run
named `baseline_<model>` (metrics `eval_em`, `eval_ex`, `eval_ex_strict`, plus per-complexity EX).

### Offline rescore (no inference)

`evaluation/rescore.py` re-applies the **current** `extract_sql` + `execution_match` to the saved
`raw_output` in the prediction files — no model is loaded. It overwrites the result files in place
(keeping a one-time `.orig` backup), is idempotent, and re-logs the corrected metrics to the
matching MLflow runs. Use it after an extractor fix instead of paying for inference again. All
scoring logic is imported from `evaluate.py` (single source of truth).

```bash
docker compose -f docker/docker-compose.yml run --rm training \
  python3 evaluation/rescore.py --config config/pipeline_config.local.yaml
# --eval-dir <dir>   rescore one model · --no-mlflow   files only · --tracking-uri <uri>
```

### Serving-efficiency benchmark

`evaluation/efficiency_benchmark.py` measures serving performance (throughput, TTFT, ITL, TPOT,
latency) of the model on the vLLM endpoint using **GuideLLM**, and logs the efficiency metrics into
the **same** MLflow run as the accuracy eval (`baseline_<model>` convention), with `think` / `nothink`
modes kept separate. It uses a **fixed cross-model config** (concurrent profile, rate 16, 100
requests, 10% warmup/cooldown) so numbers are comparable across models — do not tune it per model.

GuideLLM lives in an **isolated `/opt/guidellm` venv** (`docker/Dockerfile.training`) and is called
via subprocess, so it cannot disturb the pinned training stack (peft, transformers, torch). The
version is pinned (`guidellm[recommended]==0.6.0`) because its `benchmarks.json` schema is
version-bound.

```bash
# vLLM must be serving the model first (see Serving with vLLM):
docker compose -f docker/docker-compose.yml run --rm training \
  python3 evaluation/efficiency_benchmark.py \
  --model-id Qwen/Qwen3.5-4B \
  --target http://vllm:8000
# add --enable-thinking for thinking models
```

### Thinking close-rate probe

`tools/close_rate_probe.py` measures whether a **thinking** student actually terminates: it reports
`</think>`-close rate, loop rate, and trigram-repetition over **greedy** (temperature-0) generations
against a served vLLM endpoint — the honest test that surfaced the trace-looping the clean-distill redo
fixed.

```bash
docker compose -f docker/docker-compose.yml run --rm training \
  python3 tools/close_rate_probe.py --api-base http://vllm:8000/v1 --n 100
```

The same loose-EX scoring (`evaluate.extract_sql` + result-set match) is **reused** — with an execution
timeout — as the **GRPO reward** (`evaluation/reward.py`); it is the same metric, not a second one. See
[Reinforcement Learning (GRPO)](#reinforcement-learning-grpo).

## Reinforcement Learning (GRPO)

After SFT, an **optional** GRPO / RLVR stage (verl · LoRA · async server-mode) lifts the structurally
weak SQL categories where SDG + SFT plateau — **window functions, set operations, subqueries**. It takes
the SFT-distilled 4B **thinking** student (merged as `qwen34b_student_thinking_merged_sharded`), samples
rollouts, and reinforces the ones that produce **result-correct** SQL. The reward is
`evaluation/reward.py` — the **same loose-EX** as the eval harness (it imports `evaluate.extract_sql` and
the identical canonicalisation), wrapped as a binary RL reward with an execution timeout; it is **not** a
second metric. Runs in the dedicated **`grpo`** service (`docker/Dockerfile.grpo`: verl 0.8.0 + nightly
vLLM, isolated from the SFT stack). **Pilot (2026-06-27): 50/50 steps, ~11 h, 0 wedges → +2 pt** held-out
weak-EX (set-ops +1, subqueries +1). GB10 colocates actor+rollout → caps at ~9 B (14 B needs multi-GPU).

> **Single-GB10 / sm_121 load-bearing flags:** `VLLM_USE_FLASHINFER_SAMPLER=0` (avoids the GB10 FlashInfer
> top-k/top-p sampler race — vLLM #43885; cudagraph stays **ON**, `enforce_eager=False`) and `load_format=auto`
> (the rollout vLLM must load the **real** merged base from disk — `dummy` = random weights = garbage rollouts).
> Dr.GRPO + ref-skip. Runs log to MLflow experiment `grpo_distill` (`val-core/sql_exec/reward/mean@1` = held-out weak-EX).

**1. Build the weak-category pool** (executable gate + leakage guard; carves the held-out
`weak_test_clean.jsonl`):

```bash
docker compose -f docker/docker-compose.yml run --rm training \
  python3 training_pipeline/build_weak_pool.py
```

**2. Serve the merged student, then filter for reachability.** The reachability probe queries a
**running vLLM endpoint**, so start vLLM serving the merged thinking student first (see
[Serving with vLLM](#serving-with-vllm); merge it with `serving/merge_adapter.py`). It runs k=8 rollouts
per prompt and keeps only the learnable ones (`0 < success < 1`, ~50%-biased):

```bash
# (a) serve the merged thinking student:
docker compose -f docker/docker-compose.yml --profile vllm up -d vllm

# (b) probe reachability against that endpoint:
docker compose -f docker/docker-compose.yml run --rm training \
  python3 training_pipeline/reachability_probe.py \
  --api-base http://vllm:8000/v1 \
  --candidates data/final/grpo/weak_candidates_probe.jsonl \
  --max-tokens 4096 --target-per-cat 300
# -> data/final/grpo/weak_prompts_reachable.jsonl : 831 reachable (window 300 / subqueries 300 / set-ops 231)
```

**3. Run GRPO** via the `grpo` profile. Dry-run first to inspect the emitted verl command (no GPU), then
launch the real run under the self-heal supervisor (detects the GB10 wedge → restart → verl auto-resume
from `save_freq`). The `grpo` service entrypoint is `python3`, so pass the runner path directly:

```bash
# inspect the emitted command (no GPU):
docker compose -f docker/docker-compose.yml --profile grpo run --rm grpo \
  training_pipeline/grpo_verl_runner.py --config config/pipeline_config.local.yaml --dry-run

# tiny end-to-end verify (optional): add --smoke to the same command

# real run (recommended under the wedge watchdog):
docker compose -f docker/docker-compose.yml --profile grpo run --rm grpo \
  training_pipeline/grpo_verl_runner.py --config config/pipeline_config.local.yaml
# ops/grpo_pilot_supervised.sh wraps this with GB10 wedge detection + auto-resume
```

Full recipe, load-bearing config, the root-cause chronicle (the rollout once ran on random weights), and
the data-reproduction numbers: **[experiments-verl_RL_lora-grpo.md](docs/experiments-verl_RL_lora-grpo.md)**.

## Quantization (FP8, optional)

The deploy students can be quantized to **FP8 (W8A8)** with `tools/quantize_fp8.py` — data-free
`FP8_DYNAMIC` via an **isolated `/opt/llmcompressor` venv** (kept off the pinned training/serving stacks),
with a per-arch ignore list + a leak-assert (never quantize `lm_head` / vision / linear-attn) before save:

```bash
/opt/llmcompressor/bin/python tools/quantize_fp8.py \
  --model <merged-checkpoint-in> --out <fp8-out> --arch text   # or: --arch mm
```

FP8 serving uses a **separate** sm_121 image (not the prod `vllm`), gated by a NaN/garbage sanity check
(FP8 vs bf16). On this hardware dense FP8 is **not always servable** — a valid negative result, and the
9B is only partial-FP8 — so treat it as a pilot and read the per-model outcomes in
**[experiments-compressor.md](docs/experiments-compressor.md)** before relying on it.

## Teacher Model Backends (Modular)

The SDG teacher backend is selected by `teacher.backend` in your `config/pipeline_config.local.yaml`
(gitignored; copied from the committed `pipeline_config.yaml` template), with per-backend settings under
the same `teacher:` block:

| Backend          | `teacher.backend` | Use case                                   |
| ---------------- | ----------------- | ------------------------------------------ |
| vLLM local       | `vllm_local`      | **Production SDG teacher** — local Qwen3.6-35B-A3B (thinking MoE) |
| Azure OpenAI     | `azure`           | Committed-config default (GPT-4.1); API teacher option |
| OpenAI API       | `openai`          | Standard OpenAI or generic OpenAI-compat.  |
| Anthropic Claude | `anthropic`       | Highest-quality cloud API                  |
| Ollama local     | `ollama_local`    | Easiest local setup                        |

The actual runs use **`vllm_local`** serving **Qwen3.6-35B-A3B** (a thinking MoE) — SDG distills clean
*reasoning traces*, which needs a thinking teacher; the API backends are interchangeable alternatives.

## Model Selection (baselines)

The student/teacher choice is grounded in an untrained clean-baseline control group (14 models,
`n=100`, seed 42, extractor-v2, loose EX as headline). Headline findings:

- **Teacher: Qwen3.6-35B-A3B** (a *thinking* MoE, ~3B active), served locally via `vllm_local`. SDG
  distills **clean reasoning traces**, which needs a thinking teacher — so this MoE is the production
  teacher (every trained checkpoint encodes it: `t-qwen3635ba3b-…`). The earlier untrained baseline,
  scored on **final SQL only**, found a 14B *non-thinking* teacher already hit the ~60% EX ceiling and
  was cheapest for bulk SQL labels (no teacher beat it significantly, range 57–63% loose); that holds
  for labels, but **trace** distillation needs the thinking MoE (more cost per example, accepted for
  the reasoning data).
- **Student: Qwen3.5-4B** (primary; the ~4B knee already reaches the ~60% teacher ceiling). Trained
  lineup: **Qwen3.5 0.8B / 2B / 4B / 9B**, plus **Qwen3-4B** and **Qwen3-14B** (text-only) — each in
  **thinking** and **nothink** variants. (Qwen2.5 models were the *untrained* baseline control group,
  not trained students.)
- Structural weak spots across all teachers: **window functions**, **set operations**, and
  **subqueries** — the targets of the optional [GRPO stage](#reinforcement-learning-grpo).
- **Redo (clean distill):** cleaned reasoning traces fix the student looping (greedy `</think>`-close
  92–99%, was 4–44%); **thinking becomes competitive** (wins 4B, ties 9B), non-thinking stays the default.

Every student is trained in **two variants** — **thinking** (`<think>…</think>` reasoning before the
SQL) and **nothink** (SQL-only); `format_for_training.py` selects the variant, and eval/serving pass
`--enable-thinking` for the thinking models.

Details: [experiments-baselines.md](docs/experiments-baselines.md) (full tables/methodology),
[experiments-limits.md](docs/experiments-limits.md) (weak spots + thinking caveats),
[experiments-short.md](docs/experiments-short.md) (summary), [experiments.md](docs/experiments.md) (log),
[experiments-hardware.md](docs/experiments-hardware.md) (GB10 training speedup),
[experiments-verl_RL_lora-grpo.md](docs/experiments-verl_RL_lora-grpo.md) (GRPO/RLVR pilot),
[experiments-compressor.md](docs/experiments-compressor.md) (FP8 quantization).

## MLflow

```bash
docker compose -f docker/docker-compose.yml --profile mlflow up -d mlflow
# UI at http://localhost:5000  (backend store: ./mlruns)
```

Accuracy (`eval_*`) and serving-efficiency (`eff_nothink_*` / `eff_think_*`) metrics for a model
land in the same `baseline_<model>` run, so quality and cost are compared side by side.

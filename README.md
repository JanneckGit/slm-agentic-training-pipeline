# Agentic SLM Training Pipeline

Training pipeline for a **small-language-model orchestrator agent** (multi-step planning, tool-calling,
self-reflection/replan) — use case: an internal **Deutsche Bahn employee assistant** driving DB tools
(Fahrplan, Zugstandort, Wartung, Personal). Built on a GB10 (DGX Spark, 128 GB unified, sm_121) with
vLLM serving and a τ²-bench-based tool sandbox. Successor of the finished Text-to-SQL pipeline
(preserved in the initial commit and in [docs/text2sql-experiments/](docs/text2sql-experiments/)).

**Training plan:** Stage 1 = SFT (LoRA) on a 3-leg mix — ToolACE (tool basics) + TaskBench (planning)
+ **self-synthesized, verifier-gated German DB trajectories** (this repo's core). Stage 2 = GRPO/verl RL
(reward = trajectory verifier / τ²-bench success).

## Architecture (Stage-1 grounded synthesis)

```
gtfs.de de_fv (real timetables, CC-BY-4.0) + sha256-seeded synthetic tables
        │  sdg_pipeline/db_bahn/seed_worldstate.py
        ▼
frozen BahnDB world-state (db.json)  ──►  τ²-bench domain "db_bahn"
        │                                 (8 German tools READ+WRITE, policy.md,
        │                                  runtime-registered — tau2 source untouched)
        ▼  sdg_pipeline/db_bahn/gen_tasks.py
550 German tasks with built-in answer keys (INFO + ACTION, 119 fault-injected → replan)
        ▼  sdg_pipeline/db_bahn/rollout.py            (teacher via vLLM, prompt-and-parse,
local teacher solves tasks against the REAL tools      6 tool-call formats, resume-safe)
        ▼  evaluation/trajectory_reward.py
deterministic verifier: DB-state hash + tool-grounding + anti-hallucination → keep score==1.0 only
        ▼  data_pipeline/format_traj_for_training.py
chat JSONL (assistant tool_calls + role:"tool" turns)
        ▼  training_pipeline/train_traj.py  (+ collator_multiturn.py: assistant-only loss mask)
LoRA student (Qwen3.5-4B)
```

**Results so far:** teacher bake-off over 8 local models → winner **Qwen3.6-35B-A3B** (92 % verified
yield, ~16 s/rollout — [docs/teacher-bakeoff.md](docs/teacher-bakeoff.md)); full generation run →
**446 verified German trajectories** (92 %, zero loops/dupes); traj_sft trains cleanly (loss 0.37→0.13),
held-out before/after was flat (72.5 %→70 %, n=40 — base already solves the easy set; see the honest
analysis in [docs/agentic-db-synthesis-log.md](docs/agentic-db-synthesis-log.md)).

## Setup

Prereqs: Docker + compose (services: `sdg`, `training`, `vllm`, `grpo`, `mlflow`), plus a **Python ≥3.12
venv for τ²-bench** (its `requires-python >=3.12`; the pinned training stack stays isolated per the
repo's venv doctrine):

```bash
python3.12 -m venv .venv-tau2
git clone https://github.com/sierra-research/tau2-bench.git /tmp/tau2-bench   # pin commit 1901a30
./.venv-tau2/bin/pip install /tmp/tau2-bench
cp config/pipeline_config.yaml config/pipeline_config.local.yaml   # then fill secrets (gitignored)
```

All `ops/` scripts use `.venv-tau2/` by default (override with `TAU2PY=/path/to/python`).

## Pipeline steps

CPU steps (host python3 unless noted `TAU2PY`):

```bash
# 0) public SFT legs: ToolACE + TaskBench -> data/raw/{toolace,taskbench}/
python3 data_pipeline/prepare_agentic_data.py --config config/pipeline_config.yaml --dataset all

# 1) frozen world-state from the GTFS snapshot (byte-reproducible, seed 42)
python3 sdg_pipeline/db_bahn/seed_worldstate.py --gtfs-dir data/raw/db_sandbox/gtfs_de_fv \
  --out data/raw/db_sandbox/db.json --seed 42

# 2) tasks + answer keys + disjoint splits (bakeoff_dev / heldout_eval / sft_train)   [TAU2PY]
PYTHONPATH=. ./.venv-tau2/bin/python sdg_pipeline/db_bahn/gen_tasks.py --seed 42

# 3) GPU-free end-to-end smoke: scripted oracle through the REAL loop + verifier     [TAU2PY]
PYTHONPATH=. ./.venv-tau2/bin/python sdg_pipeline/db_bahn/rollout.py \
  --dry-run --split bakeoff_dev --n-tasks 6 --stratify --output /tmp/oracle_smoke.jsonl

# selftests
PYTHONPATH=. ./.venv-tau2/bin/python evaluation/trajectory_reward.py                 # verifier 5 cases
docker compose -f docker/docker-compose.yml run --rm -T training \
  python3 training_pipeline/collator_multiturn.py                                    # loss-mask golden test
```

GPU steps (sequential — GB10 has no MIG):

```bash
bash ops/teacher_bakeoff.sh        # compare teacher candidates -> docs/teacher-bakeoff.md
# full generation = rollout.py --split sft_train against the served winner (see ops/ + log)
bash ops/traj_sft_pipeline.sh      # BEFORE-eval -> traj_sft (assistant-only mask) -> merge -> AFTER-eval
```

## Repo layout

```
sdg_pipeline/db_bahn/        # world-state seeder, tau2 domain (tools/policy), task-gen, rollout, bake-off
evaluation/trajectory_reward.py   # deterministic trajectory verifier (Stage-2 reward seam, verl dict contract)
data_pipeline/               # prepare_agentic_data (ToolACE/TaskBench), format_traj_for_training,
                             # clean_traces (trace-filter skeleton, to adapt)
training_pipeline/           # collator_multiturn (assistant-only mask), train_traj (LoRA SFT),
                             # grpo_verl_runner + build_weak_pool + reachability_probe (Stage-2 verl recipe)
serving/                     # merge_adapter (text), merge_adapter_mm (multimodal — needed for Qwen3.5 deploys)
tools/                       # close_rate_probe (termination probe template), quantize_fp8 (deploy quant)
ops/                         # teacher_bakeoff, traj_sft_pipeline, grpo_pilot_supervised (Stage-2 watchdog)
docker/                      # sdg / training / vllm / grpo / mlflow services (GB10 sm_121 stack)
docs/                        # design docs, decision log, bake-off results + text2sql-experiments/ archive
```

## Docs

- [docs/agentic-db-synthesis-log.md](docs/agentic-db-synthesis-log.md) — **decision & bug log** (read this first)
- [docs/agentic-sft-db-synthesis.md](docs/agentic-sft-db-synthesis.md) — design + literature levers (9 papers)
- [docs/agentic-sft-data-basis.md](docs/agentic-sft-data-basis.md) — Stage-1 data basis (3 legs)
- [docs/agentic-pivot-overview.md](docs/agentic-pivot-overview.md) — what carried over from the Text2SQL base
- [docs/teacher-bakeoff.md](docs/teacher-bakeoff.md) — teacher comparison + winner validation
- [docs/text2sql-experiments/](docs/text2sql-experiments/) — archived evidence base of the predecessor pipeline

## GB10 / sm_121 load-bearing flags (inherited, verified)

`VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer sampler race), `--gdn-prefill-backend triton` (DeltaNet MoEs),
`--max-num-seqs 4` (small-batch box), sharded merges (`max_shard_size=5GB`), no MIG → serve one model at a
time. Stage-2 verl: `load_format=auto`, `attn_implementation=sdpa`, ~9B colocated cap. Details in the
[archive](docs/text2sql-experiments/) and the [decision log](docs/agentic-db-synthesis-log.md).

# Agentic Pivot — Base Analysis & Carryover Map

> **Status:** design / transition note · **Date:** 2026-07-02 · **Scope:** read-only analysis, no code changed.
>
> **Purpose.** This document records what the finished **Text-to-SQL SLM pipeline** (this repo's base)
> actually is at the code level, and maps — subsystem by subsystem — what carries over to the target
> **agentic SLM orchestrator** (multi-step planning, tool-calling, self-reflection/replan) vs. what is
> SQL-specific and must be rebuilt.
>
> The finished-base evidence logs were moved to [`docs/text2sql-experiments/`](text2sql-experiments/)
> as part of this transition. The agentic research direction (use case, datasets, benchmark, training
> plan) lives in Notion: *Knowledge Hub → "Agentic LLM / Orchestrator (Forschungsrichtung)"*.

---

## 0. TL;DR

The base is a **complete, reproducible Text-to-SQL distillation pipeline** (SDG teacher → synthetic
traces → LoRA/SFT student → eval → optional GRPO/RLVR), hardened for a single **GB10 / DGX Spark
(sm_121)**. Its value for the agentic pivot is **not the SQL** but three transferable assets:

1. a **validated verl/GRPO vehicle** (async server-mode, Dr.GRPO, ref-skip);
2. **data/reward-hygiene patterns** (leakage guard, exec/validity gate, reachability filter, trace cleaning);
3. the full **GB10 infra hardening** (sampler-race fix, DeltaNet kernels, merge/serve/quantize, self-healing supervisors).

**Key realization:** verl `rollout.mode=async` is de-risked but currently means only verl's
OpenAI-compatible **in-process vLLM server instead of colocated SPMD** — the rollout is still
**one prompt → one completion → one reward**. The **multi-turn tool loop, the tool/environment
executor, and the trajectory-level reward do not exist yet.** That is the core of what must be built.

---

## 1. What the base IS (code reality)

Central control: **one** [`config/pipeline_config.yaml`](../config/pipeline_config.yaml) (every script
parses it itself — no shared loader), global `seed: 42` through all RNG stages, 5 Docker services
(`sdg · training · grpo · vllm · mlflow`), MLflow as the shared store.

| Subsystem | Core | Key files |
|---|---|---|
| **Config / Docker** | 1 YAML + 4 images; isolation doctrine (each incompatible stack in its own venv/image, `--no-deps`, subprocess-by-path) | [`docker-compose.yml`](../docker/docker-compose.yml), [`Dockerfile.grpo`](../docker/Dockerfile.grpo) |
| **Data pipeline** | 2 paths: (a) seed+synthetic SQL labels, (b) **"thinking" clean-distill traces** (= the production path). Leakage guard + SQLite exec-gate at every stage | [`mix_datasets.py`](../data_pipeline/mix_datasets.py), [`clean_traces.py`](../data_pipeline/clean_traces.py), [`build_train_clean.py`](../data_pipeline/build_train_clean.py) |
| **SDG** | Teacher-backend abstraction (anthropic/openai/azure/vllm_local/ollama). **Production path = [`trace_capture.py`](../sdg_pipeline/trace_capture.py)** (nudge+filter+regenerate, resumable), *not* the SDG-Hub flows | [`run_sdg.py`](../sdg_pipeline/run_sdg.py) |
| **SFT training** | LoRA-SFT via **TRL SFTTrainer + PEFT** (see §3), think/nothink variants, completion-only masking | [`train.py`](../training_pipeline/train.py) |
| **GRPO / RLVR** | verl `main_ppo`, Dr.GRPO, ref-skip, LoRA-actor + async-vLLM rollout. Weak-pool carve + reachability filter (k=8, keep 0<pass<8, ~50%-bias) | [`grpo_verl_runner.py`](../training_pipeline/grpo_verl_runner.py), [`build_weak_pool.py`](../training_pipeline/build_weak_pool.py), [`reachability_probe.py`](../training_pipeline/reachability_probe.py) |
| **Eval / reward** | `extract_sql` (extractor-v2) as **single source of truth**, same loose-EX in eval *and* GRPO reward. Exec against in-memory SQLite | [`evaluate.py`](../evaluation/evaluate.py), [`reward.py`](../evaluation/reward.py) |
| **Serving / ops** | Adapter merge (**sharded save — mandatory**), FP8 quant (leak-assert), vLLM deploy, **self-healing supervisors** (power-draw wedge watchdog + verl auto-resume) | [`merge_adapter.py`](../serving/merge_adapter.py), [`grpo_pilot_supervised.sh`](../ops/grpo_pilot_supervised.sh) |
| **Docs** | Evidence base: numbers + GB10 fixes | [`text2sql-experiments/`](text2sql-experiments/) |

---

## 2. Verified results (from code / logs)

- Student knee at **~4B**; the **~60% EX ceiling is data/metric-bound, not size-bound**.
- Looping fix raised greedy `</think>`-close rate **4–44% → 92–99%** (root cause: poisoned teacher
  traces, fixed at the data root — [`text2sql-experiments/experiments.md`](text2sql-experiments/experiments.md)).
- **3.9× training speedup** via DeltaNet kernels (2355→601 ms, loss Δ ≤ 0.004 —
  [`experiments-hardware.md`](text2sql-experiments/experiments-hardware.md)).
- FP8: all 4 deploy models KEEP; only 9B-thinking −0.04 EX
  ([`experiments-compressor.md`](text2sql-experiments/experiments-compressor.md)).
- **GRPO pilot: 50/50 steps ~11 h, 0 wedges; held-out val 0.42→0.44 (+2.1 pt)**, per-category
  59→61% (set-ops +1, subqueries +1, window flat —
  [`experiments-verl_RL_lora-grpo.md`](text2sql-experiments/experiments-verl_RL_lora-grpo.md)).

---

## 3. Code-vs-docs discrepancies (gotchas to know before building on the base)

1. **[`train.py`](../training_pipeline/train.py) is NOT a "Training Hub" wrapper.** Docstring/README/config
   say "Training Hub (Red Hat), options lora_sft|sft|osft". The code imports nothing from `training_hub` —
   it is a hand-rolled **TRL `SFTTrainer` + PEFT LoRA**, and **only `lora_sft` is implemented**; `sft`/`osft`
   are dead options. (OSFT, which Notion cites for anti-forgetting when a second specialty is added, would
   need a new runner.)
2. **Production data comes from [`trace_capture.py`](../sdg_pipeline/trace_capture.py) (Path B), not the
   SDG-Hub flows (Path A).** The `SQLComplexityFilterBlock` is **dead code** (not referenced in the flow,
   `blocks/__init__.py` is empty).
3. **verl `mode=async` ≠ multi-turn.** It is the in-process OpenAI-compatible server; still single-turn.
4. **[`grpo_verl_runner.py`](../training_pipeline/grpo_verl_runner.py) defaults `load_format=dummy`**
   (random weights = the phantom-loop bug); the fix works only because the config explicitly sets
   `load_format: auto`.
5. **Stale path:** [`ops/sdg_run_supervised.sh`](../ops/sdg_run_supervised.sh) still `cd`s into the old
   repo `…/SLM-Finetuning`, not this directory.

---

## 4. Target: agentic orchestrator (from Notion)

**Goal** (Marius' master-thesis topic; Janneck = practical implementation, no own paper): not *one* big
model but a **fleet of small specialist SLMs** coordinated by an **orchestrator** that decomposes,
selects/calls tools, checks results, and **replans on surprises**. Target is the **step-by-step**
variant (code name **"Full-Trace" / "Variante C"**). The orchestrator itself needs **no SQL** — Text2SQL
is just **one tool** (already built, proves the pattern).

**Use case:** internal **Deutsche Bahn employee assistant** (not a customer chatbot) — already in prod
with a big GPT model, wired to "DB 360" + internal sources. Example trace: *"Where is ICE 1234, and can
it make tomorrow's maintenance?"* → `standort_tool` → `wartung_tool` → `fahrplan_tool` → replan →
`text2sql(team availability)` → answer. Four tools, replanning, SQL only one of them.

### Datasets & benchmark (Notion final pick — all MIT/Apache-2.0)

No ready-made internal DB dataset exists; public sets train the **multi-tool / multi-turn mechanics**,
DB specifics are generated **synthetically against the real tools**.

**Training data (orchestrator) — mixed in ONE SFT pass, shuffled, DB flows weighted higher:**

| Dataset | Purpose | License |
|---|---|---|
| **ToolACE** (Team-ACE, 26,507 APIs) | tool-call basics: call correctly, read response, call next | Apache-2.0 |
| **TaskBench** (Microsoft) | **planning** — decomposition, ordering, tool-graph, parameters | MIT |
| **τ²-bench flows** (Sierra) | **replan + self-correction** (framework generates arbitrarily many) | MIT |
| **Own DB flows** | domain adaptation, German, DB-specific — **up-weighted** | internal + DB Open Data (CC-BY) |

**Benchmark:** **τ²-bench** (main — executable env: select/call tools, react, replan + reliability) ·
**BFCL-V3 Multi-Turn** (comparison, Berkeley).

**Fleet experts (= the tools):** API/tool expert (xLAM, ToolACE) · Text-to-SQL (*existing*;
Spider/BIRD/gretelai) · data analysis (InfiAgent-DABench) · doc/RAG.

**Training method:** SFT first (LoRA/QLoRA/OSFT), mix within phase (not sequential blocks), **then**
GRPO/verl RL with reward = **τ²-bench success**. Optional 2nd SFT stage (DB-only + 10–20% general, or
OSFT) against forgetting. Paper levers (not the next step): Reusable-Modules (RL decouples tool-call
from output routing), ZPPO (cold-start), long-context over reward engineering, PACT (plan-commitment),
OpenThoughts-Agent data recipe.

---

## 5. The bridge: carryover vs. gaps

Transfer is **high for infra/vehicle/hygiene, ~zero for domain semantics**.

### Carries over (~unchanged)

- **verl-GRPO recipe:** `adv_estimator=grpo`, Dr.GRPO (`norm_adv_by_std_in_grpo=False`), ref-skip
  (`use_kl_loss=False` + `use_kl_in_reward=False` → **no reference model** = the memory saving that
  multi-turn needs), `+override_config.attn_implementation=sdpa`, `load_format=auto`, LoRA-actor +
  async-vLLM. The runner builds everything as a flat Hydra-override list → **multi-turn = add flags to
  the same list.**
- **Reward contract:** verl `compute_score(...) → dict{score, …aux}` (aux keys auto-surface as
  `val-aux/*/mean@1`). Swap the body, keep the interface → free auxiliary metrics (turns-used,
  tool-call-valid, task-solved).
- **Reachability / variance gate** ([`reachability_probe.py`](../training_pipeline/reachability_probe.py)):
  score each prompt with k rollouts, keep only `0<success<k`, bias to ~50% — task-agnostic curriculum
  selection, directly applicable to selecting learnable multi-turn tasks.
- **Data-hygiene patterns:** leakage guard (**hard-fail** if guard inputs are missing), exec/validity
  gate as keep/drop, deterministic seeded disjoint carve with a fresh RNG per newly-added category,
  disjoint held-out.
- **Trace-cleaning skeleton** ([`clean_traces.py`](../data_pipeline/clean_traces.py)): trigram-rep,
  unique-ratio, hedge-count, ritual-tail stripping — **thresholds must be re-tuned** (multi-step plans
  legitimately repeat tool names and are longer).
- **GB10 infra (verbatim):** `VLLM_USE_FLASHINFER_SAMPLER=0` (the wedge fix), `sdpa` instead of
  flash-attn-2, **sharded save (5 GB, mandatory** against the vLLM-loader stall), DeltaNet kernels,
  FP8 leak-assert, `RAY_memory_monitor_refresh_ms=0`.
- **Self-healing supervisor** ([`grpo_pilot_supervised.sh`](../ops/grpo_pilot_supervised.sh)): power-draw
  wedge watchdog + verl auto-resume — **more important** for long, wedge-prone agent rollouts.
- Teacher-backend abstraction, MLflow conventions, TRL+PEFT SFT loop, baseline-sweep harness.

### Must be rebuilt (SQL / single-turn bound)

- **The reward — the single biggest replacement:** `score_sql` (SQLite exec, result-set match) →
  **τ²-bench task verifier** (goal-state / tool-call correctness). `extract_sql` → tool-call/JSON parser.
- **Multi-turn rollout + environment:** wire verl `multi_turn.*` + `tool_config` + `interaction`; a
  **stateful tool executor / τ²-bench user simulator** the async server calls per turn. Does not exist yet.
- **Data records:** `(question, schema, sql)` → **multi-turn trajectories** `(task, tool registry,
  [tool_calls + observations], final answer)`. The chat `messages` format is already the right container,
  but it needs `role:"tool"` turns + **multi-turn loss masking** (the single hardcoded `response_template`
  completion-only collator is not enough).
- **Taxonomy:** `complexity_classes` (SQL-syntactic) → task/tool complexity (#tools, plan depth,
  replan-needed, sequential/parallel).
- **SDG flows:** SQL augmentation → **trajectory synthesis** against real tools. **Caveat:** the
  brevity/anti-hedge filters in [`trace_capture.py`](../sdg_pipeline/trace_capture.py) would delete
  exactly the **replan/reflection** behavior you want to distill — relax or invert them.
- **Serving/eval:** `--max-model-len 2048` (far too short for tool transcripts); tool-parser /
  `--enable-auto-tool-choice` / guided-json missing; [`query_model.py`](../serving/query_model.py) is
  single-shot SQL.

### Framing & hardware caveats

- The base **null result** ("thinking barely helps", ~60% ceiling) is a saturated single-turn SQL
  artifact and **does not transfer** — multi-step tool-calling with planning/reflection is exactly the
  regime where reasoning should pay off. Do not let it discourage the thinking/reflection path.
- **Memory:** verl colocated actor+rollout caps at **~9B** on one 128 GB GB10; **14B does not fit**. A
  larger orchestrator needs actor-offload or multi-GPU (unproven here). The **~4B class stays on-thesis**.

---

## 6. Suggested next steps (not started)

1. **Concrete rebuild plan:** which files/flags in order — verl `multi_turn` + `tool_config`, reward swap
   (τ²-bench verifier behind the `compute_score` contract), trajectory data schema, multi-turn loss masking.
2. **Reward spike:** dock a τ²-bench task-success verifier onto the existing `compute_score` dict contract.
3. **Dataset study:** validate ToolACE / TaskBench / τ²-bench real formats against the training + reward needs.

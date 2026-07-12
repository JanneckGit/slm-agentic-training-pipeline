# Teacher bake-off — DB trace generation

> Head-to-head comparison of thinking teachers for synthesizing German DB tool-calling traces (Plan B, Phase 3).
> **Executed 2026-07-03** with the SHORTENED protocol (user request): **12 stratified bakeoff_dev tasks × k=1**,
> max 8 turns, hard 25-min cap/candidate, **identical final harness for all** (fair pass 2 after the format-zoo
> fixes — see `agentic-db-synthesis-log.md`). Rank metric: verified-yield × German per GPU-hour ("score").
>
> **Note (dated result).** This describes the **Welle-1 domain** (10 templates, 8 tools); the domain has since
> grown to **26 templates / 12 tools**, but the selected teacher **Qwen3.6-35B-A3B** remains the production
> trace generator. Numbers below are historical by design.

## Comparison set

- 12 of the 25 `bakeoff_dev` tasks, stratified round-robin over all 10 templates (INFO + ACTION + fault-injected),
  disjoint from training; same 12 tasks for every candidate; temperature 0.7, k=1.
- Caveat: n=12 ⇒ one task ≈ 8 pp — treat yields as bands. The winner is confirmed on 100 tasks below.
- Tool-calling = pure prompt-and-parse (schemas embedded in the system prompt, no `tools` param); the parser
  handles 6 emitted call formats + 3 thinking dialects; stop-sequences cut hallucinated in-turn observations.
  **All 8 candidates served on vllm-openai:v0.21/sm_121 without kernel failures** (incl. Qwen3-Next GDN, NVFP4).

## Serving constraints (all candidates)

`--max-num-seqs 4`, `--gpu-memory-utilization 0.85`, `VLLM_USE_FLASHINFER_SAMPLER=0`, explicit `--chat-template`,
non-streaming; NVFP4/FP8 for the big ones; **no** fp8 kv-cache; DeltaNet-MoEs need `--gdn-prefill-backend triton`.
Verify sm_121 nightly has the model's kernels first.

## Results

| Teacher | n | verified-yield | German | replan (inj) | avg turns | tc-valid | s/rollout | traces/GPU-h | score |
|---|---|---|---|---|---|---|---|---|---|
| oracle | 25 | **100%** | 28% | 86% | 4.3 | 100% | 1s | 4658 | 1304 |
| nemotron-49b-nvfp4 | 12 | **92%** | 92% | 67% | 3.9 | 100% | 189s | 17 | 15 |
| q3-30b-a3b-think | 12 | **92%** | 92% | 67% | 4.0 | 100% | 74s | 45 | 37 |
| q36-27b | 12 | **92%** | 100% | 67% | 4.0 | 100% | 59s | 56 | 52 |
| q36-35b-a3b | 12 | **92%** | 100% | 67% | 4.0 | 100% | 16s | 205 | 188 |
| seed-oss-36b | 12 | **83%** | 75% | 67% | 3.7 | 100% | 268s | 11 | 7 |
| q3-next-80b-fp8 | 12 | **75%** | 92% | 67% | 3.2 | 83% | 81s | 33 | 23 |
| glm45-air-fp8 | 12 | **58%** | 92% | 0% | 3.0 | 100% | 28s | 76 | 41 |
| magistral-24b | 12 | **33%** | 100% | 0% | 3.0 | 100% | 36s | 34 | 11 |
| oracle_broken | 5 | **0%** | 100% | 0% | 1.0 | 0% | 0s | 0 | 0 |

## Winner + validation

- **Winner: `Qwen/Qwen3.6-35B-A3B`** — top yield band (92%), perfect German proxy (100%), all 3 fault-injected
  replan tasks solved, and **~12× faster** than the next 92%-yield candidate (16 s/rollout → 205 verified
  traces/GPU-h; score 188 vs 52). It is also the already-deployed base SDG teacher → zero new serving risk.
  The user's reuse instinct beat the research ranking (which had suggested Qwen3-30B first).
- Runner-ups (backup / ensemble diversity for scale-up): `q36-27b` (92%, dense control) and
  `q3-30b-a3b-think` (92%). `seed-oss-36b` is capable (83–92% across passes, strong replan) but ~17× slower.
- **Notes:** the recurring residual failure across nearly ALL candidates is `info_machbar` — the verifier's
  strict grounding/communicate flags **derived arithmetic** (computed actual-arrival times). That is a
  task/verifier calibration item, not a model gap (tracked in the decision log for Phase-4 calibration).
  `oracle` rows are harness ceilings, not models; `oracle_broken` proves the 0-score path. Pass-1 (pre-fix)
  traces are archived in `data/generated/bakeoff_pass1/` — pass-1 numbers are NOT comparable.
- **Validation run (done):** winner on **100 stratified `sft_train` tasks** (k=1, max-regen 1):
  **80% verified-yield**, avg 18 s/rollout, injected 13/21 ok. **Per-template: 8/10 templates at 100 %** —
  including ALL four ACTION templates and the fault-injected replan template (`action_ersatz` 10/10).
  The entire 20 % loss sits in the two **derived-arithmetic** templates (`info_ankunft` 1/10,
  `info_machbar` 1/10; fail modes grounding/communicate) → **verifier/template calibration item, not a
  teacher gap**. Fix before the full generation run: state planned-vs-actual arrival explicitly in those
  tickets + use robust communicate strings; deterministic task-gen means other templates' task-ids stay
  byte-identical. Note for the mix step: verified traces average 3.6 turns (1-tool tasks = 3); only 22/80
  have ≥5 turns — the OpenThoughts ≥5-turn keep-filter must be applied per-template, not globally, or it
  would drop most short INFO tasks.
- **The 80 verified traces are the first production batch** (`data/generated/db_traces_sft_train_q36-35b-a3b.jsonl`).
  Projection: 485 sft_train tasks × ~80–90 % (after the 2-template fix) ≈ **>400 verified traces in ~1–2 GPU-h**
  → the pilot's yield budget is comfortably reachable.

# RLVR / GRPO (LoRA) on GB10 — verl, the validated vehicle

**Goal of this pilot:** test whether GRPO (fresh LoRA) lifts the structurally weak SQL categories
(window functions, set operations, subqueries) where SDG-SFT plateaus. Vehicle = **verl** (LoRA + GRPO,
async server-mode) — chosen for agent-fit (verl's AgentLoop / async server is built for the later
multi-turn orchestrator) and consistency with the SFT/LoRA workflow.

**Status (2026-06-26): pilot UNBLOCKED.** The root cause of the prior "loop/too-slow" findings was
a single bug — verl's rollout vLLM ran on **random weights** (`load_format=dummy` in HYBRID mode) — fixed
with `load_format=auto`. Three-level validation (coherence / reward-variance / LoRA-sync) is green. See
**ROOT CAUSE FOUND** + **RESOLVED** below. (The separate GB10 **wedge** is NOT this bug — it's the FlashInfer
sampler race #43885, fixed by `VLLM_USE_FLASHINFER_SAMPLER=0`; see the recipe.) The configs/scripts here are
the working set.

## What runs on GB10 / SM_121
On the dense **Qwen3-4B** (thinking student), verl's full async GRPO loop runs end-to-end on sm_121:
- **Generates** real completions on sm_121 (EX-reward executes; no cubin/dispatch error). Wedge-free under
  `VLLM_USE_FLASHINFER_SAMPLER=0` (cudagraph stays ON; see recipe — the wedge is the FlashInfer sampler race
  #43885, not cudagraph).
- **LoRA-async** runs; **no OOM** (~15–17 GB on the 4B beside the FSDP actor).
- **Memory ceiling:** verl colocates actor + rollout, so the model must fit ~2× on one 128 GB GB10 →
  **works up to ~9 B; 14 B does NOT** (at rollout-worker spawn the actor leaves < the 14 B's own need
  free). For 14 B you'd need actor-offload-before-rollout (verl hybrid-engine, unproven here) or multi-GPU.

> ⚠️ **Correction (the original "validated 2026-06-24" claim was incomplete).** That run reported
> "weight-sync NON-STALE" because `actor/entropy` shifted on the served rollouts (8.41→8.86→9.79). That was
> a **proxy metric** — entropy can shift while the rollout base stays random, and the run **never checked
> the generated TEXT**. The rollout was in fact running on dummy/random weights the whole time. **Durable
> lesson: a proxy metric (entropy, timing_s/update_weights) is NOT proof the rollout has the right weights
> — check the actual decoded output.** Full story in the chronicle.

## Load-bearing run config (the reproduction fundament — current working set)
Emitted by [training_pipeline/grpo_verl_runner.py](../training_pipeline/grpo_verl_runner.py) for
`python -m verl.trainer.main_ppo`:
```
algorithm.adv_estimator=grpo
algorithm.norm_adv_by_std_in_grpo=False           # Dr.GRPO (advantage = reward − group_mean, no /std)
algorithm.use_kl_in_reward=False                  # ref-skip ┐ both false ⇒ drop the reference worker
actor_rollout_ref.actor.use_kl_loss=False         #          ┘
actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16        # fp32 default OOMs
actor_rollout_ref.rollout.tensor_model_parallel_size=1          # single GPU (verl defaults to 2)
actor_rollout_ref.rollout.mode=async                            # the agent-relevant server mode
actor_rollout_ref.rollout.enforce_eager=False                   # cudagraph ON (FULL_AND_PIECEWISE); GB10 wedge fix is env VLLM_USE_FLASHINFER_SAMPLER=0, NOT eager (see below)
actor_rollout_ref.rollout.load_format=auto                      # ROOT-CAUSE FIX: real base from disk
# (env, set by the runner + docker-compose grpo) VLLM_USE_FLASHINFER_SAMPLER=0   # THE GB10 wedge fix (vLLM #43885)
+actor_rollout_ref.model.override_config.attn_implementation=sdpa  # CRITICAL (see below)
actor_rollout_ref.rollout.top_p=0.95  actor_rollout_ref.rollout.top_k=20
+actor_rollout_ref.rollout.repetition_penalty=1.1               # needs verl SOURCE PATCH + leading `+`
actor_rollout_ref.rollout.max_num_seqs=32  actor_rollout_ref.rollout.max_num_batched_tokens=4096   # mns=32 (sampler-race fixed by the env var; mns is only a frequency lever). PILOT itself ran mns=16/eager — see PILOT RUN
data.max_response_length=2048                                   # max_model_len = 512+2048 = 2560
actor_rollout_ref.model.lora_rank=16  actor_rollout_ref.model.lora_alpha=32
reward.custom_reward_function.path=evaluation/reward.py  reward.custom_reward_function.name=compute_score
```
Load-bearing on this stack:
- **`VLLM_USE_FLASHINFER_SAMPLER=0` is THE GB10 wedge fix** (env var; set in `docker/docker-compose.yml`
  grpo env + the runner). The GB10 rollout wedge is the **FlashInfer top-k/top-p sampler kernel race**
  (`RadixTopKMaskLogitsMultiCTA`, multi-CTA barrier-arrival-counter race, sm_120/121 hit first; vLLM
  **#43885** / flashinfer#3615; the default-off PR #44405 is still open, so our pin had it ON). **NOT cudagraph
  — `enforce_eager` was a RED HERRING** (the rollout wedged at mns=32 *despite* eager — the MD's old
  "wedged despite enforce_eager" was the giveaway). cudagraph (`enforce_eager=False`, FULL_AND_PIECEWISE)
  stays **ON** for speed. The old `compilation_config={cudagraph_mode:PIECEWISE}` override was a dead end —
  PIECEWISE never escaped because the bug isn't cudagraph (it can't touch the sampler). **Verified 2026-06-30:**
  standalone A/B (sampler off + cudagraph on = 15 min / 1280 req clean; sampler on = wedge in ~2 min, py-spy
  hangs in `get_output`→stream-sync identical to #43885; CUDA_LAUNCH_BLOCKING suppresses it = timing race)
  **+** a real verl smoke (cudagraph on + sampler off + mns=32 = 3 steps wedge-free, ~10 min/step; env
  reached the colocated Ray worker, confirmed via `/proc/<worker>/environ` + the native `_topk_topp_kernel`).
- **`load_format=auto`** — HYBRID mode keeps `dummy` by default (`vllm_async_server.py:138` only auto-
  converts dummy→auto for *non*-HYBRID), so it must be forced; otherwise the rollout runs on random weights.
- **`attn_implementation=sdpa`** — the default `flash_attention_2` fails the actor forward with
  `RuntimeError: Cannot access data pointer of Tensor that doesn't have storage` (transformers-5.x /
  torch-2.11 / flash-attn). `sdpa` fixes it; rollout/vLLM is unaffected.
- **`repetition_penalty=1.1`** reaches the rollout only via the verl **source patch** (stock verl 0.8.0
  hard-codes 1.0); the runner override needs a leading `+` (the rollout.yaml schema lacks the key). It is a
  belt-and-suspenders loop-trim, not load-bearing for termination once weights are real.
- **`max_num_seqs=32`** is the default; mns is only a wedge-**frequency** lever (less concurrency = fewer/
  narrower FlashInfer-sampler CTA groups = rarer race), **not a fix** — with `VLLM_USE_FLASHINFER_SAMPLER=0`
  it's moot, so 32 (faster). `trainer.save_freq=5` + the val-resume-skip patch (`ray_trainer.py:1393` →
  baseline val only on the first start) are the wedge/crash **recovery backstop** (verl auto-resume).
  **The PILOT itself RAN at mns=16/eager — that result stands (gelaufen ist gelaufen); the corrected combo
  above is for NEW runs.** See **PILOT RUN**.

## Stack + reproduction
Validated stack is frozen into [docker/Dockerfile.grpo](../docker/Dockerfile.grpo) (image
`text2sql-grpo:verl`; backup tag `:verl-prepatch` = pre-rep-pen). Pins:
`verl 0.8.0 · vllm 0.23.1rc1.dev377+g70749fdcc (cu130 aarch64 NIGHTLY) · torch 2.11.0+cu130 ·
transformers 5.12.1 · ray 2.55.1 · torchao 0.17.0 · tensordict 0.10.0 · peft 0.19.1 · flash-attn 2.7.4`.
Build: NGC `pytorch:25.11-py3` → uv-install vLLM cu130 nightly (`--torch-backend=auto` pulls the matched
torch) → `pip install training_hub[grpo]` (verl) → `torchao>=0.16` → **two local source patches**:
rep-pen (`agent_loop.py:500` + `main_ppo_sync.py:313`: `repetition_penalty=1.0` → `=config.repetition_penalty`)
and val-resume-skip (`ray_trainer.py:1393`: `val_before_train` gated on `global_steps==0` → no re-val on
resume). Both anchored in `Dockerfile.grpo` (backup tags `:verl-prepatch`/`:verl-preval`).
Isolated from the SFT image (it replaces the pinned SFT torch/peft/transformers).

## Reusable infra (backend-independent, KEEP)
- **GB10 vLLM wedge fix = `VLLM_USE_FLASHINFER_SAMPLER=0`** (env var; cudagraph stays ON). Under sustained
  rollout load the engine hangs (throughput→0, ~13 W / 96 % util, stuck queue). Root cause = the **FlashInfer
  top-k/top-p sampler kernel race** (`RadixTopKMaskLogitsMultiCTA`, sm_120/121; vLLM **#43885** / flashinfer#3615),
  **not cudagraph** — `enforce_eager` was a red herring (wedged at mns=32 *despite* eager). Sampler off (PyTorch-
  native) held 35–57 W real compute with no hang (15 min / 1280 req standalone; 3 verl steps @ mns=32 wedge-free).
- **DURABLE LESSON — the GB10 "13 W / 96 %-util wedge" signature is OVERLOADED:** ≥4 distinct bugs look
  identical (throughput→0, ~13 W, util pinned): (1) FULL-cudagraph decode (vLLM #37729/#40969, fix: cudagraph
  off), (2) Gated-DeltaNet linear-attn kernel (the 35B SDG teacher — `ops/sdg_run_supervised.sh`; fix:
  `--gdn-prefill-backend triton`; enforce_eager useless), (3) **FlashInfer sampler race (#43885 — OUR case;
  fix: `VLLM_USE_FLASHINFER_SAMPLER=0`)**, (4) multi-rank NCCL/Ray race (#40969, fix: NCCL 2.30.4). **Always
  discriminate first** (py-spy the wedged EngineCore + a sampler-off/cudagraph A/B) **before attributing.**
- **Single-file safetensors weight-load stall** on the 0.23-nightly vLLM loader → **sharded resave**
  (`save_pretrained(max_shard_size="5GB")`) fixes it. Done by
  [serving/merge_adapter.py](../serving/merge_adapter.py) (carries `chat_template.jinja`).
- **EX-reward** [evaluation/reward.py](../evaluation/reward.py) (loose-EX, in-memory SQLite, timeout +
  row-cap) — reuses `evaluate.extract_sql`; verl-shaped `compute_score(...)` returning a dict
  `{"score","think_tokens","passed","truncated"}` + a TRL adapter.
- **Pool builders:** [training_pipeline/build_weak_pool.py](../training_pipeline/build_weak_pool.py) (executable+leakage gate,
  carves held-out `weak_test_clean.jsonl`) and [training_pipeline/reachability_probe.py](../training_pipeline/reachability_probe.py)
  (k=8, 50%-biased, variance gate; resumable via `<out>.partial.jsonl`).
- **Self-heal supervisor** [ops/grpo_pilot_supervised.sh](../ops/grpo_pilot_supervised.sh): detects the
  wedge (power<25 W sustained + log frozen) → kills the grpo container → relaunches (verl auto-resumes from
  `save_freq` checkpoint). Backstop for the multi-hour pilot.

## MLflow wiring (verl ≠ the SFT logger)
SFT/eval runs live at **`file:///app/mlruns`**. verl logs to its OWN tracker (defaults to
`sqlite:////tmp/mlruns.db`!), so the runner sets `trainer.logger=[console,mlflow]` + env
`MLFLOW_TRACKING_URI=file:///app/mlruns`, and verl maps `trainer.project_name`→experiment (`grpo_distill`,
beside `sft_distill`) and `trainer.experiment_name`→run.

## GRPO metric mapping (verified against verl 0.8.0 source)
- **Dr.GRPO:** `norm_adv_by_std_in_grpo=False` ⇒ advantage = reward − group_mean, **without** /std
  (verified in `core_algos.compute_grpo_outcome_advantage`). Avoids length/difficulty bias; safe because the
  train set is reachability-pre-filtered (0<p<1, ~50%-biased). Zero-variance groups → advantage 0 (no
  gradient). `use_kl_loss=False` ⇒ KL beta 0.
- Held-out weak EX (central metric) → **`val-core/sql_exec/reward/mean@1`** (greedy val on
  `weak_test_clean.jsonl`); the reward dict's extra keys → **`val-aux/sql_exec/{think_tokens,passed,
  truncated}/mean@1`**.
- Per-TRAIN-step gates from native verl metrics: **`response_length/clip_ratio`** = truncation rate;
  **`critic/advantages/{max,min}`≠0 ⟺ intra-group reward variance>0** (R4 gate, since Dr.GRPO advantage is
  non-zero only for variance>0 groups); `critic/rewards/mean`; `response_length/mean`;
  `timing_s/update_weights` (weight-sync ran — but NOT proof of correct weights; see the chronicle lesson).

## Reproduce the pilot data (gitignored — regenerate from committed scripts)
The pilot inputs live under `data/final/` (gitignored, root-owned → build inside the training/vllm
containers). All are reproducible from the committed scripts; a regenerator should get the **same expected
numbers** below (the loop-fix + seed-42 splits are deterministic).

1. **Merged base** `data/final/checkpoints/qwen34b_student_thinking_merged_sharded` (8.04 GB, 2 shards,
   chat_template carried) — `serving/merge_adapter.py --adapter-path <4B-thinking adapter>
   --output-path data/final/checkpoints/qwen34b_student_thinking_merged_sharded` (sharded save = vLLM-loader
   fix). FSDP actor + (via `load_format=auto`) the rollout both load this.
2. **Pools + held-out** — `training_pipeline/build_weak_pool.py` (exec-gate + leakage-guard vs SFT train/eval +
   test_clean = **0 overlap**; seed-42 so window/set-ops/subqueries splits stay byte-identical, MD5-verified):
   - `weak_test_clean.jsonl` → **240 held-out** (80×3, all 3 categories) = pilot `val_files`.
   - `weak_candidates.jsonl` → 1800 canonical (600/cat); `weak_candidates_probe.jsonl` → **3606** expanded
     (sized so window/subqueries can reach ~300 reachable; set-ops supply-capped at 606).
3. **Reachable train set** = pilot `train_files` — `training_pipeline/reachability_probe.py --api-base
   http://vllm:8000/v1 --candidates data/final/grpo/weak_candidates_probe.jsonl --max-tokens 4096
   --target-per-cat 300` (k=8, temp 1.0, **rep-pen 1.1 / top-p 0.95 / top-k 20**, 50%-bias). Run under the
   self-heal supervisor on GB10 (the served vLLM wedges ~once/50 min — this is the same FlashInfer top-k/top-p
   sampler race #43885; setting `VLLM_USE_FLASHINFER_SAMPLER=0` on the `vllm` service eliminates it). **Expected output:**

   | category | probe candidates | reachable (0<pass<8) | selected | termination |
   |---|---|---|---|---|
   | window functions | 1800 | 514 | **300** | 99.9% |
   | subqueries | 1200 | 513 | **300** | 99.9% |
   | set operations | 606 (all) | 231 | **231** (supply-capped) | 100% |
   | **total** | 3606 | 1258 | **831** | **99.9%** (n_err=0) |

   → `data/final/grpo/weak_prompts_reachable.jsonl` = **831 prompts**, **mean intra-group variance 0.200**,
   `variance_gate_pass=True`. n_pass histogram `{0:1495,1:204,2:164,3:129,4:117,5:136,6:190,7:318,8:853}`
   (healthy partial-credit spread). If a regen yields these counts (831 = 300/300/231) + variance ≈0.200 +
   termination ≈99.9%, the set matches.

## Debugging chronicle — the phantom loop (2026-06-25 → 26)
Compressed timeline of the saga that turned out to be one bug. Detail of the dead-ends is in the toggle at
the end; the **two surviving sections** (ROOT CAUSE, RESOLVED) are below this.

- **Symptom (Stage-G smoke, 2026-06-25):** every rollout (greedy val AND temp-1.0 train) hit the cap —
  `response_length` min=max=mean=2048, `clip_ratio=1.0`, no `</think>`, `critic/{score,rewards,advantages}=0`,
  `grad_norm=0`. Read as "the model loops / never terminates." Plumbing otherwise proven: data→async vLLM
  rollout→EX-reward→metrics→console+mlflow, `timing_s/update_weights` ran, no OOM.
- **Chased fix 1 — sampling/termination.** Built the loop-fix (top_p 0.95 / top_k 20 / rep-pen 1.1). On the
  **standalone** vLLM these gave 100% close `</think>` / 0% cap (close_rate probe). Reachability + pool work
  completed here (the 831-set above). But standalone ≠ verl's colocated rollout.
- **Chased fix 2 — the wedge.** verl's in-process vLLM deadlocked within minutes (96%/13 W). At the time we
  attributed it to the V1 cudagraph path and "fixed" it with `enforce_eager` — **later OVERTURNED (2026-06-30):
  the wedge is the FlashInfer top-k/top-p sampler race (#43885), NOT cudagraph; `enforce_eager` was a red
  herring** (see the recipe + Reusable-infra lesson above). The eager "~20 h/epoch too slow" number was also
  **loop-contaminated** (rollouts were marching to cap on random weights, not generating useful traces).
- **Chased fix 3 — rep-pen ignored.** Found verl 0.8.0 hard-codes `repetition_penalty=1.0`; patched the
  source (now in the image). Patch confirmed live — **rollouts STILL looped 100% to cap.** Same model+params
  gave 100% close on the standalone → the loop was "verl-rollout-specific." Wrong: it wasn't a loop at all.
- **The actual cause (see ROOT CAUSE):** the rollout ran on **random weights** the entire time
  (`load_format=dummy`); the "garbage" never closed `</think>` because it was uninitialized-weight token
  soup, not a model loop. **Durable GB10 lessons that DID survive:** the wedge is the FlashInfer sampler race
  (#43885) → `VLLM_USE_FLASHINFER_SAMPLER=0`, cudagraph stays ON (the "wedge = V1 cudagraph → `enforce_eager`"
  reading here was **OVERTURNED 2026-06-30**); external/decoupled vLLM is impossible in verl on one GB10
  (STANDALONE/PD modes are multi-GPU); SGLang on GB10 is dev-branch only; real throughput is ~456 tok/s (not
  the contaminated ~80); **proxy metrics ≠ checking decoded output.**

## ROOT CAUSE FOUND (2026-06-26): verl's rollout runs on RANDOM weights — it was NEVER a loop
Two cheap discriminators in one gated session. This **overturns the "model loops" narrative**: every verl
rollout-to-cap (Stage-G, pilot-prep, Path-A, detox, rep-pen) was **garbage from uninitialized weights**.

- **[1] PROMPT — CORRECT.** Dumped verl's actual rollout prompt via `trainer.rollout_data_dir` (writes
  `<dir>/{step}.jsonl`, `input`=prompt / `output`=completion). verl renders via
  `apply_chat_template(..., **data.apply_chat_template_kwargs)` (`agent_loop.py:297-327`); no kwargs passed →
  `enable_thinking` falls to the **template default, which IS `True`** (verified: default render ==
  enable_thinking=True render; ends bare `<|im_start|>assistant\n`, no forced think-block). verl's prompt ==
  the standalone's. Ruled out.
- **[2] vLLM 0.23 — FINE.** Offline gen with vLLM 0.23 (the image's engine) + the **REAL merged weights** +
  the exact rollout sampling (temp1.0/top_p0.95/top_k20/rep1.1, enforce_eager, cap 2048): **4/4 stop, 4/4
  close `</think>`, 596–937 tok**, coherent SQL. Not a version regression. Ruled out.
- **[3] THE CAUSE.** The dumped verl rollout `output` was **garbage token soup** (random multilingual/code/
  emoji, **0/32 closed `</think>`, all score 0**) — the signature of **uninitialized weights**. The runner
  set `load_format=dummy`, so vLLM started the rollout with random weights expecting the ParameterSynchronizer
  to load the real model — and in HYBRID mode that base-sync wasn't delivering → noise → never emits eos →
  runs to cap → clip 1.0, reward 0, zero gradient.

| discriminator | result | verdict |
|---|---|---|
| verl rollout PROMPT (dumped) | == standalone | prompt OK |
| vLLM 0.23 + REAL weights (offline) | 4/4 stop, coherent SQL | engine OK |
| verl rollout OUTPUT (dumped) | random token soup, 0/32 `</think>` | **random weights** |

Earlier work chased a phantom: the loop-fix, the wedge analysis, the eager throughput — all real findings,
but the actual rollouts were always garbage. **Fix to test:** `load_format=dummy → auto`.

## RESOLVED (2026-06-26): `load_format=auto` — three-level validation ALL GREEN
Forced `load_format=auto` (runner + both configs). Re-ran the tiny rollout-dump smoke: 12 MIXED reachable
prompts (`_verify_mix12.jsonl`, n_pass 3–6 across the 3 categories), 3 steps, n=8, temp 1.0, enforce_eager,
cap 2048. Completed clean (exit 0, ~9m40s; the trailing `DataLoader worker … Killed` is a harmless atexit
teardown). **No wedge** (eager 94 %/34 W). The rollout vLLM logged `Loading safetensors 2/2` (real merged
base **from disk** — `dummy` skipped this) + `--enable_lora` + LoRA expand/shrink kernels firing.

| level | gate | step 1 | step 2 | step 3 | verdict |
|---|---|---|---|---|---|
| **[3a] COHERENCE** (base is real) | close `</think>`, real SQL, len ≪ cap | 96/96 close, 96/96 SQL, len̄ 423 tok, **clip 0** | 94/96, 96/96, 426, clip 0 | 95/96, 96/96, 417, clip 0 | **GREEN** |
| **[3b] REAL GRADIENT** (reward signal) | reward variance ≠0, clip ≪1 | reward 0.50 (PASS 48/FAIL 48), adv ±0.875, pg_clipfrac 3e-4, grad 0.039 | 0.479 (46/50), ±0.875, 4e-4, 0.024 | 0.427 (41/55), ±0.875, 4e-4, 0.023 | **GREEN** |
| **[3c] LoRA-SYNC DELIVERS** (not frozen base) | trained updates reach rollout | `update_weights` 3.8 s + LoRA kernels; ppo_kl 3.8e-4 | 3.4 s; 1.9e-4 | 4.1 s; 5.9e-5 | **GREEN** |

- **[3a]** vs the dummy run's **0/32** closing + token-soup: ~98 % close `</think>` over 288 rollouts,
  288/288 real SQL, **0** hit the 2048 cap. The "loop" is gone — it was always random weights.
- **[3b]** genuine pass/fail mix every step; advantages span ±0.875; clip ~0 (was 1.0). A real gradient exists.
- **[3c]** proven both ways (NOT the train-but-don't-learn trap): **mechanically** the LoRA adapter is applied
  in the rollout (expand/shrink kernels) and re-synced each step (`update_weights` + decaying ppo_kl
  3.8e-4→5.9e-5); **empirically** the same 12 prompts' pass-rate moved across steps for **11/12**, and a
  tracked prompt's emitted SQL changed step→step. The synced base+LoRA reaches generation.

**Honest caveat (NOT a defect):** reward mean drifted slightly *down* 0.50→0.48→0.43 over 3 steps — a
learning-dynamics question (lr 1e-6, 3 steps, within noise for 12×8), not a sync failure (generation provably
moves). Whether GRPO *lifts* reward is what the real multi-step pilot measures.

**Pilot runtime extrapolation** (steady-state step 3 @ batch-12 = 96 rollouts → batch-32 = 256 rollouts):
- step 3: gen 79 s + log_prob 20 s + update_actor 56 s + weights 4 s ≈ **159 s/step**, throughput **379 tok/s**
  (rising 283→346→379 as gen warms; rollouts terminate at ~420 tok, not 2048).
- @ batch-32: ≈ gen 210 + log_prob 53 + update_actor 150 + 4 ≈ **~415 s/step (~7 min)**.
- 831 / 32 ≈ 26 steps/epoch → **≈ 3.0–3.5 h/epoch** (gen-bound). **`total_epochs=2` ⇒ ~6–7 h**,
  overnight-feasible. Confirm batch-32 steady-state on the first real pilot step (memory/batching may shift it).

## PILOT RUN (2026-06-26/27) — completed, 0 wedges, +2 pt
The first real GRPO pilot ran end-to-end: **50/50 steps, ~11 h, exit 0** (run
`grpo_qwen3_4b_thinking_20260626c`, MLflow `grpo_distill`). **Zero wedges.**

**Wedge-survival, not wedge-elimination** (the pilot RAN this way — the +2 pt result stands). mns=32 wedged
at step 3 (~30 min) **despite `enforce_eager`** — which, with cudagraph already OFF, *cannot* be a cudagraph
deadlock. **That self-contradiction was the clue:** the wedge is the FlashInfer top-k/top-p sampler race
(#43885), not cudagraph (confirmed 2026-06-30 — see recipe). The three levers that got the pilot through:
- **`max_num_seqs=16`** (halved from 32) — the decisive one *at the time*: lower concurrency = fewer/narrower
  FlashInfer-sampler CTA groups = rarer race, enough to luck through **11 h wedge-free**. Trade: 16 under-
  saturates the GB10 decode → **~13 min/step (~11 h)** vs ~9 min @32. **New runs don't need this: cudagraph ON
  + `VLLM_USE_FLASHINFER_SAMPLER=0` + mns=32 is wedge-free AND faster (~10 min/step verified 2026-06-30).**
- **`save_freq=1`** (was 5) — checkpoint every step → a wedge costs ≤1 step (save_freq=5 + a wedge before step 5
  = resume-from-0 = no-progress loop). *Per-step here produced 50 ckpts × 8.6 GB = 430 GB → the default
  reverted to **5** afterward; at mns=16 (wedge-free) per-step insurance is unnecessary.*
- **val-resume-skip patch** (`ray_trainer.py:1393` gated on `global_steps==0`) — baseline val only on the true
  first start, skipped on every resume (~6 min/restart saved). In the image + `Dockerfile.grpo`.
- (save_freq=1 + the patch are verified but **stayed untriggered** — no wedge occurred at mns=16.)

**Result — a small, clean, positive lift, concentrated where intended.** *Two distinct measurements, on two
different test sets, both land at +2* (they are NOT the same number):
- **In-loop held-out val** — on **`weak_test_clean` (240, the 3 weak categories)**: **0.4208 → 0.4417 (+2.1 pt)**
  over the run.
- **Offline per-category eval** — on **`test_clean` (100, all 7 categories)**, vLLM-endpoint, *same serving path
  as the SFT baseline*: **59 % → 61 % (+2 prompts)** — **set operations +1** and **subqueries +1** (2 of the 3
  target weak cats), **window functions unchanged (7.1 %)**, **no regressions** (single-join −1 is noise).
- **window flat is the EXPECTED outcome, not a miss:** it is the *hardest* category (SFT ~14 %, least headroom).
  The two **medium-hard** target cats lifted; the **hardest** stayed put — exactly what a conservative lr
  (1e-6) / 50 steps should do: a nudge, not a breakthrough.
- **Methodology:** measure the GRPO eval on the **same serving path as the baseline (vLLM)**. At ±1-prompt
  deltas the HF↔vLLM greedy confound ≈ the delta size, so mixing serving modes is invalid.

**Verdict:** pipeline proven end-to-end · wedge survived/avoided (mns=16) · a **real-but-small** lift in the
target weak categories. The lever for a bigger effect is learning dynamics (lr / steps), not the plumbing.

## Next steps (user-gated)
The pilot is **done** (outcome above). For a stronger lift: raise **lr (2e-6–5e-6)** and/or add steps/epochs —
the flat trajectory + the untouched hardest category point at too-conservative lr, not a broken signal.
**Per-category offline-eval recipe** (re-measure any checkpoint): `python -m verl.model_merger merge --backend
fsdp --local_dir <ckpt>/actor --target_dir <out>` → shard (`save_pretrained(max_shard_size="5GB")` + carry
`chat_template.jinja`) → serve on the `vllm` service (`VLLM_MODEL=<sharded>`) → `evaluation/evaluate.py
--api-base http://vllm:8000/v1 --api-model-name <id> --n-samples 100 --max-tokens 4096 --enable-thinking
--seed 42`. The `grpo_gate` stack + sharded models + weak pool/held-out are kept.

---

<details>
<summary><b>Archive: dead-ends en route to the random-weights finding (collapsed)</b></summary>

All four were correct investigations whose conclusions were superseded by ROOT CAUSE (the rollouts were
always garbage from random weights). Kept for reference / if the engine-swap question reopens.

- **Pilot-prep (2026-06-26):** found + fixed two real verl plumbing bugs — verl hard-codes
  `repetition_penalty=1.0` (rollout dict, `agent_loop.py:500` / `main_ppo_sync.py:313`), and
  `val_kwargs.repetition_penalty` ConfigKeyError-crashes (removed). Lowered cap 8192→4096. Wired save_freq=5
  + resume_mode=auto + the supervisor. Blocked on the verl-colocated wedge (a step never completed).
- **Rollout-path research (2026-06-26):** *concluded* the wedge was the V1 FULL-cudagraph decode path
  (`cudagraph_mode=FULL_AND_PIECEWISE`, `vllm_async_server.py:240`; vLLM #37729/#40969) — **WRONG, overturned
  2026-06-30: it's the FlashInfer sampler race #43885, not cudagraph.** Still-valid findings from that session:
  no external/decoupled vLLM in verl 0.8.0 on one GB10 (all rollout modes spawn in-process; STANDALONE/PD =
  multi-GPU); SGLang on GB10 = dev-branch only (HIGH bring-up risk); HFRollout/NaiveRollout are unregistered;
  `VLLM_USE_V1=0` is dead (V0 removed).
- **Path-A test (2026-06-26):** `enforce_eager` *appeared* to "kill" the wedge in a 25-min window (35–57 W) —
  **but that was luck, not a fix: the later pilot wedged at mns=32 *despite* eager** (the sampler race is
  probabilistic + concurrency-sensitive, and 25 min under-sampled it). The real fix is
  `VLLM_USE_FLASHINFER_SAMPLER=0` (#43885). The "eager ~80 tok/s → ~20 h/epoch too slow" was also
  loop-contaminated (random-weights rollouts marching to cap; real throughput ~456 tok/s).
- **Cap-2048 detox + rep-pen patch (2026-06-26):** EngineCore death-dump showed all 32 rollouts at ~1580 tok
  with `finished_req_ids=[]` (none terminated) → "verl rollout doesn't terminate like the standalone."
  Applied the rep-pen source patch (now permanent in the image) — rep-pen 1.1 confirmed live, **rollouts
  still 100% looped to cap.** Real eager throughput measured at **~456 tok/s** (corrects the ~80). All of
  this was the random-weights garbage, not a loop or a sampling problem.

*Original archive note:* the path also involved colocated-vLLM / HTTP-glue dead-ends and several 14B OOM
rounds. Two durable lessons: (1) verl's vLLM is **colocated/self-managed** — it must come up in-process on
sm_121, you can't point it at an external served image; (2) on one 128 GB GB10 the colocated topology caps
usable model size below 14 B.
</details>

# Agentic SFT — Synthetic DB Traces (literature map + build recipe)

> **Status:** ✅ **SUPERSEDED — now built** (this pre-build design note is kept for its literature rationale).
> **Date:** 2026-07-03 · **Scope:** how to synthesize the German Deutsche-Bahn tool-calling trajectories — the
> **self-synthesized DB leg** of the Stage-1 SFT mix (now **leg 4** of 4, alongside ToolACE + TaskBench +
> AReaL/τ²-bench; see [`agentic-sft-data-basis.md`](agentic-sft-data-basis.md)). The synthesis is implemented
> and run — **9,146 verified traces**, see [`agentic-db-synthesis-log.md`](agentic-db-synthesis-log.md).
>
> **Decisions locked in:** approach = **(B) grounded synthesis** (teacher drives tool calls against a *real executable* DB sandbox + verifier gate + fault-injection for replan); **teacher = local GB10 model** (a strong API teacher like GPT-5 is not guaranteed). This note sorts the literature levers **before** building, so we don't reinvent what papers already provide. Source: Notion *"Agentic LLM / Orchestrator"* → chapter *"🧪 Hebel aus der Literatur (nach Pipeline-Stufe)"* (9 papers), cross-read against the arXiv originals.

## Where this fits

Stage-1 SFT mixes **four legs**, shuffled in one pass, the German DB leg up-weighted → then Stage-2 GRPO/verl
on new, disjoint τ²-bench + db_bahn tasks.

| Leg | Source | Status |
|---|---|---|
| 1. tool-call basics | ToolACE (downloaded) | done |
| 2. planning/decomposition | TaskBench (downloaded) | done |
| 3. multi-turn dialogue/policy | AReaL / τ²-bench (downloaded) | done |
| **4. DB-specific + German + replan** | **synthesized (this note)** | **✅ built — 9,146 traces** |

## How it works — a plain-English walkthrough

**The goal:** we need training examples that are **whole conversations** — the small model plans, calls a DB tool, sees the result, replans when surprised, calls the next tool, and answers — in German, DB-specific. No such dataset exists, so we **generate** it. "Grounded" (B) means the tools **really run**, so the tool answers are real (not made up), and we can **check automatically** whether the whole thing was done right.

**Step 0 — Build the tool world (the DB sandbox).** First we need a small **runnable DB world**: the tools (`fahrplan`, `standort`, `wartung`, `mitarbeiter`, `text2sql`, …) that return real, consistent answers from a fixed synthetic data state (seeded from real Open Data where possible). *Think: a toy model railway with a database behind it — tools query it and get real answers.*

**Step 1 — Invent tasks you can check (Agents-A1 / KAG).** The trick: don't invent tasks freely (then you can't check them). Instead build a **graph** of which tool needs which other tool's output (e.g. "*Can ICE 1234 make its maintenance?*" → `standort` → `fahrplan` → `wartung` → `mitarbeiter`). Walk a path through that graph, and you already **know the correct tool order and the right answer** — so every task comes with its own **answer key**. *Think: generate a maze together with its solution — build the solution first, then phrase the question.* That answer key is what makes automatic checking possible later.

**Step 2 — Let the teacher solve it (rollout + fault-injection).** A **teacher model** (a bigger local model on the GB10) plays the assistant: reads the task, thinks a plan, calls a tool, sees the **real** result from the sandbox, thinks again, calls the next … until it answers. Because the tools really run, the observations are real. To teach **replanning** (Variante C), we deliberately throw in a problem: sometimes a tool returns an error or a surprise ("train cancelled", "no direct connection"). The teacher must notice and change plan — exactly the behavior the student should learn, and the only reliable way to get it is to **force it** (fault-injection).

**Step 3 — Filter hard (NebulaExp + BINEVAL).** The teacher isn't perfect, so we keep **only the good** trajectories, in two layers: (1) a **hard automatic check first** (NebulaExp's #1 lever) — did the teacher reach the answer that matches the answer key from Step 1? We have the ground-truth, so this is a **yes/no machine check** — no human, no judge. Drop everything that fails. (2) For the soft parts you can't run ("sensible tool? no hallucinated train number? recovered cleanly after the surprise?"), use the **BINEVAL** trick: don't ask a judge "rate 1–10" (unreliable), ask **many small yes/no questions** and count. Keep only what passes both. *(This is exactly DuoMem's "keep only successful trajectories".)*

**Step 4 — Mix smartly (OpenThoughts + Reusable Modules).** OpenThoughts: the biggest lever is **task variety, not volume** — generate many *different* DB tasks (different tools, lengths, surprises), and **keep the long (≥5-turn) ones**. Reusable Modules: keep the SFT data **separate** from the later RL data — SFT shows all the building blocks; the RL phase (Stage 2) pushes on *new* combinations.

**Step 5 — Train (DuoMem's 4B blueprint).** Finally, LoRA-finetune the small (~4B) student on those verified full-history trajectories — the *whole* conversation including the plan/replan reasoning, not just the final answer. DuoMem showed this takes a 4B model from ~4 % to ~78 % on a comparable agent benchmark, distilling from an **open** (not frontier) teacher.

**Why the local-teacher worry goes away.** We only have a local model, not a GPT-5-class API. Three papers say that's fine: a teacher that **fits the harness** (drives the same tools) beats a bigger mismatched one, and since we **throw away everything that fails the automatic check**, a so-so local teacher just needs to run **more times** to produce enough clean examples. Want more variety? Use **2–3 different local teachers**, not one giant one.

**What's not in scope yet.** The RL papers (Beyond Reward, ZPPO, DASH) belong to the **second phase** (GRPO on τ²-bench). And one deploy caveat: after training, re-check the model doesn't leak internal data or refuse oddly before it goes live ("Does Reasoning Preserve Alignment?").

**In one sentence:** build a checkable DB tool world → generate tasks with a built-in answer key → let a local teacher solve them (with injected surprises) → keep only the machine-verified conversations → train the 4B model on those.

## The 9 literature levers, classified for our use case

**Legend:** 🟢 use now for Stage-1 DB synthesis · 🟡 Stage-2 RL (later) · 🔵 deploy gate · ⚪ context/motivation

### Stage 1 — SFT method & data recipe (the 4-leg mix)

| Paper | arXiv | Verdict | Actionable takeaway for us |
|---|---|---|---|
| **OpenThoughts-Agent — Data Recipes for Agentic Models** | [2606.24855](https://arxiv.org/abs/2606.24855) | 🟢 | **Task diversity = #1 lever (up to 30 pp)**; keep **≥5-turn traces** (+3.5 pp even at fixed token budget); **strongest ≠ best teacher** (harness-compatible wins). → maximize DB task-surface diversity; keep long replan traces. |
| **NebulaExp-8B — Empirical Post-Training via Full-Scale Ablation** | [2606.26671](https://arxiv.org/abs/2606.26671) | 🟢 / 🟡 | **Execution correctness filter = lever #1** (drop ~7% bad → +7 pp); **teacher-FIT > teacher-SIZE**; multi-teacher distill lets student exceed single teachers. (pass@4 difficulty grading → Stage-2.) → exec-gate every trace first; consider **ensembling 2-3 local teachers**. |
| **From Reasoning Traces to Reusable Modules** | [2606.18089](https://arxiv.org/abs/2606.18089) | 🟢 / 🟡 | **Disjoint SFT/RL data beats overlapping**: SFT covers *all* building blocks (tool/plan/route/replan) compositionally; RL targets *new* combinations outside SFT. → make the DB SFT set compositionally complete; reserve novel tool-combos for the Stage-2 set (keep disjoint). |

### Stage 2 — Synthesizing the multi-turn DB traces (**the core of (B)**)

| Paper | arXiv | Verdict | Actionable takeaway for us |
|---|---|---|---|
| **Scaling the Horizon, Not the Parameters (Agents-A1 / KAG)** | [2606.30616](https://arxiv.org/abs/2606.30616) · [code](https://github.com/InternScience/Agents-A1) | 🟢 (data-gen half) | **Model the DB as a Knowledge-Action Graph** (typed 4-tuple: Corpus → Actions → Observations → Verifier) and synthesize verifiable multi-step tasks by **constrained graph walks** → every task ships machine-checkable ground-truth → verifier + quality-gate + clean fault-injection points **for free**. Caveat: their 79.8 τ²-bench is a **35B-MoE** result with multi-teacher OPD — adopt only the **KAG task-gen half** at 4B; OPD/SVA = Stage-2/skip on one GB10. |
| **DuoMem — On-Device Memory Agents via Dual-Space Distillation** | [2606.29961](https://arxiv.org/abs/2606.29961) | 🟢 | **The 4B blueprint.** Dual-space: (1) **parameter-space** = LoRA (~6M) on **successful FULL-HISTORY teacher trajectories** (= our trace distillation; full-history > last-5); (2) **context-space** = teacher **procedural memos** prepended to input (training-free, ~4 MB, embedding-retrieved). ALFWorld **4B 4.3 % → 77.9 %**, 89 % of the gap to a 72B, 3.4× faster, <10M params. Distills an **open 72B** → confirms a local teacher works; **nothink + distillation** best for edge. |
| **Ask, Don't Judge (BINEVAL)** | [2606.27226](https://arxiv.org/abs/2606.27226) | 🟢 | Build the **non-executable part of the verifier as atomic yes/no questions** ("right tool? args grounded in observed state? no hallucinated entity? recovered after the injected fault?") — calibrated, interpretable, no ceiling effect. Caveat: verifier strength is first-order and ours is local → **prefer deterministic/executable checks**, reserve binary-LLM for the un-checkable dimensions. |

### Stage 3 — RL (GRPO / verl) — all Stage-2, not now

| Paper | arXiv | Verdict | Actionable takeaway for us |
|---|---|---|---|
| **Beyond Reward Engineering — Data Recipe for Long-Context RL** | [2606.18831](https://arxiv.org/abs/2606.18831) | 🟡 | Plain GRPO + **difficulty filter (roll 4×, keep only mixed-outcome/medium)** beats reward-shaping; **task-balanced sampling + task-wise advantage-norm** stabilize the mixed pool. (roll-N filter can also pre-screen Stage-1 seed tasks.) |
| **ZPPO — Teacher in Prompts, Not Gradients** | [2606.18216](https://arxiv.org/abs/2606.18216) | 🟡 | **Cold-start protection** for the 4B actor — on all-fail (zero-advantage) items put the teacher **in the prompt** (BCQ/NCQ), not the gradient; biggest gains at smallest scale. |
| **Know When to Stop (DASH)** | [2607.00482](https://arxiv.org/abs/2607.00482) | 🟡 (conditional) | **Segment-level credit vs overthinking** — split a trace at intermediate-answer checkpoints, reward productive reflection, escalate-penalize answer-drift. Needs **extractable verifiable intermediate checkpoints** — trajectory-wide τ² success makes it conditional, but the **KAG verifiers can supply per-step checkpoints** → nice synergy. |

### Caveats (pointers, not levers)

- ⚪ **When in Doubt / Plan-Commitment** ([2606.16995](https://arxiv.org/abs/2606.16995)) — supports **Variante C**: commit to a plan, replan only on deviation. A design principle for the *shape* of our traces.
- 🔵 **Does Reasoning Preserve Alignment?** ([2606.11046](https://arxiv.org/abs/2606.11046)) — SFT/RL/distillation can degrade alignment → **post-RL safety/privacy re-eval is mandatory before internal DB deploy** (contextual privacy leakage + refusal miscalibration are the load-bearing risks).
- ⚪ **VibeThinker-3B / Reasoning with Sampling** ([2606.16140](https://arxiv.org/abs/2606.16140) / [2510.14901](https://arxiv.org/abs/2510.14901)) — reasoning peak ≠ orchestration; **tool-use must be trained explicitly** (motivates the whole SFT effort).

## The literature-backed (B) build recipe

The 9 levers jointly prescribe this Stage-1 pipeline (each step cites the paper that backs it):

1. **Task engine — KAG.** Model the DB domain as a Knowledge-Action Graph (schema/tools/observations/verifiers as a typed graph); generate verifiable German multi-step tasks by **constrained graph walks**, so each task carries a machine-checkable ground-truth. *(Agents-A1)*
2. **Rollout — grounded.** Local teacher(s) act against the **real executable DB sandbox** → full-history trajectories; **inject faults** (tool error / surprise / seeded bad state) to force plan → observe → **replan**. *(DuoMem full-history + our fault-injection; plan-commit shape from When-in-Doubt)*
3. **Gate hard.** **Execution-based correctness filter first** *(NebulaExp #1 lever)*, then an **atomic yes/no BINEVAL checklist** for the un-checkable parts; keep **only verified (score = 1)** trajectories. *(DuoMem "successful trajectories")*
4. **Mix for diversity.** Maximize DB task-surface diversity, keep **≥5-turn replan traces** *(OpenThoughts)*; keep the SFT set **disjoint** from the Stage-2 RL combinations *(Reusable Modules)*.
5. **Train.** LoRA on the verified full-history trajectories *(DuoMem 4B blueprint)*; optionally add teacher procedural-memos in context-space. **Built:** the tool-sandbox + verifier + multi-turn loop now live in [`sdg_pipeline/db_bahn/rollout.py`](../sdg_pipeline/db_bahn/rollout.py) (concurrency / resume / branch-on-fail / MLflow) + [`evaluation/trajectory_reward.py`](../evaluation/trajectory_reward.py) + [`training_pipeline/train_traj.py`](../training_pipeline/train_traj.py). *(The base's old `trace_capture.py` / `run_sdg.py` harness was removed in the 2026-07-03 cleanup and rebuilt fresh.)*

## Teacher question — resolved

Three papers converge that **teacher-fit / harness-compatibility beats teacher-size**: OpenThoughts (a weaker GLM beat GPT-5.3 by ~5 % as a teacher), NebulaExp ("distribution compatibility outweighs model scale"), DuoMem (distills an **open** 72B, not a frontier API model). And because filtering is **rule-based / executable rejection-sampling**, even a weak local teacher yields clean data (keep only verified). → **A local GB10 teacher driving the same DB sandbox the student will use is the recommended configuration, not a compromise.** For more diversity, **ensemble 2-3 local teachers** (NebulaExp multi-teacher), not a bigger API model.

## Open gap

No paper backs the **concrete mix ratio** (ToolACE / TaskBench / τ²-bench + up-weighted German synthetic). The levers give only generic principles (diversity, disjoint SFT/RL split, difficulty filter) — the exact weighting stays an **empirical build decision**.

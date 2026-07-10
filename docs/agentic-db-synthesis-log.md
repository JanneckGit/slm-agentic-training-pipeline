# Agentic DB-synthesis — build & decision log

> Newest entry on top. Baubegleitendes Entscheidungs- + Bug-Log. Design-Kontext:
> [agentic-sft-db-synthesis.md](agentic-sft-db-synthesis.md); Datensatz-Erklärung:
> [agentic-datasets-explained.md](agentic-datasets-explained.md).

## 2026-07-10 — ✅ Regen 2 (Split-Redesign + 1.601 Traces) + A1-Lookup-Tool (12.)

- **Split-Redesign (per-Template proportional):** Round-Robin ließ kleine Pools verhungern — `wartung_depot`
  15→0 in sft, d. h. `wartung_liste` wäre **nie** trainiert worden. Jetzt landet **jedes der 25 Templates in
  jedem disjunkten Split** (`heldout_eval`/`rl_train`/`sft_train`), `bakeoff_dev` ist ein **nicht-disjunkter**
  stratifizierter ⊆-sft-Sample. Neue Splits (HARD-FAIL-geprüft): **bakeoff_dev 25 / heldout_eval 59 /
  rl_train 295 / sft_train 1.610** (Pool weiter 1.964 unique).
- **`format_traj` split-aware:** `--split-file/--split` filtert Records, deren `task_id` nach einem Split-Regen
  in rl/heldout gewandert ist → **kein Leakage** ins SFT-Set.
- **`solve_task`-Rework — branch-first + B2-Priorität:** bei einem gescheiterten Rollout zuerst **Recovery**
  (harvest → yield-mode) statt Neustart. `choose_harvest_point` behält den Fehler **plus** seine Korrektur
  (= Selbstkorrektur-Trace), aber nur bei **nicht-mutierenden** Fehlern (READ oder abgelehnter WRITE).
  `recovery_mode ∈ {direct, harvest, clean, restart, failed}`.
- **Ergebnis:** **1.601 verifizierte Traces (99,4 % Yield), alle 25 Templates** (Coverage-Loch zu),
  **55 % Multi-Tool** (≥3 Calls), **41 % Fault/Replan**, **emergente Selbstkorrektur 0,7 %→3,1 %** (49 Traces,
  davon 33 via B2-Harvest), kein Leakage, kein `teacher_error`. Alt nach `archive/data/wave2_gen1_20260709/`.
  MLflow-Run in `db_bahn_traj_gen`.
- **Zwei dauerhafte Fixes:** (1) **Context-Overflow** — der 11-Tool-System-Prompt (~3.800 Token) sprengte das
  8192-Fenster (HTTP 400) → `max_model_len` **8192→12288**, `max_tokens_per_turn` **2048→1536**, `rollout.py`
  fängt Teacher-HTTP-Fehler graceful ab (`teacher_error`, kein Abort). Propagiert nach `gen_traces.sh` +
  `traj_sft_pipeline.sh` + `teacher_bakeoff.sh` + Config. (2) **`mlruns/` root-owned** → Host-mlflow-Schreiben
  scheiterte → via Container auf Host-User gechownt (Host + Container schreiben jetzt beide); mlflow ≥ 3.14
  braucht zusätzlich `MLFLOW_ALLOW_FILE_STORE=true`.
- **A1 — 12. Tool `mitarbeiter_details` (READ, Lookup-by-ID):** Root-Cause-Fix gegen den Über-Such-„Flail"
  (~1,5 % Traces): der Agent hatte **kein Werkzeug, eine BEKANNTE Person zu prüfen** — nur `mitarbeiter_suchen`
  (Kategorie-Filter, bei 10 Treffern abgeschnitten); stand die ID im Ticket, „verifizierte" der Teacher blind
  und schloss aus einer gekürzten Trefferliste falsch auf „nicht qualifiziert". Das neue Tool gibt Stammdaten
  per ID (`ValueError` bei unbekannter ID). `policy.md` 11→12 Tools + Verifikations-Regel (bekannte ID →
  `mitarbeiter_details`, nie aus abgeschnittener Liste auf Abwesenheit schließen). **Rein additiv** (Gold-Pfade
  unverändert, `expected_tools ⊆ called`), **READ** → Verifier/rollout unberührt. CPU-Gates grün: Env-Smoke
  **12 Tools**, Tool funktional (ID→Stammdaten, unbekannt→`ValueError`, whitespace-tolerant), Verifier-Selftest
  8/8, Oracle-Dry-Run `bakeoff_dev` 100 %.
- **Datenlage / Nuance:** die 1.601 Traces entstanden auf der **11-Tool**-Domäne (`mitarbeiter_details` in
  keiner Trace) → ein **einheitlicher 12-Tool-Regen** ist geplant (User, Wochenende).
- **Offen:** Training (`traj_sft_pipeline.sh`) + **Re-Baseline** auf `heldout_eval` (59), danach Stage-2 GRPO
  re-wire (Config-GRPO-Block zeigt noch auf alte Text2SQL-Artefakte).

## 2026-07-08 — ✅ WELLE 2: Clean Rebuild der Domäne + Task-Pool (1.964 Tasks, alle Gates grün)

- **Decision — clean rebuild statt Welle-1-Schonung (User-Vorgabe):** kein Byte-Identitäts-Gefrickel, um die
  446 alten Traces zu retten — Domäne/Templates/Splits sauber nach Merit neu gebaut; **Welle-1-Artefakte
  archiviert** nach `data/archive/wave1_20260708/` (tasks/splits/keys + alle db_traces + db_traces_chat).
  Why: synthetische Daten sind billig reproduzierbar (deterministischer Generator + eigene GPU), alte Traces
  wären mit dem neuen 11-Tool-Prompt eh inhomogen. Welt (`db.json`) unverändert — kein Re-Seed.
- **Domäne erweitert (tools.py, 8 → 11 Tools):** 3 Such-READ-Tools (`zuege_suchen`, `mitarbeiter_suchen`,
  `wartung_liste`; ≥1 Filter Pflicht, Cap 10 Zeilen, deterministische Sortierung — „erster Treffer = kleinste
  ID" ist der Tiebreak, den Tickets referenzieren) + **Business-Regeln in den WRITE-Tools** (Laufzeit-Fehler
  per deutscher ValueError → Error-Observation → Replan): Rollen-Gate, Produkt-Qualifikations-Gate (nur
  ICE/IC/EC), Duplikat-Gate, Endstatus „abgeschlossen", `faellig_am`-Format, Depot-Whitelist. Brachliegende
  Weltdaten (qualifications, shifts) damit erstmals agentisch erreichbar. policy.md auf 11 Tools + Regeln 4/5
  (Zuteilung/Wartung) + „abgelehnte Aufrufe nie wiederholen" erweitert.
- **gen_tasks.py neu:** EINE Registry `Spec(fn, pool, n, injectable, fault_rate)`, 25 Templates (9 polierte
  Welle-1-Formen, `info_wartung_machbar` ersatzlos raus [15/49-Freiform-Schwäche]; 16 neue: Suche ohne
  vorgekaute IDs, 3–4-Tool-Ketten, bedingte Writes, Multi-Write, 3 Laufzeit-Fehler-Replan-Formen). Einheitliches
  Key-Schema für ALLE Tasks: `fault/expected_calls/oracle_calls` — `make_oracle` läuft nur noch über
  `oracle_calls` (Zugnummer-Heuristik gelöscht). Neue Pools inkl. „ein Trip pro Fahrzeug" (Dedup gegen
  Beinahe-Duplikate) und deterministische Filter-Kombo-Pools mit 1–3 Treffern.
- **Ergebnis (seed 42, Gate-d-kalibriert):** **1.964 Tasks** — Multi-Tool (expected_calls ≥3) **52 %**
  (Ziel ≥50), Fault **41 %** (Ziel ~40; 538 state / 128 runtime / 140 state+runtime), Single-Tool 27 %
  (Welle 1: 64 %). Splits frisch + disjunkt (HARD-FAIL-geprüft): bakeoff_dev 25 / heldout_eval 60 /
  **rl_train 300 (GRPO-Reserve, wird nie für SFT gerollt)** / sft_train 1.579.
- **Gates:** (a) Oracle-Dry-Run 100 % verified, 0× gold_replay_failed (validiert Keys + neue WRITE-Regeln,
  bakeoff 25/25; große Splits s. validation_w2/); (c) Verifier-Selftest 8/8 inkl. Runtime-Fault-Roundtrip
  (Rejection→Suche→valide Zuweisung→1,0 mit `replan_occurred`; ignorierte Rejection→0,0); (d) Stats-Gate s. o.;
  (e) Env-Smoke 11 Tools + alle 6 Ablehnungs-Gates funktional. **Determinismus: zweiter Lauf byte-identisch.**
- **Eval-Bruch beabsichtigt:** 11-Tool-Prompt + neues 60er-Heldout → alte 72,5 %/70 % nur noch historisch;
  Re-Baseline des Basis-Modells gehört in den Rollout-Abend. `replan_occurred` zählt jetzt auch
  Laufzeit-Fehler-Replans (`fault∈{runtime,state+runtime}` + ≥1 Tool-Error + ≥2 Planungs-Turns).
- **Offen (Etappe 3, GPU auf Zuruf):** Rollout-Abend mit Qwen3.6-35B-A3B über sft_train (1.579; k=2-Top-up
  auf Multi/Fault-Teilmenge für das 1.500–2.000-Band), Re-Baseline auf heldout_eval.
  Drive-by gefixt: `seed_worldstate.py` Manifest-KeyError (`db["_meta"]`→`db["meta"]`).

## 2026-07-03 — Implementation started

- **Decision:** build phase-by-phase (Phase 0 → 6) with a smoke test after each phase before proceeding; document
  here as we go; work in the git working tree only (no commits/pushes).
- **Why:** the plan's working principles; keeps each stage independently verifiable.
- **Status:** Phase 0 in progress.

## 2026-07-03 — ✅ PILOT END-TO-END COMPLETE — honest null accuracy result (pipeline proven)

- **AFTER-eval (trained student, base+LoRA, 40 heldout tasks): 70.0%** vs BEFORE 72.5% → **28/40 vs 29/40 =
  a 1-task difference = statistically identical.** Per-template: changes are pure noise (3 tasks 0→1, 4 tasks
  1→0, all in the near-boundary templates action_ersatz/action_wartung/info_ankunft/info_machbar at n=4 each);
  behavior unchanged (3.5 turns, 97-98% valid tool-calls, ~25/40 use `<plan>`, **0 loops** — termination fine).
- **Honest read (the point of a vertical slice):** on THIS deliberately-simple, same-distribution held-out set
  traj_sft does not move accuracy — because (a) the base Qwen3.5-4B already scores 72.5% (little headroom),
  (b) n=40 (4/template) only detects large effects, (c) the SFT data teaches nothing the base can't already do.
  This mirrors the base project's own honest null ("distillation barely lifts a saturated benchmark"). **What
  a real lift needs:** a base that CANNOT already tool-call, and/or a HARDER eval with headroom (more tools,
  deeper plans, more replan), and/or the full Stage-1 MIX (ToolACE+TaskBench+DB) — not this easy DB-only slice.
- **What IS proven end-to-end (the actual deliverable):** grounded synthesis → 446 verified German multi-turn
  traces (92%) → assistant-only-masked traj_sft (clean loss 0.37→0.13, 0 loops) → deployable student. Every
  stage smoke/gate-tested. The machinery is the result; the accuracy proof is a separate, harder experiment.

## 2026-07-03 — Repo cleanup: text2sql-only code removed, agentic-only tree

- **Rationale:** repo is agentic-only now; every deleted file is preserved in the initial commit
  (`683f311`) and in the old repo (`JanneckGit/SLM-Finetuning`). Deletion decided by a grep-verified
  import/reference graph: the agentic code is a closed graph; the only old file in the active path is
  `serving/merge_adapter.py`.
- **Deleted (SQL-only / superseded):** data_pipeline/{prepare_data,prepare_sdg_input,mix_datasets,
  build_train_clean,complexity_taxonomy,format_for_training}.py · sdg_pipeline/{run_sdg,trace_capture}.py
  + blocks/ + flows/ · evaluation/{rescore,efficiency_benchmark}.py · training_pipeline/train.py ·
  serving/{query_model.py,deploy_vllm.sh} · tools/bench_deltanet.py · ops/{run_baseline_pipeline,
  run_all_baselines,sdg_run_supervised,setup_remote}.sh · root debris (3 empty root-owned docker-mount
  artifacts: tmpl_probe*.py, lencheck.py).
- **Kept for Stage-2/later (B-set):** grpo_verl_runner + build_weak_pool + reachability_probe (verl
  recipe), grpo_pilot_supervised.sh (watchdog), reward.py+evaluate.py (verl contract pair; reward imports
  evaluate.extract_sql), merge_adapter_mm.py (MM deploy merge), clean_traces.py (filter skeleton),
  close_rate_probe.py, quantize_fp8.py. Docker: all services/images kept (Dockerfile.grpo carries the
  rep-pen source patch).
- **Config slimmed in BOTH yamls** (template + local, key-synchronous): removed complexity_classes,
  sdg:, infra:, and the SQL data-subkeys; grpo: kept as marked Stage-2 template; teacher_candidates
  synced to the final bake-off list (drift fixed). Dockerfile.sdg/.training CMDs updated (pointed at
  deleted scripts); compose header examples updated; .gitignore: docs/text2sql-experiments now TRACKED
  (user decision — evidence base referenced by the agentic docs), nohup.out + .venv-tau2/ added.
- **ops scripts:** TAU2PY now defaults to repo-local `.venv-tau2/` (was a session-temp path), overridable.
- **README rewritten** for the agentic pipeline (architecture, results, setup incl. tau2 venv, verified
  quickstart commands, no dead links).
- **Post-cleanup smoke suite PASSED:** tau2-venv import sweep 8/8 (fresh repo-local `.venv-tau2` built
  exactly per the new README setup — verifies those instructions for real) · host sweep 1/1 · training-
  container sweep 7/7 · world-state + tasks byte-identical · oracle dry-run 6/6 (100%) · verifier selftest
  5/5 · collator golden test OK · README: all links exist, all CPU commands ran verbatim, GPU-command
  flags match the scripts' argparse · grep proof: no surviving code references any deleted module.
  Note: `data_pipeline/clean_traces.py` is a hand-run SCRIPT (top-level logic, opens its input on import) —
  kept as filter TEMPLATE, deliberately excluded from import sweeps; guard with __main__ when adapting it.
- **Disk:** broken 8 GB text-merged checkpoint removed (recreatable from the kept 85 MB LoRA adapter via
  merge_adapter_mm); stray docker/.env.save removed.
- **NOT committed/staged/pushed** (user does that later). Suggested commit split:
  (1) `chore: remove text2sql-only pipeline code (preserved in initial commit)` — the deletions;
  (2) `feat: agentic DB-trace synthesis pipeline (tau2 db_bahn, verifier, rollout, bake-off, traj_sft)` — new code + ops;
  (3) `docs: agentic docs, teacher bake-off, text2sql experiment archive, new README`;
  (4) `chore: slim config to agentic pipeline; gitignore + docker CMD fixes`.

## 2026-07-03 — Phase 5 + 6: formatter, assistant-only mask, traj_sft trained

- **Phase 5 — `data_pipeline/format_traj_for_training.py`:** 446 verified traces → chat JSONL. tool_call
  `arguments` string→**dict** (the Qwen template renders arguments as a mapping, not a JSON string — else
  `apply_chat_template` raises "Can only get item pairs from a mapping"). 446/446 kept.
- **Phase 5 — `training_pipeline/collator_multiturn.py` (loss mask):** confirmed Qwen3.5-4B ships
  `{% generation %}`-less → `return_assistant_tokens_mask` is all-zero (plan P1-3 fallback needed). Built an
  explicit ChatML span scan: unmask `<|im_start|>assistant\n … <|im_end|>` blocks only. Qwen renders
  `role:"tool"` observations as a `user` turn (`<tool_response>`), so tool outputs are masked for free.
  **Golden self-test PASS:** trains ~52% of tokens (assistant + `<plan>` + tool_call), masks user/tool-obs/system.
- **Phase 6 — `training_pipeline/train_traj.py`:** standalone traj_sft (base run_lora_sft untouched); pre-tokenize
  with the mask, plain HF Trainer + TrajSFTCollator. Token lengths max 3502 → max_seq_len 4096 drops nothing.
  **Training clean: loss 0.37 → 0.13 over 2 epochs (28 steps, ~54 min), no NaN**; LoRA adapter (85 MB) saved.
- **BEFORE-eval (base Qwen3.5-4B, untrained, on 40 heldout tasks): 72.5% verified-yield** — the base is already
  decent on the deliberately-simple 1-2-tool tasks; headroom for "after" is therefore limited on this set.
- **🐛 Merge bug + fix:** `serving/merge_adapter.py` (text merge via AutoModelForCausalLM) produced a
  **text-only config** (`Qwen3_5TextConfig`) that vLLM's Qwen3.5 loader rejects (wants the full multimodal
  `Qwen3_5Config` with vision_config) — exactly the base pipeline's MM-merge gotcha (`merge_adapter_mm.py`
  exists for this). **Fix for the eval:** skip merge, serve base MM model + the LoRA adapter directly via
  vLLM `--enable-lora --lora-modules db_bahn=<adapter>` (VLLM_ALLOW_RUNTIME_LORA_UPDATING already on).
  (For a deployable single-file student, use `merge_adapter_mm.py` later.)

## 2026-07-03 — ✅ LEG-3 GENERATION COMPLETE: 446 verified traces (92%), Gate-2b PASSED

- **Full run result:** 485 sft_train tasks → **446 verified (92%)**; new-rollout yield 90.4% (was 80%
  pre-calibration). GPU released. Dataset: `data/generated/db_traces_sft_train_q36-35b-a3b.jsonl`.
- **Calibration effect:** `info_ankunft` 1/10 → **45/48**. `info_machbar` 1/10 → 15/49 — still the hardest
  (free-form feasibility judgment); acceptable for the pilot, revisit only if the mix needs more of it.
  **All 4 ACTION templates 48/48 (100%), including the fault-injected replan template.**
- **Gate-2b batch check (all PASS):** 0 loopy finals (trigram>10) · **100% German finals** · 0 duplicate tasks ·
  balanced tool distribution (8 tools, 683 calls) · 85 injected verified, **72 with an explicit replan turn** ·
  avg 4.1 turns (159 traces ≥5 turns) · trace length p95 ≈ 9.1k chars (~3k tokens) → traj_sft max_seq_len of
  ~6-8k tokens suffices (better than the feared 12-16k).
- **Leg 3 status: RAW GENERATION DONE.** Next: Phase 5 (chat formatter + assistant-only loss mask) and
  Phase 6 (traj_sft smoke + held-out German eval before/after).

## 2026-07-03 — Calibration of the 2 derived-arithmetic templates + FULL generation run

- **Template fix (`gen_tasks.py`):** `info_ankunft` + `info_machbar` tickets now explicitly require stating
  the PLANNED arrival time, next stop, and delay (minutes/grund or 'pünktlich') — so communicate_info
  matches stated-observed values instead of hoping the model doesn't paraphrase.
- **Verifier fix (`trajectory_reward.py`):** grounding corpus now includes **derived times** = any observed
  HH:MM ± any observed delay-minutes (from `verspaetung_minuten` / "+N Min") — computed arrival arithmetic
  is legitimate reasoning, not hallucination. Times only; ids/dates stay strict. Unit test: 18:30+45→19:15
  grounded, random 21:47 still rejected; full selftest still 5/5.
- **Consistency proof:** task regen with the fixes → id set unchanged; **only the 110 tasks of the 2
  calibrated templates differ, all other 440 byte-identical** (deterministic content-derived ids pay off).
  Production trace file purged to the 78 still-valid verified records (fails + recalibrated-template records
  removed → resume re-rolls exactly those).
- **Full generation run launched:** winner q36-35b-a3b over ALL 485 sft_train tasks (k=1, max-regen 2,
  conc 4, 2h cap). Target ≥400 verified.

## 2026-07-03 — Bake-off COMPLETE: winner Qwen3.6-35B-A3B; validated 80% on 100 tasks

- **Fair pass 2 (identical final harness, 12 stratified tasks × k=1):** q36-35b-a3b **92 / 100 % DE / 16 s**
  (score 188) · q36-27b 92/100 %/59 s · q3-30b-think 92 %/74 s · nemotron-49b 92 %/189 s · seed-oss 83 %/268 s ·
  q3-next-80b 75 % · glm45-air 58 % (tight 6144 ctx) · magistral 33 %. Full table: `docs/teacher-bakeoff.md`.
- **Winner: Qwen/Qwen3.6-35B-A3B** — best yield band + perfect German + all replan tasks + ~12× throughput
  advantage; also the already-deployed base teacher (user's reuse instinct beat the research ranking).
- **Additional harness fixes in pass 2/cleanup:** Qwen3-30B-Thinking emits **bare-JSON calls with NO wrapper**
  (format #6; 8 %→92 % after parse branch) · Magistral 400 root cause = **tool_call ids must be 9 alphanumeric
  chars** (mistral tokenizer) → global id scheme `c00000001` (33 % after fix — genuinely weak, not blocked) ·
  GLM context overflow fixed via 6144 ctx/1024-per-turn (58 %).
- **Winner validation (100 stratified sft_train tasks, k=1 + 1 regen): 80 % verified-yield**, 18 s/rollout.
  **8/10 templates = 100 %** (incl. ALL ACTION + the injected replan template 10/10). The whole 20 % loss =
  the two derived-arithmetic INFO templates (`info_ankunft`, `info_machbar` — grounding/communicate on
  computed arrival times) → **calibration item for the full run** (ticket wording + robust communicate
  strings), not a teacher gap. ≥5-turn share is 22/80 → apply the ≥5-turn keep-filter per-template in the mix.
- **First production batch exists:** 80 verified traces in `data/generated/db_traces_sft_train_q36-35b-a3b.jsonl`.
  Projection: >400 verified in ~1–2 GPU-h after the 2-template fix. GPU released (vllm down).

## 2026-07-03 — Bake-off pass 1 complete: a zoo of tool-call formats; final parser; fair re-run pass

- **Pass-1 results (12 stratified tasks × k=1 each; NOT comparable — parser evolved mid-pass):**
  seed-oss-36b **92%** (all 3 replan tasks solved; slow 228 s/rollout) · q36-27b 75% · q36-35b-a3b 67%
  (4× faster than 27b) · nemotron-49b 67% · q3-30b-think 8% · magistral-24b 0% (400s) · q3-next-80b 0% ·
  glm45-air 0%. All 8 candidates SERVE on vLLM v0.21/sm_121 (incl. Qwen3-Next GDN + NVFP4) — no kernel failures.
- **Every 0%/low score was a HARNESS gap, found by reading decoded traces (base lesson), not model inability:**
  (1) Qwen XML drift `<function=x><parameter=k>v` · (2) thinking-only models emit `reasoning</think>answer`
  with NO opening tag (+ `<seed:think>`, `[THINK]` dialects) · (3) Qwen3-Next wraps calls in `<tools>` AND
  **hallucinates the tool RESPONSE in-turn** → fixed with stop-sequences (`</tool_call>`, `</tools>`,
  `</TOOLCALL>` + include_stop_str_in_output) · (4) Nemotron uses `<TOOLCALL>[{…}]` · (5) GLM's template
  injects the OPENING `<tool_call>` (content = `{json}</tool_call>`) · (6) GLM 400 = context overflow
  (max_len 4096 < prompt 2k + max_tokens 2k) → GLM re-run at 6144/mns2/1024-per-turn · (7) magistral 400
  root cause still unknown (error-body capture added; re-run will reveal).
- **Final parser handles 5 call formats + 3 think dialects + stop-cut tails; 10/10 regression tests pass.**
  Pass-1 traces archived to `data/generated/bakeoff_pass1/`. **Fair pass 2 = all 8 candidates re-run with
  the identical final harness**; per-candidate maxtok (thinking-only get 3072; GLM 1024).

## 2026-07-03 — Parser gap found via trace inspection: Qwen XML tool-call drift format

- **Diagnosis (verify decoded output, not metrics — the base lesson applied):** both Qwen3.6 models "failed"
  the action_ersatz replan tasks the same way. Reading the trace showed the model's 2nd call came in the
  **Qwen XML drift format** (`<function=name><parameter=key>value</parameter></function>` inside
  `<tool_call>`), not JSON — our parser only knew JSON → call never executed → db_match fail. This is the
  documented Qwen3.x XML↔JSON format oscillation; owning the parse step made the fix trivial.
- **Fix:** `parse_tool_calls` now parses BOTH formats (+ mixed); unit-tested against the exact failing block.
  **Consequence:** q36-27b (75%) and q36-35b-a3b (67%) yields are UNDERSTATED → re-run both after the
  current pass (delete traces → runner resume re-runs). Verifier calibration note stays: strict grounding
  flags derived arithmetic (e.g. computed arrival '09:58') — uniform across candidates.

## 2026-07-03 — Bake-off started; 400-bug found + fixed (embedded tools); first real teacher works

- **Bug (candidate 0, first real call):** vLLM 400 — bisect against the live server confirmed:
  `"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set` — sending the
  `tools` param requires exactly the server-side parser dependency the plan wanted to avoid.
  **Fix (pure prompt-and-parse):** tool schemas now EMBEDDED in the system prompt (native Qwen
  `<tools>…</tools>` block, model-agnostic) and the `tools` param is NOT sent. Also restored
  **per-request error isolation** in the worker (a trace_capture feature lost in the fork — one HTTP error
  had killed the whole eval) and fixed a self-inflicted edit bug (dropped `sys_prompt` line). Oracle re-check
  after fixes: 6/6 verified.
- **Shortened bake-off protocol** (user: "keine 3h pro Modell"): 12 stratified tasks × k=1, max 8 turns,
  2048 tok/turn, hard 25-min cap/candidate; next model downloads in background during the current eval.
  Runner: `ops/teacher_bakeoff.sh` (resume-safe, skip-on-failure); summary → `docs/teacher-bakeoff.md`
  via `bakeoff_summary.py`. Candidate swap: **Qwen3.6-27B (cached) replaces Qwen3-32B** as dense control.
- **First real result — q36-27b: 75% verified-yield (9/12)**, avg 3.4 turns, clean German plan→tool→answer
  traces. Fails: both injected `action_ersatz` replan tasks (model answered without performing the write →
  replan weakness signal) + one grounding fail on a **derived** arrival time ('09:58' = plan + delay,
  computed not observed). **Verifier calibration note:** strict grounding flags derived arithmetic values;
  uniform across candidates (fair for the bake-off), revisit for production synthesis (allow derived values
  or instruct models to state observed values only).

## 2026-07-03 — Phase 4 (verifier) + Phase 3 CPU (rollout harness) built + smoke PASSED

- **`evaluation/trajectory_reward.py`** (Phase 4, pulled forward — the bake-off needs it for scoring):
  deterministic, never-raises `score_trajectory(...) -> dict{score, ...aux}` (verl-shaped, mirrors reward.py).
  Components: ACTION → **db_match** (replay trajectory tool calls on fresh env vs gold init+reference-actions
  hash, tau2 semantics) + **asserts_pass**; INFO → **no_write** invariant; both → **actions_pass** (expected-tools
  set-membership, order-free), **communicate** (case-insensitive substrings), **grounding** (anti-hallucination:
  every id/time/date/Zugnummer token in the final answer must appear in ticket+observations). Aux: turns_used,
  n_plan_turns, tool_calls_valid, **replan_occurred** (injected & ≥2 plan turns). verl `compute_score` adapter
  as the Stage-2 seam. **✅ Self-test:** good-action 1.0 · wrong-write 0.0 (db_match) · good-info 1.0 ·
  **hallucination 0.0 (grounding catches invented MA-99999)** · injected+replan 1.0.
- **`sdg_pipeline/db_bahn/rollout.py`** (Phase 3 CPU part): trace_capture-style scaffolding (ThreadPool +
  write-lock, append+flush, **resume by (task_id, sample_idx)**, regen loop) around a manual multi-turn agent
  loop: teacher → **prompt-and-parse** (`<tool_call>{json}</tool_call>` parsed by us; native `tool_calls` field
  also accepted) → env.use_tool (real observations, errors as tool messages) → repeat → inline verifier scoring.
  `<think>` stripped from context; `<plan>` kept (Variante C). ALL rollouts written with score → yield measurable.
  German system prompt = policy.md + concise-plan nudge + tool-call format.
- **✅ CPU smoke (oracle, no GPU):** scripted oracle teacher emits `<tool_call>` TEXT (exercises the real parser)
  over bakeoff_dev: **25/25 verified (100% yield)**, replan-rate 24% (= injected share), avg 4.3 turns;
  **broken oracle (hallucinating): 0/5 (0%)**. Resume verified (rerun → todo=0). Sample trace has the exact
  target shape: system→user→assistant(plan+tool_call)→tool→assistant→tool→assistant(final).
- **Config:** new `db_bahn` + `trajectory` blocks in pipeline_config.yaml (incl. `teacher_candidates` bake-off
  list with per-candidate quant/serve flags).
- **Note:** tau2 venv has no mlflow → tracking deferred to the bake-off runner (best-effort there).

## 2026-07-03 — Phase 2: task generation built + smoke PASSED

- **`sdg_pipeline/db_bahn/gen_tasks.py`:** 10 German templates (6 INFO + 4 ACTION) over the frozen world-state,
  each task with a built-in machine-checkable answer-key (KAG principle). **550 tasks** (55/template),
  **119 fault-injected** (inject_verspaetung / inject_lokfuehrer_ausfall via `initialization_actions`),
  0 near-dups (unique (template, Zugnummer)). Splits **bakeoff_dev 25 / heldout_eval 40 / sft_train 485**,
  disjoint-by-construction + hard-fail assert. Content-derived task ids (no uuid) → **byte-reproducible**
  (verified: identical tasks.json across two runs). Files: tasks.json / split_tasks.json / **answer_keys.json**
  (side-channel for our Phase-4 grounding checker; tau2 sees only its own schema).
- **API facts verified in tau2 source:** `run_env_function_call` uses `getattr(toolkit, name)` → injections can be
  plain NON-tool methods (agent can't call them); the evaluator applies `initialization_actions` to BOTH predicted
  and gold envs before replaying reference actions → injections stay consistent between rollout and target-hash.
- **Reward wiring per task kind:** ACTION → reward_basis [DB, ENV_ASSERTION] (reference actions + assert_*);
  INFO → [COMMUNICATE] with few distinctive substrings (station/employee/order-ids/cause words — no weak numeric
  substrings), strict fact-check deferred to our own verifier (P0-1/P0-2 fix).
- **✅ Phase 2 smoke PASSED:** 550/550 `Task.model_validate`; splits disjoint+complete; **ACTION replay 12/12**
  (init → reference actions → env_assertions pass → DB-hash changed); injected INFO facts == answer key;
  German tickets read correctly. (One smoke-script selector bug fixed — generator itself was correct.)

## 2026-07-03 — Phase 1: tau2 `db_bahn` domain built + smoke PASSED

- **Env:** host has **Python 3.12.3 (aarch64)**; tau2-bench cloned (pin commit **1901a30**, MIT) into scratchpad,
  `pip install -e` into an isolated 3.12 venv — **installs cleanly on aarch64** (pure-Python deps). No Docker needed
  for dev; a `Dockerfile.tau2` (Py-3.12) is the later reproducible artifact.
- **Decision — runtime registration:** tau2's `registry` is a global singleton with `register_domain/register_tasks`.
  We keep ALL `db_bahn` code in **our repo** (`sdg_pipeline/db_bahn/tau2_domain/{data_model,tools,environment,__init__}.py`
  + `policy.md`) and register on import — **never edit the pip-installed tau2 source**. Data (db.json/tasks.json) lives
  under `$DB_BAHN_DATA` (default `data/raw/db_sandbox`, gitignored); `policy.md` is authored source in the repo.
- **Tools:** 5 READ (`fahrplan, verspaetung, zugstandort, wartung_status, mitarbeiter_info`) + 3 WRITE
  (`wartung_einplanen, crew_zuweisen, wartung_status_setzen`) + 3 `assert_*` for env_assertions. German docstrings →
  German tool schema. (P0-1 fix: WRITE tools give the DB-state gate real teeth.)
- **✅ Phase 1 smoke PASSED** (tau2 venv + repo on PYTHONPATH, `LOGURU_LEVEL=ERROR`): domain+taskset registered;
  `get_environment(solo_mode=True)` builds; 8 tools with valid OpenAI schema (name/description/parameters); READ tools
  return correct German data (fahrplan 7 Halte, verspaetung, 3 crew); **WRITE `wartung_einplanen` mutates the DB and
  `db.get_hash()` changes** → the deterministic DB-state reward works; `assert_maintenance_exists` True.
- **Note:** `BahnDB` is `BaseModelNoExtra` → db.json must match fields exactly; seeder now emits entity tables as
  dicts-by-pk + a `meta` field (no leading-underscore key). db.json re-seeded, still byte-reproducible.

## 2026-07-03 — Phase 0: gtfs.de de_fv inspected (open items resolved)

- **Downloaded** `de_fv/latest.zip` (396 KB, CC-BY-4.0) → `data/raw/db_sandbox/gtfs_de_fv/`. Feed generated 2026-06-27,
  9 files: agency/feed_info/stops/routes/calendar/calendar_dates/trips/stop_times/attributions.
- **Open items resolved on the real file:**
  - `stops.txt.stop_id` = **gtfs.de internal integer** (e.g. 22776), **not EVA/IFOPT**. Real station names + lat/lon
    present. **523 parent stations** (location_type=1) out of 1239 rows → use parents as the `stations` table.
  - **`shapes.txt` is ABSENT** → no polylines → `zugstandort` positions are **interpolated from stop_times / synthetic**,
    labeled mock, excluded from answer-keys (as the plan foresaw).
  - `trips.txt` has **no train number** (only route_id/service_id/trip_id) → **synthesize a deterministic Zugnummer**
    per trip. `routes.txt.route_short_name` = product/line (ICE, IC, EC, ECE, RJ, EN; some like "ICE 42").
- **Scale:** ~5479 trips, 52834 stop_times, 848 services; per-product: ICE ~313, IC ~236, RJ ~158, EC ~110 base routes.
  Calendar window from 20260627 (7-day). **Decision:** freeze `SIM_DATE = 2026-06-29` (Monday, in-window) + `SIM_NOW = 12:00`.
- **Decision:** seeder emits `db.json` (tau2 world-state) from real (stations/lines/trips/schedule/calendar) + sha256-seeded
  synthetic (zugnummer, delays, positions, vehicles, maintenance_orders, employees, shifts, assignments); reproducible via
  `random.Random(sha256(SEED|table|pk))`; standalone (no tau2 import) so it runs under any Python.
- **✅ Phase 0 smoke PASSED** (`sdg_pipeline/db_bahn/seed_worldstate.py`): `db.json` (4.7 MB) builds; **byte-reproducible**
  (identical sha256 across two runs). Counts @ SIM_DATE 2026-06-29 / SIM_NOW 12:00: stations 576, lines 97, trips 1070,
  schedule 9650, delays 9650, positions 245 (en-route), vehicles 548, maintenance 450, employees 2140, assignments 2678.
  Sample coherent (e.g. "ICE 1562", real coords, German delay remarks).

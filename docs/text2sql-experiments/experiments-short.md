# Experiments Short

Kurzübersicht (neueste oben). Details: [experiments.md](experiments.md) · Baselines: [experiments-baselines.md](experiments-baselines.md) · Limits: [experiments-limits.md](experiments-limits.md) · Hardware: [experiments-hardware.md](experiments-hardware.md).

---

## GRPO-Pilot durch — +2 Pkt auf schwachen Kategorien (2026-06-27)

verl GRPO (LoRA) auf dem 4B-thinking-Studenten: **50/50 Steps, ~11 h, 0 Wedges**. Lift klein aber sauber, **zwei verschiedene Sets, beide +2**: Held-out-Val **0.4208→0.4417 (+2,1 Pkt, weak_test_clean 240)**; Per-Kategorie-Offline-Eval (vLLM, test_clean 100) **59→61 %** — **set-ops +1, subqueries +1** (Ziel-Kats), **window flach** (härteste Kat, erwartbar bei lr 1e-6), keine Regression. Pipeline bewiesen, Effekt klein → stärker via höhere lr/mehr Steps. Detail + Recipe: [experiments-verl_RL_lora-grpo.md](experiments-verl_RL_lora-grpo.md).

**Wedge-Korrektur (2026-06-30, verifiziert):** der GB10-„Wedge" war NICHT cudagraph, sondern der FlashInfer-top-k/top-p-Sampler-Race (vLLM #43885) — Fix `VLLM_USE_FLASHINFER_SAMPLER=0`; `enforce_eager` war ein Red Herring, mns nur Frequenz-Hebel. Pilot lief unter mns=16/eager (Ergebnis gilt); künftige Läufe: cudagraph AN + Sampler aus + mns=32 (schneller).

---

## ✅ Redo erfolgreich — Looping behoben, thinking konkurrenzfähig (2026-06-18)

Alle 10 Studenten auf **721 saubere Traces** re-trainiert + unter **reinem greedy (kein penalty)** evaluiert. **`</think>`-Close-Rate 92–99 %** (kaputt: 4–44 %) → Looping behoben. **EX:** sauberes thinking gewinnt bei **4B (0.61 vs 0.58)**, Gleichstand **9B (0.63)**; nothink führt an den Rändern (0.8B/2B, **14B 0.64** = bestes Einzel-EX). **Lehre:** die frühere thinking-Niederlage war **Daten-Vergiftung**, nicht fehlender Reasoning-Nutzen. Voll: [experiments.md](experiments.md).

---

## ⚠️ Distill-Serie 1 KAPUTT + archiviert (2026-06-11) — durch Redo aufgelöst ↑

Die erste trainierte Serie (Qwen3.5 0.8B–9B + Qwen3-14B × thinking/nothink) liegt in `archive/broken_distill_2026-06-11/`. **thinking-Studenten loopen** (Repetition bis Token-Limit), Wurzel = **schlechte SDG-thinking-Traces** (verbose thinking-Teacher, „Wait" 2945× / „Actually" 2407×). nothink war sauber. Voller Befund + Redo-Plan (1a): [experiments.md](experiments.md), [experiments-limits.md](experiments-limits.md).

**Was als Aussage hält:** nothink ist der Deploy-Pfad (sauber, schneller, EX ≥ thinking); bestes Qualität/Param **Qwen3.5-4B nothink**. Ob *sauberes* thinking konkurriert, klärt der Redo. (Die alten thinking-Zahlen waren loop-gedrückt → nicht zitieren.)

---

## Clean-Baselines (untrainiert, n=100, rescored 2026-05-29)

Kontrollgruppe (untrainierte Base-Modelle) auf `test_clean.jsonl`. Details: [experiments-baselines.md](experiments-baselines.md).

**Teacher-Kandidaten (groß, non-th außer markiert):** 14B 59% · 32B 62% · 72B 60% · 27B(th) 57% · 35B-A3B(th) 63% — **Decke flach**, kein Paar signifikant (McNemar p>0.15). → **14B-Teacher genügt.**

**Student-Kandidaten (Baseline EX loose):** 0.5B 35 · 1.5B 37 · 2B(th) 51 · 3B 56 · 4B(th) 60 · 7B 60 · 9B(th) 60 · 14B(th) 62 — **Knie bei ~4B**, ab da flach. Baseline sagte schon: thinking lohnt nicht, schadet auf Mini-Modellen.

> **Der (archivierte) nothink-Lauf bestätigte die Baseline-Prognose** — 4B als Sweet-Spot, Decke ~60%, monoton flach ab 4B. Die thinking-Zahlen des Laufs sind loop-kontaminiert (s. Callout oben) → nicht zitieren; saubere thinking-Zahlen kommen aus dem Redo.

**Daumenregeln (robust):**
- **non-thinking** ist der Deploy-Default (sauber, EX ≥ thinking, vielfach schneller).
- EX-Decke ~60 % ist daten-/metrik-limitiert, nicht modellgrößen-limitiert (ab 4B flach).
- Distillation hebt die gesättigte Benchmark kaum — der Wert liegt im deploybaren nothink-Verhalten.

**Offene TODOs:** Redo 1a (saubere SDG-Traces → re-train → re-eval); SDG-Anreicherung window functions / set operations; fairere EX-Metrik (Spalten-Teilmengen / Multi-Gold); härteres, größeres Test-Set (n≫100).

# Übersicht: SFT-Training DB-Agent (Qwen3-4B)

**Kontext:** Wir trainieren einen 4B-SLM (Qwen3-4B, dense/thinking) per LoRA zum
agentischen DB-Assistenten — erst **SFT** (Verhalten klonen), später **RL**
(GRPO, Belohnung über den tau2-Verifier). Läuft lokal auf einer **NVIDIA GB10**.

---

## 1. Daten für SFT (der „Mix")

Gemischter Datensatz aus 3 Quellen: **15.687 Train + 301 Val**
(unified Chat-Format, Loss nur auf Assistant-Turns).

| Quelle | Train | Anteil Tokens | Was / wofür |
|---|---|---|---|
| **db_bahn** | 8.964 | **59 %** | Selbst-synthetisierte, verifizierte deutsche DB-Agent-Traces (Kern-Domäne), ≤5,9k Tokens |
| **AReaL (τ²-bench)** | 2.013 | 34 % | tau2-Dialoge (Airline/Retail/Telecom), lange Episoden **bis 12k Tokens** |
| **ToolACE** | 4.710 | 6 % | Tool-/API-Breite + „Irrelevanz"-Fälle, kurz (~0,8k) |

- 58 % Multi-Turn (≥3 Assistant-Züge). Val pro Quelle stratifiziert.

## 2. Daten für RL (später) — Task-Pools, kein Traces-Training

Beim RL rollt das Modell selbst aus, Belohnung kommt vom Verifier:

| Quelle | RL-Tasks | Domäne |
|---|---|---|
| **db_bahn** (`rl_train`) | 998 | Deutsche DB |
| **AReaL (τ²-bench)** (`tau2_rl_train`) | 1.982 | Airline / Retail / Telecom |
| **Summe** | **~2.980** | 4 Domänen |

## 3. Daten für Evaluation

- **`heldout_eval`: 276 Tasks — reine db_bahn-Evaluation.** 
- Benchmarks folgen

---

## 3b. Ergebnis (Stand 15.07.2026) — das SFT hat funktioniert

Training: 2 Epochen in **41,7 h** (752 Traces/h). Eval auf `heldout_eval` (276 Tasks, nie trainiert):

| Modell | verified yield | gelöst |
|---|---|---|
| Base Qwen3-4B (untrainiert) | 89,1 % | 246/276 |
| SFT Epoche 1 | 98,6 % | 272/276 |
| **SFT Epoche 2** ⭐ (gewählt) | **99,6 %** | **275/276** |

**+10,5 pp klingt wenig — ist aber irreführend.** Das Base-Modell konnte **18 der 26 Aufgabentypen schon zu
100 %**; die verwässern den Schnitt. Auf den **8 Aufgabentypen mit echtem Spielraum (91 Tasks): 67 % → 98,9 %**.
Darunter **drei Typen, die das Base-Modell überhaupt nicht konnte** (0 % → 100 %). **Keine einzige
Verschlechterung** — alles, was vorher schon lief, läuft weiter.

Nebenbefund: die Replan-Fähigkeit bleibt erhalten (`replan_rate` ~0,39), aber die Selbstkorrektur-Rate fällt
auf 0 — der trainierte Agent macht die Fehler gar nicht mehr, aus denen er sich retten müsste.

---

## 4. Was wir optimiert haben

**Speed (Software — voll ausgereizt):**
FlashAttention-2 · **Liger-Kernel** (fused Cross-Entropy, verhindert einen
55-GB-Tensor → größere Batch @12k passt erst dadurch) · `group_by_length`
(weniger Padding) · größere Batch (micro 8) + fused Optimizer.
- **softwareseitig ist auf dem GB10-Chip praktisch kein Speed mehr rauszuholen**.

**Qualität:**
LoRA-Rang **r=32** (von 16 → mehr Kapazität) · **NEFTune** (leichtes
Embedding-Rauschen (konservative 5%) → bessere Generalisierung) · **Checkpoint-Auswahl** (Epoche 1
& 2 gespeichert, der bessere gewinnt per Eval) · 12k Kontext · 2 Epochen ·
MLflow-Tracking.

> **NEFTune-Quelle:** Jain et al., *„NEFTune: Noisy Embeddings Improve Instruction
> Finetuning"*, arXiv:**2310.05914** (2023, ICLR 2024). Kernbefund: LLaMA-2-7B auf
> Alpaca → AlpacaEval-Win-Rate **29,8 % → 64,7 %**. Wirkt nur zur Trainingszeit,
> kein Serving-Overhead.

---

## 5. Warum das Training so lange dauert (41,7 h für 2 Epochen)

**a) Datensatz — die langen AReaL-Episoden:** bis **12.000 Tokens**;
Attention-Aufwand wächst **quadratisch** mit der Länge → ein langer Step **~89 s**
vs. ein kurzer **~7,6 s** (Faktor ~12). Nur 34 % der Daten, aber der Großteil der
Zeit.

**b) Hardware — GB10:** kompakter **Dev-/Kapazitäts-Chip**, kein
Trainings-Beschleuniger. Unified-Speicher (LPDDR5X, ~273 GB/s) hat **~12× weniger
Bandbreite** als eine H100 (3,35 TB/s) → real **~7-8× weniger Trainings-Durchsatz**.

> **Vergleich:** Denselben Lauf schafft eine Cloud-**H100 in ~8-11 h** (~$20-30).

**Fazit:** Der Lauf ist gesund und korrekt (Loss fällt sauber, kein Overfit) — nur
langsam, weil lange Daten × langsamere Dev-Hardware. Erwartbar, kein Fehler.

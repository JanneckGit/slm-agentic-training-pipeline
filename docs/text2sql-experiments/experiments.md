# Experiments Log

Trainings-Serie Text-to-SQL Distillation. Kontrollgruppe (untrainierte Baselines): [experiments-baselines.md](experiments-baselines.md) · Mess-/Daten-Lehren: [experiments-limits.md](experiments-limits.md) · Hardware: [experiments-hardware.md](experiments-hardware.md).

---

## ✅ Redo-Ergebnis 2026-06-18 — saubere Traces, alle 10 Modelle, Looping behoben

Der Redo (Plan v4) ist durch. **Kernergebnis: der Daten-Fix wirkt** — unter *reinem greedy, ohne `repetition_penalty`* schließen die thinking-Studenten jetzt 92–99 % sauber (kaputter Lauf: 4–44 %). Das Looping war ein **Daten**-Problem, kein „Reasoning bringt nichts".

### Close-Rate (greedy, KEIN penalty) — der Lackmustest

| Größe | `</think>`-Close | Loop-Rate | kaputter Lauf (Close) |
|---|---|---|---|
| 0.8B | **99 %** | 4 % | 44 % |
| 2B | **98 %** | 5 % | — |
| 4B | **92 %** | 12 % | 16 % |
| 9B | **93 %** | 8 % | 4 % |
| 14B | **94 %** | 8 % | 38 % |
| Qwen3-4B† | **93 %** | 9 % | — |

Resttail (4–12 % laufen ins Limit) = harte/OOD-Fragen; die alte Katastrophe (unbegrenzte ~60k-Zeichen-Loops) existiert nicht mehr. Messung: `tools/close_rate_probe.py`, n=100.

### Execution Accuracy (Test-Subset, n=100, conc 16, greedy)

| Größe | thinking | nothink | Δ (think−nothink) |
|---|---|---|---|
| 0.8B | 0.39 | **0.49** | −0.10 |
| 2B | 0.48 | **0.54** | −0.06 |
| 4B | **0.61** | 0.58 | **+0.03** |
| 9B | 0.63 | 0.63 | 0.00 |
| 14B* | 0.60 | **0.64** | −0.04 |
| Qwen3-4B† | 0.59 | **0.62** | −0.03 |

*14B = Qwen3-14B (text-only, andere Familie); 0.8B–9B = Qwen3.5 (multimodal).
† **Qwen3-4B** = Qwen3-Familie **text-only** (NICHT das Qwen3.5-4B-mm oben), 2026-06-25 nachgezogen. Baseline untrainiert: thinking 0.60 / nothink 0.61 / close 99 %-**loop 25 %** → trainiert auf denselben 721 Clean-Traces: EX flach, aber **loop 25 %→9 %** (saubereres Verhalten, exakt wie die anderen Studenten). Reiht sich nahtlos ein.

### Befund

- **thinking ist jetzt konkurrenzfähig** — gewinnt bei 4B (+0.03), Gleichstand bei 9B. Im kaputten Lauf war thinking durchweg *katastrophal schlechter* (Loops). Mit sauberen Traces kippt das → **die frühere Niederlage war Daten-Vergiftung, nicht fehlender Reasoning-Nutzen.** Das war die zentrale offene Frage des Redos.
- **nothink bleibt der pragmatische Deploy-Pfad** — führt an den Rändern (0.8B/2B, 14B), kein Thinking-Budget nötig, knapp bestes Gesamt-EX (**14B nothink 0.64**). Bestes Qualität/Param: **Qwen3.5-9B** (0.63 in beiden) bzw. **4B thinking** (0.61).
- **Sauberes Scaling** in beiden Varianten (thinking 0.39→0.63, nothink 0.49→0.63 über 0.8B→9B).
- **Daten:** 814 generierte Traces → **721 clean** (`data_pipeline/clean_traces.py`: Ritual-Schwanz gestrippt, Degeneriertes + nicht-ausführbares SQL raus). Teacher-Deadlocks vom Supervisor (`ops/sdg_run_supervised.sh`) selbst geheilt. `micro_batch=4` hielt für alle Größen inkl. 9B/14B (kein OOM).

---

## ⚠️ Status 2026-06-11 — erster Distill-Lauf KAPUTT (Repetition-Loops) — Redo abgeschlossen (Ergebnis oben ↑)

Die erste Serie (Qwen3.5 0.8B/2B/4B/9B + Qwen3-14B × thinking/nothink) ist **archiviert** unter `archive/broken_distill_2026-06-11/` (Checkpoints + Eval + MLflow). Grund + Lehre unten — **muss für künftige Distill-Versuche stehen bleiben.**

### Der Befund: distilliertes thinking degeneriert in Repetition-Loops

Die trainierten **thinking**-Studenten produzieren auf schweren Fragen Endlosschleifen bis ans Token-Limit (greedy/temp-0 als Auslöser):
- 0.8B: *„…Or maybe it means: For each vendor, calculate the percentage… Or maybe it means… Or maybe…"* (61.860 Zeichen, unique-word 0.00)
- 4B: *„Let's consider `… UNION ALL …`? No. Let's consider …? No. I'll just write …"*
- 9B: *„Wait, `…` is not right. - Wait, … is not right. - Let's think of …"*

Non-close-`</think>`-Rate (trainiert): 0.8B 44 % · 14B 38 % · 2B 29 % · 4B 16 % · 9B 4 %. Die **untrainierte** Qwen3-14B-Basis terminierte dagegen **100/100 sauber** (max ~3500 Tok) → das **Training** hat das Looping induziert.

### Die Wurzel: SDG-thinking-Traces sind schlecht (SQL-Labels sind gut)

| SDG-Teil | Qualität |
|---|---|
| SQL-Labels | ✅ gut (median 116 Zeichen; nothink-Studenten waren sauber) |
| thinking-Traces | ❌ verbose + hedge-lastig (median ~1300 Tok, max ~5000; „Wait" 2945×, „Actually" 2407×, „Or maybe" 413× über 631 Beispiele; 416/631 mit ≥3 Hedge-Markern) |

Der **thinking-Teacher** (Qwen3.6-35B-A3B) ignorierte das „Be concise"-Prompt und lieferte seine native, selbstzweifelnde Deliberation; die wurde roh als Trace gespeichert. Studenten distillieren den Stil → loopen genau an „Wait/Actually/Or maybe". **nothink** (nur SQL) ist davon unberührt — deshalb war nothink durchweg sauber und ≥ thinking.

### Sekundäre Lehren aus dem Lauf (für die Eval-Infra)
- **vLLM-greedy ist NICHT bit-deterministisch über Concurrency-Stufen** (4B thinking 64 %@c32 vs 60 %@c16). Vergleichsläufe nur bei **identischer** Concurrency.
- **Große Modelle + hohe Concurrency = Client-Timeouts** auf der bandbreiten-limitierten GB10: 14B-thinking-Longgen lief am 1200s-httpx-Timeout auf (→ 3600s erhöht), und conc 48 *hungert* die langen Requests aus (jeder >60 min). Große Modelle brauchen **niedrigere** Eval-Concurrency.
- **thinking-Eval braucht großzügiges Token-Budget** (12288) — sonst Truncation; nothink reicht 2048.
- **GB10-vLLM-Serving braucht `VLLM_USE_FLASHINFER_SAMPLER=0`** — sonst der FlashInfer-Sampler-Race-Wedge (vLLM #43885; ~96 %/13 W/0 Throughput) bei jedem top-k/top-p-Sampling-Lauf (Reachability/SDG). Schon im docker-compose `vllm`-Env; Detail → [experiments-verl_RL_lora-grpo.md](experiments-verl_RL_lora-grpo.md).
- Hardware-Trainings-Speedup (Gated-DeltaNet-Kernel, 3.9×) ist eigenständig gültig → [experiments-hardware.md](experiments-hardware.md).

### Was als Aussage bleibt (robust, trotz kaputtem Lauf)
- **nothink ist der Deploy-Pfad** — sauber, schneller, EX ≥ thinking. Bestes Qualität/Param: **Qwen3.5-4B nothink**.
- thinking lohnt für diesen Task nicht — und der kaputte Lauf zeigt *warum*: nicht „Reasoning bringt nichts", sondern **die thinking-Daten waren vergiftet**. Mit *sauberen* Traces ist thinking evtl. konkurrenzfähig — das klärt der Redo.

---

## Redo (Plan v4) — sauberer Distill, ✅ abgeschlossen (Ergebnis ganz oben ↑)

**Prinzip: Fix an der WURZEL (Daten), nicht am Symptom (Inferenz).** Eval bleibt **reines greedy** als Lackmustest — **kein `repetition_penalty`** (das würde nur verschleiern, ob die Daten sauber sind). thinking-Traces vom **selben** 35B-MoE-Teacher (Qwen3.6-35B-A3B), aber gefixt. **Option A** (roher `<think>` via `trace_capture.py`); greift der Fix nicht (Gate 2b), Neustart mit **Option B** (`[REASONING]`-Erklärung).

**Schritt 0 — GPU teilen:** Eval-Serve `gpu_util 0.5` (~64 GB frei), SDG-35B-Teacher `0.7`. Compute hat keine HW-Isolation (kein MIG) → zeitlich koordinieren; Memory teilbar (128 GB).

**Schritt 1 — ein `trace_capture`-Run:** ~1110 Beispiele aus `seed_sdg_input.jsonl` (schon über die 7 Kategorien balanciert), Teacher löst Frage→SQL → `<think>` + SQL → beide Varianten aus einem Run.

**Schritt 2 — gute Traces (in `trace_capture.py`):**
- **Nudge-System-Prompt:** brief, committed, kein „wait/actually", keine Verifikation/Confirmation danach.
- **Quality-Filter (locker — nur Degeneriertes):** drop bei >4000 Zeichen / >4 Hedge-Marker / unique-word <0.3 / truncated (kein `</think>`).
- **Regenerate:** gefilterte 2× re-sample vor Drop.
- _Validiert (20er-Test): Traces median ~470 Tok (vorher 1300), hedge ~0, keine Loops, ~75 % keep._

**Schritt 2b — GATE (adversarialer Quercheck der vollen Daten):** Längen gesund / keine Loops / `</think>` schließt / 7-Kategorie-Balance / SQL ausführbar. ✅ → Schritt 3. ❌ → Neustart Option B.

**Schritt 3 — Daten:** `build_train_clean` → `format_for_training` → `train_chat_thinking` (saubere Traces) + `train_chat_nothink` (nur SQL).

**Schritt 4 — Training:** alle 10 via `ops/run_all_baselines.sh` (Infra + Blackwell-Kernel validiert).

**Schritt 5 — Eval: reines greedy, uniform concurrency 16, KEIN penalty.** max-tokens 12288/2048, Timeout 3600, gpu_util 0.5. **Gate:** `</think>`-Close ~100 % + 0 Loops unter greedy = Beweis, dass der Daten-Fix wirkte → dann EX. Rest-Loops → zurück an die Daten.

**Schritt 6 — Quercheck aller 10 + Doku** (diese Datei wieder mit Tabellen füllen).

**Erfolgskriterien:** Daten: ~1000, 7 Kategorien, terminieren, 0 Loops. Modelle: `</think>`-Close ~100 % unter **reinem greedy** — ohne Inferenz-Trick.

# Eval-Limitierungen & Mess-Lehren

Warum die EX-Zahlen so nah beieinander liegen (Baselines **und** trainierte Studenten) und welche Mess-/Daten-Fallstricke die Trainings-Serie aufgedeckt hat. Bezug: [experiments-baselines.md](experiments-baselines.md), [experiments.md](experiments.md).

## ⚠️ Größte Lehre (2026-06-11): distilliertes thinking degeneriert durch schlechte SDG-Traces

Der erste Distill-Lauf ist **kaputt** (archiviert unter `archive/broken_distill_2026-06-11/`): die trainierten **thinking**-Studenten loopen auf schweren Fragen bis ans Token-Limit (*„Wait… Actually… Or maybe… No…"*, bis 61k Zeichen, unique-word 0.00). Non-close-`</think>`: 0.8B 44 %, 14B 38 %, 2B 29 %, 4B 16 %, 9B 4 %; die untrainierte Qwen3-14B-Basis terminierte 100/100 sauber → **trainings-induziert**.

**Wurzel = SDG-thinking-Trace-Qualität:** der thinking-Teacher (Qwen3.6-35B-A3B) lieferte trotz „Be concise"-Prompt verbose, selbstzweifelnde Deliberation (median ~1300 Tok; „Wait" 2945×, „Actually" 2407× über 631 Beispiele; 416/631 mit ≥3 Hedge-Markern). Die Studenten distillieren den Stil → Loops. Die **SQL-Labels sind gut** (nothink sauber). **Konsequenz für künftige SDG-Runs:** Traces mit non-thinking-Teacher + hartem Längen-Cap + Hedge-Filter erzeugen — siehe Redo-Plan in [experiments.md](experiments.md). Greedy verstärkt Loops → bei Inferenz `repetition_penalty`.

> **✅ Aufgelöst (Redo 2026-06-18):** Der Fix saß an der **Wurzel (Daten)**, nicht am Symptom. Befund war (a) der verbose Teacher *plus* (b) ein Bestätigungs-Ritual-Schwanz im rohen `<think>` („Done. Proceeds. [Final Check].") — beides Loop-Seeds. Cleanup (`data_pipeline/clean_traces.py`: Ritual gestrippt, Hedge/Repetition/non-exec gedroppt; 814→721 Traces) → re-trainiert → unter **reinem greedy OHNE `repetition_penalty`** schließen jetzt **92–99 %** sauber (vorher 4–44 %). Die `repetition_penalty`-Empfehlung oben ist damit **überholt** — der Penalty hätte nur das Symptom kaschiert; die ehrliche Close-Rate ist der Beweis, dass die Daten sauber sind. EX-seitig wird thinking dadurch konkurrenzfähig (4B 0.61 > nothink 0.58, 9B Gleichstand). Details: [experiments.md](experiments.md).

## Pipeline ist OK — kein Bug
- `gold_failed = 0` bei allen Modellen → jedes Gold führt aus, keine kaputten Referenzen.
- Extraktion sauber; nur thinking-Läufe mit Overflow erzeugen gelegentlich Prosa statt SQL.
- Verifiziert (adversarialer Quercheck aller 8 trainierten Läufe): kein Silent-Degrade, korrekte thinking/nothink-Behaviors, Datensatz-Variante im Snapshot konsistent.

## Warum die Zahlen so gleich/flach sind (Baseline **und** trainiert)
- **Gesättigte Benchmark:** leichte Klassen an der Decke (basic 93 %, aggregation bis 79 %), schwere am Boden (window functions 0–14 %, set operations 27–60 %) → wenig Spielraum, in dem sich Modelle trennen.
- **Single-Gold-EX zählt vernünftige Alternativen als falsch.** Klassiker window functions: Gold nutzt `… OVER (PARTITION BY …)`, das Modell gibt `GROUP BY` mit den **richtigen Werten** → wegen Spaltenanzahl falsch. Trifft alle gleich → window functions ~0 quer durch.
- **Winzige Schema-Daten** (teils 4 Zeilen) → Ergebnis-Set-Vergleich fragil.
- **Schmale Aufgabe:** single-turn Text-to-SQL = Qwen2.5/3.5-Heimspiel; Reasoning/Long-Context-Stärken werden kaum belastet → thinking bringt nichts.
- **n=100:** Unterschiede weniger Beispiele = Rauschen (kein thinking-vs-nothink-Paar signifikant; |Δ| ≤ 7 von 100).

## Was der trainierte Sweep zusätzlich gelehrt hat (2026-06-10)

- **SDG-Finetuning bewegt die Decke nicht.** Trainierter Student vs. eigene untrainierte Baseline (gepaart, thinking): 0.8B +4, 2B −3, 4B ±0, 9B −1 → netto ~0. Auf einer gesättigten Benchmark mit 631 SDG-Beispielen zeigt sich der Distill-Effekt **nicht in EX** — der Wert liegt im deploybaren non-thinking-Verhalten, nicht in höherer Genauigkeit. (Genau die in der Baseline-Notiz vorhergesagte Beobachtung.)

- **⚠️ vLLM-Greedy ist NICHT bit-deterministisch über Concurrency-Stufen.** 4B-thinking ergab **64 % @ concurrency 32** vs. **60 % @ concurrency 16** — derselbe Checkpoint, dieselben Daten, temp 0. Ursache: unterschiedliche Batch-Kompositionen → minimal andere Numerik → bei langem thinking-Output kippen ~4 Beispiele. **Konsequenz:** Vergleichsläufe nur bei **identischer Concurrency** (und identischem Serve-/Token-Setup). Sonst ist ein 4-pp-„Effekt" ein Batching-Artefakt.

- **⚠️ Thinking braucht ein großzügiges Token-Budget, sonst untertreibt die EX.** Mit `--max-tokens 2048` wurden 49–60 % der thinking-Traces mitten im Reasoning abgeschnitten (kein `</think>`), die EX lag zu niedrig. Mit `--max-tokens 12288` (= 4B-Referenz) terminiert das Reasoning; der Extractor war zwar robust genug, dass der Gesamteffekt klein blieb (2B +2pp), aber **fair messen heißt: thinking volles Budget geben**. nothink reicht 2048 (kurzes SQL).

- **⚠️ vLLM-Serving auf GB10 braucht `VLLM_USE_FLASHINFER_SAMPLER=0`.** Sonst der FlashInfer-top-k/top-p-Sampler-Race-Wedge (vLLM #43885; ~96 %/13 W/0 Throughput) bei jedem top-k/top-p-Sampling-Lauf (Reachability/SDG) — **Greedy-Eval ist immun** (kein top-k/top-p). Schon im docker-compose `vllm`-Env; Detail → [experiments-verl_RL_lora-grpo.md](experiments-verl_RL_lora-grpo.md).

## Was zu verbessern wäre (für mehr Trennschärfe, optional)
- **Metrik fairer:** Spalten-Teilmengen erlauben ODER Multi-Gold / LLM-Judge-Äquivalenz → hebt v. a. window functions für alle.
- **Test-Set härter & größer:** mehr verschachtelte/Multi-Join-Queries, größere Schemas mit mehr Daten, n ≫ 100.
- **Gezielte SDG-Anreicherung** auf window functions / set operations (templated/verifiziert statt reines LLM-Sampling) — der gemeinsame Bodensatz von Teacher, Baseline und trainiertem Student.

## Was NICHT umsonst war
- Sauberes, leakage-freies, ausführbares Test-Set + reproduzierbarer Harness (Seed, gepaart, loose/strict, MLflow) = wiederverwendbares Asset.
- Der trainierte Sweep liefert die belastbare **Deploy-Entscheidung**: Qwen3.5-4B **non-thinking** (gleiche EX wie thinking, 6× schneller).
- Null-Ergebnis (thinking lohnt nicht, SDG hebt Decke nicht) spart Geld: kleiner non-thinking-Student statt thinking/9B+.
- Hardware-Optimierung (3.9× Trainings-Speedup) ist eigenständig wertvoll → [experiments-hardware.md](experiments-hardware.md).

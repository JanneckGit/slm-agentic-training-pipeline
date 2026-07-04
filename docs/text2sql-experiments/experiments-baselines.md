# Baselines: Untrainierte Modelle (Clean-Set)

> **Status 2026-06-18 — Kontrollgruppe (untrainiert), bleibt gültig.** Dieses Dokument ist die **Baseline** der Serie. Der erste trainierte Distill-Lauf war kaputt (thinking loopte, Wurzel = schlechte SDG-Traces); der **Redo ist durch** ([experiments.md](experiments.md)): saubere Traces → Looping behoben (Close **92–99 %**), und sauberes thinking ist bei **4B/9B EX-konkurrenzfähig** (4B 0.61 > nothink 0.58, 9B Gleichstand). Die Baseline-Prognose „thinking lohnt nicht" galt für *loop-gedrückte* Zahlen → mit sauberer Distillation ist thinking auf Augenhöhe; **nothink bleibt der pragmatische Default** (führt an den Rändern, kein Thinking-Budget). Was die Baseline robust richtig sah: Knie/Decke bei ~4B, gesättigte Benchmark (EX bewegt sich kaum, trainiert wie untrainiert).

## Zweck

Diese Datei dokumentiert die **Kontrollgruppe** zur Trainings-Serie in [experiments.md](experiments.md): die **untrainierten** Base-Modelle, evaluiert auf `data/final/test_clean.jsonl` (**n=100**), mit festem **Seed (42)** und derselben [evaluation/evaluate.py](../evaluation/evaluate.py) wie alle Runs. Dies ist die **erste belastbare Baseline der ganzen Serie**. Neben den ursprünglichen **Qwen2.5-Instruct** (non-thinking) sind jetzt auch die **Qwen3 / Qwen3.5 / Qwen3.6** Reasoning-Modelle (thinking, `--enable-thinking`) auf demselben Set evaluiert — sauber getrennt, weil thinking und non-thinking nicht 1:1 vergleichbar sind.

Was dieses Set sauber macht:
- **7 Komplexitätsklassen** (CTEs bewusst ausgeschlossen — der gretelai-Datensatz hat dort nur PostgreSQL-DML-CTEs `WITH … AS (UPDATE/DELETE/INSERT …)`, in SQLite nicht ausführbar und für Text-to-SQL off-task).
- **Jedes Gold-SQL ist in SQLite ausführbar und liefert ≥1 Ergebniszeile** — kein „ungewinnbares" Beispiel. Die theoretische Decke je Klasse ist damit **100%**; jede Abweichung ist echte Modell-Fähigkeit, kein Datenartefakt.
- **Leakage-frei** — `test_clean.jsonl` ist über `(question, sql)`-Dedup aus train/eval ausgeschlossen ([mix_datasets.py](../data_pipeline/mix_datasets.py)).

> **Ersetzt alle früheren Baseline-Zahlen.** Die vorherigen Werte standen auf einem kontaminierten Set, bei dem **40% der Gold-SQLs nicht ausführbar** waren (CTEs sogar 14/15). Kaputte Golds zählen für jedes Modell als Fehlschlag und **verwischen so echte Modell-Unterschiede**. Die Zahlen hier sind nicht mit den archivierten Baselines vergleichbar.

---

## Methodik / Provenance (reproduzierbar)

Alle Zahlen unten stammen **ausschließlich** aus `data/final/eval/<modell>/clean_*.json` (Aggregat) bzw. `clean_*_predictions.jsonl` (per-Beispiel). Stand: **rescored 2026-05-29**.

- **Test-Set:** `data/final/test_clean.jsonl`, **n=100**, **Seed=42**. Das Set ist über alle 14 Modelle **identisch** (verifiziert: die 100 `(question, gold_sql)`-Paare sind set-gleich). Vergleiche sind damit **gepaart**. Achtung: die Zeilen-**Reihenfolge** im `predictions.jsonl` unterscheidet sich zwischen der Qwen2.5-Familie (+ Qwen3-14B) und der Qwen3.5/3.6-Familie — ein per-`index`-Vergleich über Familien hinweg ist falsch; gepaart wird **über den Fragetext**, nicht über den Index.
- **Metrik EX (Execution Accuracy):** Ergebnis-Set-Gleichheit nach Ausführung in SQLite.
  - **loose (kanonisch, Headline):** zeilen- **und** spaltenreihenfolge-insensitiv; Unterschiede in der **Spaltenanzahl** bleiben Mismatch.
  - **strict:** positionsgenau (Spalten in exakter Reihenfolge).
  - Beide Spalten werden mitgeführt; loose−strict ist hier durchweg ≤ 2pp.
- **Rescore (Extraktor-v2 + loose-Metrik):** Punktevergabe wurde für **alle** Läufe — inkl. Qwen2.5 — neu bewertet. Der v2-Extraktor entfernt Markdown-Fences (```` ```sql … ``` ````) und zieht SQL robust aus Klartext/Reasoning. Das **löst zwei alte Artefakte** (s.u. 1.5B, 7B).
- **thinking AN/AUS** (am `raw_output` verifiziert): Qwen2.5-* = **non-thinking** (kein Reasoning, ø ~150 Zeichen Output). Qwen3/3.5/3.6 = **thinking** (Reasoning vorhanden; Qwen3-14B mit `<think>…</think>`-Tags, die Qwen3.5/3.6 emittieren Reasoning ohne öffnendes Tag, schließen mit `</think>`). **Caveat Overflow** (Reasoning läuft in `max_tokens`, `</think>` fehlt): Qwen3.6-27B 28%, Qwen3.5-2B 25%, Qwen3.5-0.8B 20%, Qwen3.6-35B-A3B 8%, Qwen3.5-4B/9B je 6%, Qwen3-14B 0%.
- **n=100 → 1 Beispiel = 1pp.** Pro Klasse n=14–15 → 1 Beispiel ≈ 6.7–7.1pp.

> **Warum alle Modelle so nah beieinander liegen** (gesättigte Benchmark, Single-Gold-EX, window functions ~0 = Metrik-Limit, kein Bug) und was zu verbessern wäre: siehe [experiments-limits.md](experiments-limits.md).

---

## Übersicht — non-thinking (Qwen2.5, n=100)

| Modell | Params | EX (loose) | EX (strict) | EM |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 0.5B | 35% | 35% | 8% |
| Qwen2.5-1.5B-Instruct | 1.5B | 37% | 35% | 7% |
| Qwen2.5-3B-Instruct | 3B | 56% | 56% | 12% |
| Qwen2.5-7B-Instruct | 7B | 60% | 59% | 9% |
| Qwen2.5-14B-Instruct | 14B | 59% | 59% | 8% |
| Qwen2.5-32B-Instruct | 32B | 62% | 61% | 16% |
| Qwen2.5-72B-Instruct-GPTQ-Int8 | 72B | 60% | 59% | 12% |

> **Rescore-Auflösung der alten Caveats (jetzt behoben):**
> - **1.5B:** stand vorher auf **0%** (Markdown-Fence-Extraktions-Artefakt — Modell umrahmte SQL mit ```` ```sql … ```` , alter Extraktor entfernte es nicht → SQLite-Syntaxfehler). Der v2-Extraktor entfernt die Fences → **37% loose** (echte Fähigkeit, wie damals vermutet).
> - **7B:** stand vorher auf **41%** (partiell dasselbe Format-Artefakt, nicht-monoton unter dem 3B). Rescored **60% loose** — der Knick verschwindet, 7B liegt jetzt erwartungsgemäß auf 14B-Niveau.
> Die EX-Spalte ist jetzt monoton/plausibel; die früheren „unzuverlässig"-Markierungen für 1.5B/7B sind damit gegenstandslos.

---

## Übersicht — thinking (Qwen3 / 3.5 / 3.6, n=100)

| Modell | Params | thinking | EX (loose) | EX (strict) | EM |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.8B | ✅ | 39% | 38% | 5% |
| Qwen3.5-2B | 2B | ✅ | 51% | 51% | 10% |
| Qwen3.5-4B | 4B | ✅ | 60% | 60% | 14% |
| Qwen3.5-9B | 9B | ✅ | 60% | 59% | 13% |
| Qwen3-14B | 14B | ✅ | 62% | 62% | 15% |
| **Qwen3-4B**‡ | 4B | ✅ | 60% | 58% | 11% |
| Qwen3.6-27B | 27B | ✅ | 57% | 56% | 13% |
| Qwen3.6-35B-A3B | 35B (A3B aktiv) | ✅ | 63% | 63% | 12% |

(EX = Execution Accuracy, loose = Headline. EM = Exact Match — durchweg niedrig (5–16%), weil äquivalentes SQL syntaktisch variiert; EX ist die aussagekräftige Metrik. n=100 → 1 Beispiel = 1pp.)

‡ **Qwen3-4B** (Qwen3-Familie, **text-only**, ≠ Qwen3.5-4B-mm) — 2026-06-25 als Student aufgenommen. Als einzige zusätzlich mit **nothink-Baseline** gemessen (loose **61%** / strict 60% / EM 12%); die übrigen Qwen3-Baselines sind nur thinking. Trainiert (gleiche 721 Clean-Traces): EX flach (thinking 0.59 / nothink 0.62), aber **loop 25%→9%** — Details + Close-Rate in [experiments.md](experiments.md).

---

## Per-Komplexität-Matrix (EX loose, %) — theoretische Decke je Klasse = 100%

### non-thinking (Qwen2.5)

| Komplexität | n | 0.5B | 1.5B | 3B | 7B | 14B | 32B | 72B |
|---|---|---|---|---|---|---|---|---|
| basic SQL | 15 | 86.7 | 93.3 | 93.3 | 93.3 | 93.3 | 93.3 | 86.7 |
| single join | 14 | 71.4 | 50.0 | 78.6 | 85.7 | 78.6 | 85.7 | 85.7 |
| aggregation | 14 | 14.3 | 50.0 | 85.7 | 85.7 | 92.9 | 92.9 | 78.6 |
| subqueries | 14 | 35.7 | 28.6 | 42.9 | 50.0 | 50.0 | 42.9 | 50.0 |
| set operations | 15 | 13.3 | 6.7 | 26.7 | 26.7 | 33.3 | 33.3 | 40.0 |
| window functions | 14 | 7.1 | 0.0 | 7.1 | 14.3 | 0.0 | 14.3 | 21.4 |
| multiple_joins | 14 | 14.3 | 28.6 | 57.1 | 64.3 | 71.4 | 71.4 | 57.1 |

### thinking (Qwen3 / 3.5 / 3.6)

| Komplexität | n | 3.5-0.8B | 3.5-2B | 3.5-4B | 3.5-9B | 3-14B | 3.6-27B | 3.6-35B-A3B |
|---|---|---|---|---|---|---|---|---|
| basic SQL | 15 | 93.3 | 86.7 | 93.3 | 93.3 | 93.3 | 93.3 | 93.3 |
| single join | 14 | 57.1 | 50.0 | 71.4 | 71.4 | 64.3 | 64.3 | 71.4 |
| aggregation | 14 | 50.0 | 78.6 | 85.7 | 85.7 | 85.7 | 71.4 | 78.6 |
| subqueries | 14 | 28.6 | 42.9 | 57.1 | 57.1 | 71.4 | 57.1 | 64.3 |
| set operations | 15 | 26.7 | 33.3 | 33.3 | 26.7 | 33.3 | 26.7 | 46.7 |
| window functions | 14 | 0.0 | 7.1 | 14.3 | 21.4 | 21.4 | 21.4 | 14.3 |
| multiple_joins | 14 | 14.3 | 57.1 | 64.3 | 64.3 | 64.3 | 64.3 | 71.4 |

(Pro Klasse n=14–15 → 1 Beispiel ≈ 6.7–7.1pp; Klassen-Differenzen ≤ ~7pp sind 1-Beispiel-Rauschen.)

---

## Interpretation (Teacher-Entscheidung)

Teacher-Kandidaten = große Modelle, die den **gesamten** SDG-Satz (tausende Beispiele) erzeugen: **Qwen2.5-14B / 32B / 72B** (non-thinking) sowie **Qwen3.6-27B / 35B-A3B** (thinking). Bewertet wird **Label-Qualität gegen Durchsatz/Kosten**, nicht reines `argmax(EX)`.

### 1. Kein Teacher schlägt einen anderen signifikant (gepaart, n=100)

Die Overall-EX aller Teacher liegt in einem schmalen Band **57–63% loose**. Gepaart (über den Fragetext aligniert) sind die Abstände **Rauschen**:

| Vergleich (loose) | nur A richtig | nur B richtig | abweichend | net (A−B) | McNemar p |
|---|---|---|---|---|---|
| 14B vs 32B | 4 | 7 | 11 | −3 | ~0.55 |
| 14B vs 72B | 7 | 8 | 15 | −1 | ~1.0 |
| 14B vs 35B-A3B | 6 | 10 | 16 | −4 | ~0.45 |
| 32B vs 27B | 8 | 3 | 11 | +5 | ~0.23 |
| 27B vs 35B-A3B | 4 | 10 | 14 | −6 | ~0.18 |

**Kein einziger Vergleich ist signifikant** (alle p > 0.15, |net| ≤ 6 von 100). Der nominell höchste Wert (Qwen3.6-35B-A3B, 63%) liegt nur **4 Beispiele** über dem 14B (59%) — innerhalb des Rausch-Bands.

### 2. 14B → 32B → 72B bleibt flach (bestätigt, jetzt rescored)

`14B 59% → 32B 62% → 72B 60%` — Sprünge **+3 / −2pp**, reines n=100-Rauschen. Eine Ver-5-fachung der Parameter (14B→72B) bringt **+1pp**. 72B-GPTQ-Int8 fällt auf einzelnen Klassen sogar leicht hinter 14B zurück (aggregation 78.6% vs. 92.9%, multiple_joins 57.1% vs. 71.4% — plausibel Int8-Quant-Degradation), jedenfalls kein Vorteil. Die frühere Schlussfolgerung **„14B genügt, 32B/72B kein Gewinn" gilt unverändert** und ist nach Rescore sauberer denn je.

| Modell | kontaminiert (n=88, historisch) | clean rescored (n=100, loose) |
|---|---|---|
| 14B | 45% | 59% |
| 32B | 44% | 62% |
| 72B | 43% | 60% |

Das Clean-Set hebt das **absolute** Niveau um ~+15pp (die unlösbaren 40%-Golds sind weg), aber die **relative** Reihung 14B≈32B≈72B bleibt flach → die flache Teacher-Decke ist **real**, kein Mess-Artefakt.

### 3. Die neuen Qwen3.6-Thinking-Zahlen überschreiben das nicht

Qwen3.6-35B-A3B (63%) ist der nominell beste Teacher, aber (a) der Vorsprung auf 14B ist **nicht signifikant** (net −4, p≈0.45) und (b) für **Bulk-SDG** ist der Kostenunterschied drastisch: thinking-Modelle grübeln pro Beispiel ~6.800–9.500 Zeichen (vs. ~150 beim non-thinking 14B) und **terminieren teils gar nicht** (27B 28% Overflow, 35B-A3B 8%). Für tausende Beispiele ist das ein Vielfaches an Generierungszeit/Kosten **ohne** verlässlichen Qualitätsgewinn. Qwen3.6-27B ist zudem der **schwächste** Teacher (57%) bei **gleichzeitig höchstem** Overflow (28%) → dominiert, fällt raus.

Nebeneffekt der Rolle: ein **non-thinking** Teacher liefert direkte SQL-Labels (passend für einen non-thinking-Deploy-Student); ein thinking-Teacher müsste man auf den finalen Query reduzieren.

### 4. Per-Klasse: wo trennt sich überhaupt etwas?

- **window functions:** der einzige Punkt, an dem 14B klar schwächer ist — **14B = 0/14**, während 32B/72B/27B/35B-A3B **2–3/14** holen. Aber: **alle** Teacher sind hier desaströs (Decke 0–21%). Der Unterschied ist 0 vs. 2–3 Beispiele — kein Modell „kann" window functions.
- **set operations:** 35B-A3B am besten (46.7%), 72B 40%, Rest 27–33% — ebenfalls flächendeckend schwach.
- **subqueries:** thinking leicht vorn (27B/35B-A3B 57–64% vs. non-thinking 43–50%).
- **aggregation / single join:** non-thinking 14B/32B vorn (14B aggregation 92.9%, 32B single join 85.7%); 27B fällt bei aggregation auf 71.4% ab.
- Netto heben sich die thinking-Stärken (subqueries) und -Schwächen (single join, aggregation) gegen die non-thinking-Profile auf → identische Overall-Decke.

### 5. Empfehlung Teacher

**Qwen2.5-14B-Instruct bleibt der SDG-Teacher.** Begründung:
- Im gepaarten Test schlägt **kein** Kandidat das 14B signifikant; die Spanne 57–63% ist Rauschen.
- Für die Teacher-Rolle (Bulk-SDG, tausende Beispiele) ist **Durchsatz/Kosten** der Tiebreaker → schnelles non-thinking-Modell. 14B ist das günstigste, das die ~60%-Decke erreicht.
- thinking-Teacher (27B/35B-A3B) liefern trotz nominell 57–63% **keinen** signifikanten Mehrwert, kosten aber pro Beispiel ein Vielfaches und überschreiten teils `max_tokens`.

**Marginaler Upgrade-Pfad (optional, nicht erforderlich):** Wer maximale non-thinking-Qualität will, nimmt **32B** (62%, net +3 vs. 14B, NS) — minimal bessere window functions/single join, aber ~2,3× Compute. **Strukturelle Schwachpunkte bleiben window functions (selbst 35B-A3B nur 14%, 72B 21%, 14B 0%) und set operations** — diese Klassen löst **kein** Teacher zuverlässig und sind die sinnvollsten Ziele für gezielte SDG-Anreicherung (templated/verifizierte Generierung statt reiner LLM-Sampling).

---

## Interpretation (Student-Entscheidung)

Student-Kandidaten = kleine, fine-tunebare und **deploybare** Modelle: Qwen2.5-0.5B/1.5B/3B/7B (non-thinking), Qwen3.5-0.8B/2B/4B/9B + Qwen3-14B (thinking). Ziel: **kleinste Größe, deren Baseline trägt** — denn der Student wird deployed, klein/günstig zählt.

### 1. Die Größen-Kurve hat einen klaren Knick bei ~4B

`0.5B 35% → 1.5B 37% → 2B 51% → 3B 56% → 4B 60% → 7B 60% → 9B 60% → 14B 62%`

- **0.5B ≈ 1.5B** (35/37%, gepaart NS) — beide zu schwach.
- **0.8B → 2B** ist ein **echter** Sprung (39%→51%, gepaart net +12, McNemar p≈0.02, **signifikant**).
- **2B → 4B** (+9pp, net +9, p≈0.06) — knapp unter Signifikanz, aber klarer Trend.
- **ab 4B flach:** 4B = 7B = 9B = 60%, 14B 62% (alle Abstände NS). **Die ~60%-Decke der 14B/32B/72B-Teacher wird bereits bei 4B erreicht.**

→ Bestes **Qualität-pro-Parameter** liegt bei **4B** (Knie der Kurve), dicht gefolgt von 3B.

### 2. Thinking lohnt hier nicht — und schadet auf kleinen Modellen

Bei gleicher/ähnlicher Größe bringt thinking **keinen** EX-Vorteil, kostet aber massiv Inferenz-Tokens:
- **9B-think (60%) = 7B-non-thinking (60%)** — gepaart net 0. thinking kauft bei 7–9B exakt nichts, generiert aber ~4.700 statt ~165 Zeichen/Beispiel.
- **2B-think (51%) < 3B-non-thinking (56%)** (gepaart net −5) — das non-thinking 3B schlägt das thinking 2B.
- **Auf Mini-Modellen schädlich:** Qwen3.5-0.8B (39%) und 2B (51%) laufen in **20–25% der Fälle in `max_tokens`** (kein `</think>`), und ihre **closed-Genauigkeit** (0.8B 44%, 2B 61%) liegt deutlich über der unclosed-Genauigkeit (je 20%) — d.h. ein großer Teil der Calls verschwendet das Budget auf nicht-terminierendes Reasoning. Für einen deployten Student ist das ein K.O.-Kriterium.

### 3. Per-Klasse-Profil der Top-Kandidaten (was das Finetuning heben muss)

| Klasse | Qwen3.5-4B | Qwen2.5-3B | Qwen2.5-7B |
|---|---|---|---|
| basic SQL | 93.3 | 93.3 | 93.3 |
| aggregation | 85.7 | 85.7 | 85.7 |
| single join | 71.4 | 78.6 | 85.7 |
| multiple_joins | 64.3 | 57.1 | 64.3 |
| subqueries | 57.1 | 42.9 | 50.0 |
| set operations | 33.3 | 26.7 | 26.7 |
| window functions | 14.3 | 7.1 | 14.3 |

basic SQL und aggregation sind bei allen schon „fertig". **Finetuning-Ziele: window functions, set operations, subqueries, multiple_joins** — exakt die Klassen, die auch die Teacher schwach abdecken (→ SDG-Anreicherung dort ist der gemeinsame Hebel für Teacher **und** Student).

### 4. Empfehlung Student

**Primär: Qwen3.5-4B.** Bestes Qualität-pro-Parameter — erreicht mit nur 4B die **60%-Decke** der 14B/32B/72B-Teacher und das beste Per-Klasse-Profil der kleinen Modelle (subqueries 57%, multiple_joins 64%). Es ist ein thinking-Modell mit **niedrigstem** Overflow seiner Klasse (6%); für günstigen Deploy das Finetuning auf **direkte SQL-Ausgabe** treiben und **non-thinking** servieren — die Daten zeigen, dass thinking hier keinen EX-Vorteil bringt (9B-think = 7B-nonthink), das Abschalten kostet also nichts Messbares und entfernt den Rumination-/Overflow-Aufschlag.

**Fallback (günstigster, voll non-thinking): Qwen2.5-3B-Instruct.** 56% loose bei 3B, kein Reasoning-Overhead, keine Overflow-Gefahr; der Abstand zum 4B ist ~4 Beispiele (NS). Erste Wahl, wenn der Deploy strikt non-thinking/minimal sein muss oder das 4B-Finetuning nicht trägt.

**Nicht sinnvoll als Student:** Qwen2.5-7B (60%), Qwen3.5-9B (60%), Qwen3-14B (62%) — kein nennenswerter Baseline-Gewinn über das 4B, aber spürbar größer/teurer. Qwen2.5-0.5B/1.5B und Qwen3.5-0.8B sind mit 35–39% zu schwach als Deploy-Basis.

---

## Quellen

non-thinking (Qwen2.5):
- `data/final/eval/Qwen2.5-0.5B-Instruct/clean_Qwen2.5-0.5B-Instruct.json`
- `data/final/eval/Qwen2.5-1.5B-Instruct/clean_Qwen2.5-1.5B-Instruct.json`
- `data/final/eval/Qwen2.5-3B-Instruct/clean_Qwen2.5-3B-Instruct.json`
- `data/final/eval/Qwen2.5-7B-Instruct/clean_Qwen2.5-7B-Instruct.json`
- `data/final/eval/Qwen2.5-14B-Instruct/clean_Qwen2.5-14B-Instruct.json`
- `data/final/eval/Qwen2.5-32B-Instruct/clean_Qwen2.5-32B-Instruct.json`
- `data/final/eval/Qwen2.5-72B-Instruct-GPTQ-Int8/clean_Qwen2.5-72B-Instruct-GPTQ-Int8.json`

thinking (Qwen3 / 3.5 / 3.6):
- `data/final/eval/Qwen3.5-0.8B/clean_thinking_Qwen3.5-0.8B.json`
- `data/final/eval/Qwen3.5-2B/clean_thinking_Qwen3.5-2B.json`
- `data/final/eval/Qwen3.5-4B/clean_thinking_Qwen3.5-4B.json`
- `data/final/eval/Qwen3.5-9B/clean_thinking_Qwen3.5-9B.json`
- `data/final/eval/Qwen3-14B/clean_thinking_Qwen3-14B.json`
- `data/final/eval/Qwen3.6-27B/clean_thinking_Qwen3.6-27B.json`
- `data/final/eval/Qwen3.6-35B-A3B/clean_thinking_Qwen3.6-35B-A3B.json`

(per-Beispiel: jeweils `…_predictions.jsonl` im selben Ordner.)

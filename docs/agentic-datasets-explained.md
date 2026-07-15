# Die Datensätze — einfach erklärt

> Was ist welcher Datensatz, wie sieht er aus, was bringt er dem Modell bei — und wie genau nutzen wir
> τ²-bench dabei. Alle Beispiele unten sind **echt** aus unseren Dateien (bzw. für τ²-bench aus dem Framework).
> Verwandt: [agentic-sft-data-basis.md](agentic-sft-data-basis.md) (Zahlen), [agentic-sft-db-synthesis.md](agentic-sft-db-synthesis.md) (DB-Design).
>
> ⚠️ **Der gebaute Mix hat 3 Legs, nicht 4: TaskBench ist NICHT im SFT-Mix** (→ Eval-Regal, Begründung §2.2).
> Dieses Doc erklärt weiterhin alle **vier** Datensätze — TaskBench bleibt als Vergleich + späteres Eval
> relevant. **Alle Zahlen + Ist-Stand:** [SFT-Training-Uebersicht.md](SFT-Training-Uebersicht.md).

---

## 0. Das große Bild in einem Satz

Wir bringen einem **kleinen Orchestrator-Modell** bei, eine Anfrage in Schritte zu zerlegen, die richtigen
**Werkzeuge** aufzurufen, deren Antworten zu lesen und **bei Überraschungen umzuplanen**. Dafür mischen wir
mehrere Datensätze, die jeweils *ein Teilstück* dieser Fähigkeit üben — und obendrauf einen selbst erzeugten,
deutschen, DB-spezifischen Satz.

```
                          Was der Agent können muss
   ┌───────────────┬────────────────┬──────────────────────┬─────────────────────┐
   │ Tool korrekt  │  Aufgabe in     │  bei Fehler/Über-     │  in UNSERER Domäne   │
   │ aufrufen &    │  Schritte       │  raschung UMPLANEN    │  (Bahn), auf DEUTSCH │
   │ Antwort lesen │  zerlegen       │  + selbst korrigieren │                      │
   └──────┬────────┴───────┬─────────┴──────────┬───────────┴──────────┬──────────┘
          │                │                    │                      │
       ToolACE         TaskBench          τ²-bench-Abläufe          db_bahn-Traces
     (Grundlagen)      (Planung)         (Replan/Korrektur)       (Domäne + Deutsch)
          │        ⚠ NICHT im Mix              │                      │
          │        (→ Eval-Regal, §2.2)        │                      │
          └───────────────────┬─────────────────┴──────────────────────┘
                              │
                    Stage-1 SFT (gemischt)   ← Zahlen: siehe Übersicht
                                       │
                             Stage-2 GRPO (RL)  ← neue, disjunkte Aufgaben, Reward = Verifier
                                       │
                                Eval: τ²-bench-Testsplit + db_bahn-Heldout + BFCL-V3
```

**Merksatz:** ToolACE = *Wortschatz*, τ²-bench-Abläufe = *auf Fehler reagieren*, db_bahn = *unser Dialekt
(Bahn, Deutsch)*. (TaskBench = *Grammatik* — die wir am Ende **nicht** trainieren, weil sie nur Pläne
notiert statt Tools auszuführen; §2.2.)

---

## 1. Die vier Datensätze im Vergleich

Jeder Datensatz „kann" andere Dinge. Diese Eigenschaften unterscheiden sie:

| Eigenschaft | **ToolACE** | **TaskBench** | **τ²-bench-Abläufe** | **db_bahn-Traces** |
|---|---|---|---|---|
| **Domäne** | 26.507 zufällige APIs (Krypto, Finanzen, Wetter…) | KI-Modelle / Multimedia / Alltags-APIs | Kundenservice: Airline, Retail, Telecom | **Deutsche Bahn** (intern) |
| **Sprache** | Englisch | Englisch | Englisch | **Deutsch** |
| **Tools (Werkzeuge)** | sehr viele, wechselnd | pro Domäne ein fester Tool-Katalog | fester Domänen-Katalog (buchen, stornieren…) | 12 feste (6 Lookup + 3 **Suche** + 3 Write mit Ablehnungs-Regeln) |
| **Planung (Zerlegung)** | einfach (1–wenige Calls) | **Kern**: Tool-Graph, Reihenfolge, Parameter | mehrstufig, regelbasiert | mehrstufig (1–4 Tools) |
| **Fehler / injected mismatches** | nein (nur saubere Calls) | nein (nur der Soll-Graph) | **ja** — Überraschungen aus der Umgebung | **ja** — bewusst injiziert (Ausfall, Verspätung) |
| **Rationales (Denk-Schritte)** | teils | nein (nur Struktur) | ja (Teacher denkt) | **ja** — `<plan>…</plan>` pro Schritt |
| **Revision / Replan** | nein | nein | **ja** — reagiert auf Überraschung | **ja** — „Plan A scheitert → Plan B" |
| **Echte Tool-Antworten?** | erfunden (im Datensatz) | keine (nur Graph) | **echt** (Framework führt aus) | **echt** (unser Sandbox führt aus) |
| **Format** | ShareGPT (`conversations`) | Parquet (Instruktion + Graph als JSON-Strings) | per-turn: `messages`-Vorkontext + `answer` (thinking + flache `tool_calls`) | OpenAI-Messages (`messages` + `tool_calls`) |
| **Herkunft** | Download (fertig) | Download (fertig) | **Download (AReaL-Shortcut)** statt selbst erzeugen | **selbst erzeugt** ✅ |
| **Menge** | 11.300 | 17.331 | 33.531 per-turn (**74,5 % correct**) | Pool: **10.473 Tasks**; **9.146 verifizierte Traces** (99,4 %) |
| **Rolle** | SFT | SFT | SFT (+ 1.982 RL-Tasks) | SFT (+ Domäne für RL/Eval) |
| **Status bei uns** | ✅ gezogen | ✅ gezogen | ✅ **gezogen + validiert** | ✅ **9.146 12-Tool-Traces fertig** (W2.5) |

Kurz: Nur die **rechten zwei** (τ²-bench, db_bahn) haben echte Umgebungen mit *Fehlern* und *Umplanen* — das
ist genau die „Variante C"-Fähigkeit, um die es uns geht. Die linken zwei sind statische Bausteine (Grundlagen).

---

## 2. Jeder Datensatz einzeln (mit echtem Beispiel)

### 2.1 ToolACE — „der Wortschatz" (ein Tool korrekt bedienen)

- **Was es übt:** Ein einzelnes Werkzeug richtig aufrufen (Name + Argumente), die Antwort lesen, ggf. das
  nächste rufen. Die *Grundmechanik* des Tool-Callings — kein tiefes Planen, keine Überraschungen.
- **Wie es aussieht:** Ein `system`-Text mit den erlaubten Funktionen als JSON, dann ein Gespräch
  (`user → assistant → tool → assistant …`). Die Tool-Aufrufe stehen als **Klammer-Text** (nicht JSON):

```
system:    "You are an expert in composing functions … [{"name": "newAddress", …}, {"name": "Market Trends API", …}]"
user:      "I'd like to know what's happening in the market right now…"
assistant: [Market Trends API(trend_type="MARKET_INDEXES", country="us")]        ← Tool-Aufruf (Klammer-DSL)
tool:      [{"name":"Market Trends API","results":{"trends":[{"name":"S&P 500", …}]}}]   ← (erfundene) Antwort
assistant: "Here are the top Market Trends in the US: 1. S&P 500 …"              ← Endantwort
```

- **Eigenheit / Aufpassen:** Aufrufe als Klammer-DSL statt OpenAI-`tool_calls`; Tools stecken im `system`-Text.
  Beim Vereinheitlichen müssen wir das umformen. Tool-Antworten sind **erfunden** (kein echtes Env).

### 2.2 TaskBench — „die Grammatik" ⚠️ **aus dem Mix geflogen (2026-07-13) → Eval-Regal**

> **Warum raus:** genau wegen des Punktes direkt darunter — TaskBench zeigt **nur den Bauplan, nie die
> Ausführung**. Trainiert man darauf, lernt das Modell eine **Planungs-Notation zu emittieren, die es zur
> Serve-Zeit nie braucht** (dort emittiert es `tool_calls`). Es gibt keine Assistant-/Tool-Turns zum Maskieren,
> also auch kein sauberes Loss-Signal für unser Format. Die Rohdaten bleiben liegen (bewusst **kein**
> `convert_taskbench.py`) — als *Evaluations*-Satz für Zerlegung bleibt es interessant.

- **Was es übt:** **Planung** — eine Anfrage in Teilschritte zerlegen, das richtige Tool je Schritt wählen,
  die Reihenfolge/Abhängigkeiten festlegen („Tool-Graph"). Es zeigt **nicht** das Ausführen, sondern nur den
  *Bauplan*.
- **Wie es aussieht:** Eine Instruktion + der Soll-Graph (als JSON-Strings gespeichert). Echtes Beispiel:

```
instruction: "I have an image 'example.jpg' … extract the key content and provide a synopsis."
n_tools: 4, type: "chain"
tool_steps: ["Step 1: Segment the image", "Step 2: Ask 'What is the main subject?'", "Step 3: Summarize"]
tool_links:  Image Segmentation → Document Question Answering → Summarization      ← die Abhängigkeitskette
```

- **Eigenheit:** Reiner **Plan/Graph**, keine Ausführung, keine Antworten, keine Fehler. Lehrt „*welche* Schritte
  in *welcher* Reihenfolge", nicht „was tun, wenn Schritt 2 schiefgeht".

### 2.3 τ²-bench-Abläufe — „auf Fehler reagieren" (Replan, Selbstkorrektur) — ✅ **gezogen (AReaL-Shortcut)**

- **Was es übt:** Das, was ToolACE/TaskBench *fehlt* — **Umplanen & Selbstkorrektur**, wenn ein Tool eine
  Überraschung liefert (nicht verfügbar, Regel verletzt, Zustand anders als gedacht). Über **diverse** Domänen
  (Airline/Retail/Telecom) → **Task-Diversität**, der stärkste Daten-Hebel (OpenThoughts).
- **Wichtig:** τ²-bench selbst liefert **keinen Download**, nur die *Umgebung*; die „Abläufe" (Trajektorien)
  erzeugt man normalerweise selbst — ein starker Teacher spielt die Aufgaben durch, der Verifier filtert die
  korrekten. Genau das kann unser Harness (siehe §3) — diesen Schritt erspart uns aber der AReaL-Shortcut
  (Kasten unten): dort hat das inclusionAI-Team genau diese Erzeugung schon gemacht.
- **Wie es aussähe** (Airline-Domäne, illustrativ):

```
user:      "Bitte storniere Flug AA123 und buche mich auf den Nachmittagsflug um."
assistant: <plan>Erst Buchung prüfen, dann stornieren, dann umbuchen.</plan>  →CALL get_reservation(id="AA123")
tool:      {"status":"confirmed","flight":"AA123","fare":"basic_economy"}
assistant: <plan>Überraschung: basic_economy ist nicht umbuchbar (Policy).      ← REPLAN
            Also: Regel prüfen, dann Alternative anbieten.</plan>  →CALL get_policy(topic="rebooking")
tool:      {"rule":"basic_economy not changeable"}
assistant: "Basic-Economy ist nicht umbuchbar. Möchten Sie stattdessen stornieren + neu buchen?"
```

- **Eigenheit:** **Echte** Tool-Antworten (Framework führt aus), echte Regel-/Zustands-Überraschungen → echte
  Replan-Beispiele. Englisch, Kundenservice.

> **📌 Finding — Shortcut statt selbst generieren:** [`inclusionAI/AReaL-tau2-data`](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data)
> liefert diese Abläufe **fertig** (Apache-2.0, kommerziell nutzbar): **33.531 SFT-Beispiele** (airline/retail/
> telecom) + Bonus **1.982 RL-Tasks inkl. DB-Snapshots + Eval-Kriterien** — eine fertige Stage-2-Startrampe.
> Planning bleibt erhalten (`answer.thinking` = CoT + `tool_calls` pro Beispiel).
> **Aber:** **Per-Turn-Format** (1 Assistant-Zug + Vorkontext = 1 Beispiel; darum 33k aus nur ~hunderten
> Gesprächen), nicht Full-Episode wie db_bahn. Folge: beim Mischen **ein** Format wählen (AReaL zu Episoden
> fügen *oder* db_bahn/ToolACE auch per-turn schneiden) — reine Buchhaltung, kein Qualitätsverlust (der
> Assistant-only-Loss-Mask trainiert ohnehin nur die Assistant-Tokens). Erspart die τ²-User-Sim-Anpassung.

> **📌 Update (2026-07-08) — gezogen + validiert ✅:** liegt unter `data/raw/areal/` (926 MB, Revision
> `86971dc0` gepinnt), geprüft mit `data_pipeline/validate_areal.py --deep`: **PASS, 0 fail / 0 warn / 34 Checks**
> (Report: `data/raw/areal/validation_report.json`). Alle Card-Zahlen exakt bestätigt (33.531 = 12.842 airline /
> 11.395 retail / 9.294 telecom; RL 1.982 = 1.148/563/271); jeder `db_path` löst auf, alle 9 DB-Snapshots parsen
> **und laden im installierten tau2-Package** — die Stage-2-Startrampe funktioniert. Drei Funde für den Konverter:
> 1. **Die SFT-Datei enthält auch fehlgeschlagene Turns:** nur **74,5 %** haben `metadata.correct == 1`.
>    Der Konverter **muss** darauf filtern (analog zum `score==1.0`-Gate in `format_traj_for_training.py`) —
>    sonst trainiert man auf ~8.500 bekannt-falschen Beispielen. Die `correct==0`-Turns roh behalten:
>    potenziell Negativ-Beispiele für DPO/Preference später.
> 2. **Format-Details:** `answer.tool_calls` sind **flach** (`{name, arguments}`, nicht OpenAI-`function`-nested);
>    `thinking` zu 97,5 % befüllt; 37,3 % der Turns rufen Tools, der Rest ist Nutzer-Kommunikation.
>    *Einordnung der ~63 % Kommunikations-Turns:* **nicht wegfiltern** — das sind Rückfragen, Pflicht-
>    Bestätigungen vor Write-Aktionen, Policy-Ablehnungen und Fakten-Rückmeldung (der Verifier scored
>    `communicate_info`; τ²-bench-Eval läuft gegen einen User-Sim, misst das also direkt), und mit 97,5 %
>    `thinking` tragen auch sie Planungs-CoT („mir fehlt Z → erst fragen"). **Aber:** Kandidat für einen
>    **Mix-Ratio-Knopf** im Konverter — falls der Gesamt-Mix zu konversationslastig wird, Kommunikations-
>    Turns untersampeln (trivialer Filter auf `answer.tool_calls` dank Per-Turn-Format).
> 3. **Kein Domain-Feld:** die Domain steckt in drei per-Domain-Metadata-Keysets (airline: `seed_pattern_task_id`,
>    retail: `scenario_id`, telecom: `task_id`+`difficulty`) — Ableitung fertig implementiert in
>    `validate_areal.py::_derive_domain`, kann der Konverter wiederverwenden.

### 2.4 db_bahn-Traces — „unser Dialekt" (Bahn-Domäne, Deutsch) — ✅ fertig

- **Was es übt:** Dasselbe wie τ²-bench-Abläufe, aber in **unserer** Domäne und auf **Deutsch** — macht das
  Modell bahn-spezifisch. Höher gewichtet, weil einziges Deutsch + einzige Zieldomäne.
- **Wie es aussieht** (echt, eine Replan-Aufgabe mit injizierter Überraschung):

```
user:      "Der eingeteilte Lokführer von EC 290 ist ausgefallen. Prüfe die Besatzung und teile
            Paul Schmidt (MA-4551) als Ersatz-Lokführer zu."
assistant: <plan>1. Besatzung von EC 290 prüfen. 2. MA-4551 als Lokführer zuweisen.</plan>
                                                    →CALL mitarbeiter_info(zugnummer="EC 290")
tool:      {"besatzung":[{"mitarbeiter_id":"MA-5989","name":"Felix Schäfer","rolle":"Zugbegleiter"}]}  ← KEIN Lokführer!
assistant: <plan>Kein Lokführer eingeteilt → MA-4551 zuweisen.</plan>
                                                    →CALL crew_zuweisen(zugnummer="EC 290", mitarbeiter_id="MA-4551", rolle="Lokführer")
tool:      {"assignment_id":"AS-2677","emp_id":"MA-4551","role":"Lokführer"}
assistant: "Paul Schmidt (MA-4551) wurde als Lokführer für EC 290 zugewiesen (ID AS-2677)."
```

- **Eigenheit:** Der **`<plan>`** vor jedem Schritt ist die *sichtbare Orchestrierung* (Variante C). Die
  **Überraschung** (Ausfall) ist bewusst **injiziert** — so entsteht ein echtes „Plan A → beobachten → Plan B".
  Echte Tool-Antworten aus unserem Sandbox. Jede Trace wurde vom Verifier auf 1,0 geprüft.

> **📌 Update (2026-07-08 → 07-12) — Welle 2 + 2.5 umgesetzt: Clean Rebuild + Weltvergrößerung ✅ (10.473er-Pool, 9.146 Traces).**
> Welle 1 (446 Traces, 64 % Ein-Tool, flacher Eval) war zu einfach; statt sie zu schonen wurde sauber neu
> gebaut und **Welle 1 komplett archiviert** (`archive/data/wave1_20260708/`). Dafür wuchs die **Domäne
> selbst**: 8 → **12 Tools** (3 Such-Tools `zuege_suchen`/`mitarbeiter_suchen`/`wartung_liste` für Aufgaben
> **ohne vorgekaute IDs**, + Lookup-by-ID `mitarbeiter_details` als 12.) und die WRITE-Tools **lehnen jetzt
> regelwidrige Aufrufe ab** (Rolle/Qualifikation/Duplikat/Endstatus → echte **Laufzeit-Fehler-Replans**, z. B.
> „Zuweisung abgelehnt: … fehlt die Qualifikation ICE"). 25 Templates (9 poliert übernommen,
> `info_wartung_machbar` gestrichen, 16 neue).

| | Welle 1 (archiviert) | Welle 2 (archiviert) | **Welle 2.5 (ist, validiert)** |
|---|---|---|---|
| Welt | 548 Fzg / 450 Aufträge / 10 Depots | wie W1 | **1.070 Fzg / 1.949 Aufträge / 20 Depots** |
| Templates | 10 | 25 | **26** (+ `info_mitarbeiter`: Lookup-by-ID-Gold-Pfad fürs A1-Tool) |
| Task-Pool | 550 | 1.964 | **10.473** — Splits: bakeoff 26 / heldout 276 / **rl_train 998 (GRPO)** / sft 9.199 |
| Multi-Tool (≥3 Calls, Pool) | 20% | 52% | **51%** |
| Fault/Replan (Pool) | 22% | 41% | **41%** (2.977 state, 237 runtime, 1.030 state+runtime) |
| Verifizierte Traces | 446 | 1.601 (11-Tool) | **9.146** (99,4 %; 57% Multi-Tool, 40% Fault; branch-on-fail + k=2-Top-up + B2-Harvest) |

> Alle CPU-Gates grün: Oracle-Replay **10.473/10.473 = 100 %** / 0× gold_replay_failed, Verifier-Selftest 8/8,
> Seeder + Generator byte-deterministisch (je 2× gelaufen, identisch). **Achtung Eval-Bruch (beabsichtigt):**
> 12-Tool-Prompt + neues 276er-Heldout → alte 72,5 %/70 %-Zahlen nur noch historisch (Re-Baseline inzwischen
> gelaufen → [Übersicht](SFT-Training-Uebersicht.md)).
> **Welle 2.5 (2026-07-12): der einheitliche 12-Tool-Regen ist gelaufen** — Welt vergrößert (Weg B,
> replikations-validiert), 9.146 Traces. **A1-Befund:** der Teacher nutzt `mitarbeiter_details` in **16,8 %**
> der Traces (davon 1.271 **organisch**, d. h. außerhalb des Lookup-Templates — Person vor der Zuweisung
> prüfen), der Über-Such-„Flail" fiel von ~1,5 % auf **0,11 %**. Details:
> [agentic-db-synthesis-log.md](agentic-db-synthesis-log.md) (Einträge 2026-07-08 → 07-13).

---

## 3. τ²-bench — was ist das, wie funktioniert es, wie nutzen wir es?

### 3.1 Was τ²-bench IST (und was nicht)

τ²-bench (von Sierra Research) ist **kein Datensatz**, sondern ein **Framework / Prüfstand**: ein Baukasten, um
*ausführbare Tool-Umgebungen* („Domänen") zu definieren und darin Agenten laufen zu lassen und zu bewerten.
Man lädt keine fertigen Trajektorien herunter — **man erzeugt sie**. Unsere **`db_bahn` ist eine solche Domäne**,
selbst gebaut (siehe [agentic-sft-db-synthesis.md](agentic-sft-db-synthesis.md)).

### 3.2 Die vier Bausteine einer Domäne

```
   ┌──────────────────────────────────────────────────────────────────────┐
   │  τ²-bench-Domäne (z.B. airline, retail, telecom … oder unser db_bahn)  │
   ├──────────────┬──────────────┬───────────────┬─────────────────────────┤
   │ 1) DB        │ 2) Tools      │ 3) Tasks       │ 4) Verifier             │
   │ Welt-Zustand │ Funktionen    │ Aufgaben mit   │ prüft: ist der End-     │
   │ (Pydantic-   │ READ/WRITE    │ eingebautem    │ Zustand richtig? +      │
   │  Objekt,     │ (@is_tool)    │ Lösungs-       │ wurden die richtigen    │
   │  hat Hash)   │               │ schlüssel      │ Tools genutzt?          │
   └──────────────┴──────────────┴───────────────┴─────────────────────────┘
```

1. **DB (Welt-Zustand):** alle Daten der Domäne (bei uns: Bahnhöfe, Züge, Fahrpläne, Wartung, Personal). τ²-bench
   kann davon jederzeit einen **Fingerabdruck (Hash)** bilden — das ist der Trick fürs Prüfen.
2. **Tools:** Python-Funktionen, markiert als `READ` (nur lesen) oder `WRITE` (verändern die DB → Hash ändert
   sich). Aus den Signaturen + Docstrings entstehen automatisch die JSON-Schemas, die das Modell sieht.
3. **Tasks:** Aufgaben, die ihren eigenen **Lösungsschlüssel** mitbringen — eine Referenz-Aktionsfolge,
   Zustands-Prüfungen (`assert_…`), Pflicht-Fakten für die Antwort. Optional **Initialisierungs-Aktionen**, die
   die Welt *vor* der Aufgabe manipulieren → so bauen wir **Überraschungen** ein (Lokführer entfernen).
4. **Verifier:** die Bewertung (siehe §3.4).

### 3.3 Wie ein Ablauf entsteht (der Rollout-Loop, „Solo-Mode")

```
Task (Ticket)  ─►  ┌─────────────────────────────────────────────┐
                   │  AGENT (Teacher-Modell)                      │
                   │  1. <plan> denken                            │
                   │  2. Tool aufrufen ───────────────┐           │
                   └──────────────────────────────────│───────────┘
                             ▲                         ▼
                             │            ┌─────────────────────────┐
                   Beobachtung (echt)     │  τ²-Domäne führt Tool    │
                             │            │  gegen die DB aus        │
                             └────────────│  → echtes Ergebnis       │
                                          └─────────────────────────┘
      … wiederholen (plan → tool → beobachten → ggf. UMPLANEN) …
                             │
                             ▼
                   Endantwort (kein Tool mehr)  ─►  VERIFIER  ─►  1,0 behalten / 0,0 verwerfen
```

„Solo-Mode" = die Aufgabe kommt als **Ticket** (Text), es gibt keinen simulierten Nutzer, der Agent löst sie
allein. Genau so läuft unser `sdg_pipeline/db_bahn/rollout.py`.

### 3.4 Der Verifier — wie geprüft wird (deterministisch, kein „Gefühl")

```
Frische Welt  ──(Init-Aktionen)──►  Referenz-Aktionen abspielen   →  ZIEL-Hash
Frische Welt  ──(Init-Aktionen)──►  Aktionen DES MODELLS abspielen →  IST-Hash
                                                                        │
                            ZIEL-Hash == IST-Hash  ?  →  richtig / falsch
```

- **Aktions-Aufgaben:** Modell darf jeden Weg gehen — es zählt nur, ob am Ende der **DB-Zustand** stimmt.
- **Info-Aufgaben:** zusätzlich unser eigener Check — stehen die richtigen **Fakten** in der Antwort, und stammt
  **jede** ID/Zeit/Zugnummer aus einer echten Tool-Beobachtung? (Anti-Halluzination.) Siehe
  `evaluation/trajectory_reward.py`.

### 3.5 Die drei Rollen von τ²-bench (das, was oft verwechselt wird)

Ein und dieselbe Domäne bedient — auf **disjunkten** Task-Splits — drei Zwecke:

| Rolle | Wer handelt | Wozu | Was rein-/rausgeht |
|---|---|---|---|
| **(a) SFT-Abläufe** | **Teacher** löst Aufgaben | Trainingsdaten erzeugen | Tasks (`sft-gen`) → verifizierte **Trajektorien** |
| **(b) GRPO-Reward** | **Student** würfelt selbst Rollouts | RL Stage 2 | Tasks (`rl-train`) → Verifier gibt **Reward** live |
| **(c) Eval** | Student wird gemessen | Benchmark | Tasks (`test`) → **Score** |

**Faustregel:** SFT frisst *fertige Trajektorien*. GRPO frisst *Aufgaben* und würfelt seine Rollouts **selbst**
(on-policy, der Verifier bewertet). Deshalb muss man auch **verschiedene** Aufgaben nehmen (siehe §4).

---

## 4. Warum der erwartete Output der richtige ist (Korrektheit „by construction")

**Das Prinzip in einem Satz:** Frage und Soll-Antwort werden **nicht zweimal unabhängig erzeugt** (das wäre
fehleranfällig — zwei Schätzungen können auseinanderlaufen), sondern **beide mechanisch aus derselben Quelle
abgeleitet**: der eingefrorenen, deterministischen Datenbank. *Wie ein Lehrer, der die Klausur mit
aufgeschlagenem Buch schreibt — die Antwort wird nachgeschlagen, nicht erinnert.*

```
                    EINE Quelle: die eingefrorene DB (sha256-geseedet, Uhr steht)
                          │                                    │
        INFO-Aufgabe:     ▼                                    ▼   ACTION-Aufgabe:
        1. ECHTES Tool aufrufen → Antwort            1. Ticket + Referenz-Aktion aus
           (z.B. verspaetung() → "35 Min")              DENSELBEN Variablen bauen
        2. DAS wird der Lösungsschlüssel             2. Referenz einmal ausführen →
        3. DANN die Frage formulieren                   „Foto" der Welt danach (Hash)
                          │                                    │
        richtig = Modell nennt die Fakten,           richtig = Modell-Aktionen führen
        die das Tool wirklich liefert                zum GLEICHEN Foto (jeder Weg ok)
```

- **Kein LLM rät den Lösungsschlüssel** — er ist Tool-Ausgabe (Info) bzw. Ausführungs-Ergebnis (Action).
- **Fault-Injection bleibt konsistent:** die Überraschungs-Aktionen (`initialization_actions`) werden auf
  *alle* Welten gleich angewandt — Answer-Key-Berechnung, Gold-Replay und Rollout sehen dieselbe Störung.
- **Drei Fangnetze**, falls ein Key doch kaputt wäre:
  1. **Gold-Replay** — *die Musterlösung wird selbst getestet:* Vor jeder Bewertung spielt der Prüfer die
     Referenz-Lösung der Aufgabe einmal selbst durch. Läuft schon *die* nicht (z. B. Verweis auf eine ID,
     die es nicht gibt), ist die **Aufgabe** kaputt → Abbruch mit `gold_replay_failed`, statt das Modell
     fälschlich mit 0 zu bestrafen. *(Der Lehrer rechnet seine Musterlösung nach, bevor er korrigiert.)*
  2. **Cross-Teacher-Detektor** — *wenn alle durchfallen, ist die Prüfung schuld:* 8 verschiedene Teacher
     lösten im Bake-off dieselben Aufgaben. Scheitert *einer* → Modell-Schwäche. Scheitern **alle acht** an
     derselben Aufgabe → der **Key/die Formulierung** ist fehlerhaft (mehrdeutig, zu streng). So fanden wir
     die 2 Kalibrier-Templates (`info_ankunft`/`info_machbar`: 1/10 → nach Fix 45/48).
  3. **Grounding-Check** — *kein Glückstreffer ohne Arbeit:* Auch eine zufällig richtig klingende Antwort
     fällt durch, wenn die Fakten nicht **belegt** sind — jede ID/Zeit/Zugnummer muss aus einer echten
     Tool-Beobachtung (oder der Aufgabe) stammen, und die erwarteten Tools müssen wirklich gerufen worden sein.

  *Zusammen:* Netz 1 fängt kaputte **Aufgaben**, Netz 2 kaputte **Lösungsschlüssel**, Netz 3 unverdiente
  **Glückstreffer**.
- **Reicht das? Braucht es einen LLM-as-a-Judge?** Für die **Korrektheit: nein** — der deterministische
  Nachrechner ist einem Judge (Meinungsgeber, nicht reproduzierbar) überlegen (NebulaExp: Exec-Filter =
  Hebel #1). Ein kleiner **Ja/Nein-Judge** (BinEval-Stil) ist nur als *Zusatz* für Weiches sinnvoll
  (Stil, Plan-Qualität, Relevanz) — als Berichts-Schicht, nie als Ersatz des harten Gates.
- **Und die finale Evaluation?** Dieselbe Prüf-Maschine, drei Einsätze: Daten-Filter (✅) →
  Held-out-Messung auf ungesehenen Aufgaben (✅ — Zahlen in der [Übersicht](SFT-Training-Uebersicht.md)) →
  **offizielle** τ²-bench-Testsplits + BFCL-V3 (❌ steht noch aus — das ist die papervergleichbare Endmessung).

## 5. Die sinnvolle Aufteilung (SFT / RL / Eval)

Zwei einfache Regeln entscheiden alles:

1. **Statisch vs. ausführbar:** ToolACE & TaskBench haben *keine* Umgebung → nur **SFT** möglich (kein
   Live-Reward). τ²-bench & db_bahn *haben* eine Umgebung → können **SFT + RL + Eval**.
2. **Disjunkt splitten** (Paper „Reusable Modules"): SFT-Aufgaben, RL-Aufgaben und Eval-Aufgaben müssen
   **getrennte** Mengen sein. Sonst lernt RL nur nach, was SFT schon kann — und der Benchmark lügt.

```
STATISCH (nur SFT)          AUSFÜHRBAR (Task-Pool in 3 disjunkte Teile splitten)
┌──────────┬──────────┐     ┌───────────────────────────────┬──────────────────┐
 ToolACE    TaskBench        τ²-Domänen (airline/retail/tel)   db_bahn
    │      ⚠ NICHT im Mix     sft-gen │ rl-train │ test         sft │ rl │ heldout
    │      (nur Eval-Regal)      │    │    │     │               │   │    │
    ▼                            ▼    │    │     │               ▼   │    │
  ┌──────────── STAGE-1 SFT (gemischt: ToolACE + τ²-Abläufe + db_bahn) ┐  │    │
  └────────────────────────────────────────────────────────────────────┘  │    │
                                        ┌─── STAGE-2 GRPO (Aufgaben, on-policy) ─┘    │
                                        │    τ²-rl-train + db_bahn-rl                 │
                                        │    (Reward = Verifier)                      │
                                        └────────────────────────────────────────────┘
                                                 ┌─── EVAL (nie trainiert) ───────────┘
                                                 │    db_bahn-heldout (+ τ²-test,
                                                 │    TaskBench, BFCL-V3 als Regal)
```

**Ist-Stand** (Mix-Zahlen, RL-Pools, Trainings-Ergebnis): → [SFT-Training-Uebersicht.md](SFT-Training-Uebersicht.md).
Dieses Doc erklärt die Datensätze; den jeweils aktuellen Stand hält die Übersicht.

---

## 6. Spickzettel

| | ToolACE | TaskBench | τ²-bench-Abläufe | db_bahn |
|---|---|---|---|---|
| lehrt | Tool bedienen | zerlegen/ordnen | **umplanen** | umplanen + **Domäne/Deutsch** |
| Fehler/Replan | – | – | ✅ | ✅ |
| `<plan>`-Rationales | teils | – | ✅ | ✅ |
| echtes Env | – | – | ✅ | ✅ |
| Sprache | EN | EN | EN | **DE** |
| bei uns | ✅ | ✅ | ✅ (AReaL, `correct==1` filtern) | ✅ Pool W2.5 · 9.146 Traces ✅ |

**In einem Satz:** ToolACE + TaskBench liefern die Bausteine (statisch, nur SFT); τ²-bench und db_bahn liefern
*echte* Umgebungen mit Fehlern und Umplanen — und dieselbe Umgebung dient, auf **getrennten** Aufgaben-Splits,
gleichzeitig als SFT-Trace-Fabrik, GRPO-Reward und Eval-Benchmark.

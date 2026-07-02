# Experiments: FP8-Quantisierung (llm-compressor)

Autonom gestartet **2026-06-19** (über Nacht). Ziel: isoliertes llm-compressor-venv im Image, **4 Modelle FP8-quantisiert**, je **FP8 + bf16-Rebaseline** auf **EX + Efficiency** (thinking zusätzlich Close-Rate), alles in **MLflow**. Reihenfolge **14B-nothink → 14B-thinking → 9B-nothink → 9B-thinking**. Plan: `~/.claude/plans/eventual-beaming-hedgehog.md` (v3).

## Fixe Vorgaben (User)
- Separates FP8-Image **`eugr/spark-vllm-docker`** (baut sm_121a; offizielle Images haben keins). Prod-`vllm` (0.21) NICHT anfassen.
- bf16-Rebaseline auf **demselben** FP8-Image (apples-to-apples; greedy nicht versionsstabil).
- **NaN/Garbage-Gate** Pflicht vor jeder EX (5-Prompt-Sanity FP8 vs bf16). Bei Fail: auto `--linear-backend torch`; bleibt's Garbage → Modell als **„dense FP8 auf sm_121 nicht servierbar"** (valides Negativergebnis) loggen + **WEITER** zum nächsten.
- Ein gescheitertes Modell bricht den Rest **nicht** ab.
- 9B: leak-assert grün + als **partial-FP8** markieren (nur MLP + 8 full-attn; DeltaNet+Vision bf16).
- Params: max-tokens **4096** (thinking) / **2048** (nothink), serve-len **8192**, conc **16**.
- Code lokal, **NICHT pushen**.

## Entscheidungen (chronologisch)

### D-001 — torch-Pin gedroppt (venv)
**Was:** Kein `cons.txt`/torch-Pin; bare venv (`python3 -m venv /opt/llmcompressor`, KEIN `--system-site-packages`), normaler `pip install "llmcompressor==0.12.*"`.
**Warum:** `torch==2.10.0a0+…` (NGC) kollidiert mit llmcompressors `torch>=2.10.0` — PEP-440 ordnet `2.10.0a0 < 2.10.0` → unlösbar, Build bräche. Quant ist **CPU-only** → das venv braucht den SM_121-NGC-torch nicht; ein pip-eigener torch genügt. Bare venv hält zudem den gepinnten Training-Stack (torch 2.10 / transformers 5.6.2) komplett unberührt.
**Risiko/Adaption:** Falls pip keinen `torch>=2.10` für aarch64 findet → adaptiv. **De-risk vorab im Wegwerf-Container.**
**Ergebnis (✅ getestet):** bare venv zog sauber **torch 2.12.0+cu130**, **transformers 5.10.1**, **llmcompressor 0.12.0**, compressed-tensors 0.17.1; Imports ok; `Qwen3_5ForConditionalGeneration` + `Qwen3ForCausalLM` beide vorhanden. Ansatz validiert, NGC-torch (System 2.10.0a0 / transformers 5.6.2) unberührt.

### D-002 — Quant CPU-erzwungen (`CUDA_VISIBLE_DEVICES=""`)
**Warum:** venv-torch 2.12+cu130 ist generisch (keine sm_121-Kernel). FP8_DYNAMIC ist data-free (kein Forward), aber llm-compressors DataFreePipeline ruft `dispatch_model()` → könnte Gewichte auf die GB10-GPU legen und dort auf SM_121 scheitern. CPU-Force umgeht jeden GPU-Pfad im venv; RTN-Quant ist CPU-billig (128 GB RAM reichen für 18–30 GB Modell).

### D-003 — Serving-Strategie für FP8 (Abweichung vom „eugr primär", begründet, wird an Stage 4 getestet)
**Befund:** `eugr/spark-vllm-docker` ist **kein Pull** — es **baut vLLM** (20–40 min) und startet über einen eigenen `launch-cluster.sh`-Wrapper, **inkompatibel mit unserem compose-Harness** (evaluate.py erwartet `http://vllm:8000/v1` im docker-Netz).
**Plan (schlauer-Pfad-zuerst, autonom):** Der Kern-Grund für eugr war „offizielle Images haben kein sm_121a-CUTLASS-Cubin". Der **`--linear-backend torch`-Fallback** (torch._scaled_mm) braucht **gar kein** CUTLASS-Cubin → umgeht das Problem an der Wurzel und passt in unser compose-Harness. Daher an Stage 4: separates compose-`vllm`-Override mit frisch gezogenem offiziellem `vllm/vllm-openai` + NaN-Gate; bei Garbage automatisch `--linear-backend torch`. **eugr-Build nur als letzter Fallback**, falls beides scheitert (Build+Wrapper-Integration ist der teure Pfad). Ergebnis je Modell wird geloggt; „nicht servierbar" bleibt valides Negativergebnis.
**Image-Hinweis:** vorhandenes `vllm/vllm-openai:latest` = f023269abe06, **~4 Wochen alt (~22.05.)** → vermutlich inkl. #41215 (per-tensor-FP8 sm_121, merged 20.05.) + #38093/#39538. Kein Re-Pull (Prod-`:latest`-Referenz unberührt).

### D-004 — Stage 1 Quant fertig (✅ alle 4)
Alle 4 in ~2 min (CPU). leak-asserts grün. Quantisierte Linears: **14B=280** (40×7, lm_head aus), **9B=128** (MLP 32×3 + full-attn 8×4; DeltaNet 24× + Vision + lm_head bf16). Größen: 14B 28→**16G**, 9B 18→**13G** (9B kleiner Anteil FP8 = partial, erwartet). compressed-tensors `quantization_config` + (9B) Preprocessor-Configs vorhanden.

### D-005 — MLflow-Naming (Kollision bewusst in Kauf genommen)
evaluate.py leitet den Run-Namen aus `--api-model-name` ab. FP8: served-name = `…_fp8` → Run `baseline_…_fp8` (eindeutig). bf16-Rebaseline: served-name = bf16-Pfad → Run-Name **kollidiert** mit den historischen bf16-Runs (gleicher Name, aber neuer run_id/timestamp). **Quelle der Wahrheit = die JSON-Dateien** (FP8 → `…_fp8/student_<v>.json`, Rebaseline → `…/student_<v>_rebaseline.json`), MLflow sekundär. Vergleichstabelle kommt aus den JSONs.

### D-006 — ✅ FP8 serviert auf SM_121 mit EXISTIERENDEM Image + Default-Backend
**Test (14B-nothink FP8, 16G):** `vllm/vllm-openai:latest` (f023269, ~22.05.), Default-Backend → lädt sauber (HEALTHY ~195s, **kein** sm120/sm121-Trap, **kein** „Error Internal"), **NaN-Gate 5/5 kohärent** (echtes `SELECT`-SQL). Das Image (inkl. #41215 per-tensor-FP8-sm_121-Fix, ~20.05.) serviert dense **per-channel-FP8** korrekt auf GB10. → **D-003 bestätigt: kein eugr-Build, kein torch-Fallback nötig.** `--linear-backend torch` bleibt Auto-Fallback im Master (v.a. fürs partial-FP8-9B-Hybrid). Prod-`:latest` unberührt (kein Re-Pull). **R2/D1/D2 aus dem Plan damit entschärft.**

### D-007 — Efficiency-Modus nothink = `--disable-thinking` (nicht leer)
Damit die FP8- und bf16-Rebaseline-Efficiency-Messung der bf16-Methodik (run_baseline_pipeline EFF_THINK) entspricht; thinking = `--enable-thinking` (langer Decode = aussagekräftiges TPOT).

---

## MORGEN-SUMMARY (Ergebnis, 2026-06-19)

**Alle 4 Modelle FP8-quantisiert + FP8 & bf16-Rebaseline auf EX + Close (thinking) + Efficiency gemessen, alles in MLflow. KEINE Fehlschläge, KEINE Negativergebnisse, kein Eingriff nötig.**

### Vergleichstabelle (FP8 vs bf16-Rebaseline, gleiches FP8-fähiges Image, apples-to-apples)
| Modell | Var | EX FP8 | EX bf16 | **EX-Δ** | Close FP8 | Close bf16 | **Close-Δ** | Throughput FP8 | bf16 | **Eff** | Verdikt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 14B | nothink | 0.63 | 0.64 | **−0.01** | — | — | — | 172 | 98 tok/s | **+76 %** | **KEEP** |
| 14B | thinking | 0.62 | 0.62 | **0.00** | 92 % | 95 % | −3pp | 110 | 78 tok/s | **+41 %** | **KEEP** |
| 9B | nothink | 0.64 | 0.64 | **0.00** | — | — | — | 218 | 75 tok/s | **+191 %** | **KEEP ⭐** |
| 9B | thinking | 0.60 | 0.64 | **−0.04** | 98 % | 94 % | +4pp | 217 | 161 tok/s | **+35 %** | **KEEP** (EX-Δ grenzwertig) |

(Throughput = guidellm Output-tok/s, rate 16; Detail-TPOT/TTFT in MLflow `eff_*`. bf16-Rebaseline auf demselben Image gemessen → konfundierungsfrei; bestätigt die historischen bf16-Zahlen ±0.02.)

### Aussage
- **Accuracy hält** unter FP8: 3/4 mit EX-Δ ≤ 0.01 (Rauschen); nur **9B-thinking −0.04** (4/100, grenzwertig — Close ist dort sogar besser, also kein Loop-Effekt, eher minimaler Reasoning-Verlust durch partial-FP8 auf langem Decode).
- **Close-Rate bleibt** bei beiden thinking-Modellen im Rauschen (±3–4pp), **kein FP8-induziertes Looping**.
- **Efficiency: FP8 durchweg schneller** (+35 % bis +191 %); **9B-nothink ist der Star** (EX identisch, ~2,9× Throughput).
- **9B = partial-FP8** (nur MLP + 8 full-attn FP8; 24 DeltaNet-Layer + Vision + lm_head bf16) → kleinere FP8-Coverage als 14B, deshalb 9B-FP8 nur 13G (vs 14B 16G).

### Plattform-Befund (die größte Unbekannte — entschärft)
Das **vorhandene `vllm/vllm-openai:latest` (~22.05., inkl. PR #41215) serviert dense per-channel-FP8 korrekt auf SM_121** — alle 4 Gates 5/5 mit **Default-Backend**. Die schweren Plan-Contingencies (**eugr-Build, `--linear-backend torch`-Fallback, w8a8-Truncate-Patch**) waren **NICHT nötig** (blieben als Auto-Fallback scharf, nie ausgelöst). Damit sind R2/R3/R4/D1/D2 aus dem Plan empirisch erledigt.

> **Serving-Hinweise (GB10):** Beim FP8-Servieren auf dem `vllm`-Service (Eval/Reachability) `VLLM_USE_FLASHINFER_SAMPLER=0` setzen — sonst der FlashInfer-Sampler-Race-Wedge (vLLM #43885); schon im docker-compose `vllm`-Env hinterlegt. Getestet wurde `f023269` (~22.05.); bei viel neuerem `:latest` FP8-Servierbarkeit re-validieren.

### Empfehlung
**FP8 generell KEEP** — gratis Throughput (1,35–2,9×) bei gehaltener Accuracy. Für Deploy ist **9B-nothink FP8** der Sweet-Spot (EX 0.64 = bf16, ~2,9× schneller). Einziger Vorbehalt: 9B-thinking-FP8 −4/100 EX (mit größerem Test-Set re-prüfen, falls 9B-thinking deployt werden soll).

### Fehlschläge / Fallbacks ausgelöst: **keine.** Alle 4 FP8 first-try mit Default-Backend.

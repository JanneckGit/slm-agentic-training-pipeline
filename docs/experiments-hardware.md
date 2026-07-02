# Hardware-Optimierung: Training auf NVIDIA GB10 (Blackwell)

**Frage:** Wie macht man das LoRA-SFT-Training von Qwen3.5 auf der DGX-Spark-Hardware (GB10, Blackwell) schneller? **Antwort in einem Satz:** Die fehlenden Gated-DeltaNet-Fast-Path-Kernel (`flash-linear-attention` + `causal_conv1d`) nachinstallieren → **3.9× schnelleres Long-Sequence-Training, numerisch identisch.** Alles andere (FP8, torch.compile, größere Batches) bringt hier nichts und ist mit Begründung verworfen.

Bezug: Trainings-Ergebnisse in [experiments.md](experiments.md). Stand 2026-06-10.

---

## 1. Plattform (verifiziert per Introspektion)

| | |
|---|---|
| GPU | **NVIDIA GB10** (Grace-Blackwell Superchip), Compute Capability **12.1 (SM_121)** |
| CPU/Arch | **aarch64 / ARM64** (Grace) |
| Speicher | **128 GB unified / coherent** (CPU+GPU teilen sich den Pool) |
| Treiber / CUDA | 580.95.05 / **CUDA 13.0** |
| Training-Image | `nvcr.io/nvidia/pytorch:25.11-py3` → **PyTorch 2.10.0a0 nv25.11** (einzige SM_121-kompatible Quelle) |
| Schon im Image | **triton 3.5.0**, **flash_attn 2.7.4.post1**, **transformer_engine 2.9.0** (alle ungenutzt vor der Optimierung) |
| Gepinnter Stack (fragil) | peft 0.13.2 (`--no-deps`, kein torchao), transformers 5.6.2, trl 0.12.0, accelerate ≥1.3.0 |

Besonderheit der Plattform: ARM64 **und** Blackwell **und** CUDA 13 **und** PyTorch-Preview — für viele Kernel-Libs gibt es **keine vorgebauten Wheels**, sie müssen aus Quelltext für `sm_120/sm_121` gebaut werden. Das ist der eigentliche Aufwand.

---

## 2. Der Engpass (Root Cause)

Qwen3.5 ist ein **Hybrid-Attention-Modell**: von 32 Layern sind **24 `linear_attention` (Gated-DeltaNet)** und nur 8 `full_attention` (alle 4 Layer einer). Das native `transformers/models/qwen3_5/modeling_qwen3_5.py` schaltet die schnellen Kernel über ein Gate:

```python
is_fast_path_available = all((
    causal_conv1d_fn, causal_conv1d_update,        # aus  causal_conv1d
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule,  # aus  fla
))
```

Beide Libs fehlten im Image → Gate = `False` → die 24 DeltaNet-Layer liefen im **reinen PyTorch-Fallback** `torch_chunk_gated_delta_rule` (viele kleine bf16-Matmuls + fp32-Up/Downcasts, kein Kernel-Fusing). Laufzeit-Warnung:

> `the fast path is not available because one of the required library is not installed. Falling back to torch implementation.`

**Symptom:** Der Fallback skaliert schlecht mit der Sequenzlänge → **thinking-Training (lange `<think>`-Sequenzen, ~2–6k Tokens) war 3–5× langsamer als nothink** (~270 Tokens). GPU bei 87 % Auslastung, aber memory-/launch-bound auf den unfused Ops.

---

## 3. Vorgehen (wie der Hebel gefunden wurde)

1. **Profiling/Introspektion** im Container: torch/triton/Kernel-Verfügbarkeit, `device_capability`, welche Libs da/fehlen.
2. **Parallele Recherche** (6 Optimierungs-Achsen) + **adversariale Machbarkeitsprüfung** der unsicheren Kernel-/FP8-Achsen auf genau dieser Box.
3. **Empirischer Test in Wegwerf-Containern** (`docker compose run --rm`, Image unverändert): installiert/baut es überhaupt auf sm_121/aarch64/cu13? → entscheidend, statt zu spekulieren.
4. **A/B-Microbenchmark** ([tools/bench_deltanet.py](../tools/bench_deltanet.py)): identische echte thinking-Batch, Fallback vs. Fast-Path, ms/Schritt **und** Loss-Abgleich.

---

## 4. Die Optimierung (was wirkt)

### Tier 2 — Gated-DeltaNet-Fast-Path-Kernel (der eigentliche Hebel)

Beide Libs `--no-deps` installiert (→ der gepinnte torch/peft/transformers-Stack bleibt unberührt, verifiziert):

```dockerfile
# flash-linear-attention: NUR von GitHub — das PyPI-Paket liefert kein fla.ops
RUN pip install --no-deps "git+https://github.com/fla-org/flash-linear-attention"   # -> fla 0.5.1

# causal_conv1d: kein aarch64/cu13-Wheel -> Source-Build für Blackwell.
# nvcc kommt aus dem NGC-Image; GPU beim Build NICHT nötig (Arch-Liste statt Query).
RUN CAUSAL_CONV1D_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST="12.0" \
    pip install --no-deps --no-build-isolation causal-conv1d                        # -> 1.6.2.post1
```

- **fla** ist Triton-basiert → läuft dank **triton 3.5** direkt auf SM_121, kein C-Build nötig (Wheel-Build ~3 s).
- **causal_conv1d** ist eine CUDA-C++-Extension → **Source-Build ~5,5 min** (`TORCH_CUDA_ARCH_LIST="12.0"` zwingt nvcc auf Blackwell; `--no-build-isolation` nutzt das vorhandene torch).
- Danach: `is_fast_path_available = True`, alle vier Symbole importierbar, Stack-Versionen unverändert (torch 2.10 / peft 0.13.2 / transformers 5.6.2 / trl 0.12.0).

**Gemessen** (`bench_deltanet.py`, Qwen3.5-0.8B, echte thinking-Batch, micro_batch 1, seq ≤ 8192, bf16, grad-ckpt):

| | ms/Schritt (median) | ms/Schritt (mean) | `is_fast_path` |
|---|---|---|---|
| Fallback (vorher) | 2355 | 2426 | False |
| **fla + causal_conv1d** | **601** | **585** | True |
| **Speedup** | **3.9×** | **4.1×** | |

**Numerisch identisch:** Per-Schritt-Loss Fallback `[0.694, 0.936, 0.643, 0.754, 0.665, 0.620]` vs. Fast-Path `[0.692, 0.938, 0.646, 0.757, 0.661, 0.621]` — Δ ≤ 0.004 = reines bf16-Rauschen. Kein Genauigkeitsverlust.

### Tier 1 — Config-Hebel (kein Image-Umbau)

- **`gradient_checkpointing=off`** für die kleinen Modelle (0.8B/2B): spart den Recompute-Forward (~1.2–1.3× zusätzlich), passt locker in 128 GB, **ergebnis-invariant**. 9B bleibt sicherheitshalber AN. Konfigurierbar via `--grad-checkpointing on|off`.

---

## 5. Was VERWORFEN wurde (mit harter Begründung)

| Achse | Verfügbar? | Warum hier nutzlos |
|---|---|---|
| **FP8-Training (transformer_engine 2.9)** | ✅ im Image | Der Engpass sind **Non-GEMM** SSM/Conv-Scan-Ops — FP8 beschleunigt nur GEMMs. Nur 8/32 Layer haben FP8-fähige Projektionen, und bei **LoRA sind die Basisgewichte eingefroren** (kein Weight-Grad-GEMM). `micro_batch=1` → winzige GEMM-M-Dimension, in der FP8-Quantisierungs-Overhead dominiert. → realistisch ~1.0× / Slowdown. (Die „46–55 % FP8"-Zahlen von NVIDIA sind full-parameter, dense-attention, large-batch — nichts davon trifft hier zu.) |
| **torch.compile / Inductor** | ✅ | Full-Model-Compile scheitert am `sm_121a`-ptxas-Codegen bzw. recompiliert pro Seq-Längen-Bucket (Minuten je Shape) und riskiert still-falsche Numerik. Nur die sicheren Teil-Flags (TF32, dataloader-workers) bringen ~5–12 %, fast nur auf nothink. |
| **liger_kernel** | installierbar | Kein `qwen3_5`-Patch in der Version; würde nur RMSNorm/SwiGLU/CE fusen (Norm/MLP-Anteil), **nicht** den DeltaNet-Scan = den eigentlichen Engpass. |
| **micro_batch hochskalieren** | trivial | **Gemessen:** per-Sequenz-Zeit flach (1538→1704 ms von batch 1→8) — die Linear-Attn-Kosten sind pro Sequenz, nicht pro Optimizer-Schritt; größere Batches sparen nur etwas Python-/Dataloader-Overhead (~1.05–1.2×). *(Redo 2026-06-18 lief dennoch mit `micro_batch=4` / `grad_accum=8` = eff. Batch 32 — nicht für Speed, sondern weil es nach Kernel-Install bequem in 128 GB passt; result-neutral, kein OOM bis 14B.)* |
| **flash_attention_2** | ✅ im Image | Greift nur die 8 `full_attention`-Layer, und SDPA-flash war dort ohnehin aktiv → ~1.0×. |

**Kurz:** Auf diesem Workload (LoRA, hybrid-linear-attn, micro_batch 1) ist der **einzige** echte Hebel der DeltaNet-Kernel. Blackwell-FP8 klingt verlockend, trifft aber den falschen Teil der Rechnung.

---

## 6. Netto-Wirkung

- thinking-Training ~**4× schneller**, numerisch identisch.
- Realer 6-Modell-Sweep (0.8B/2B/9B × 2): Training z. B. 0.8B-thinking **~13 min** (vorher ~63 min), 2B ~28 min, 9B ~70 min.
- Die ursprüngliche „~5–7 h"-Schätzung war damit deutlich unterboten; der Engpass ist jetzt eher Merge/Serve/Eval-Overhead als das Training selbst.

---

## 7. Reproduktion

```bash
# Image mit Kerneln bauen (enthält die Dockerfile-Schritte 4b)
docker compose -f docker/docker-compose.yml build training

# Fast-Path verifizieren
docker compose -f docker/docker-compose.yml run --rm --entrypoint python3 training -c \
  "import transformers.models.qwen3_5.modeling_qwen3_5 as M; print('fast_path:', M.is_fast_path_available)"
# -> fast_path: True

# A/B-Benchmark (vorher: vanilla Image -> False; nachher: gebautes Image -> True)
docker compose -f docker/docker-compose.yml run --rm training python3 tools/bench_deltanet.py
```

**Versionen (verifiziert lauffähig auf GB10/sm_121/cu13, 2026-06-10):** torch 2.10.0a0 nv25.11 · triton 3.5.0 · flash-linear-attention 0.5.1 (GitHub main) · causal_conv1d 1.6.2.post1 (Source) · transformers 5.6.2 · peft 0.13.2 · trl 0.12.0.

> **Wartungs-Hinweis:** `fla` wird von GitHub-`main` (unpinned) gezogen — bei einem Rebuild kann sich die Version ändern. Falls der Fast-Path bricht, auf den hier verifizierten Stand pinnen. `causal_conv1d` baut nur mit gesetztem `TORCH_CUDA_ARCH_LIST` (sonst kennt nvcc `sm_120/121` nicht).

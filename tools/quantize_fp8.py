#!/usr/bin/env python3
"""FP8_DYNAMIC-Quantisierung (data-free, CPU) der Deploy-Students.

Per-Arch ignore + Leak-Assert (kein FP8 in linear_attn/visual/lm_head) VOR dem Speichern.
Lauf via isoliertem llm-compressor-venv:
    /opt/llmcompressor/bin/python tools/quantize_fp8.py --model <in> --out <out> --arch text|mm
"""
import argparse, json, os, shutil, sys
import transformers
from transformers import AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--arch", choices=["text", "mm"], required=True)
args = ap.parse_args()

# per-Architektur ignore-Liste
if args.arch == "mm":
    IGNORE = ["lm_head", "re:.*visual.*", "re:.*linear_attn.*"]   # Vision+Merger, lm_head, Gated-DeltaNet bf16
else:
    IGNORE = ["lm_head"]

arch = json.load(open(os.path.join(args.model, "config.json")))["architectures"][0]
print(f"[quant] model={args.model} arch={arch} ignore={IGNORE}", flush=True)

# exakte Arch-Klasse (NICHT AutoModelForCausalLM fuers MM-Modell)
Cls = getattr(transformers, arch, None)
if Cls is None:
    Cls = transformers.AutoModelForImageTextToText if args.arch == "mm" else transformers.AutoModelForCausalLM
    print(f"[quant] {arch} nicht in transformers -> fallback {Cls.__name__}", flush=True)

model = Cls.from_pretrained(args.model, torch_dtype="bfloat16", low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained(args.model)

recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)
oneshot(model=model, recipe=recipe)   # kein dataset -> data-free, kein Forward

# ----- Leak-Assert: kein verbotenes Modul quantisiert -----
q = [n for n, m in model.named_modules() if getattr(m, "quantization_scheme", None) is not None]
bad = [n for n in q if ("linear_attn" in n) or ("visual" in n) or n.endswith("lm_head")]
print(f"[quant] quantisierte Linears: {len(q)}", flush=True)
if q:
    print(f"[quant]   z.B. {q[:2]} … {q[-2:]}", flush=True)
assert not bad, f"LEAK: verbotene Module quantisiert: {bad[:10]}"
print("[quant] leak-assert OK (0 verbotene quantisiert)", flush=True)

os.makedirs(args.out, exist_ok=True)
model.save_pretrained(args.out, save_compressed=True)
tok.save_pretrained(args.out)

# MM: Preprocessor-Configs kopieren (save_pretrained schreibt nur model+tok)
if args.arch == "mm":
    for f in ("preprocessor_config.json", "video_preprocessor_config.json",
              "processor_config.json", "chat_template.jinja"):
        src = os.path.join(args.model, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, f))
            print(f"[quant] copied {f}", flush=True)

print(f"[quant] DONE -> {args.out}", flush=True)

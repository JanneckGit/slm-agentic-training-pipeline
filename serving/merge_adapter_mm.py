"""Remap text-only LoRA adapter into the FULL multimodal Qwen3.5-4B and merge.

Hard-aborts (sys.exit(1)) on any mismatch. Steps:
  1. Remap adapter keys: base_model.model.model.layers. -> base_model.model.model.language_model.layers.
  2. ASSERT (pre-merge): every remapped target module exists in the full MM model
     as an nn.Linear with shape-compatible LoRA factors.
  3. Merge: W += (alpha/r) * (B @ A) in-place, vision tower untouched. Save MM checkpoint + tokenizer.
  4. ASSERT (post-merge): sampled attn weight in the SAVED artifact differs from base
     and equals base + computed delta (delta really applied, not a no-op).
"""
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file

# Overridable via env so thinking/nothink share one script (defaults = thinking run).
ADAPTER = os.environ.get(
    "MERGE_ADAPTER",
    "data/final/checkpoints/t-qwen3635ba3b-bf16_s-qwen354b_sdg750_2ep_seed42_20260608_2245_thinking",
)
BASE = os.environ.get("MERGE_BASE", "Qwen/Qwen3.5-4B")
OUT = os.environ.get("MERGE_OUT", "/data/hf_cache/qwen354b_student_thinking_mm_merged")

OLD_PREFIX = "base_model.model.model.layers."
NEW_PREFIX = "base_model.model.model.language_model.layers."
PEFT_WRAP = "base_model.model."  # stripped to get the real module path


def die(msg):
    print(f"\n!!! STOP: {msg}", flush=True)
    sys.exit(1)


# --- adapter config sanity -------------------------------------------------
with open(os.path.join(ADAPTER, "adapter_config.json")) as f:
    acfg = json.load(f)
if acfg.get("peft_type") != "LORA":
    die(f"peft_type != LORA: {acfg.get('peft_type')}")
if acfg.get("use_dora"):
    die("use_dora=True -> simple linear merge math is invalid")
if acfg.get("modules_to_save"):
    die(f"modules_to_save set ({acfg['modules_to_save']}) -> not handled by this merge")
if acfg.get("bias", "none") != "none":
    die(f"bias != none ({acfg.get('bias')})")
r, alpha = acfg["r"], acfg["lora_alpha"]
scaling = alpha / (r ** 0.5) if acfg.get("use_rslora") else alpha / r
print(f"[cfg] r={r} alpha={alpha} use_rslora={acfg.get('use_rslora')} -> scaling={scaling}")

# --- 1. load + remap adapter keys -----------------------------------------
sd = load_file(os.path.join(ADAPTER, "adapter_model.safetensors"))
bad = [k for k in sd if not k.startswith(OLD_PREFIX)]
if bad:
    die(f"{len(bad)} adapter key(s) don't start with {OLD_PREFIX!r}, e.g. {bad[:3]}")
remapped = {k.replace(OLD_PREFIX, NEW_PREFIX, 1): v for k, v in sd.items()}
print(f"[remap] {len(remapped)} keys: {OLD_PREFIX}* -> {NEW_PREFIX}*")

# group into modules -> {module_path: {A:.., B:..}}
modules = {}
for k, v in remapped.items():
    body = k[len(PEFT_WRAP):]  # model.language_model.layers.N.<...>.q_proj.lora_X.weight
    if body.endswith(".lora_A.weight"):
        mod, which = body[: -len(".lora_A.weight")], "A"
    elif body.endswith(".lora_B.weight"):
        mod, which = body[: -len(".lora_B.weight")], "B"
    else:
        die(f"unexpected adapter key suffix: {k}")
    modules.setdefault(mod, {})[which] = v
incomplete = {m: list(d) for m, d in modules.items() if set(d) != {"A", "B"}}
if incomplete:
    die(f"modules missing an A/B factor: {incomplete}")
print(f"[remap] {len(modules)} target modules (each with A+B)")

# --- load full MM model (CPU, bf16, no forward pass) ----------------------
import transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

print(f"[load] transformers {transformers.__version__}; loading full MM base on CPU ...")
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
)
model.eval()
named = dict(model.named_modules())

# --- 2. ASSERT (pre-merge): every target exists, is Linear, shapes fit -----
missing, not_linear, shape_bad = [], [], []
for mod, ab in modules.items():
    m = named.get(mod)
    if m is None:
        missing.append(mod)
        continue
    if not isinstance(m, torch.nn.Linear):
        not_linear.append((mod, type(m).__name__))
        continue
    out_f, in_f = m.weight.shape
    A, B = ab["A"], ab["B"]          # A:(r,in)  B:(out,r)
    if A.shape != (r, in_f) or B.shape != (out_f, r):
        shape_bad.append((mod, tuple(A.shape), tuple(B.shape), (out_f, in_f)))
if missing or not_linear or shape_bad:
    print(f"\n[ASSERT-PRE] FAILED  missing={len(missing)} not_linear={len(not_linear)} shape_bad={len(shape_bad)}")
    for x in missing[:20]:
        print(f"    MISSING   {x}")
    for x in not_linear[:20]:
        print(f"    NOT-LINEAR {x}")
    for x in shape_bad[:20]:
        print(f"    SHAPE     mod={x[0]} A={x[1]} B={x[2]} W={x[3]}")
    die("pre-merge module mapping mismatch")
# how many of these target modules live under the vision tower? (should be 0)
vis = [m for m in modules if "language_model" not in m]
print(f"[ASSERT-PRE] OK: all {len(modules)} targets exist as Linear with matching shapes; "
      f"{len(vis)} outside language_model (expect 0)")
attn = sorted(int(m.split('.layers.')[1].split('.')[0]) for m in modules if 'self_attn' in m)
mlp = sorted(int(m.split('.layers.')[1].split('.')[0]) for m in modules if 'mlp' in m)
print(f"[ASSERT-PRE] attn-adapted layers ({len(set(attn))}): {sorted(set(attn))}")
print(f"[ASSERT-PRE] mlp-adapted  layers ({len(set(mlp))}): {sorted(set(mlp))[:8]}... ")

# --- 3. merge (in-place) ---------------------------------------------------
# sample: first attn q_proj by layer index, snapshot original for post-check
sample_mod = next(m for m in sorted(modules,
                                    key=lambda x: int(x.split('.layers.')[1].split('.')[0]))
                  if m.endswith("self_attn.q_proj"))
orig_sample = named[sample_mod].weight.detach().clone().float()
sample_delta = None

with torch.no_grad():
    for mod, ab in modules.items():
        W = named[mod].weight
        delta = scaling * (ab["B"].float() @ ab["A"].float())
        if mod == sample_mod:
            sample_delta = delta.clone()
        W.add_(delta.to(W.dtype))
print(f"[merge] applied {len(modules)} LoRA deltas; sample module = {sample_mod}")

# post-merge in-process check
merged_sample = named[sample_mod].weight.detach().float()
if torch.allclose(merged_sample, orig_sample):
    die(f"sample {sample_mod} unchanged after merge (no-op!)")

print(f"[save] writing MM checkpoint -> {OUT}")
os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(OUT)

# --- verify saved config + tokenizer --------------------------------------
with open(os.path.join(OUT, "config.json")) as f:
    scfg = json.load(f)
archs = scfg.get("architectures")
if archs != ["Qwen3_5ForConditionalGeneration"]:
    die(f"saved architectures != [Qwen3_5ForConditionalGeneration]: {archs}")
if scfg.get("model_type") != "qwen3_5":
    die(f"saved model_type != qwen3_5: {scfg.get('model_type')}")
vis_cfg_key = next((k for k in scfg if "vision" in k or "visual" in k), None)
if vis_cfg_key is None:
    die(f"no vision config in saved config.json (keys: {list(scfg)})")
has_template = os.path.exists(os.path.join(OUT, "chat_template.jinja")) or \
    ("chat_template" in json.load(open(os.path.join(OUT, "tokenizer_config.json"))))
if not has_template:
    die("no chat_template in saved tokenizer")
print(f"[verify] config OK: arch={archs} model_type=qwen3_5 vision_key={vis_cfg_key!r} chat_template=yes")

# --- 4. ASSERT (post-merge): SAVED artifact differs from base by delta -----
disk_key = sample_mod + ".weight"
idx_path = os.path.join(OUT, "model.safetensors.index.json")
if os.path.exists(idx_path):
    shard = json.load(open(idx_path))["weight_map"][disk_key]
else:
    shard = "model.safetensors"
with safe_open(os.path.join(OUT, shard), framework="pt") as f:
    disk_w = f.get_tensor(disk_key).float()
if torch.allclose(disk_w, orig_sample):
    die(f"SAVED {disk_key} identical to base -> delta not persisted")
got_delta = disk_w - orig_sample
max_err = (got_delta - sample_delta).abs().max().item()
delta_norm = sample_delta.abs().max().item()
print(f"[ASSERT-POST] SAVED {disk_key} DIFFERS from base.")
print(f"[ASSERT-POST]   |applied delta|max={delta_norm:.4g}  reconstruction max_err={max_err:.4g} (bf16 rounding)")
if max_err > max(1e-2, delta_norm * 0.05):
    die(f"saved delta doesn't match computed LoRA delta (max_err={max_err})")
print("\n=== ALL ASSERTS PASSED ===")
print(f"merged MM model: {OUT}")

"""Tier-0 microbenchmark: Gated-DeltaNet torch-fallback vs fla+causal_conv1d fast path.

Times real forward+backward LoRA-SFT steps on Qwen3.5-0.8B over a fixed set of
real THINKING examples (the slow regime), at the exact training settings
(micro_batch=1, seq<=8192, bf16, grad-checkpointing, LoRA r16). Run it twice:
  1) vanilla training image  -> is_fast_path_available=False (fallback)
  2) after installing fla+causal_conv1d -> True (fast path)
Compare ms/step (speedup) AND the per-step losses (numerical sanity: the fast
kernel must match the fallback within bf16 noise). Same examples both runs.
"""
import json, time, statistics, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3_5.modeling_qwen3_5 as M
from peft import LoraConfig, get_peft_model, TaskType

print(f"is_fast_path_available = {M.is_fast_path_available}", flush=True)

MODEL = "Qwen/Qwen3.5-0.8B"
N_EX = 6          # fixed example set (same both runs)
WARMUP = 2        # first calls JIT-compile Triton kernels on the fast path

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True).to(torch.bfloat16).cuda()
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
                  bias="none", task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lcfg)
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)

# fixed real thinking examples
ids_list = []
with open("data/final/train_chat_thinking.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        msgs = json.loads(line)["messages"]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=8192)
        ids_list.append(enc["input_ids"].cuda())
        if len(ids_list) >= N_EX + WARMUP:
            break
print("seq lens:", [t.shape[1] for t in ids_list], flush=True)

def step(ids):
    out = model(input_ids=ids, labels=ids)
    out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    return out.loss.item()

for ids in ids_list[:WARMUP]:        # warmup / kernel compile
    step(ids)
torch.cuda.synchronize()

times, losses = [], []
for ids in ids_list[WARMUP:WARMUP + N_EX]:
    torch.cuda.synchronize(); t0 = time.time()
    l = step(ids)
    torch.cuda.synchronize(); times.append((time.time() - t0) * 1000); losses.append(l)

print(f"RESULT ms/step: mean={statistics.mean(times):.0f}  median={statistics.median(times):.0f}  (n={len(times)})", flush=True)
print(f"RESULT losses: {[round(x,4) for x in losses]}", flush=True)

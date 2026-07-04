"""
training_pipeline/collator_multiturn.py
=======================================
Phase 5 of Plan (B): assistant-only loss masking for MULTI-TURN tool-calling SFT.

Qwen3.x chat templates ship WITHOUT effective `{% generation %}` tags, so
`return_assistant_tokens_mask` yields an all-zero mask (verified). Fallback (plan P1-3): an explicit
ChatML span scan — train on tokens inside every `<|im_start|>assistant\n … <|im_end|>` block, mask
everything else. Qwen renders `role:"tool"` observations as a `user` turn (`<tool_response>…`), so tool
outputs are naturally masked too; assistant `tool_call` tokens ARE trained (they're inside the assistant
block). Robust to interleaving because it keys on the special-token ids, not on template string matching.

build_masked_labels(tokenizer, messages) -> (input_ids, labels) with labels=-100 off assistant spans.
TrajSFTCollator: pads a batch of pre-tokenized {input_ids, labels} for the HF Trainer.
"""

import torch


def _assistant_header_ids(tokenizer) -> list[int]:
    # "<|im_start|>assistant\n" -> the exact token id sequence emitted by the template
    return tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)


def build_masked_labels(tokenizer, messages: list[dict], max_len: int | None = None):
    """Tokenize a full conversation and return (input_ids, labels) with only assistant-generated tokens
    (content + tool_call, through the closing <|im_end|>) contributing to the loss."""
    enc = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if not isinstance(enc, list):                  # transformers 5.x BatchEncoding (UserDict, not dict)
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], list):           # batched shape [1, seq]
        enc = enc[0]
    input_ids = [int(x) for x in enc]
    if max_len:
        input_ids = input_ids[:max_len]
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    hdr = _assistant_header_ids(tokenizer)
    h = len(hdr)
    labels = [-100] * len(input_ids)
    i = 0
    n = len(input_ids)
    while i <= n - h:
        if input_ids[i:i + h] == hdr:
            j = i + h
            while j < n and input_ids[j] != im_end:
                j += 1
            end = min(j, n - 1)                 # include the <|im_end|> that closes the turn
            for k in range(i + h, end + 1):
                labels[k] = input_ids[k]
            i = end + 1
        else:
            i += 1
    return input_ids, labels


class TrajSFTCollator:
    """Pads pre-tokenized {input_ids, labels} examples; builds attention_mask. labels pad = -100."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features: list[dict]) -> dict:
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            ids, lab = list(f["input_ids"]), list(f["labels"])
            padn = maxlen - len(ids)
            input_ids.append(ids + [self.pad] * padn)
            labels.append(lab + [-100] * padn)
            attn.append([1] * len(ids) + [0] * padn)
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


def _selftest():
    """Run in the training container: asserts the mask covers assistant+tool_call, excludes user+tool obs."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Wo ist ICE 7?"},
        {"role": "assistant", "content": "<plan>Standort prüfen.</plan>",
         "tool_calls": [{"id": "c00000001", "type": "function",
                         "function": {"name": "zugstandort", "arguments": {"zugnummer": "ICE 7"}}}]},
        {"role": "tool", "tool_call_id": "c00000001", "content": '{"lat": 50.1, "naechster_halt": "Kassel"}'},
        {"role": "assistant", "content": "ICE 7 ist kurz vor Kassel."},
    ]
    ids, labels = build_masked_labels(tok, msgs)
    trained = tok.decode([i for i, l in zip(ids, labels) if l != -100])
    masked = tok.decode([i for i, l in zip(ids, labels) if l == -100])
    assert "zugstandort" in trained, "tool_call must be trained"
    assert "kurz vor Kassel" in trained, "final answer must be trained"
    assert "Standort prüfen" in trained, "<plan> must be trained"
    assert "Wo ist ICE 7" in masked and "Wo ist ICE 7" not in trained, "user must be masked"
    assert "naechster_halt" in masked and "naechster_halt" not in trained, "tool observation must be masked"
    assert "SYS" in masked, "system must be masked"
    frac = sum(1 for l in labels if l != -100) / len(labels)
    print(f"collator self-test OK — trained {frac:.0%} of tokens (assistant+tool_call only); "
          f"user/tool-obs/system masked")


if __name__ == "__main__":
    _selftest()

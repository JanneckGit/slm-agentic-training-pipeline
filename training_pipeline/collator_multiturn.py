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

CONTEXT-TURN MASKING (final_turns_only, default ON — 2026-07-22)
The Qwen3 template keeps a `<think>` block ONLY for assistant turns AFTER the last user message; every
earlier assistant turn is silently rendered think-less (verified against the real tokenizer). Training
those spans therefore teaches "answer immediately, no thinking" — measured on the AReaL leg: 18,763 of
24,144 assistant turns (78%). So we take gradient only on spans after the last real user message. The
think-less context itself STAYS in the sequence — that is exactly what the model sees at inference.

The rule is universal (no leg names in here): it describes a property of the chat template, not of a
dataset, and so it keeps holding when a leg's shape changes. Measured impact: db_bahn 0 turns (all 41,382
assistant turns follow its single user query), ToolACE <=509 of 11,300 raw rows (95.5% are single-query).

The cut is computed in MESSAGE space, never in token space: Qwen renders role:"tool" observations as
`<|im_start|>user\n<tool_response>`, so scanning token ids for the user header would count tool results as
user turns — and would collapse the rule to "only the very last turn" on every db_bahn trace.

build_masked_labels(tokenizer, messages) -> (input_ids, labels) with labels=-100 off assistant spans.
TrajSFTCollator: pads a batch of pre-tokenized {input_ids, labels} for the HF Trainer.
"""

import torch


def _assistant_header_ids(tokenizer) -> list[int]:
    # "<|im_start|>assistant\n" -> the exact token id sequence emitted by the template
    return tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)


def _render_ids(tokenizer, messages: list[dict]) -> list[int]:
    enc = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if not isinstance(enc, list):                  # transformers 5.x BatchEncoding (UserDict, not dict)
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], list):           # batched shape [1, seq]
        enc = enc[0]
    return [int(x) for x in enc]


def context_cut(tokenizer, messages: list[dict], full_ids: list[int]) -> int:
    """Token index where the last real user turn ends; assistant spans before it are context (think-less).
    0 if the conversation has no user message at all (-> train everything).

    Relies on the render being prefix-stable (rendering msgs[:k] yields a token prefix of rendering msgs).
    Verified on 50 real records (25 AReaL + 25 db_bahn, 0 violations); the check stays in as a hard error
    because it runs during pre-tokenization, i.e. before the GPU has spent anything."""
    lastq = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1)
    if lastq < 0:
        return 0
    prefix = _render_ids(tokenizer, messages[:lastq + 1])
    if full_ids[:len(prefix)] != prefix:
        raise ValueError("context_cut: chat template is not prefix-stable for this record — "
                         "the message-space cut would land on the wrong token")
    return len(prefix)


def build_masked_labels(tokenizer, messages: list[dict], max_len: int | None = None,
                        final_turns_only: bool = True):
    """Tokenize a full conversation and return (input_ids, labels) with only assistant-generated tokens
    (content + tool_call, through the closing <|im_end|>) contributing to the loss.

    final_turns_only=True additionally masks assistant turns BEFORE the last user message (context turns,
    which the template renders without their <think> block). See the module docstring."""
    input_ids = _render_ids(tokenizer, messages)
    cut = context_cut(tokenizer, messages, input_ids) if final_turns_only else 0
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
            if i >= cut:                        # context turns (before the last user msg) stay masked
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
    from data_pipeline.common import STUDENT_MODEL_DEFAULT
    tok = AutoTokenizer.from_pretrained(STUDENT_MODEL_DEFAULT, trust_remote_code=True)
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

    # (2) the db_bahn shape must be IDENTICAL with and without the context-turn rule: one user query up
    # front, so every assistant turn already follows it (the tool observations must NOT count as user turns)
    ids_all, labels_all = build_masked_labels(tok, msgs, final_turns_only=False)
    assert (ids, labels) == (ids_all, labels_all), "final_turns_only must be a no-op on single-query traces"
    assert context_cut(tok, msgs, ids) > 0, "cut must sit after the single user query, not at 0"

    # (3) multi-user dialogue (AReaL/tau2 shape): the template drops the FIRST turn's <think>, so that
    # span must be masked; the last turn keeps its think and is trained.
    dlg = [
        {"role": "user", "content": "FRAGE 1"},
        {"role": "assistant", "content": "<think>\nDENKEN A\n</think>\n\nANTWORT A"},
        {"role": "user", "content": "FRAGE 2"},
        {"role": "assistant", "content": "<think>\nDENKEN B\n</think>\n\nANTWORT B"},
    ]
    d_ids, d_lab = build_masked_labels(tok, dlg)
    d_tr = tok.decode([i for i, l in zip(d_ids, d_lab) if l != -100])
    d_ms = tok.decode([i for i, l in zip(d_ids, d_lab) if l == -100])
    assert "DENKEN A" not in tok.decode(d_ids), "template is expected to DROP the context turn's think"
    assert "ANTWORT A" in d_ms and "ANTWORT A" not in d_tr, "think-less context turn must be masked"
    assert "DENKEN B" in d_tr and "ANTWORT B" in d_tr, "the turn after the last user msg must be trained"
    d_ids2, d_lab2 = build_masked_labels(tok, dlg, final_turns_only=False)
    assert "ANTWORT A" in tok.decode([i for i, l in zip(d_ids2, d_lab2) if l != -100]), \
        "without the flag the think-less context turn WOULD be trained (that is the bug we fix)"

    print(f"collator self-test OK — trained {frac:.0%} of tokens (assistant+tool_call only); "
          f"user/tool-obs/system masked; db_bahn unaffected by final_turns_only; "
          f"multi-user context turns masked")


if __name__ == "__main__":
    _selftest()

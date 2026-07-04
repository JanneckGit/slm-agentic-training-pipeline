# Agentic SFT — Data Basis (Stage 1: mixed SFT)

> **Status:** raw data pulled ✅ · **Date:** 2026-07-02 · **Scope:** acquisition only (faithful copy + source tags).
>
> This document shows **what the Stage-1 SFT data basis is** for the agentic orchestrator. The raw sets
> live under `data/raw/` (gitignored → not committed), so **this doc is the shareable artifact.**

## Where this fits — the two-stage training plan

1. **SFT (LoRA first)** — mixed on **ToolACE + TaskBench + synthetic DB flows** (one shuffled pass, DB up-weighted).
2. **RL / GRPO (LoRA)** — on **τ²-bench**.

This doc covers the **raw basis for Stage 1, legs 1 & 2** (the two public sets). Leg 3 (synthetic DB flows)
and Stage 2 (τ²-bench) are separate, later workstreams. Converting these raw records into the unified
chat/training format is the **next** step (see the last section).

## What was pulled

| Dataset | HF id | License | Rows | Split | On disk | Role in the mix |
|---|---|---|---:|---|---:|---|
| **ToolACE** | `Team-ACE/ToolACE` | Apache-2.0 | **11,300** | `train` | ~35 MB | tool-call **basics** |
| **TaskBench** | `microsoft/Taskbench` | MIT | **17,331** | `test` | ~29 MB | **planning** / decomposition |
| **— total** | | | **28,631** | | ~64 MB | |

Both are **public (not gated)** → no HF token needed. Pulled by
[`data_pipeline/prepare_agentic_data.py`](../data_pipeline/prepare_agentic_data.py); counts verified on disk.

---

## Leg 1 — ToolACE (tool-call basics)

**Format:** ShareGPT-style. Each row = `{ system, conversations }` (+ our `source`, `split` tags).
- `system` — a fixed instruction **with the callable tools embedded as a JSON array** in the prompt text.
- `conversations` — `[{from, value}]`, `from ∈ user / assistant / tool`. Genuinely multi-turn (row 0 alone
  has 10 turns: `user → assistant → tool → assistant → user → …`).

**One embedded tool definition** (parsed out of the `system` string):
```json
{"name": "newAddress", "description": "Generates a new Ethereum address …",
 "parameters": {"type": "dict", "properties": {"password": {"type": "string", "description": "…"}},
 "required": ["password"]}}
```

**A real turn — assistant tool call, then tool result:**
```
assistant:  [Market Trends API(trend_type="MARKET_INDEXES", country="us")]
tool:       [{"name": "Market Trends API", "results": {"trends": [{"name": "S&P 500", …}]}}]
```
> ⚠️ **Format note for the conversion step:** assistant calls are the **bracket-DSL** (`[Func(arg=…)]`,
> BFCL/ToolACE style), **not** OpenAI `tool_calls` JSON, and tool defs live inside the `system` string.

**Teaches:** call a tool correctly, read its response, call the next — the core tool-use loop.

---

## Leg 2 — TaskBench (planning / decomposition)

**3 domain configs** (all pulled), each with its own tool inventory:

| Domain | Rows | # tools (`tool_desc.json`) |
|---|---:|---:|
| `huggingface` | 7,458 | 23 |
| `multimedia` | 5,555 | 40 |
| `dailylifeapis` | 4,318 | 40 |

**Format:** each row = `{ id, seed, n_tools, type, instruction, tool_steps, tool_nodes, tool_links,
sampled_nodes, sampled_links }` (+ `source`, `domain`, `split`). The graph columns are **JSON-encoded
strings** (`json.loads` them). Two **sidecar files** per domain carry what's *not* in the rows:
- `tool_desc.json` — the tool/API inventory, e.g. `{"id": "get_weather", "desc": "…", "parameters": [{"name":"location","type":"string",…}]}`
- `graph_desc.json` — the tool-dependency graph (nodes + links).

**A real multi-tool example** (`huggingface`, `n_tools=4`, a dependency chain):
```json
{"instruction": "… extract the key content from the document 'example.jpg' and provide a synopsis …",
 "tool_nodes": [{"task": "Image Segmentation", "arguments": ["example.jpg"]},
                {"task": "Document Question Answering", "arguments": ["<output_of_Image_Segmentation>", "What is the main subject matter?"]},
                {"task": "Summarization", "arguments": ["<output_of_Document_Question_Answering>"]}],
 "tool_links": [{"source": "Image Segmentation", "target": "Document Question Answering"},
                {"source": "Document Question Answering", "target": "Summarization"}]}
```

**Teaches:** decompose a request into steps, pick the right tool + parameters, order them (the tool-graph).
> ⚠️ **Note:** TaskBench rows are **request → plan/graph** — there are **no assistant/tool *execution*
> turns**. So it trains **planning/decomposition**, not the execute-observe loop (that's ToolACE's job,
> and τ²-bench's in Stage 2).

---

## Leg 3 — synthetic DB flows (later, not in this pull)

Domain adaptation for the Deutsche-Bahn assistant: German + DB-specific flows generated **against the
real DB tools** (Fahrplan, Zugstandort, Wartung, …). **Up-weighted** in the mix (only German + only
DB-specific leg). Separate synthesis workstream — not part of this acquisition.

---

## How to (re)fetch

Public sets → no token needed. Run in the `sdg` container (reproducible, persistent HF cache):
```bash
docker compose -f docker/docker-compose.yml run --rm sdg \
  python data_pipeline/prepare_agentic_data.py \
  --config config/pipeline_config.local.yaml --dataset all
```
Or on any host with `datasets` + `huggingface_hub`:
```bash
python data_pipeline/prepare_agentic_data.py --config config/pipeline_config.yaml --dataset all
# --dataset toolace|taskbench to pull one; --n-samples N for a quick subset
```
Dataset IDs/configs are in `config/pipeline_config.yaml` under `data.agentic` (the script also has
built-in defaults, so it runs without config edits).

## Placement & git

```
data/raw/
├── toolace/toolace.jsonl                                   # 11,300
├── taskbench/{huggingface,multimedia,dailylifeapis}/
│   ├── data.jsonl                                          # 7,458 / 5,555 / 4,318
│   ├── tool_desc.json  graph_desc.json                     # tool inventory + dep-graph
└── agentic_manifest.json                                   # counts/paths/columns
```
`data/raw/` is **gitignored** — the pulled data is never committed; this doc + the fetch script are the
committed, shareable artifacts.

## Toward the unified training format (for the next step)

The mix step must reconcile the two into **one chat `messages` format** (system with a tool registry;
assistant turns with tool calls; `role:"tool"` result turns; multi-turn loss masking on assistant turns):
- **ToolACE:** convert bracket-DSL calls → the target tool-call format; lift tool defs out of `system`.
- **TaskBench:** `json.loads` the graph columns; render the plan/graph into the target format (no
  execution turns to mask).
- **Mixing:** one shuffled SFT pass (not sequential blocks); **DB flows up-weighted** once added.

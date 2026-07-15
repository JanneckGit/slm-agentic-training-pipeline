# Agentic SFT — Data Basis (Stage 1: mixed SFT)

> **Status:** raw data pulled ✅ · **Date:** 2026-07-02 (AReaL leg added 2026-07-08) · **Scope:** acquisition
> record (faithful copy + source tags).
>
> This document shows **what the public Stage-1 SFT data basis is** for the agentic orchestrator. The raw sets
> live under `data/raw/` (gitignored → not committed), so **this doc is the shareable artifact.**

## Where this fits — the two-stage training plan

> ⚠️ **This is the ACQUISITION record, not the mix.** The counts below are **pull counts**, not mix
> contributions, and the built mix has **3 legs, not 4** — **TaskBench is not in it** (eval shelf, see its
> section). Mix numbers: [SFT-Training-Uebersicht.md](SFT-Training-Uebersicht.md).

1. **Stage 1 — SFT (LoRA)** — mixed on a **3-leg** basis: **ToolACE + τ²-bench flows (AReaL) +
   self-synthesized German DB trajectories** (one shuffled pass, the DB leg dominant by count).
2. **Stage 2 — RL / GRPO (LoRA)** — on **new, disjoint** τ²-bench + db_bahn tasks (reward = trajectory verifier).

This doc is the **acquisition record for the public sets** (ToolACE, TaskBench, AReaL — TaskBench pulled but
shelved). The self-synthesized DB leg has its own docs → [agentic-datasets-explained.md](agentic-datasets-explained.md)
+ [agentic-db-synthesis-log.md](agentic-db-synthesis-log.md). Note: τ²-bench enters as an **SFT leg** (via the
AReaL per-turn data), **not** RL-only.

## What was pulled

| Dataset | HF id | License | Rows | Split | On disk | Role in the mix |
|---|---|---|---:|---|---:|---|
| **ToolACE** | `Team-ACE/ToolACE` | Apache-2.0 | **11,300** | `train` | ~35 MB | tool-call **basics** |
| **TaskBench** | `microsoft/Taskbench` | MIT | **17,331** | `test` | ~29 MB | ~~planning / decomposition~~ → **not in the mix** (eval shelf) |
| **AReaL (τ²-bench)** | `inclusionAI/AReaL-tau2-data` | Apache-2.0 | **33,531** SFT (+1,982 RL) | — | ~926 MB | **dialogue / policy** (multi-turn) |
| **— total** | | | **62,162** SFT | | ~990 MB | |

All three are **public (not gated)** → no HF token needed. Pulled by
[`data_pipeline/prepare_agentic_data.py`](../data_pipeline/prepare_agentic_data.py) and validated by
[`data_pipeline/validate_areal.py`](../data_pipeline/validate_areal.py); counts verified on disk.

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

## TaskBench (planning / decomposition) — **pulled, then shelved**

> **Decision 2026-07-13 — dropped from the SFT mix → eval shelf.** Reason: the ⚠️ note at the end of this
> section turned out to be disqualifying. TaskBench rows are **request → plan/graph**, i.e. a *planning
> notation* with no assistant/tool **execution** turns. Training on them would teach the agent to emit a
> notation it never produces at serve time (it emits tool calls). The raw data stays on disk; there is
> deliberately **no `convert_taskbench.py`**. It remains usable as an *evaluation* set for decomposition.

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

## Leg 3 — AReaL (τ²-bench flows)

τ²-bench customer-service dialogues (airline / retail / telecom), pulled via the **AReaL shortcut**
([`inclusionAI/AReaL-tau2-data`](https://huggingface.co/datasets/inclusionAI/AReaL-tau2-data), Apache-2.0)
instead of self-generated: **33,531 per-turn SFT rows** (12,842 airline / 11,395 retail / 9,294 telecom) +
**1,982 RL tasks** with DB snapshots. Per-turn format (`messages` context + `answer`). Teaches **multi-turn
dialogue + policy adherence**. **Caveat:** only ~74.5 % of SFT turns carry `metadata.correct==1` — the mix
step **must filter on `correct==1`**. Validated by [`validate_areal.py`](../data_pipeline/validate_areal.py)
(streaming schema / integrity / referential checks → `data/raw/areal/validation_report.json`).

## Leg 4 — synthetic DB flows (own workstream)

Domain adaptation for the Deutsche-Bahn assistant: German + DB-specific trajectories generated **against the
real DB tools** (Fahrplan, Zugstandort, Wartung, …), verifier-gated. **Up-weighted** in the mix (only German +
only DB-specific leg). **Done** — 9,146 verified 12-tool traces; details in
[agentic-datasets-explained.md](agentic-datasets-explained.md) + [agentic-db-synthesis-log.md](agentic-db-synthesis-log.md).

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
# --dataset toolace|taskbench|areal to pull one; --n-samples N for a quick subset
# areal is a ~970 MB snapshot — validate afterwards with data_pipeline/validate_areal.py --deep
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
├── areal/  (per-turn SFT + RL tasks + DB snapshots)        # 33,531 SFT + 1,982 RL
│   └── validation_report.json                              # validate_areal.py output
└── agentic_manifest.json                                   # counts/paths/columns
```
`data/raw/` is **gitignored** — the pulled data is never committed; this doc + the fetch script are the
committed, shareable artifacts.

## The unified training format — **built** (2026-07-13)

All legs are reconciled into **one chat `messages` format** (system with a tool registry; assistant turns with
tool calls; `role:"tool"` result turns; assistant-only loss masking). What was actually implemented:
- **ToolACE** → [`convert_toolace.py`](../data_pipeline/convert_toolace.py): bracket-DSL → OpenAI tool-calls;
  tool defs lifted out of `system` into a `<tools>` block. **4,800 kept** (all multi-call + 500 irrelevance + fill).
- **TaskBench:** ~~render the plan/graph~~ — **dropped**, see above (no execution turns to learn from).
- **AReaL (τ²)** → [`convert_areal.py`](../data_pipeline/convert_areal.py): per-turn rows **reassembled into
  episodes**, `metadata.correct==1` filter (episode-level), tau2 tool schemas injected, trimmed at 12k on the
  last assistant speech turn. **2,052 episodes**.
- **db_bahn:** already in the target format via `format_traj_for_training.py` (verified traces only).
- **The mix** → [`build_sft_mix.py`](../data_pipeline/build_sft_mix.py): flail-drop, stratified per-source val
  cut, one shuffle. Resulting counts: [SFT-Training-Uebersicht.md](SFT-Training-Uebersicht.md).
- **Mixing:** one shuffled SFT pass (not sequential blocks); **the German DB leg up-weighted**.

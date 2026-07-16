#!/usr/bin/env python3
"""Build the Stage-2 GRPO task pool (rl_train subset) as verl parquet.

Usage (in the grpo container):  python3 training_pipeline/build_grpo_pool.py [--n-train 32] [--n-val 8]

Pool design — GRPO learns from INTRA-GROUP reward variance: a task the policy always solves (or never
solves) yields advantage 0 under Dr.GRPO, i.e. no gradient. Base+think already solves 96 % of the heldout,
so an unfiltered rl_train sample would be mostly all-correct groups. This picks the templates where
base+think measurably fails (heldout, docs/agentic-db-synthesis-log.md 2026-07-16) plus fault-injected
tasks (replan pressure), which is the cheap stand-in for the reachability probe the SQL-era pilot ran.

verl schema (verified against the frozen image): the reward manager reads reward_model.ground_truth
(naive.py:43) and hands extra_info through untouched; RLDataset promotes extra_info.tools_kwargs
(rl_dataset.py:385-391) into per-sample tool create_kwargs. Nested dicts go in as JSON strings so
pyarrow can't trip over heterogeneous task schemas.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from data_pipeline.common import TOOLS_BLOCK_TMPL

DATA_DIR = Path("data/raw/db_sandbox")
OUT_DIR = Path("data/final/grpo/db_bahn_pool")
SEED = 42

# base+think heldout yields < 100 % (the only templates with usable reward variance)
WEAK_TEMPLATES = ["t_info_zugsuche_status", "t_info_zug_komplett",
                  "t_info_ankunft_suche", "t_info_verspaetung_suche"]

# No <plan> instruction here (unlike the SDG system prompt): this run trains WITH thinking, and the
# chat template injects the tool schemas itself (verl passes tools= to apply_chat_template).
SYSTEM_PROMPT = """{policy}

## Arbeitsweise (wichtig)

- Rufe Werkzeuge in genau diesem Format auf (ein Block pro Aufruf, Argumente als JSON):
<tool_call>
{{"name": "werkzeug_name", "arguments": {{"argument": "wert"}}}}
</tool_call>
- Nach jedem Werkzeug-Ergebnis: prüfen, ob der Plan noch passt; bei Überraschungen umplanen.
- Wenn die Aufgabe gelöst ist: KEIN Tool-Aufruf mehr, sondern eine kurze deutsche Schlussantwort mit den
  belegten Fakten (nur Werte, die ein Werkzeug geliefert hat)."""


def _row(task: dict, key: dict, idx: int, sys_prompt: str, tool_names: list[str]) -> dict:
    init_json = json.dumps(((task.get("initial_state") or {}).get("initialization_actions")) or [],
                           ensure_ascii=False)
    return {
        "data_source": "db_bahn",
        "prompt": [{"role": "system", "content": sys_prompt},
                   {"role": "user", "content": task.get("ticket") or ""}],
        "reward_model": {"style": "rule", "ground_truth": task["id"]},
        "extra_info": {
            "index": idx,
            "need_tools_kwargs": True,
            "task": json.dumps(task, ensure_ascii=False),
            "answer_key": json.dumps(key, ensure_ascii=False),
            # every tool gets the same create_kwargs: whichever fires first builds the episode env
            "tools_kwargs": {n: {"create_kwargs": {"initialization_actions_json": init_json}}
                             for n in tool_names},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=32)
    ap.add_argument("--n-val", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    tasks = {t["id"]: t for t in json.loads((DATA_DIR / "tasks.json").read_text())}
    keys = json.loads((DATA_DIR / "answer_keys.json").read_text())
    rl_ids = json.loads((DATA_DIR / "split_tasks.json").read_text())["rl_train"]

    from sdg_pipeline.db_bahn.tau2_domain import get_environment
    env = get_environment(solo_mode=True)
    tool_names = [t.name for t in env.get_tools()]
    tools_block = "\n".join(json.dumps(t.openai_schema, ensure_ascii=False) for t in env.get_tools())
    # policy only — the tools block goes through the chat template, not the system text
    sys_prompt = SYSTEM_PROMPT.format(policy=env.get_policy())

    weak = [i for i in rl_ids if keys[i]["template"] in WEAK_TEMPLATES]
    fault = [i for i in rl_ids if keys[i].get("injected") and i not in set(weak)]
    rng = random.Random(SEED)
    rng.shuffle(weak)
    rng.shuffle(fault)

    n_total = args.n_train + args.n_val
    half = n_total // 2
    picked = weak[:half] + fault[:n_total - min(half, len(weak))]
    if len(picked) < n_total:  # top up from the rest of rl_train if a bucket runs dry
        rest = [i for i in rl_ids if i not in set(picked)]
        rng.shuffle(rest)
        picked += rest[:n_total - len(picked)]
    rng.shuffle(picked)

    rows = [_row(tasks[i], keys[i], n, sys_prompt, tool_names) for n, i in enumerate(picked[:n_total])]
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows[:args.n_train]).to_parquet(args.out / "train.parquet")
    pd.DataFrame(rows[args.n_train:]).to_parquet(args.out / "val.parquet")

    from collections import Counter
    tmpl = Counter(keys[i]["template"] for i in picked[:n_total])
    n_weak = sum(v for k, v in tmpl.items() if k in WEAK_TEMPLATES)
    print(f"train={args.n_train} val={args.n_val} | weak-template share={n_weak}/{n_total} "
          f"| injected={sum(1 for i in picked[:n_total] if keys[i].get('injected'))}")
    print("templates:", dict(tmpl.most_common()))
    print("out:", args.out)


if __name__ == "__main__":
    main()

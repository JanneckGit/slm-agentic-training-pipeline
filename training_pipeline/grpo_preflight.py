#!/usr/bin/env python3
"""Preflight gates for a Stage-2 GRPO run — runs INSIDE the grpo container (CPU only), called by
ops/grpo_run.sh before verl starts. Every gate prints OK/FAIL/INFO; any FAIL exits 1.

    docker compose --profile grpo run --rm -T grpo training_pipeline/grpo_preflight.py default --run <run>

Gates (docs/stage2-RL.md, Block 2):
  config      profile renders, memory budget < limit
  init        init model dir exists, fingerprint == config (the ep3 the heldout/tau2 numbers belong to)
  pool        parquets + manifest exist; manifest corridor sha == corridor file; row counts; roles;
              train ⊂ rl_train, val ⊂ heldout_eval, train ∩ val = ∅; tools_kwargs cover all tools
  registry    grpo_tool_config.yaml names == live tau2 env tools (13/13)
  flags       dataloader_num_workers=0, turns >= 15 (both), max_parallel_calls >= 10, val sampling set,
              custom dataset wired, ledger/run dir wired, no repetition_penalty override
  imports     CorridorDataset, grpo_reward.compute_score, DbBahnTool importable in this image
  roundtrip   grpo_reward parser vs stored scores on the ep3 heldout traces (--roundtrip-limit)
  prompt      INFO: unified-diff size between verl's rendered system prompt (template + tools) and
              rollout.py's eval prompt — known divergence, the step-0 validation is the real gate
  state       fresh run: no stale ledger/sampler state; resume: checkpoint dir present
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training_pipeline.grpo_config import load_profile, memory_budget, render_flags, run_paths  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, str, str]] = []


def gate(name: str, ok: bool | None, msg: str) -> None:
    status = "INFO" if ok is None else ("OK" if ok else "FAIL")
    RESULTS.append((status, name, msg))
    print(f"[{status:4s}] {name:10s} {msg}", flush=True)


def sha256_16(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def fingerprint_model(d: Path) -> str:
    """Same recipe as evaluation/benchmarks/tau2/preflight.py:fingerprint_model."""
    h = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "generation_config.json"):
        f = d / name
        if f.exists():
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--run", default=None)
    ap.add_argument("--roundtrip-limit", type=int, default=200)
    ap.add_argument("--roundtrip-file", type=Path,
                    default=REPO / "data/generated/eval/db_traces_heldout_eval_after_qwen3-4b_ep3_k4.jsonl")
    a = ap.parse_args()

    r = load_profile(a.profile, run=a.run)
    rp = run_paths(r)
    flags = render_flags(r)
    fl = {f.lstrip("+").split("=", 1)[0]: f.split("=", 1)[1] for f in flags}

    # --- config / memory ---------------------------------------------------------------------------
    m = memory_budget(r)
    gate("config", True, f"profile={r['profile']} run={r['run']} image={r['image']} flags={len(flags)}")
    gate("memory", m["ok"], f"rollout {m['rollout_gb']} + actor {m['actor_gb']} + overhead {m['overhead_gb']} "
                            f"= {m['total_gb']} GB (limit {m['limit_gb']})")

    # --- init model ----------------------------------------------------------------------------------
    init = Path(r["paths"]["init_model"])
    fp = fingerprint_model(init) if init.is_dir() else None
    gate("init", init.is_dir() and fp == r["paths"]["init_fingerprint"],
         f"{init} fingerprint={fp} expected={r['paths']['init_fingerprint']}")

    # --- pool ----------------------------------------------------------------------------------------
    import pandas as pd
    pool = Path(r["paths"]["pool_dir"])
    train_f, val_f, man_f = Path(rp["train_file"]), Path(rp["val_file"]), pool / "pool_manifest.json"
    ok_files = train_f.exists() and val_f.exists() and man_f.exists()
    gate("pool", ok_files, f"{train_f.name} {val_f.name} pool_manifest.json in {pool}")
    if not ok_files:
        return finish()
    man = json.loads(man_f.read_text())
    corr = Path(r["paths"]["corridor_file"])
    sha_ok = corr.exists() and sha256_16(corr) == man.get("corridor_sha256_16")
    gate("pool", sha_ok, f"manifest corridor sha {man.get('corridor_sha256_16')} == {corr.name} "
                         f"{sha256_16(corr) if corr.exists() else 'missing'}")
    train, val = pd.read_parquet(train_f), pd.read_parquet(val_f)
    t_ids = [x["ground_truth"] for x in train["reward_model"]]
    v_ids = [x["ground_truth"] for x in val["reward_model"]]
    splits = json.loads((REPO / "data/raw/db_sandbox/split_tasks.json").read_text())
    gate("pool", len(train) == man["n_train"] and len(train) > 0,
         f"train rows {len(train)} (manifest {man['n_train']}, corridor {man['n_corridor']}, templates {man['n_templates']})")
    exp_val = man["n_val"] if val_f.name == "val.parquet" else man["n_smoke_val"]
    gate("pool", len(val) == exp_val, f"val rows {len(val)} (expected {exp_val} for {val_f.name})")
    roles_t = {e.get("pool_role") for e in train["extra_info"]}
    roles_v = {e.get("pool_role") for e in val["extra_info"]}
    gate("pool", roles_t == {"train"} and "train" not in roles_v and not any("ledger_dir" in e for e in val["extra_info"]),
         f"roles train={roles_t} val={roles_v}, val carries no ledger_dir")
    gate("pool", set(t_ids) <= set(splits["rl_train"]) and set(v_ids) <= set(splits["heldout_eval"])
         and not (set(t_ids) & set(v_ids)),
         f"train ⊂ rl_train, val ⊂ heldout_eval, overlap {len(set(t_ids) & set(v_ids))}")
    for col in ("rate0", "template_rate0", "ever_failed", "template", "signal"):
        if not all(col in e for e in train["extra_info"]):
            gate("pool", False, f"extra_info.{col} missing in train rows")

    # --- registry: yaml tool names == live env tools == parquet tools_kwargs -----------------------
    import yaml
    from sdg_pipeline.db_bahn.tau2_domain import get_environment
    env_tools = sorted(t.name for t in get_environment(solo_mode=True).get_tools())
    reg = yaml.safe_load(Path(r["paths"]["tool_config"]).read_text())
    reg_tools = sorted(t["config"]["tool_name"] for t in reg["tools"])
    pq_tools = sorted(train["extra_info"].iloc[0]["tools_kwargs"].keys())
    gate("registry", env_tools == reg_tools == pq_tools,
         f"env {len(env_tools)} / registry {len(reg_tools)} / parquet {len(pq_tools)} tools identical")

    # --- flags ---------------------------------------------------------------------------------------
    mt = "actor_rollout_ref.rollout.multi_turn."
    checks = {
        "dataloader_num_workers=0": fl.get("data.dataloader_num_workers") == "0",
        "assistant_turns>=15": int(fl.get(mt + "max_assistant_turns", 0)) >= 15,
        "user_turns>=15": int(fl.get(mt + "max_user_turns", 0)) >= 15,
        "parallel_calls>=10": int(fl.get(mt + "max_parallel_calls", 0)) >= 10,
        "val_kwargs sampled": fl.get("actor_rollout_ref.rollout.val_kwargs.do_sample") == "True"
                              and int(fl.get("actor_rollout_ref.rollout.val_kwargs.n", 0)) >= 1,
        "custom dataset": fl.get("data.custom_cls.name") == "CorridorDataset",
        "run dir wired": fl.get("data.grpo_run_dir") == rp["run_dir"],
        "no rep-pen override": not any("repetition_penalty" in f for f in flags),
        "tool_agent loop": fl.get("actor_rollout_ref.rollout.agent.default_agent_loop") == "tool_agent",
        "load_format=auto": fl.get("actor_rollout_ref.rollout.load_format") == "auto",
        # an unbatched validation keeps every episode's tau2 env alive at once -> host OOM
        "val batched (<=64 tasks)": 0 < int(fl.get("data.val_batch_size", 0) or 0) <= 64,
        # updates per step = train_batch_size / ppo_mini_batch_size must be a whole number
        "mini-batch divides batch": (lambda tb, mb: tb > 0 and mb > 0 and tb % mb == 0)(
            int(fl.get("data.train_batch_size", 0) or 0), int(fl.get("actor_rollout_ref.actor.ppo_mini_batch_size", 0) or 0)),
        "warmup set": int(fl.get("actor_rollout_ref.actor.optim.lr_warmup_steps", -1)) >= 0,
    }
    bad = [k for k, v in checks.items() if not v]
    gate("flags", not bad, "all decision values present" if not bad else f"violations: {bad}")

    # --- imports -------------------------------------------------------------------------------------
    try:
        from training_pipeline.grpo_dataset import CorridorDataset  # noqa: F401
        from evaluation.grpo_reward import compute_score  # noqa: F401
        from training_pipeline.grpo_db_bahn_tools import DbBahnTool  # noqa: F401
        gate("imports", True, "CorridorDataset, grpo_reward.compute_score, DbBahnTool")
    except Exception as e:
        gate("imports", False, f"{type(e).__name__}: {e}")

    # --- round-trip ----------------------------------------------------------------------------------
    if a.roundtrip_limit > 0 and a.roundtrip_file.exists():
        from evaluation.grpo_reward import hermes_to_messages, render_verl_response
        from evaluation.trajectory_reward import score_trajectory
        tasks = {t["id"]: t for t in json.loads((REPO / "data/raw/db_sandbox/tasks.json").read_text())}
        keys = json.loads((REPO / "data/raw/db_sandbox/answer_keys.json").read_text())
        n = same = 0
        with open(a.roundtrip_file) as f:
            for line in f:
                if n >= a.roundtrip_limit:
                    break
                rec = json.loads(line)
                n += 1
                res = score_trajectory(tasks[rec["task_id"]], hermes_to_messages(render_verl_response(rec["messages"])),
                                       keys.get(rec["task_id"]))
                same += res["score"] == rec["score"]["score"] and res["components"] == rec["score"]["components"]
        gate("roundtrip", same == n, f"{same}/{n} identical scores ({a.roundtrip_file.name})")
    else:
        gate("roundtrip", None, "skipped")

    # --- prompt parity (informational) ----------------------------------------------------------------
    try:
        from transformers import AutoTokenizer
        from sdg_pipeline.db_bahn.rollout import SYSTEM_TEMPLATE
        tok = AutoTokenizer.from_pretrained(str(init))
        envx = get_environment(solo_mode=True)
        schemas = [t.openai_schema for t in envx.get_tools()]
        row = train.iloc[0]
        verl_prompt = tok.apply_chat_template(list(row["prompt"]), tools=schemas, tokenize=False, add_generation_prompt=True)
        eval_sys = SYSTEM_TEMPLATE.format(policy=envx.get_policy(),
                                          tools_block="\n".join(json.dumps(t, ensure_ascii=False) for t in schemas))
        eval_prompt = tok.apply_chat_template([{"role": "system", "content": eval_sys}, list(row["prompt"])[1]],
                                              tokenize=False, add_generation_prompt=True)
        diff = list(difflib.unified_diff(eval_prompt.splitlines(), verl_prompt.splitlines(), lineterm="", n=0))
        changed = sum(1 for d in diff if d[:1] in "+-" and not d.startswith(("+++", "---")))
        gate("prompt", None, f"verl prompt {len(tok(verl_prompt).input_ids)} tok vs eval prompt "
                             f"{len(tok(eval_prompt).input_ids)} tok; {changed} differing lines (known divergence)")
        # hard gate: every rendered prompt (train + val) must fit data.max_prompt_length with margin
        limit = int(fl["data.max_prompt_length"])
        longest = 0
        for df in (train, val):
            for pr in df["prompt"]:
                longest = max(longest, len(tok(tok.apply_chat_template(list(pr), tools=schemas, tokenize=False,
                                                                      add_generation_prompt=True)).input_ids))
        gate("prompt_len", longest + 64 <= limit, f"longest rendered prompt {longest} tok vs max_prompt_length {limit}")
    except Exception as e:
        gate("prompt", None, f"parity check skipped: {type(e).__name__}: {e}")

    # --- run state -----------------------------------------------------------------------------------
    ckpt = Path(rp["ckpt_dir"])
    steps = sorted(ckpt.glob("global_step_*")) if ckpt.exists() else []
    ledger = Path(rp["ledger_dir"])
    ledger_files = list(ledger.glob("*.jsonl")) if ledger.exists() else []
    if steps:
        gate("state", True, f"RESUME: {len(steps)} checkpoints in {ckpt}, ledger files {len(ledger_files)}")
    else:
        gate("state", not ledger_files and not Path(rp["sampler_dir"]).exists(),
             f"FRESH: no checkpoints; ledger files {len(ledger_files)}, sampler state "
             f"{'present' if Path(rp['sampler_dir']).exists() else 'absent'} (must be empty for a fresh run)")
    return finish()


def finish() -> int:
    fails = [x for x in RESULTS if x[0] == "FAIL"]
    print(f"\npreflight: {len([x for x in RESULTS if x[0] == 'OK'])} ok, {len(fails)} fail, "
          f"{len([x for x in RESULTS if x[0] == 'INFO'])} info")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

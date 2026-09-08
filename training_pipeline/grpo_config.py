#!/usr/bin/env python3
"""Render a profile of config/grpo_db_bahn.yaml into verl Hydra flags (+ memory budget, paths) and
persist/load the resolved configuration of a run.

    python3 training_pipeline/grpo_config.py default --run <run> --print-flags   # one flag per line (for the runner)
    python3 training_pipeline/grpo_config.py smoke --print-json                  # resolved profile
    python3 training_pipeline/grpo_config.py default --run <run> --memory        # budget check
    python3 training_pipeline/grpo_config.py default --run <run> --write-resolved <file>   # runs/<run>/resolved_config.json
    python3 training_pipeline/grpo_config.py --run-env <run>                     # KEY=value lines from a run's resolved config

A profile is a configuration variant; the run name is the identity. Every run keeps the resolved
configuration it was started with (`resolved_config.json`), so metrics and export read from there and
a later change of the defaults never re-interprets an old run.

Plain python3 + PyYAML. The mapping below is the single place where our settings meet verl's
config keys; the load-bearing GB10 flags (load_format=auto, sdpa, Dr.GRPO, tool_agent loop) are
inherited from the validated SQL-era pilot.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG_DEFAULT = REPO / "config/grpo_db_bahn.yaml"


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_profile(profile: str, config_path: Path = CONFIG_DEFAULT, run: str | None = None) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    if profile not in cfg["profiles"]:
        raise SystemExit(f"unknown profile {profile!r}; have {sorted(cfg['profiles'])}")
    prof = cfg["profiles"][profile] or {}
    resolved = _deep_merge(cfg["base"], {k: v for k, v in prof.items() if k != "run_default"})
    run = run or prof.get("run_default")
    if not run:
        raise SystemExit(f"profile {profile!r} has no default run name — give one: "
                         f"bash ops/grpo_run.sh {profile} <run-name>")
    resolved["run"] = run
    resolved["profile"] = profile
    resolved["image"] = cfg["image"]
    resolved["paths"] = copy.deepcopy(cfg["paths"])
    return resolved


def run_paths(r: dict) -> dict:
    p, run = r["paths"], r["run"]
    return {
        "run_dir": f"{p['runs_root']}/{run}",
        "ledger_dir": f"{p['runs_root']}/{run}/ledger",
        "sampler_dir": f"{p['runs_root']}/{run}/sampler",
        "rollout_dump": f"{p['runs_root']}/{run}/rollout_dump",
        "val_dump": f"{p['runs_root']}/{run}/val_dump",
        "ckpt_dir": f"{p['ckpt_root']}/{run}",
        "train_file": f"{p['pool_dir']}/train.parquet",
        "val_file": f"{p['pool_dir']}/{r['pool']['val_file']}",
    }


def memory_budget(r: dict) -> dict:
    m, gmu = r["memory"], r["rollout"]["gpu_memory_utilization"]
    rollout_gb = gmu * m["total_gb"]
    total = rollout_gb + m["actor_gb"] + m["overhead_gb"]
    return {"rollout_gb": round(rollout_gb, 1), "actor_gb": m["actor_gb"], "overhead_gb": m["overhead_gb"],
            "total_gb": round(total, 1), "limit_gb": m["limit_gb"], "ok": total <= m["limit_gb"]}


def resolved_config(r: dict) -> dict:
    """What a run was started with — written to runs/<run>/resolved_config.json by ops/grpo_run.sh."""
    import datetime
    import subprocess
    try:
        head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        head = None
    return {"schema": 1, "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "git_head": head, "flags": render_flags(r), "resolved": r}


def write_resolved(r: dict, path: Path) -> str:
    """Create the file, or verify an existing one (resume): the knobs must match, else refuse."""
    doc = resolved_config(r)
    if path.exists():
        old = json.loads(path.read_text()).get("resolved", {})
        diff = [k for k in set(old) | set(doc["resolved"]) if old.get(k) != doc["resolved"].get(k) and k != "profile"]
        if diff:
            raise SystemExit(f"{path} exists and differs in {sorted(diff)} — a resume must use the same "
                             f"configuration (or start a new run name)")
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1))
    return "written"


def load_run(run: str, config_path: Path = CONFIG_DEFAULT) -> dict:
    """The resolved configuration of an existing run (host path derived from paths.runs_root)."""
    cfg = yaml.safe_load(config_path.read_text())
    runs_root = cfg["paths"]["runs_root"]
    host_root = REPO / runs_root[len("/app/"):] if runs_root.startswith("/app/") else Path(runs_root)
    f = host_root / run / "resolved_config.json"
    if not f.exists():
        raise SystemExit(f"no resolved_config.json for run {run!r} under {host_root} — "
                         f"is the run name right? (runs: {sorted(p.name for p in host_root.iterdir() if p.is_dir()) if host_root.exists() else []})")
    return json.loads(f.read_text())["resolved"]


def render_flags(r: dict) -> list[str]:
    p, rp, t, ro, po, va = r["paths"], run_paths(r), r["train"], r["rollout"], r["pool"], r["val"]
    b = lambda x: "True" if x else "False"  # noqa: E731
    flags = [
        # --- algorithm: Dr.GRPO, no reference model ---
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=False",
        "algorithm.use_kl_in_reward=False",
        # --- data + the corridor dataset ---
        f"data.train_files={rp['train_file']}",
        f"data.val_files={rp['val_file']}",
        f"data.val_batch_size={va['batch_size']}",
        f"data.train_batch_size={t['prompts_per_step']}",
        f"data.max_prompt_length={ro['prompt_len']}",
        f"data.max_response_length={ro['response_len']}",
        "data.dataloader_num_workers=0",
        "data.return_raw_chat=True",
        "data.shuffle=True",
        f"data.seed={po['seed']}",
        f"data.custom_cls.path={p['dataset_cls']}",
        "data.custom_cls.name=CorridorDataset",
        f"+data.grpo_run_dir={rp['run_dir']}",
        f"+data.grpo_sampling={po['sampling']}",
        f"+data.grpo_epoch_size={po['epoch_size']}",
        f"+data.grpo_eps={po['eps']}",
        f"+data.grpo_mix={po['mix']}",
        f"+data.grpo_max_wait={po['max_wait']}",
        f"+data.grpo_seed={po['seed']}",
        # --- model: merged ep3 + fresh LoRA ---
        f"actor_rollout_ref.model.path={p['init_model']}",
        f"actor_rollout_ref.model.lora_rank={t['lora_rank']}",
        f"actor_rollout_ref.model.lora_alpha={t['lora_alpha']}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.use_remove_padding=False",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        # --- actor ---
        f"actor_rollout_ref.actor.optim.lr={t['lr']}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps={t.get('warmup_steps', -1)}",   # -1 = none (verl: ratio 0)
        f"actor_rollout_ref.actor.ppo_mini_batch_size={t.get('mini_batch', t['prompts_per_step'])}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={t['micro_bs']}",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.entropy_coeff=0",
        f"actor_rollout_ref.actor.use_kl_loss={b(t['kl'])}",
        f"actor_rollout_ref.actor.kl_loss_coef={t['kl_coef']}",
        "actor_rollout_ref.actor.kl_loss_type=low_var_kl",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        # --- rollout (in-process vLLM, tool agent loop) ---
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.n={t['n']}",
        f"actor_rollout_ref.rollout.temperature={ro['temperature']}",
        f"actor_rollout_ref.rollout.top_p={ro['top_p']}",
        f"actor_rollout_ref.rollout.top_k={ro['top_k']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={ro['gpu_memory_utilization']}",
        f"actor_rollout_ref.rollout.max_model_len={ro['prompt_len'] + ro['response_len']}",
        "actor_rollout_ref.rollout.load_format=auto",
        "actor_rollout_ref.rollout.enforce_eager=False",
        f"actor_rollout_ref.rollout.max_num_seqs={ro['max_num_seqs']}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={ro['max_num_batched_tokens']}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={t['logprob_micro_bs']}",
        "actor_rollout_ref.rollout.multi_turn.enable=True",
        f"actor_rollout_ref.rollout.multi_turn.tool_config_path={p['tool_config']}",
        f"actor_rollout_ref.rollout.multi_turn.max_assistant_turns={ro['max_assistant_turns']}",
        f"actor_rollout_ref.rollout.multi_turn.max_user_turns={ro['max_user_turns']}",
        f"actor_rollout_ref.rollout.multi_turn.max_parallel_calls={ro['max_parallel_calls']}",
        f"actor_rollout_ref.rollout.multi_turn.max_tool_response_length={ro['max_tool_response_length']}",
        "actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right",
        "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
        # --- validation sampling (verl default would be greedy, n=1) ---
        f"actor_rollout_ref.rollout.val_kwargs.n={va['n']}",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={va['temperature']}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={va['top_p']}",
        f"actor_rollout_ref.rollout.val_kwargs.top_k={va['top_k']}",
        # --- reward ---
        f"reward.custom_reward_function.path={p['reward_fn']}",
        "reward.custom_reward_function.name=compute_score",
        # --- trainer ---
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        f"trainer.total_epochs={t['total_epochs']}",
        f"trainer.save_freq={t['save_freq']}",
        f"trainer.test_freq={t['test_freq']}",
        f"trainer.val_before_train={b(t['val_before_train'])}",
        "trainer.resume_mode=auto",
        f"trainer.default_local_dir={rp['ckpt_dir']}",
        f"trainer.rollout_data_dir={rp['rollout_dump']}",
        f"trainer.validation_data_dir={rp['val_dump']}",
        "trainer.logger=[console,mlflow]",
        f"trainer.project_name={r['mlflow']['project']}",
        f"trainer.experiment_name={r['run']}",
    ]
    if t.get("total_training_steps"):
        flags.append(f"trainer.total_training_steps={t['total_training_steps']}")
    return flags


def main() -> None:
    ap = argparse.ArgumentParser(description="render a GRPO profile into verl flags / read a run's resolved config")
    ap.add_argument("profile", nargs="?", default=None)
    ap.add_argument("--config", type=Path, default=CONFIG_DEFAULT)
    ap.add_argument("--run", default=None, help="run name (required for profiles without run_default)")
    ap.add_argument("--write-resolved", type=Path, default=None, help="write/verify runs/<run>/resolved_config.json")
    ap.add_argument("--run-env", default=None, metavar="RUN", help="KEY=value lines from an existing run's resolved config")
    ap.add_argument("--print-flags", action="store_true")
    ap.add_argument("--print-json", action="store_true")
    ap.add_argument("--memory", action="store_true")
    ap.add_argument("--paths", action="store_true")
    ap.add_argument("--env", action="store_true", help="shell KEY=value lines (run, image, run_dir, ckpt_dir)")
    a = ap.parse_args()
    if a.run_env:
        r = load_run(a.run_env, a.config)
        rp = run_paths(r)
        for k, v in (("RUN", r["run"]), ("PROFILE", r.get("profile")), ("IMAGE", r["image"]), ("RUN_DIR", rp["run_dir"]),
                     ("CKPT_DIR", rp["ckpt_dir"]), ("INIT_MODEL", r["paths"]["init_model"]), ("VAL_DUMP", rp["val_dump"])):
            print(f"{k}={shlex.quote(str(v))}")
        return
    if not a.profile:
        ap.error("profile required (or --run-env RUN)")
    r = load_profile(a.profile, a.config, a.run)
    if a.write_resolved:
        print(f"resolved_config: {write_resolved(r, a.write_resolved)} -> {a.write_resolved}")
    if a.print_flags:
        print("\n".join(render_flags(r)))
    if a.print_json:
        print(json.dumps({k: v for k, v in r.items() if k != "paths"}, indent=1))
    if a.paths:
        print(json.dumps({**r["paths"], **run_paths(r), "run": r["run"], "image": r["image"]}, indent=1))
    if a.env:
        rp = run_paths(r)
        for k, v in (("RUN", r["run"]), ("IMAGE", r["image"]), ("RUN_DIR", rp["run_dir"]), ("CKPT_DIR", rp["ckpt_dir"]),
                     ("LEDGER_DIR", rp["ledger_dir"]), ("VAL_DUMP", rp["val_dump"]), ("ROLLOUT_DUMP", rp["rollout_dump"])):
            print(f"{k}={shlex.quote(str(v))}")
    if a.memory:
        m = memory_budget(r)
        print(json.dumps(m))
        sys.exit(0 if m["ok"] else 1)


if __name__ == "__main__":
    main()

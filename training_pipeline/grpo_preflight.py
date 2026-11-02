#!/usr/bin/env python3
"""Preflight gates for a Stage-2 GRPO run — runs INSIDE the grpo container (CPU only), called by
ops/grpo_run.sh before verl starts. Every gate prints OK/FAIL/INFO; any FAIL exits 1. Each gate catches
its own exceptions (the runner shows only the gate lines, no tracebacks).

    docker compose --profile grpo run --rm -T grpo training_pipeline/grpo_preflight.py default --run <run>

Gates:
  config      profile renders, memory budget < limit
  init        init model dir exists, config fingerprint == config (identical for every 4B merge)
  weights     sha256_16 over the init model's *.safetensors == paths.init_weights_fingerprint (= ep3)
  pool        parquets + manifest exist; manifest corridor sha == corridor file; row counts; roles;
              train ⊂ rl_train, val ⊂ heldout_eval, train ∩ val = ∅; tools_kwargs cover all tools
  registry    grpo_tool_config.yaml names == live tau2 env tools (13/13)
  flags       grpo_config.flag_checks: dataloader workers, turn/parallel caps, val sampling, dataset +
              run dir wired, no repetition_penalty; KL -> ref micro-batch, cosine horizon, raw log-probs,
              confidence weighting wired iff enabled
  hydra       verl composes ppo_trainer with exactly these flags, the values arrive, validate_config passes,
              actor/rollout/model configs build (the model config imports model.external_lib)
  sched       verl's own lr scheduler, stepped like the trainer, == grpo_config.lr_factor for every step
  logprobs    rollout.logprobs_mode is a mode the image's vLLM accepts
  loss        conf_weighted registered + equals vanilla on hand-weighted advantages (hard iff enabled)
  imports     CorridorDataset, grpo_reward.compute_score, DbBahnTool importable in this image
  roundtrip   reward path (render -> parse -> verifier, and the full compute_score seam with constant
              keys) vs stored scores: first --roundtrip-limit rows + every failed row (--roundtrip-all)
  prompt      INFO: unified-diff size between verl's rendered system prompt (template + tools) and
              rollout.py's eval prompt — known divergence, the step-0 validation is the real gate
  prompt_len  every rendered train/val prompt fits data.max_prompt_length
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

from training_pipeline.grpo_config import (CONF_LOSS_LIB, CONF_LOSS_MODE, conf_weight, flag_checks,  # noqa: E402
                                           flag_map, load_profile, lr_factor, memory_budget, render_flags,
                                           run_paths)

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


def weights_fingerprint(d: Path) -> str | None:
    """Same recipe as evaluation/benchmarks/tau2/preflight.py:weights_fingerprint (sha256_16 over the
    *.safetensors bytes, sorted by name)."""
    files = sorted(d.glob("*.safetensors"))
    if not files:
        return None
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        with open(f, "rb") as fh:
            while chunk := fh.read(16 << 20):
                h.update(chunk)
    return h.hexdigest()[:16]


def guarded(name: str, fn) -> None:
    """fn() -> (ok | None, msg); an exception is a FAIL with its message."""
    try:
        ok, msg = fn()
    except Exception as e:
        ok, msg = False, f"{type(e).__name__}: {str(e)[:300]}"
    gate(name, ok, msg)


def compose_verl(flags: list[str]):
    """verl's ppo_trainer config with our flags, the way main_ppo builds it (hydra + legacy-reward migration)."""
    import os

    import verl
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from verl.experimental.reward_loop import migrate_legacy_reward_impl
    with initialize_config_dir(config_dir=os.path.join(os.path.dirname(verl.__file__), "trainer", "config"),
                               version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=list(flags))
    cfg = migrate_legacy_reward_impl(cfg)
    OmegaConf.resolve(cfg)
    return cfg


def hydra_check(r: dict, cfg) -> tuple[bool, str]:
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import omega_conf_to_dataclass, validate_config
    t, ro, cw = r["train"], r["rollout"], conf_weight(r)
    arr, act = cfg.actor_rollout_ref, cfg.actor_rollout_ref.actor
    want = {
        "use_kl_loss": (act.use_kl_loss, bool(t["kl"])),
        "kl_loss_coef": (float(act.kl_loss_coef), float(t["kl_coef"])),
        "lr_scheduler_type": (act.optim.lr_scheduler_type, t.get("lr_schedule") or "constant"),
        "total_training_steps": (cfg.trainer.total_training_steps, t.get("total_training_steps") or None),
        "calculate_log_probs": (bool(arr.rollout.calculate_log_probs), bool(ro.get("calculate_log_probs", False))),
        "logprobs_mode": (arr.rollout.logprobs_mode, ro.get("logprobs_mode") or "processed_logprobs"),
        "loss_mode": (act.policy_loss.loss_mode, CONF_LOSS_MODE if cw["enabled"] else "vanilla"),
        "external_lib": (arr.model.external_lib, CONF_LOSS_LIB if cw["enabled"] else None),
    }
    if t.get("lr_schedule"):
        want["min_lr_ratio"] = (float(act.optim.min_lr_ratio), float(t.get("min_lr_ratio", 0.0)))
    bad = {k: v for k, v in want.items() if v[0] != v[1]}
    if bad:
        return False, f"values did not arrive: {bad}"
    use_ref = need_reference_policy(cfg)
    if use_ref != bool(t["kl"]):
        return False, f"need_reference_policy={use_ref} but train.kl={t['kl']}"
    validate_config(config=cfg, use_reference_policy=use_ref, use_critic=need_critic(cfg))
    omega_conf_to_dataclass(arr.rollout)
    omega_conf_to_dataclass(arr.model)          # imports model.external_lib (the confidence loss) if set
    return True, (f"composed + validate_config ok; ref policy {'on' if use_ref else 'off'}, scheduler "
                  f"{act.optim.lr_scheduler_type}, loss {act.policy_loss.loss_mode}, logprobs {arr.rollout.logprobs_mode}")


def sched_check(r: dict, cfg) -> tuple[bool | None, str]:
    import torch
    from verl.utils.torch_functional import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
    t = r["train"]
    total = int(t.get("total_training_steps") or 0)
    warmup, sched, min_ratio = int(t.get("warmup_steps", -1)), t.get("lr_schedule") or "constant", float(t.get("min_lr_ratio", 0.0))
    if total <= 0:
        return None, "no total_training_steps — horizon = epochs x steps, not checked"
    opt = cfg.actor_rollout_ref.actor.optim
    if sched == "cosine" and (float(opt.num_cycles) != 0.5 or not bool(opt.zero_indexed_step)):
        return False, f"optim.num_cycles={opt.num_cycles} zero_indexed_step={opt.zero_indexed_step} — lr_factor assumes 0.5/True"
    sgd = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    s = (get_cosine_schedule_with_warmup(sgd, num_warmup_steps=max(warmup, 0), num_training_steps=total, min_lr_ratio=min_ratio)
         if sched == "cosine" else get_constant_schedule_with_warmup(sgd, num_warmup_steps=max(warmup, 0)))
    got = []
    for _ in range(total):
        got.append(sgd.param_groups[0]["lr"])   # the lr the trainer's step k uses
        sgd.step()
        s.step()
    exp = [lr_factor(k, warmup, total, min_ratio, sched) for k in range(1, total + 1)]
    dev = max(abs(a - b) for a, b in zip(got, exp))
    shown = [k for k in (1, warmup, warmup + 1, 24, 32, total) if 1 <= k <= total]
    table = " ".join(f"{k}:{got[k - 1]:.4f}" for k in dict.fromkeys(shown))
    if warmup >= total:
        return None, f"horizon {total} <= warmup {warmup}: the run ends inside the warmup ({table}); verl == lr_factor: {dev < 1e-9}"
    return dev < 1e-9, f"{sched} warmup {warmup} -> {total} steps, min {min_ratio}: lr factor per step {table} (max dev {dev:.1e})"


def logprobs_check(r: dict) -> tuple[bool | None, str]:
    import typing
    mode = r["rollout"].get("logprobs_mode")
    if not mode:
        return None, "logprobs_mode not set (verl default processed_logprobs)"
    from vllm.config.model import LogprobsMode
    modes = typing.get_args(LogprobsMode)
    return mode in modes, f"{mode} in vLLM LogprobsMode {modes}; calculate_log_probs={r['rollout'].get('calculate_log_probs', False)}"


def loss_check(r: dict) -> tuple[bool | None, str]:
    cw = conf_weight(r)
    import training_pipeline.grpo_conf_loss as cl
    problems = cl.verl_check(tau=cw["tau"] if cw["enabled"] else 0.7, factor=cw["factor"])
    if cw["enabled"]:
        return not problems, (f"{CONF_LOSS_MODE} tau={cw['tau']} factor={cw['factor']}: registered, == vanilla on weighted "
                              f"advantages" if not problems else f"{problems}")
    return None, f"confidence weighting off (vanilla loss); module check {'ok' if not problems else problems}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--run", default=None)
    ap.add_argument("--roundtrip-limit", type=int, default=200)
    ap.add_argument("--roundtrip-all", action="store_true", help="round-trip every row of --roundtrip-file")
    ap.add_argument("--roundtrip-file", type=Path,
                    default=REPO / "data/generated/eval/db_traces_heldout_eval_after_qwen3-4b_ep3_k8.jsonl")
    a = ap.parse_args()

    r = load_profile(a.profile, run=a.run)
    rp = run_paths(r)
    flags = render_flags(r)
    fl = flag_map(flags)

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
    exp_w = r["paths"].get("init_weights_fingerprint")
    if exp_w:
        guarded("weights", lambda: (lambda w: (w == exp_w, f"safetensors sha256_16={w} expected={exp_w}"))(
            weights_fingerprint(init) if init.is_dir() else None))
    else:
        gate("weights", None, "paths.init_weights_fingerprint not set — skipped")

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
    try:
        bad = [k for k, v in flag_checks(r, flags).items() if not v]
        gate("flags", not bad, f"all decision values present ({len(flags)} flags)" if not bad else f"violations: {bad}")
    except Exception as e:
        gate("flags", False, f"{type(e).__name__}: {e}")

    # --- verl's own view of the flags: hydra compose, validate_config, scheduler, log-probs, loss ------
    cfg = None
    try:
        cfg = compose_verl(flags)
    except Exception as e:
        gate("hydra", False, f"compose failed: {type(e).__name__}: {str(e)[:300]}")
    if cfg is not None:
        guarded("hydra", lambda: hydra_check(r, cfg))
        guarded("sched", lambda: sched_check(r, cfg))
    guarded("logprobs", lambda: logprobs_check(r))
    guarded("loss", lambda: loss_check(r))

    # --- imports -------------------------------------------------------------------------------------
    try:
        from training_pipeline.grpo_dataset import CorridorDataset  # noqa: F401
        from evaluation.grpo_reward import compute_score  # noqa: F401
        from training_pipeline.grpo_db_bahn_tools import DbBahnTool  # noqa: F401
        gate("imports", True, "CorridorDataset, grpo_reward.compute_score, DbBahnTool")
    except Exception as e:
        gate("imports", False, f"{type(e).__name__}: {e}")

    # --- round-trip ----------------------------------------------------------------------------------
    if (a.roundtrip_limit > 0 or a.roundtrip_all) and a.roundtrip_file.exists():
        def _rt():
            from evaluation.grpo_reward import roundtrip_file
            rt = roundtrip_file(a.roundtrip_file, None if a.roundtrip_all else a.roundtrip_limit, include_fails=True, verbose=3)
            return rt["ok"] == rt["n"] and rt["failed_rows"] > 0, (
                f"{rt['ok']}/{rt['n']} identical incl. {rt['failed_rows']} failed rows, keys constant "
                f"({a.roundtrip_file.name}, {rt['seconds']}s)" + (f" problems {rt['problems']} e.g. {rt['examples']}"
                                                                   if rt["problems"] else ""))
        guarded("roundtrip", _rt)
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
    except Exception as e:
        gate("prompt", None, f"parity check skipped: {type(e).__name__}: {e}")

    # hard gate: every rendered prompt (train + val) must fit data.max_prompt_length with margin
    def _prompt_len():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(init))
        schemas = [t.openai_schema for t in get_environment(solo_mode=True).get_tools()]
        limit = int(fl["data.max_prompt_length"])
        longest = 0
        for df in (train, val):
            for pr in df["prompt"]:
                longest = max(longest, len(tok(tok.apply_chat_template(list(pr), tools=schemas, tokenize=False,
                                                                      add_generation_prompt=True)).input_ids))
        return longest + 64 <= limit, f"longest rendered prompt {longest} tok vs max_prompt_length {limit}"
    guarded("prompt_len", _prompt_len)

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

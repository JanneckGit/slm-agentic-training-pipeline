"""
training_pipeline/grpo_verl_runner.py
=====================================
GRPO (LoRA) runner for the dense Qwen3-4B SFT-thinking student via **verl** (async server-mode)
on GB10/sm_121. Reads the `grpo:` block of config/pipeline_config.yaml, builds verl's Parquet data
+ the load-bearing Hydra override command, wires MLflow into the SFT store, and launches
`python -m verl.trainer.main_ppo`.

RUN IN THE GRPO CONTAINER (Dockerfile.grpo / `grpo` compose service) with the repo mounted at /app
and data at /app/data:
    # full pilot (needs reachability output as train_files):
    python3 training_pipeline/grpo_verl_runner.py --config config/pipeline_config.yaml --date 20260625
    # end-to-end SMOKE (raw weak_candidates, ~4 steps) — proves the pipeline, NOT the training signal:
    python3 training_pipeline/grpo_verl_runner.py --smoke --date 20260625
    # inspect the exact verl command without launching:
    python3 training_pipeline/grpo_verl_runner.py --smoke --dry-run

Load-bearing recipe (verbatim, see docs/experiments-verl_RL_lora-grpo.md): model_dtype=bfloat16,
ref-skip (use_kl_loss=False + use_kl_in_reward=False), rollout TP=1, mode=async,
cudagraph ON (FULL_AND_PIECEWISE) + VLLM_USE_FLASHINFER_SAMPLER=0 (GB10 wedge fix, vLLM #43885),
attn_implementation=sdpa, Dr.GRPO (norm_adv_by_std_in_grpo=False).
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent

# EXACT system prompt the 4B-thinking student was SFT'd on (close_rate_probe / reachability_probe).
SYS_THINK = """You are an expert SQL query writer. Given a natural language question and a database schema, write the correct SQL query that answers the question.

Think through the problem step by step before writing the SQL:
1. Identify which tables are needed
2. Determine what joins are required
3. Figure out what filters, aggregations, or ordering to apply
4. Write the final SQL

Output your reasoning in <think>...</think> tags, then the SQL query."""


def _to_parquet(jsonl_path, parquet_path, g, n_rows=None):
    """Convert a weak-pool JSONL (question/schema/gold_sql/complexity) to verl's Parquet schema:
    prompt(chat) + reward_model{style:rule, ground_truth} + extra_info{schema, reward_timeout_s, row_cap}."""
    import pandas as pd
    rows = [json.loads(l) for l in open(jsonl_path) if l.strip()]
    if n_rows:
        rows = rows[:n_rows]
    recs = []
    for i, r in enumerate(rows):
        recs.append({
            "data_source": "sql_exec",
            "prompt": [
                {"role": "system", "content": SYS_THINK},
                {"role": "user", "content": f"Database schema:\n{r['schema']}\n\nQuestion: {r['question']}"},
            ],
            "reward_model": {"style": "rule", "ground_truth": r["gold_sql"]},
            "extra_info": {"index": i, "schema": r["schema"],
                           "reward_timeout_s": g["reward_timeout_s"], "row_cap": g["row_cap"]},
        })
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_parquet(parquet_path, index=False)
    return len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline_config.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end smoke (raw candidates, ~4 steps)")
    ap.add_argument("--date", default="run", help="appended to the MLflow run name")
    ap.add_argument("--dry-run", action="store_true", help="print the verl command, do not launch")
    args = ap.parse_args()

    g = yaml.safe_load(open(args.config))["grpo"]
    if args.smoke:
        g = {**g, **g.get("smoke", {})}   # smoke overrides win

    # --- build Parquet data --------------------------------------------------------------
    out_dir = REPO / "data/final/grpo" / ("verl_smoke" if args.smoke else "verl_run")
    train_pq, val_pq = out_dir / "train.parquet", out_dir / "val.parquet"
    n_train = _to_parquet(REPO / g["train_files"], train_pq, g, n_rows=g.get("n_prompts"))
    n_val = _to_parquet(REPO / g["val_files"], val_pq, g, n_rows=g.get("n_val"))
    max_model_len = g["max_prompt_length"] + g["max_completion_length"]
    tbs = g["train_batch_size"]
    run_name = f"{g['run_name_prefix']}{'_smoke' if args.smoke else ''}_{args.date}"
    print(f"[data] train={n_train} ({train_pq})  val={n_val} ({val_pq})  max_model_len={max_model_len}")
    print(f"[mlflow] uri={g['mlflow_tracking_uri']}  experiment={g['mlflow_experiment']}  run={run_name}")

    # --- verl Hydra override command (load-bearing recipe) -------------------------------
    cmd = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=False",          # Dr.GRPO: advantage = reward - group_mean
        "algorithm.use_kl_in_reward=False",                 # ref-skip (beta 0)
        f"data.train_files={train_pq}", f"data.val_files={val_pq}",
        f"data.train_batch_size={tbs}",
        f"data.max_prompt_length={g['max_prompt_length']}",
        f"data.max_response_length={g['max_completion_length']}",
        f"actor_rollout_ref.model.path={REPO / g['base_model']}",
        f"actor_rollout_ref.model.lora_rank={g['lora_rank']}",
        f"actor_rollout_ref.model.lora_alpha={g['lora_alpha']}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.model.use_remove_padding=False",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",   # CRITICAL (flash_attn_2 crashes)
        f"actor_rollout_ref.actor.optim.lr={g['learning_rate']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={tbs}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.use_kl_loss=False",        # beta 0
        "actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",   # 4B fits without offload
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.n={g['num_generations']}",
        f"actor_rollout_ref.rollout.temperature={g['temperature']}",
        # loop-fix: tighten the permissive verl rollout defaults (top_p 1.0 / top_k -1) toward Qwen3,
        # AND apply repetition_penalty — the lever that actually terminates (standalone: rep-pen alone
        # closes the loop). Stock verl 0.8.0 HARD-CODES repetition_penalty=1.0 in the rollout dict
        # (agent_loop.py:500 / main_ppo_sync.py:313), so this image carries a SOURCE PATCH
        # (docker/Dockerfile.grpo: rep_pen patch step) changing those to `config.repetition_penalty`.
        # → `rollout.repetition_penalty` now reaches the rollout sampler. (val_kwargs.repetition_penalty
        # still MUST stay out — SamplingConfig has no such field → ConfigKeyError; the base
        # config.repetition_penalty flows to val too, which is fine at greedy.)
        f"actor_rollout_ref.rollout.top_p={g['top_p']}",
        f"actor_rollout_ref.rollout.top_k={g['top_k']}",
        # '+' is REQUIRED: rollout.yaml schema has no repetition_penalty key (struct-locked → plain
        # override ConfigAttributeError), but the RolloutConfig dataclass DOES have the field (rollout.py:182),
        # so '+' adds the key, dataclass-merge accepts it, and the patched agent_loop reads config.repetition_penalty.
        f"+actor_rollout_ref.rollout.repetition_penalty={g['repetition_penalty']}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={g['gpu_memory_utilization']}",
        f"actor_rollout_ref.rollout.max_model_len={max_model_len}",
        # load_format: dummy = rollout vLLM starts with RANDOM weights, relies ENTIRELY on the
        # ParameterSynchronizer weight-sync (base via load_weights then LoRA via add_lora). The base-sync
        # was the bug (rollout dumped GARBAGE token-soup). `auto` loads the real merged base from disk
        # (model.path) so the rollout is coherent from step 1; the LoRA still syncs on top (add_lora).
        f"actor_rollout_ref.rollout.load_format={g.get('load_format', 'dummy')}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0",        # greedy held-out EX
        "actor_rollout_ref.rollout.val_kwargs.do_sample=False",
        # NB: NO val_kwargs.repetition_penalty — SamplingConfig has no such field → ConfigKeyError at
        # config load (verl 0.8.0). Greedy val also can't get an anti-loop rep-pen without a verl patch.
        # cudagraph: enforce_eager=False keeps verl's default cudagraph_mode=FULL_AND_PIECEWISE (cudagraph ON).
        # The GB10 rollout wedge was NOT the cudagraph path — enforce_eager was a RED HERRING (the rollout
        # wedged at mns=32 *despite* eager). Root cause = the FlashInfer top-k/top-p sampler kernel race
        # (RadixTopKMaskLogitsMultiCTA, sm_120/121; vLLM #43885), fixed by VLLM_USE_FLASHINFER_SAMPLER=0
        # (set in env above). max_num_seqs is only a wedge-FREQUENCY lever, not a fix. Verified 2026-06-30:
        # cudagraph ON + sampler off + mns=32 ran 3 verl steps wedge-free (diagnosis: see experiments doc).
        f"actor_rollout_ref.rollout.enforce_eager={g['enforce_eager']}",
        f"actor_rollout_ref.rollout.max_num_seqs={g['rollout_max_num_seqs']}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={g['rollout_max_num_batched_tokens']}",
        f"reward.custom_reward_function.path={REPO / 'evaluation/reward.py'}",
        "reward.custom_reward_function.name=compute_score",
        "trainer.n_gpus_per_node=1", "trainer.nnodes=1",
        f"trainer.total_epochs={g['total_epochs']}",
        # [C] wedge-recovery: verl has NO rollout watchdog and its vLLM is in-process/colocated (not
        # separately restartable), so the only recovery from the GB10 vLLM wedge is a process-level
        # kill+restart. Checkpoint every save_freq steps + resume_mode=auto (restores step/dataloader/
        # LoRA/optimizer) so a wedge-restart loses <= save_freq steps. default_local_dir must be on the
        # host-mounted volume (/app -> repo) and STABLE across relaunches (same --date) for auto-resume.
        f"trainer.save_freq={g.get('save_freq', 5)}",
        f"trainer.resume_mode={g.get('resume_mode', 'auto')}",
        f"trainer.default_local_dir={REPO / 'data/final/grpo/verl_ckpt' / run_name}",
        f"trainer.test_freq={g.get('test_freq', 1)}",
        f"trainer.val_before_train={g.get('val_before_train', True)}",
        "trainer.logger=[console,mlflow]",
        f"trainer.project_name={g['mlflow_experiment']}",     # -> MLflow experiment
        f"trainer.experiment_name={run_name}",                # -> MLflow run name
    ]
    # diagnostic (optional): dump the rendered rollout PROMPT (input) + COMPLETION (output) per train
    # step to <dir>/{step}.jsonl (decoded text). Set grpo.rollout_data_dir to enable. Fires every
    # train step, independent of validation (ray_trainer.py:1681).
    if g.get("rollout_data_dir"):
        cmd.append(f"trainer.rollout_data_dir={REPO / g['rollout_data_dir']}")

    env = dict(os.environ)
    env["MLFLOW_TRACKING_URI"] = g["mlflow_tracking_uri"]      # same store as the SFT runs
    env["REWARD_TOKENIZER"] = g["reward_tokenizer"]            # reward.py think-token count
    env.setdefault("RAY_memory_monitor_refresh_ms", "0")      # Ray OOM-killer false-positives on unified mem
    env.setdefault("HYDRA_FULL_ERROR", "1")
    # GB10 sm_121 FlashInfer top-k/top-p sampler race (vLLM #43885) -> PyTorch-native sampler. THE GB10
    # rollout-wedge fix (enforce_eager was a red herring; the wedge sat in the sampler kernel, not cudagraph).
    # setdefault so an explicit =1 from outside can still override. Reaches verl's colocated Ray rollout
    # worker (verified via /proc/<worker>/environ, 2026-06-30). Belt-and-suspenders with docker-compose grpo env.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    print("\n[verl command]\n" + " \\\n  ".join(cmd) + "\n")
    if args.dry_run:
        print("[dry-run] not launching.")
        return
    subprocess.run(cmd, env=env, cwd=str(REPO), check=True)


if __name__ == "__main__":
    main()

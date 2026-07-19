#!/usr/bin/env bash
# =============================================================================
# Stage-2 GRPO on db_bahn — verl tool-agent loop against the REAL tau2 tools.
#
#   bash ops/grpo_smoke.sh                     # smoke: 8 prompts x n=4, 2 steps
#   TBS=16 N=8 STEPS=4 SAVE=2 bash ops/grpo_smoke.sh   # the short run
#
# Init = base Qwen3-4B and rollouts THINK (Qwen3 templates think unless told otherwise; verl's agent
# loop passes no enable_thinking kwarg -> nothing to configure). Reward = the SAME deterministic
# verifier the SDG/eval path uses, via evaluation/grpo_reward.py (Hermes text -> messages -> replay).
#
# The load-bearing GB10/verl config below is inherited verbatim from the validated SQL-era pilot:
# Dr.GRPO + ref-skip, load_format=auto
# (else the rollout serves RANDOM weights), attn_implementation=sdpa (FA2 fails the actor forward on
# sm_121), rep-pen 1.1 (needs the image's source patch + the leading `+`), and the wedge fix
# VLLM_USE_FLASHINFER_SAMPLER=0 (set in the compose service).
# Agentic delta vs. that pilot: rollout.agent.default_agent_loop=tool_agent (REQUIRED — multi_turn.enable
# alone is a no-op) + the 12-tool config + a multi-turn response budget.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

export GRPO_IMAGE="${GRPO_IMAGE:-agentic-grpo:verl}"   # frozen verl stack + tau2 (docker/Dockerfile.grpo-agentic)
COMPOSE="docker compose -f docker/docker-compose.yml"
RUN="${RUN:-grpo_smoke_$(date +%Y%m%d)}"
TBS="${TBS:-8}"          # prompts per step
N="${N:-4}"              # rollouts per prompt (the GRPO group)
STEPS="${STEPS:-2}"
SAVE="${SAVE:-1}"
RESP="${RESP:-12288}"    # TOTAL response budget per episode (all turns + tool observations)
PROMPT_LEN="${PROMPT_LEN:-4096}"
GMU="${GMU:-0.45}"       # rollout vLLM share; the FSDP actor needs the rest
mkdir -p logs

echo "== GRPO run=$RUN | TBS=$TBS n=$N steps=$STEPS resp=$RESP gmu=$GMU | image=$GRPO_IMAGE"

$COMPOSE --profile grpo run --rm -T grpo -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.use_kl_in_reward=False \
  data.train_files=/app/data/final/grpo/db_bahn_pool/train.parquet \
  data.val_files=/app/data/final/grpo/db_bahn_pool/val.parquet \
  data.train_batch_size="$TBS" \
  data.max_prompt_length="$PROMPT_LEN" \
  data.max_response_length="$RESP" \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="$TBS" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n="$N" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.top_k=20 \
  +actor_rollout_ref.rollout.repetition_penalty=1.1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GMU" \
  actor_rollout_ref.rollout.max_model_len=$((PROMPT_LEN + RESP)) \
  actor_rollout_ref.rollout.load_format=auto \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.max_num_seqs=32 \
  actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=/app/training_pipeline/grpo_tool_config.yaml \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=8 \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=8 \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4000 \
  actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  reward.custom_reward_function.path=/app/evaluation/grpo_reward.py \
  reward.custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.total_epochs=10 \
  trainer.total_training_steps="$STEPS" \
  trainer.save_freq="$SAVE" \
  trainer.resume_mode=auto \
  trainer.default_local_dir="/app/data/final/grpo/verl_ckpt/$RUN" \
  trainer.test_freq=100 \
  trainer.val_before_train=False \
  trainer.rollout_data_dir="/app/data/final/grpo/rollout_dump/$RUN" \
  trainer.logger=[console,mlflow] \
  trainer.project_name=db_bahn_grpo \
  trainer.experiment_name="$RUN" \
  2>&1 | tee -a "logs/${RUN}.log" | grep -vE "^\s*$"

echo "== done: ckpt=data/final/grpo/verl_ckpt/$RUN | dump=data/final/grpo/rollout_dump/$RUN | log=logs/${RUN}.log"

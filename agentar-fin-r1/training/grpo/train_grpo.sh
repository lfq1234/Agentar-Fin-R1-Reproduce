#!/usr/bin/env bash
# ============================================================================
# Agentar-Fin-R1 — Stage 2 GRPO (verl)
# ----------------------------------------------------------------------------
# 在 Stage 1 merge 后的完整模型（outputs/sft_merged）上做 GRPO 强化学习攻坚。
# 基座 Qwen3-8B + LoRA(r=32, alpha=64, all-linear)，vLLM rollout，FSDP 训练，
# 外部 72B LLM 裁判（fin_judge reward manager，见 fin_judge_reward.py）。
#
# 与旧 ms-swift 版等价映射：
#   num_generations=8          → actor_rollout_ref.rollout.n=8        (group size K)
#   beta=0.04                  → actor_rollout_ref.actor.kl_loss_coef=0.04
#   temperature=0.9 / top_p=0.9→ rollout.temperature / rollout.top_p
#   max_completion_length=1024 → data.max_response_length=1024
#   max_prompt_length=2048     → data.max_prompt_length=2048
#   learning_rate=5e-6         → actor.optim.lr=5e-6
#   gradient_accumulation=8    → data.train_batch_size=64 + ppo_mini_batch_size=64
#   use_vllm / colocate        → rollout.name=vllm（verl 原生 vLLM rollout）
#   reward_funcs=judge_reward  → custom_reward_function.name=compute_score
#
# 运行顺序：① SFT → ② merge_lora.py → ③ 本脚本
#
# 用法：
#   NPROC=1 \
#   SFT_MERGED=./outputs/sft_merged \
#   GRPO_DATA=./data/verl/grpo.parquet \
#     bash training/grpo/train_grpo.sh
# ============================================================================
set -xeuo pipefail

SFT_MERGED=${SFT_MERGED:-./outputs/sft_merged}
GRPO_DATA=${GRPO_DATA:-./data/verl/grpo.parquet}
NPROC=${NPROC:-1}
REWARD_SCRIPT=${REWARD_SCRIPT:-./training/grpo/fin_judge_reward.py}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${GRPO_DATA} \
    data.val_files=${GRPO_DATA} \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=${SFT_MERGED} \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.04 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.param_dtype=float16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=float16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_dtype=float16 \
    actor_rollout_ref.rollout.reward_model.enable=False \
    custom_reward_function.path=${REWARD_SCRIPT} \
    custom_reward_function.name=compute_score \
    trainer.balance_batch=True \
    trainer.logger='["console"]' \
    trainer.project_name=agentar_fin_r1 \
    trainer.experiment_name=grpo_stage2 \
    trainer.n_gpus_per_node=${NPROC} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=./outputs/grpo_lora_adapter

#!/usr/bin/env bash
# ============================================================================
# Agentar-Fin-R1 — Stage 2 DAPO (verl) 两阶段启动壳
# ----------------------------------------------------------------------------
# Phase 1: 数据预处理（原始 JSON/JSONL → verl DAPO parquet）
# Phase 2: DAPO 训练（在 SFT merge 后模型上做强化学习）
#
# 基座 Qwen3.5-9B + LoRA(r=32, alpha=64, all-linear)，vLLM rollout，FSDP 训练，8×A800，
# 外部 DeepSeek V4 Flash API 裁判（fin_judge_reward.py，走 OpenAI 兼容 /v1）。
#
# DAPO vs GRPO 关键差异：
#   - adv_estimator=dapo：DAPO 解耦 KL 惩罚，将其从 loss 移至 reward
#   - use_kl_in_reward=True：KL 散度作为 reward 惩罚项，而非 actor loss 正则
#   - 去掉 actor.use_kl_loss / kl_loss_coef / kl_loss_type
#   - 新增 algorithm.kl_ctrl 控制 reward 中 KL 惩罚强度
#
# 运行顺序：① SFT → ② merge_lora.py → ③ 本脚本
#
# 用法：
#   # 一键：原始数据 → 预处理 → 训练
#   RAW_DATA=./data/raw/train.json bash training/dapo/train_dapo.sh
#
#   # 已有 parquet，跳过预处理
#   DAPO_DATA=./data/verl/dapo.parquet bash training/dapo/train_dapo.sh
#
#   # 指定 merge 后模型
#   SFT_MERGED=./training/sft/merged DAPO_DATA=./data/verl/dapo.parquet \
#     bash training/dapo/train_dapo.sh
# ============================================================================
set -xeuo pipefail

SCRIPT_DIR="$(dirname "$0")"

# ---- Phase 1: 数据预处理（若指定了 RAW_DATA） ----
if [ -n "${RAW_DATA:-}" ]; then
    export DAPO_DATA="${DAPO_DATA:-./data/verl/dapo.parquet}"
    echo "[train_dapo.sh] Phase 1: 数据预处理  $RAW_DATA → $DAPO_DATA"
    python "$SCRIPT_DIR/prepare_dapo_data.py" \
        --input "$RAW_DATA" \
        --output "$DAPO_DATA"
    echo "[train_dapo.sh] Phase 1: 完成 → $DAPO_DATA"
fi

# ---- Phase 2: DAPO 训练 ----
SFT_MERGED=${SFT_MERGED:-./training/sft/merged}
DAPO_DATA=${DAPO_DATA:-./data/verl/dapo.parquet}
NPROC=${NPROC:-8}
REWARD_SCRIPT=${REWARD_SCRIPT:-./training/dapo/fin_judge_reward.py}

echo "[train_dapo.sh] Phase 2: DAPO 训练  merged=$SFT_MERGED  data=$DAPO_DATA"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=dapo \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    data.train_files=${DAPO_DATA} \
    data.val_files=${DAPO_DATA} \
    data.messages_key=messages \
    data.train_batch_size=64 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=${SFT_MERGED} \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=64 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.param_dtype=bfloat16 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_dtype=bfloat16 \
    actor_rollout_ref.rollout.reward_model.enable=False \
    custom_reward_function.path=${REWARD_SCRIPT} \
    custom_reward_function.name=compute_score \
    trainer.balance_batch=True \
    trainer.logger='["console"]' \
    trainer.project_name=agentar_fin_r1 \
    trainer.experiment_name=dapo_stage2 \
    trainer.n_gpus_per_node=${NPROC} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=./training/dapo/outputs

#!/usr/bin/env bash
# ============================================================================
# Agentar-Fin-R1 — Stage 3 OPD (verl) 两阶段启动壳
# ----------------------------------------------------------------------------
# On-Policy Distillation (OPD)：学生自采样 rollout，教师给 token 级 logprob 监督，
# 学生 = Qwen3-0.6B，教师 = Qwen3-8B。走 verl.trainer.main_ppo + distillation.* 配置块。
#
# Phase 1: 数据预处理（原始 JSON/JSONL → verl parquet，仅保留 prompt）
# Phase 2: OPD 训练（学生 FSDP 训练 + 教师 vLLM 推理，独立资源池）
#
# 资源池（8×A800 单机拆分建议）：
#   学生训练   trainer.n_gpus_per_node=4
#   教师推理   distillation.n_gpus_per_node=4（8B 用 tensor_parallel=2 → 2 replicas）
#   （学生 0.6B 极小，4 卡 FSDP 足够；8B 教师 4 卡 TP 也很轻松）
#
# 蒸馏损失：PG OPD（loss_mode=k1 + use_policy_gradient=True，官方默认），
#   纯蒸馏不叠加任务 reward（use_task_rewards=False），故无需 reward 裁判脚本。
#
# 用法：
#   # 一键：原始数据 → 预处理 → 训练
#   RAW_DATA=./data/raw/train.json bash training/opd/train_opd.sh
#
#   # 已有 parquet，跳过预处理
#   OPD_DATA=./data/verl/opd.parquet bash training/opd/train_opd.sh
#
#   # 指定教师/学生模型
#   TEACHER_MODEL=./Qwen3-8B STUDENT_MODEL=./Qwen3-0.6B \
#     OPD_DATA=./data/verl/opd.parquet bash training/opd/train_opd.sh
# ============================================================================
set -xeuo pipefail

SCRIPT_DIR="$(dirname "$0")"

# ---- Phase 1: 数据预处理（若指定了 RAW_DATA） ----
if [ -n "${RAW_DATA:-}" ]; then
    export OPD_DATA="${OPD_DATA:-./data/verl/opd.parquet}"
    echo "[train_opd.sh] Phase 1: 数据预处理  $RAW_DATA → $OPD_DATA"
    python "$SCRIPT_DIR/prepare_opd_data.py" \
        --input "$RAW_DATA" \
        --output "$OPD_DATA"
    echo "[train_opd.sh] Phase 1: 完成 → $OPD_DATA"
fi

# ---- Phase 2: OPD 训练 ----
STUDENT_MODEL=${STUDENT_MODEL:-./Qwen3-0.6B}
TEACHER_MODEL=${TEACHER_MODEL:-./Qwen3-8B}
OPD_DATA=${OPD_DATA:-./data/verl/opd.parquet}

NPROC=${NPROC:-4}                          # 学生训练 GPU 数
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-4} # 教师资源池 GPU 数
TEACHER_TP=${TEACHER_TP:-2}                 # 教师 tensor 并行度

LOSS_MODE=${LOSS_MODE:-k1}                  # k1 / k2 / k3 / kl / low_var_kl / forward_kl_topk
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
DISTILL_TOPK=${DISTILL_TOPK:-64}

MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-2048}
MAX_MODEL_LEN=$((MAX_PROMPT_LEN + MAX_RESPONSE_LEN + 1))

echo "[train_opd.sh] Phase 2: OPD 训练  student=$STUDENT_MODEL  teacher=$TEACHER_MODEL  data=$OPD_DATA"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files=${OPD_DATA} \
    data.val_files=${OPD_DATA} \
    data.messages_key=messages \
    data.train_batch_size=128 \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=${STUDENT_MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
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
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    distillation.enabled=True \
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE} \
    distillation.nnodes=1 \
    distillation.teacher_models.teacher_model.model_path=${TEACHER_MODEL} \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP} \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=0.6 \
    distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_MODEL_LEN} \
    distillation.distillation_loss.loss_mode=${LOSS_MODE} \
    distillation.distillation_loss.topk=${DISTILL_TOPK} \
    distillation.distillation_loss.use_task_rewards=False \
    distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT} \
    distillation.distillation_loss.loss_max_clamp=10.0 \
    distillation.distillation_loss.log_prob_min_clamp=-10.0 \
    trainer.balance_batch=True \
    trainer.logger='["console"]' \
    trainer.project_name=agentar_fin_r1 \
    trainer.experiment_name=opd_stage3 \
    trainer.n_gpus_per_node=${NPROC} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.default_local_dir=./training/opd/outputs

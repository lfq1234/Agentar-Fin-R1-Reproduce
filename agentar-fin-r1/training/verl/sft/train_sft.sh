#!/usr/bin/env bash
# ============================================================================
# Agentar-Fin-R1 — Stage 1 SFT (verl)
# ----------------------------------------------------------------------------
# 基座 Qwen3-8B + LoRA(r=64, alpha=128, all-linear)，FSDP 后端。
# 产物：LoRA adapter 目录（outputs/sft_lora_adapter），供 merge_lora.py 合并。
#
# 与旧 ms-swift 版（training/sft/src/train_sft.py）等价映射：
#   tuner_type=lora + lora_rank=64 + lora_alpha=128 + target_modules=all-linear
#   → model.lora_rank / model.lora_alpha / model.target_modules
#   learning_rate=1e-4 → optim.lr
#   num_train_epochs=3 → trainer.total_epochs
#   max_length=8192    → data.max_length
#
# 用法：
#   NPROC=1 MODEL_PATH=Qwen/Qwen3-8B SFT_DATA=./data/verl/sft.parquet \
#     bash training/verl/sft/train_sft.sh
# ============================================================================
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
SFT_DATA=${SFT_DATA:-./data/verl/sft.parquet}
SAVE_PATH=${SAVE_PATH:-./outputs/sft_lora_adapter}
NPROC=${NPROC:-1}

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC} \
    -m verl.trainer.sft_trainer \
    data.train_files=${SFT_DATA} \
    data.val_files=${SFT_DATA} \
    data.micro_batch_size_per_gpu=1 \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=True \
    data.max_length=8192 \
    optim.lr=1e-4 \
    optim.lr_warmup_steps=50 \
    engine=fsdp \
    engine.ulysses_sequence_parallel_size=1 \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    model.lora_rank=64 \
    model.lora_alpha=128 \
    model.target_modules=all-linear \
    model.use_gradient_checkpointing=true \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.total_epochs=3 \
    trainer.logger='["console"]'

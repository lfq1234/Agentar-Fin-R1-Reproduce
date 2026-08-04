#!/usr/bin/env bash
# ============================================================================
# Agentar-Fin-R1 — Stage 1 SFT (verl) 启动壳
# ----------------------------------------------------------------------------
# 仅负责设置路径环境变量，然后调用 train_sft.py（超参在此文件中持有）。
# 真正的训练逻辑见同目录 train_sft.py。
#
# 用法：
#   NPROC=1 MODEL_PATH=Qwen/Qwen3-8B SFT_DATA=./data/verl/sft.parquet \
#     bash training/sft/train_sft.sh
# 产物 → ./outputs/sft_lora_adapter
# ============================================================================
set -xeuo pipefail

# —— 路径环境变量（可被外部覆盖） ——
export MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
export SFT_DATA=${SFT_DATA:-./data/verl/sft.parquet}
export SAVE_PATH=${SAVE_PATH:-./outputs/sft_lora_adapter}
export NPROC=${NPROC:-1}

# —— 调用 python 启动器（超参在 train_sft.py 中） ——
python "$(dirname "$0")/train_sft.py"

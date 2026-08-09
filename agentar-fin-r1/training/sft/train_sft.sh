#!/usr/bin/env bash
# ============================================================================
# Agentar-Fin-R1 — Stage 1 SFT (verl) 两阶段启动壳
# ----------------------------------------------------------------------------
# Phase 1: 数据预处理（原始 JSON/JSONL → verl parquet）
# Phase 2: SFT 训练
#
# 用法：
#   # 一键：原始数据 → 预处理 → 训练
#   RAW_DATA=./data/raw/train.json bash training/sft/train_sft.sh
#
#   # 已有 parquet，跳过预处理直接训练
#   SFT_DATA=./data/verl/sft.parquet bash training/sft/train_sft.sh
#
#   # 指定本地模型
#   NPROC=4 MODEL_PATH=./Qwen3-8B RAW_DATA=./data/raw/train.json \
#     bash training/sft/train_sft.sh
# ============================================================================
set -xeuo pipefail

SCRIPT_DIR="$(dirname "$0")"

# ---- Phase 1: 数据预处理（若指定了 RAW_DATA） ----
if [ -n "${RAW_DATA:-}" ]; then
    export SFT_DATA="${SFT_DATA:-./data/verl/sft.parquet}"
    echo "[train_sft.sh] Phase 1: 数据预处理  $RAW_DATA → $SFT_DATA"
    python "$SCRIPT_DIR/prepare_sft_data.py" \
        --input "$RAW_DATA" \
        --output "$SFT_DATA"
    echo "[train_sft.sh] Phase 1: 完成 → $SFT_DATA"
fi

# ---- Phase 2: SFT 训练 ----
echo "[train_sft.sh] Phase 2: SFT 训练  data=${SFT_DATA:-./data/verl/sft.parquet}  model=${MODEL_PATH:-./Qwen3-8B}"
python "$SCRIPT_DIR/train_sft.py"

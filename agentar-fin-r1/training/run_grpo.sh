#!/usr/bin/env bash
#
# Stage 2 GRPO 启动脚本 (Agentar-Fin-R1 reproduction)
# =====================================================
# 实现：verl 库 + LoRA（GRPO 由 verl 原生提供，见 grpo/__init__.py）
#
# 用法:
#   bash run_grpo.sh --hard-subset data/golden/hard_subset.jsonl \
#                    --stage1-adapter checkpoints/stage1 --max-samples 50
#   bash run_grpo.sh --help               # 查看全部参数
#
# 必填参数:
#   --hard-subset <jsonl>   每行 {question, answer} 的难题集
#                           来源：归因闭环 grpo/attribution.py，或数据流水线，或手工小文件
#
# 行为:
#   1. 优先用 uv；若没装 uv，则在本目录建 .venv 并 pip install -e .
#   2. 启动 grpo 模块（training/grpo/__init__.py）：
#      - 把 hard_subset.jsonl 转成 verl 的 parquet 格式
#      - 用 grpo/config.yaml 生成 verl 的 hydra 覆盖参数
#      - 调用 verl.trainer.main_ppo 跑 GRPO（group_size=8，LoRA）
#      产出在 ./checkpoints/stage2-grpo（仅保存 LoRA 增量）
#
set -euo pipefail

# 切到 training/ 目录
cd "$(dirname "$0")"

# ---- 1. 环境 ----
if command -v uv >/dev/null 2>&1; then
  echo "[run_grpo] 使用 uv"
  PY_RUN="uv run"
else
  if [ ! -d .venv ]; then
    echo "[run_grpo] 未检测到 venv，创建并安装依赖..."
    python -m venv .venv
    # shellcheck disable=SC1091
    source .venv/Scripts/activate
    pip install -e . -q
  else
    # shellcheck disable=SC1091
    source .venv/Scripts/activate
  fi
  PY_RUN="python"
fi

# ---- 2. 启动 Stage 2 GRPO ----
if ! echo " $* " | grep -q -- "--hard-subset"; then
  echo "[run_grpo] 错误: 必须提供 --hard-subset <jsonl>"
  echo "示例: bash run_grpo.sh --hard-subset data/golden/hard_subset.jsonl \\"
  echo "                       --stage1-adapter checkpoints/stage1 --group-size 8 --max-samples 50"
  exit 1
fi

echo "[run_grpo] 启动训练 (模块: grpo / verl backend)..."
$PY_RUN -m grpo "$@"

echo "[run_grpo] 完成。检查点: ./checkpoints/stage2-grpo"

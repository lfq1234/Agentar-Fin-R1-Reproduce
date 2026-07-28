#!/usr/bin/env bash
#
# Stage 1 SFT 启动脚本 (Agentar-Fin-R1 reproduction)
# =====================================================
# 实现：peft + LoRA + trl.SFTTrainer（掉包式实现，见 sft/__init__.py）
#
# 用法:
#   bash run_sft.sh                      # 用默认 sft/config.yaml 完整跑
#   bash run_sft.sh --max-financial 50   # 小规模试跑，快速验证流程
#   bash run_sft.sh --help               # 查看全部参数
#
# 行为:
#   1. 优先用 uv；若没装 uv，则在本目录建 .venv 并 pip install -e .
#   2. 启动 sft 模块（training/sft/__init__.py）
#      默认模型 Qwen/Qwen3.5-9B，金融 CoT 数据 antgroup/Agentar-DeepFinance-100K
#      难度加权默认 complexity，产出在 ./checkpoints/stage1
#
set -euo pipefail

# 切到 training/ 目录，保证 configs/、checkpoints/ 相对路径正确
cd "$(dirname "$0")"

# ---- 1. 环境 ----
if command -v uv >/dev/null 2>&1; then
  echo "[run_sft] 使用 uv"
  PY_RUN="uv run"
else
  if [ ! -d .venv ]; then
    echo "[run_sft] 未检测到 venv，创建并安装依赖..."
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

# ---- 2. 启动 Stage 1 SFT ----
echo "[run_sft] 启动训练 (模块: sft)..."
$PY_RUN -m sft "$@"

echo "[run_sft] 完成。检查点: ./checkpoints/stage1"

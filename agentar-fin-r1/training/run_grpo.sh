#!/usr/bin/env bash
#
# Stage 2 GRPO 启动脚本 (Agentar-Fin-R1 reproduction)
# =====================================================
# 用法:
#   bash run_grpo.sh --hard-subset data/golden/hard_subset.jsonl \
#                    --stage1-adapter checkpoints/stage1 --max-samples 50
#   bash run_grpo.sh --help               # 查看 train 全部参数
#
# 必填参数:
#   --hard-subset <jsonl>   每行 {question, answer} 的难题集
#                           来源：归因闭环 attribution.py，或数据流水线，或手工造一个小文件
#
# 行为:
#   1. 优先用 uv；若没装 uv，则在本目录建 .venv 并 pip install -e .
#   2. 启动 grpo 模块（training/grpo/__init__.py）
#      默认 group_size=8（每次 rollout 8 条比较），模型 Qwen/Qwen3.5-9B
#      默认读 grpo/config.yaml，CLI 参数覆盖配置
#      产出在 ./checkpoints/stage2-grpo（含停滞时回退的 targeted SFT 步骤）
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
# 未提供 --hard-subset 时给出明确提示再退出（train 本身也会校验）
if ! echo " $* " | grep -q -- "--hard-subset"; then
  echo "[run_grpo] 错误: 必须提供 --hard-subset <jsonl>"
  echo "示例: bash run_grpo.sh --hard-subset data/golden/hard_subset.jsonl \\"
  echo "                       --stage1-adapter checkpoints/stage1 --group-size 8 --max-samples 50"
  exit 1
fi

echo "[run_grpo] 启动训练 (模块: grpo)..."
$PY_RUN -m grpo "$@"

echo "[run_grpo] 完成。检查点: ./checkpoints/stage2-grpo"

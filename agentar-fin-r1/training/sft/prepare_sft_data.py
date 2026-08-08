#!/usr/bin/env python3
# ============================================================================
# Agentar-Fin-R1 — SFT 数据预处理：原始对话 JSON → verl parquet
# ----------------------------------------------------------------------------
# 将原始训练数据（messages 格式，role: HUMAN/ASSISTANT）转换为 verl 标准
# parquet 格式，供 verl.trainer.sft_trainer 消费。
#
# 输入格式支持：
#   1. JSON 数组：[[{"role":"HUMAN",...},{"role":"ASSISTANT",...}], ...]
#   2. JSONL：每行一个 messages 列表
#
# 用法：
#   python training/sft/prepare_sft_data.py \
#       --input ./data/raw/train.json --output ./data/verl/sft.parquet
# ============================================================================
import argparse
import json
import os
from pathlib import Path

import pandas as pd

# ---- 角色映射 ----
ROLE_MAP = {
    "HUMAN": "user",
    "human": "user",
    "USER": "user",
    "ASSISTANT": "assistant",
    "assistant": "assistant",
    "GPT": "assistant",
}


def _map_role(role: str) -> str:
    return ROLE_MAP.get(role, role)


def load_messages(input_path: str) -> list[list[dict]]:
    """从 JSON 数组或 JSONL 文件加载 messages 列表。"""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    with open(path, encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON 顶层必须是数组")
            return data
        else:
            return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description="原始对话 JSON/JSONL → verl SFT parquet")
    p.add_argument("--input", required=True, help="输入文件（.json 或 .jsonl）")
    p.add_argument("--output", required=True, help="输出 parquet 路径")
    args = p.parse_args()

    print(f"[prepare_sft] 加载: {args.input}")
    messages_list = load_messages(args.input)
    print(f"[prepare_sft] 共 {len(messages_list)} 条对话")

    records = []
    for messages in messages_list:
        mapped = [{"role": _map_role(m["role"]), "content": m["content"]} for m in messages]
        records.append({"messages": mapped})

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[prepare_sft] 完成 → {args.output} ({len(df)} 条)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Agentar-Fin-R1 — OPD 数据预处理：原始对话 JSON → verl parquet。

OPD 是纯蒸馏（学生自采样 + 教师 token 级 logprob 监督，不叠加任务 reward），
因此只需 prompt（messages），无需 gold answer：
  - messages：仅保留 user 消息（prompt）
  - data_source：固定 "agentar_fin"（单教师模式下用于多教师路由，此处仅占位）

用法：
  python training/opd/prepare_opd_data.py \
      --input ./data/raw/train.json --output ./data/verl/opd.parquet
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

ROLE_MAP = {
    "HUMAN": "user", "human": "user", "USER": "user",
    "ASSISTANT": "assistant", "assistant": "assistant", "GPT": "assistant",
}

HUMAN_ROLES = {"HUMAN", "human", "USER"}


def _map_role(role: str) -> str:
    return ROLE_MAP.get(role, role)


def load_messages(input_path: str) -> list[list[dict]]:
    path = Path(input_path)
    with open(path, encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        else:
            return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print(f"[prepare_opd] 加载: {args.input}")
    messages_list = load_messages(args.input)
    print(f"[prepare_opd] 共 {len(messages_list)} 条")

    records = []
    for messages in messages_list:
        prompt_msgs = [
            {"role": _map_role(m["role"]), "content": m["content"]}
            for m in messages if m.get("role") in HUMAN_ROLES
        ]
        records.append({
            "messages": prompt_msgs,
            "data_source": "agentar_fin",
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[prepare_opd] 完成 → {args.output} ({len(df)} 条)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Agentar-Fin-R1 — GRPO 数据预处理：原始对话 JSON → verl parquet。

将原始数据（messages 格式，role: HUMAN/ASSISTANT）转换为 GRPO 格式：
  - messages：仅保留 user 消息（prompt）
  - extra_info：原题 + 标准思维链 + 标准答案输出（供 RLAIF 裁判对标）

用法：
  python training/grpo/prepare_grpo_data.py \
      --input ./data/raw/train.json --output ./data/verl/grpo.parquet
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

ROLE_MAP = {
    "HUMAN": "user", "human": "user", "USER": "user",
    "ASSISTANT": "assistant", "assistant": "assistant", "GPT": "assistant",
}

HUMAN_ROLES = {"HUMAN", "human", "USER"}
ASSISTANT_ROLES = {"ASSISTANT", "assistant", "GPT"}


def _map_role(role: str) -> str:
    return ROLE_MAP.get(role, role)


def _split_thinking_and_output(content: str) -> tuple[str, str]:
    m = re.search(r"<think>(.*?)</think>(.*)", content, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", content.strip()


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

    print(f"[prepare_grpo] 加载: {args.input}")
    messages_list = load_messages(args.input)
    print(f"[prepare_grpo] 共 {len(messages_list)} 条")

    records = []
    for messages in messages_list:
        prompt_msgs = [
            {"role": _map_role(m["role"]), "content": m["content"]}
            for m in messages if m.get("role") in HUMAN_ROLES
        ]
        assistant_content = next(
            (m["content"] for m in messages if m.get("role") in ASSISTANT_ROLES), ""
        )
        gold_thinking, gold_output = _split_thinking_and_output(assistant_content)
        question = next(
            (m["content"] for m in messages if m.get("role") in HUMAN_ROLES), ""
        )
        records.append({
            "messages": prompt_msgs,
            "extra_info": json.dumps({
                "question": question,
                "gold_thinking": gold_thinking,
                "gold_output": gold_output,
            }),
        })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"[prepare_grpo] 完成 → {args.output} ({len(df)} 条)")


if __name__ == "__main__":
    main()

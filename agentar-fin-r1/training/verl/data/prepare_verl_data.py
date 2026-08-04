"""Prepare verl-ready parquet from Agentar-Fin-R1 data sources.

把训练数据转成 verl 要求的 parquet 格式：

  SFT  (training/verl/sft/train_sft.sh 用)
      schema: {"messages": [{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}
      assistant 内容 = <think>{thinking}</think><answer>{answer}</answer>
                  —— thinking 来自 Solution（DeepFinance）或 triplet.thinking（pipeline）

  GRPO (training/verl/grpo/train_grpo.sh 用)
      schema: {"prompt":[...user turn...],
               "reward_model":{"ground_truth":"<answer>"},
               "data_source":"agentar_fin"}
      ground_truth 供 fin_judge_reward.py 的裁判比对使用。

兼容两种输入：
  A) pipeline 产物 golden.jsonl —— ReasoningTriplet(query, thinking, answer)
  B) DeepFinance-100K        —— Question / Solution / Answer（+ 元数据列）
  C) 已含 messages 字段的 jsonl —— 直接透传为 SFT，并从 assistant 抽取答案作 GRPO ground_truth

用法：
  python training/verl/data/prepare_verl_data.py \
      --input ./data/golden/golden.jsonl \
      --out-dir ./data/verl

  # 用 DeepFinance-100K 本地副本（parquet/jsonl）：
  python training/verl/data/prepare_verl_data.py \
      --input /path/to/deepfinance.parquet --out-dir ./data/verl
"""

from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd


DEFAULT_SYSTEM = (
    "你是一名专业的金融推理助手。面对用户的问题，请先进行严谨、分步的推理，"
    "把思考过程写在 <think>...</think> 标签内，再在 <answer>...</answer> 标签内给出"
    "最终结论。推理需基于给定的事实与金融常识，避免无依据臆测。"
)


def _wrap_answer(thinking: str, answer: str) -> str:
    """构造带 <think>/<answer> 边界的 assistant 内容（判定标准③要求 <think> 边界）。"""
    thinking = (thinking or "").strip()
    answer = (answer or "").strip()
    parts = []
    if thinking:
        parts.append(f"<think>{thinking}</think>")
    parts.append(f"<answer>{answer}</answer>")
    return "\n".join(parts)


def _extract_messages_from_row(row: dict) -> dict | None:
    """从一行数据抽取 (system, query, thinking, answer)。返回 None 表示该行无法转换。"""
    # C) 已有 messages：直接透传 SFT；从最后一个 assistant 抽答案作 ground_truth
    if "messages" in row and isinstance(row["messages"], list):
        msgs = row["messages"]
        query = ""
        answer = ""
        for m in msgs:
            if m.get("role") == "user":
                query = m.get("content", "")
            if m.get("role") == "assistant":
                answer = m.get("content", "")
        return {
            "system": next((m["content"] for m in msgs if m.get("role") == "system"), ""),
            "query": query,
            "thinking": "",
            "answer": answer,
        }

    # A) ReasoningTriplet
    if "query" in row or "thinking" in row or "answer" in row:
        return {
            "system": row.get("system", "") or "",
            "query": row.get("query") or row.get("question") or "",
            "thinking": row.get("thinking") or "",
            "answer": row.get("answer") or "",
        }

    # B) DeepFinance-100K
    q = row.get("Question") or row.get("question") or row.get("query") or ""
    sol = row.get("Solution") or row.get("solution") or ""
    ans = row.get("Answer") or row.get("answer") or ""
    # DeepFinance 的 Answer 有时嵌在 Solution 末尾 "Answer: ..."，兜底抽取
    if not ans and "Answer:" in sol:
        ans = sol.split("Answer:")[-1].strip()
    return {"system": row.get("system", "") or "", "query": q, "thinking": sol, "answer": ans}


def _load_rows(path: str) -> list[dict]:
    if path.endswith((".jsonl", ".json")):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    # parquet（DeepFinance-100K 常见）
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def build(input_path: str, out_dir: str, system: str, valid_frac: float = 0.05):
    os.makedirs(out_dir, exist_ok=True)
    rows = _load_rows(input_path)
    print(f"[prepare] loaded {len(rows)} rows from {input_path}")

    sft_records, grpo_records = [], []
    skipped = 0
    for row in rows:
        ex = _extract_messages_from_row(row)
        if not ex or not ex["query"] or not ex["answer"]:
            skipped += 1
            continue
        sys_msg = ex["system"] or system
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": ex["query"]},
            {"role": "assistant", "content": _wrap_answer(ex["thinking"], ex["answer"])},
        ]
        sft_records.append({"messages": messages})
        grpo_records.append(
            {
                "prompt": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": ex["query"]},
                ],
                "reward_model": {"ground_truth": ex["answer"]},
                "data_source": "agentar_fin",
            }
        )

    print(f"[prepare] usable={len(sft_records)} skipped={skipped}")

    # SFT：train + 小 val
    sft_df = pd.DataFrame(sft_records)
    n_val = max(1, int(len(sft_df) * valid_frac))
    sft_df.to_parquet(os.path.join(out_dir, "sft.parquet"), index=False)
    sft_df.head(n_val).to_parquet(os.path.join(out_dir, "sft_val.parquet"), index=False)

    # GRPO：train + 小 val
    grpo_df = pd.DataFrame(grpo_records)
    grpo_df.to_parquet(os.path.join(out_dir, "grpo.parquet"), index=False)
    grpo_df.head(n_val).to_parquet(os.path.join(out_dir, "grpo_val.parquet"), index=False)

    print(f"[prepare] wrote: sft.parquet({len(sft_df)}), grpo.parquet({len(grpo_df)}) → {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Build verl SFT/GRPO parquet for Agentar-Fin-R1")
    p.add_argument("--input", required=True, help="golden.jsonl / deepfinance.parquet / 含 messages 的 jsonl")
    p.add_argument("--out-dir", default="./data/verl")
    p.add_argument("--system", default=DEFAULT_SYSTEM, help="金融推理 system prompt")
    p.add_argument("--valid-frac", type=float, default=0.05)
    args = p.parse_args()
    build(args.input, args.out_dir, args.system, args.valid_frac)


if __name__ == "__main__":
    main()

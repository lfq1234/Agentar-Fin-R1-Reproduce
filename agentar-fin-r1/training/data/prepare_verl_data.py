"""Prepare verl-ready parquet from Agentar-Fin-R1 data sources.

真实数据格式（主格式，本文件优先支持）：
  [
    {"role": "HUMAN",    "content": "<context>\nQuestion: ...\nAnswer:"},
    {"role": "ASSISTANT","content": "<think>...</think>\n...\\boxed{-606}"}
  ]
  —— role 是大写 HUMAN/ASSISTANT；最终答案写在 \\boxed{} 里（金融数值题常见）。
  —— SFT 直接用归一化后的 messages；GRPO 从 ASSISTANT 的 \\boxed{} 抽 gold 作 ground_truth。

其余兼容格式：
  A) pipeline 产物 golden.jsonl —— ReasoningTriplet(query, thinking, answer)
  B) DeepFinance-100K        —— Question / Solution / Answer
  C) 已含 messages 字段的 jsonl —— 同样走主格式解析

GRPO ground_truth 写入 JSON：{"answer": <gold>, "verifiable": <bool>}
  verifiable=True  → 阶段1 RLVR 规则比对（数值/日期/选项/是非，0/1）
  verifiable=False → 阶段2 RLGHAI 交给 72B 裁判打质量分
  判定逻辑见 training/grpo/fin_judge_reward.py。

用法：
  python training/data/prepare_verl_data.py \
      --input ./data/golden/golden.jsonl --out-dir ./data/verl
  # 或整段 messages 数组直接作为一行 jsonl：
  python training/data/prepare_verl_data.py \
      --input ./data/qa_messages.jsonl --out-dir ./data/verl
"""

from __future__ import annotations

import argparse
import json
import os
import re

import pandas as pd


DEFAULT_SYSTEM = (
    "你是一名专业的金融推理助手。面对用户的问题，请先进行严谨、分步的推理，"
    "把思考过程写在 <think>...</think> 标签内，再在 \\boxed{} 中给出最终结论，"
    "推理需基于给定的事实与金融常识，避免无依据臆测。"
)

# —— 可验证题型识别（干净、确定，不做松散兜底）——
_NUM_RE = re.compile(
    r"^[\$¥€£]?\s*-?\d[\d,]*(\.\d+)?\s*(%|万|亿|倍|k|m|bn|trn|元|美元|美圆)?$",
    re.I,
)
_DATE_RE = re.compile(r"\d{4}[-/年]\d{1,2}[-/月]\d{1,2}")
_YESNO = {"TRUE", "FALSE", "YES", "NO", "对", "错", "是", "否", "Y", "N"}


def _normalize_role(r: str) -> str:
    """HUMAN→user, ASSISTANT→assistant, SYSTEM→system（兼容大小写/别名）。"""
    r = (r or "").strip().upper()
    if r in ("HUMAN", "USER"):
        return "user"
    if r in ("ASSISTANT", "BOT", "AI"):
        return "assistant"
    if r == "SYSTEM":
        return "system"
    return (r or "").lower()


def _extract_gold_answer(content: str) -> str:
    """从 ASSISTANT 内容抽取标准答案（用于 GRPO ground_truth）。

    优先级：\\boxed{} → <answer>…</answer> → 行内 "Answer: X" → 整段兜底。
    """
    if not content:
        return ""
    m = re.search(r"\\boxed\{([^{}]*)\}", content)
    if m:
        return m.group(1).strip()
    m = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"[Aa]nswer\s*[:：]\s*([^\n]+)", content)
    if m:
        a = m.group(1).strip().rstrip(". ")
        if a:
            return a
    return content.strip()


def _is_verifiable(ans: str) -> bool:
    """判断标准答案是否可被规则验证。

    仅当答案是：数值（含单位/百分比/货币）、日期、选项(A-E)、是非(TRUE/FALSE/是/否)
    时才标 verifiable=True；长文本论述（>40字）一律视为开放题走 RLGHAI。
    """
    a = (ans or "").strip()
    if not a or len(a) > 40:
        return False
    if _NUM_RE.match(a) or _DATE_RE.search(a):
        return True
    if a.upper() in _YESNO:
        return True
    if re.fullmatch(r"[A-Ea-e]", a):
        return True
    return False


def _wrap_answer(thinking: str, answer: str) -> str:
    """构造带 <think>/<answer> 边界的 assistant 内容（用于非主格式来源的 SFT）。"""
    thinking = (thinking or "").strip()
    answer = (answer or "").strip()
    parts = []
    if thinking:
        parts.append(f"<think>{thinking}</think>")
    parts.append(f"<answer>{answer}</answer>")
    return "\n".join(parts)


def _from_messages(messages: list[dict]) -> dict | None:
    """主格式解析：messages 数组 → {system, query, answer, sft_messages}。"""
    msgs = [{"role": _normalize_role(m.get("role")), "content": m.get("content", "")}
            for m in messages if isinstance(m, dict)]
    if not msgs:
        return None
    system = next((m["content"] for m in msgs if m["role"] == "system"), "")
    query = next((m["content"] for m in msgs if m["role"] == "user"), "")
    assistant = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    if not query or not assistant:
        return None
    gold = _extract_gold_answer(assistant)
    sft_messages = [{"role": "system", "content": system}] if system else []
    sft_messages += [{"role": "user", "content": query},
                     {"role": "assistant", "content": assistant}]
    return {
        "system": system,
        "query": query,
        "answer": gold,
        "sft_messages": sft_messages,
        "thinking": "",
    }


def _extract_messages_from_row(row: dict | list) -> dict | None:
    """从一行数据抽取统一结构。返回 None 表示该行无法转换。"""
    # 主格式：整行就是 messages 数组
    if isinstance(row, list):
        return _from_messages(row)
    # 主格式：含 messages 字段
    if isinstance(row.get("messages"), list):
        return _from_messages(row["messages"])
    # A) ReasoningTriplet
    if "query" in row or "thinking" in row or "answer" in row:
        return {
            "system": row.get("system", "") or "",
            "query": row.get("query") or row.get("question") or "",
            "answer": row.get("answer") or "",
            "sft_messages": None,
            "thinking": row.get("thinking") or "",
        }
    # B) DeepFinance-100K
    q = row.get("Question") or row.get("question") or row.get("query") or ""
    sol = row.get("Solution") or row.get("solution") or ""
    ans = row.get("Answer") or row.get("answer") or ""
    if not ans and "Answer:" in sol:
        ans = sol.split("Answer:")[-1].strip()
    if not q or not ans:
        return None
    return {
        "system": row.get("system", "") or "",
        "query": q,
        "answer": ans,
        "sft_messages": None,
        "thinking": sol,
    }


def _load_rows(path: str) -> list:
    if path.endswith((".jsonl", ".json")):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    df = pd.read_parquet(path)  # DeepFinance-100K 常见 parquet
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

        # SFT：优先用原始 messages（保留 <think>+\\boxed{} 真实格式）
        if ex.get("sft_messages"):
            sft_messages = ex["sft_messages"]
        else:
            sft_messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": ex["query"]},
                {"role": "assistant",
                 "content": _wrap_answer(ex["thinking"], ex["answer"])},
            ]
        sft_records.append({"messages": sft_messages})

        # GRPO：prompt + ground_truth(含 verifiable 标注)
        grpo_records.append(
            {
                "prompt": [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": ex["query"]},
                ],
                "reward_model": {
                    "ground_truth": json.dumps(
                        {"answer": ex["answer"],
                         "verifiable": _is_verifiable(ex["answer"])},
                        ensure_ascii=False,
                    )
                },
                "data_source": "agentar_fin",
            }
        )

    print(f"[prepare] usable={len(sft_records)} skipped={skipped} "
          f"(verifiable 占比见统计)")
    n_verif = sum(
        1 for g in grpo_records
        if json.loads(g["reward_model"]["ground_truth"]).get("verifiable")
    )
    print(f"[prepare] grpo: {len(grpo_records)} 题中 {n_verif} 走 RLVR，"
          f"{len(grpo_records) - n_verif} 走 RLGHAI")

    n_val = max(1, int(len(sft_records) * valid_frac))
    sft_df = pd.DataFrame(sft_records)
    sft_df.to_parquet(os.path.join(out_dir, "sft.parquet"), index=False)
    sft_df.head(n_val).to_parquet(os.path.join(out_dir, "sft_val.parquet"), index=False)

    grpo_df = pd.DataFrame(grpo_records)
    grpo_df.to_parquet(os.path.join(out_dir, "grpo.parquet"), index=False)
    grpo_df.head(n_val).to_parquet(os.path.join(out_dir, "grpo_val.parquet"), index=False)

    print(f"[prepare] wrote sft.parquet({len(sft_df)}), grpo.parquet({len(grpo_df)}) → {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Build verl SFT/GRPO parquet for Agentar-Fin-R1")
    p.add_argument("--input", required=True,
                   help="golden.jsonl / deepfinance.parquet / 含 messages 的 jsonl")
    p.add_argument("--out-dir", default="./data/verl")
    p.add_argument("--system", default=DEFAULT_SYSTEM, help="金融推理 system prompt")
    p.add_argument("--valid-frac", type=float, default=0.05)
    args = p.parse_args()
    build(args.input, args.out_dir, args.system, args.valid_frac)


if __name__ == "__main__":
    main()

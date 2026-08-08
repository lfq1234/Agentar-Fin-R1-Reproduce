"""Agentar-Fin-R1 GRPO 奖励函数 —— RLAIF + rubric 极简版（verl 0.8.0 自定义奖励）。

奖励信号来自 **RLAIF（Reinforcement Learning from AI Feedback）**：由外部金融裁判模型
按固定 **rubric（评分量规）** 对回答打分，作为 reward。

裁判模型：**外部 DeepSeek V4 Flash API**（OpenAI 兼容 /v1）。

  Step 1 · 格式闸门（format gate，确定性，免费）
    响应必须含 <think>…</think> 与（\\boxed{} 或 <answer>…</answer>）。
    缺任一标签 → 直接 0 分。

  Step 2 · RLAIF rubric 打分（对所有格式合格样本统一走裁判）
    裁判按 4 个维度各打 0~10 分，代码按权重聚合为 0~1 reward：
        correctness      正确性       0.40  结论与参考答案一致、事实/计算准确
        reasoning        推理严谨性    0.30  推理链完整、逻辑自洽、与标准思维链对标
        compliance_risk  合规风险意识  0.15  提示风险/合规约束、避免误导
        clarity_format   表达与结构    0.15  清晰结构化、符合格式约定

extra_info 传入三部分（由 prepare_grpo_data.py 生成）：
    question        — 原题
    gold_thinking   — 标准思维链（参考 reasoning 维度）
    gold_output     — 标准答案输出（参考 correctness 维度）

只暴露一个函数 compute_score(data_source, solution_str, ground_truth, extra_info)，
由 verl 逐样本调用。

接线（见 train_grpo.sh）：
    custom_reward_function.path=<本文件>
    custom_reward_function.name=compute_score

运行前需导出 API 凭证：
    export JUDGE_API_KEY=<你的 DeepSeek API key>
"""

import json
import os
import re

import requests

# —— 裁判模型服务（外部 DeepSeek V4 Flash，OpenAI 兼容 /v1）——
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-v4-flash")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "")
JUDGE_TIMEOUT = int(os.environ.get("JUDGE_TIMEOUT", "60"))

# —— rubric：维度名 + 权重（和=1.0）——
RUBRIC = [
    ("correctness", 0.40),
    ("reasoning", 0.30),
    ("compliance_risk", 0.15),
    ("clarity_format", 0.15),
]
RUBRIC_KEYS = [k for k, _ in RUBRIC]


# ============================================================================
# 工具：解析 / 抽取 / 格式闸门
# ============================================================================
def _parse_gt(raw) -> dict:
    """ground_truth 可能是 JSON 字符串或 dict。"""
    if isinstance(raw, dict):
        return raw
    try:
        d = json.loads(raw)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {"answer": str(raw)}


def _has_valid_format(response: str) -> bool:
    """格式闸门：<think> 与（\\boxed{} 或 <answer>）同时存在。"""
    has_think = "<think>" in response and "</think>" in response
    has_answer = ("\\boxed{" in response) or (
        "<answer>" in response and "</answer>" in response
    )
    return has_think and has_answer


def _parse_extra_info(extra_info) -> dict:
    """从 extra_info 中提取三部分参考信息。"""
    if isinstance(extra_info, dict):
        return extra_info
    if isinstance(extra_info, str):
        try:
            d = json.loads(extra_info)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return {}


# ============================================================================
# RLAIF：裁判按 rubric 打分 → 加权聚合为 0~1
# ============================================================================
_RUBRIC_DESC = "\n".join(
    f"- {k}: {desc}（权重 {w:.2f}）"
    for (k, w), desc in zip(
        RUBRIC,
        [
            "结论是否与参考答案一致、事实与计算是否准确",
            "推理链是否完整、逻辑自洽、与标准思维链对标",
            "是否体现风险/合规意识、避免误导性陈述",
            "是否清晰结构化、符合 <think>/<answer> 或 \\boxed{} 约定",
        ],
    )
)


def _rlaif_rubric_score(
    question: str, response: str,
    gold_output: str, gold_thinking: str,
) -> float:
    """外部金融裁判按 rubric 给各维度 0~10 分，代码加权聚合成 0~1。

    任何异常（无服务/解析失败）→ 0.0，不污染训练。
    """
    prompt_parts = []
    if question:
        prompt_parts.append(f"[Question]\n{question}")
    prompt_parts.append(f"[Gold thinking / reference reasoning]\n{gold_thinking}")
    prompt_parts.append(f"[Gold output / correct answer]\n{gold_output}")
    prompt_parts.append(f"[Assistant answer to evaluate]\n{response}")
    prompt_parts.append(
        f"Rubric (score each dimension 0-10, higher is better):\n{_RUBRIC_DESC}\n\n"
        f'Return ONLY JSON: {{"dimensions": {{"correctness": x, "reasoning": x, '
        f'"compliance_risk": x, "clarity_format": x}}, "rationale": "..."}}'
    )
    prompt = "\n\n".join(prompt_parts)

    try:
        resp = requests.post(
            f"{JUDGE_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {JUDGE_API_KEY}"},
            json={
                "model": JUDGE_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
            timeout=JUDGE_TIMEOUT,
        )
        content = resp.json()["choices"][0]["message"]["content"]
        scores = _parse_rubric_scores(content)
        if not scores:
            return 0.0
        total = 0.0
        for key, weight in RUBRIC:
            if key in scores:
                total += weight * max(0.0, min(10.0, scores[key])) / 10.0
        return max(0.0, min(1.0, total))
    except Exception:
        return 0.0


def _parse_rubric_scores(content: str) -> dict:
    """尽力从裁判输出中解析出 4 个维度分数（0~10）。"""
    try:
        d = json.loads(content)
        dims = d.get("dimensions", d) if isinstance(d, dict) else None
        if isinstance(dims, dict):
            out = {}
            for k in RUBRIC_KEYS:
                if k in dims:
                    try:
                        out[k] = float(dims[k])
                    except Exception:
                        pass
            if out:
                return out
    except Exception:
        pass
    out = {}
    for key in RUBRIC_KEYS:
        m = re.search(rf'(?:"{key}"|{key})\s*[:=]\s*(\d+(?:\.\d+)?)', content)
        if m:
            out[key] = float(m.group(1))
    return out


# ============================================================================
# 主入口：格式闸门 → RLAIF rubric 打分（verl 逐样本调用）
# ============================================================================
def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """Agentar-Fin-R1 奖励：格式闸门 → RLAIF rubric 加权打分。"""
    # Step 1 · 格式不合格 → 0 分
    if not _has_valid_format(solution_str):
        return 0.0

    # Step 2 · RLAIF：裁判按 rubric 对格式合格的回答打分（0~1）
    ref = _parse_extra_info(extra_info)
    return _rlaif_rubric_score(
        question=ref.get("question", ""),
        response=solution_str,
        gold_thinking=ref.get("gold_thinking", ""),
        gold_output=ref.get("gold_output", ""),
    )

"""Agentar-Fin-R1 GRPO 奖励函数 —— RLAIF + rubric 极简版（verl 0.8.0 自定义奖励）。

奖励信号来自 **RLAIF（Reinforcement Learning from AI Feedback）**：由外部金融裁判模型
按固定 **rubric（评分量规）** 对回答打分，作为 reward。比单一 0/1 规则更平滑、可解释，
也比纯偏好对更省标注。

  Step 1 · 格式闸门（format gate，确定性，免费）
    响应必须含 <think>…</think> 与（\\boxed{} 或 <answer>…</answer>）。
    缺任一标签 → 直接 0 分（论文强调「verifiable / auditable」输出）。

  Step 2 · RLAIF rubric 打分（对所有格式合格样本统一走裁判）
    裁判按 4 个维度各打 0~10 分，代码按权重聚合为 0~1 reward：
        correctness      正确性       0.35  结论与参考答案一致、事实/计算准确
        reasoning        推理严谨性    0.30  推理链完整、逻辑自洽、无原则性错误
        compliance_risk  合规风险意识  0.20  提示风险/合规约束、避免误导
        clarity_format   表达与结构    0.15  清晰结构化、符合格式约定
    参考标准答案与（可选）原题一并喂给裁判，correctness 维度据此比对。

只暴露一个函数 compute_score(data_source, solution_str, ground_truth, extra_info)，
由 verl 逐样本调用。无 RewardManager 子类、无 ThreadPool、无单例客户端——
逻辑是单条自上而下的 if/return，避免回调地狱。

接线（见 train_grpo.sh）：
    actor_rollout_ref.rollout.reward_model.enable=False
    custom_reward_function.path=<本文件>
    custom_reward_function.name=compute_score

ground_truth 为 JSON 字符串：{"answer": "<标准答案>", "verifiable": true|false}
（verifiable 不再决定走规则还是裁判，仅作元信息；RLAIF 对所有样本统一打分）
extra_info 可选含 "question"/"prompt" 字段，作为原题上下文喂给裁判。
"""

import json
import os
import re

import requests

# —— 裁判模型服务（OpenAI 兼容 /v1，独立部署，不与训练抢显存）——
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "EMPTY")
JUDGE_TIMEOUT = int(os.environ.get("JUDGE_TIMEOUT", "60"))

# —— rubric：维度名（用于解析 JSON）+ 权重（和=1.0）——
RUBRIC = [
    ("correctness", 0.35),
    ("reasoning", 0.30),
    ("compliance_risk", 0.20),
    ("clarity_format", 0.15),
]
RUBRIC_KEYS = [k for k, _ in RUBRIC]


# ============================================================================
# 工具：解析 / 抽取 / 格式闸门
# ============================================================================
def _parse_gt(raw) -> dict:
    """ground_truth 可能是 JSON 字符串或纯文本，统一成 {answer, verifiable}。"""
    if isinstance(raw, dict):
        return raw
    try:
        d = json.loads(raw)
        if isinstance(d, dict):
            return d
    except Exception:
        pass
    return {"answer": str(raw), "verifiable": False}


def _has_valid_format(response: str) -> bool:
    """格式闸门：<think> 与（\\boxed{} 或 <answer>）同时存在。"""
    has_think = "<think>" in response and "</think>" in response
    has_answer = ("\\boxed{" in response) or (
        "<answer>" in response and "</answer>" in response
    )
    return has_think and has_answer


def _extract_question(extra_info) -> str:
    """优先从 extra_info 取原题上下文（key 兼容 question / prompt）。"""
    if isinstance(extra_info, dict):
        return extra_info.get("question") or extra_info.get("prompt") or ""
    return ""


# ============================================================================
# 阶段2 RLAIF：裁判按 rubric 打分 → 加权聚合为 0~1
# ============================================================================
_RUBRIC_DESC = "\n".join(
    f"- {k}: {desc}（权重 {w:.2f}）"
    for (k, w), desc in zip(
        RUBRIC,
        [
            "结论是否与参考答案一致、事实与计算是否准确",
            "推理链是否完整、逻辑自洽、无原则性错误",
            "是否体现风险/合规意识、避免误导性陈述",
            "是否清晰结构化、符合 <think>/<answer> 或 \\boxed{} 约定",
        ],
    )
)


def _rlaif_rubric_score(question: str, response: str, gold_answer: str) -> float:
    """外部金融裁判按 rubric 给各维度 0~10 分，代码加权聚合成 0~1。

    任何异常（无服务/解析失败）→ 0.0，不污染训练。
    """
    q_part = f"[Question]\n{question}\n" if question else ""
    prompt = (
        f"{q_part}"
        f"[Reference answer / gold]\n{gold_answer}\n\n"
        f"[Assistant answer to evaluate]\n{response}\n\n"
        f"Rubric (score each dimension 0-10, higher is better):\n{_RUBRIC_DESC}\n\n"
        f'Return ONLY JSON: {{"dimensions": {{"correctness": x, "reasoning": x, '
        f'"compliance_risk": x, "clarity_format": x}}, "rationale": "..."}}'
    )
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
    # 1) 优先解析 JSON
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
    # 2) 退化解析：正则抓 key（可带或不带引号）后跟 : 或 = 再跟数字
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
    gt = _parse_gt(ground_truth)

    # Step 1 · 格式不合格 → 0 分
    if not _has_valid_format(solution_str):
        return 0.0

    # Step 2 · RLAIF：裁判按 rubric 对格式合格的回答打分（0~1）
    question = _extract_question(extra_info)
    return _rlaif_rubric_score(question, solution_str, gt.get("answer", ""))

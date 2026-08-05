"""Agentar-Fin-R1 GRPO 奖励函数 —— 简洁版（verl 0.8.0 自定义奖励）。

对齐论文的两阶段混合奖励（verl 官方称 RLVR「规则可验证奖励」+ LLM 裁判打分）：

  Step 1 · 格式闸门（format gate）
    响应必须含 <think>…</think> 与（\\boxed{} 或 <answer>…</answer>）。
    缺任一标签 → 直接 0 分（论文强调「verifiable / auditable」输出）。

  Step 2 · 可验证题 → 规则比对（RLVR）
    verifiable=True（数值/日期/选项/是非/百分比）：抽预测答案归一化比对，
    正确 1.0，错误 0.0。这是确定性硬信号，GRPO 收得最稳。

  Step 3 · 开放题 → LLM 裁判打分（RLGHAI）
    verifiable=False（分析/论述类）：规则判不了对错，
    交给外部 72B 裁判按 正确性+推理严谨性+格式 打 0~1 质量分。

只暴露一个函数 compute_score(data_source, solution_str, ground_truth, extra_info)，
由 verl 逐样本调用。没有 RewardManager 子类、没有 ThreadPool、没有单例客户端——
逻辑是单条自上而下的 if/return，避免在奖励里堆回调。

接线（见 train_grpo.sh）：
    actor_rollout_ref.rollout.reward_model.enable=False
    custom_reward_function.path=<本文件>
    custom_reward_function.name=compute_score

ground_truth 为 JSON 字符串：{"answer": "<标准答案>", "verifiable": true|false}
（由 training/data/prepare_verl_data.py 自动标注 verifiable）。
"""

import json
import os
import re

import requests

# —— 裁判模型服务（OpenAI 兼容 /v1，独立部署，不与训练抢显存）——
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "EMPTY")


# ============================================================================
# 工具：解析 / 归一化 / 抽取 / 比对
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


def _normalize(s: str) -> str:
    """金融答案归一化：去单位/符号/空白，便于比对。"""
    s = (s or "").lower().strip()
    for ch in ["¥", "$", "元", "人民币", "，", ",", "%", "约", "approximately",
               "approx", "≈", " "]:
        s = s.replace(ch, "")
    return re.sub(r"\s+", "", s)


def _to_num(s: str):
    try:
        return float(_normalize(s))
    except Exception:
        return None


def _extract_pred_answer(response: str) -> str:
    """抽取模型回答里的预测答案：优先 \\boxed{}，其次 <answer>。"""
    m = re.search(r"\\boxed\{([^{}]*)\}", response)
    if m:
        return m.group(1).strip()
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def _verify_answer(pred: str, gold: str) -> bool:
    """RLVR 核心：预测 vs 标准答案。数值近似（容差 1e-3）或归一化字符串相等。"""
    if not pred or not gold:
        return False
    pn, gn = _to_num(pred), _to_num(gold)
    if pn is not None and gn is not None:
        if gn == 0:
            return abs(pn) < 1e-6
        return abs(pn - gn) / abs(gn) < 1e-3
    return _normalize(pred) == _normalize(gold)


def _has_valid_format(response: str) -> bool:
    """格式闸门：<think> 与（\\boxed{} 或 <answer>）同时存在。"""
    has_think = "<think>" in response and "</think>" in response
    has_answer = ("\\boxed{" in response) or (
        "<answer>" in response and "</answer>" in response
    )
    return has_think and has_answer


# ============================================================================
# 阶段3 RLGHAI：LLM 裁判质量打分（逐样本，单条最简洁）
# ============================================================================
def _judge_quality(response: str, gold_answer: str) -> float:
    """外部 72B 裁判综合打 0~1 质量分；任何异常 → 0（不污染训练）。"""
    prompt = (
        "你是金融推理质量评分裁判。对给定 (模型回答, 参考标准答案)，"
        "从①结论正确性 ②推理严谨性 ③<think>/<answer>格式合规性 三方面综合打分，"
        "只返回一个 0~1 之间的小数（1 最优，0 最差），不要任何额外解释。\n"
        f"参考标准答案：{gold_answer}\n模型回答：{response}"
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
            timeout=60,
        )
        content = resp.json()["choices"][0]["message"]["content"].strip()
        m = re.search(r"(\d+(?:\.\d+)?)", content)
        if not m:
            return 0.0
        return max(0.0, min(1.0, float(m.group(1))))
    except Exception:
        return 0.0


# ============================================================================
# 主入口：三步式混合奖励（verl 逐样本调用）
# ============================================================================
def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """Agentar-Fin-R1 混合奖励：格式闸门 → RLVR 规则 → RLGHAI 裁判。"""
    gt = _parse_gt(ground_truth)

    # Step 1 · 格式不合格 → 0 分
    if not _has_valid_format(solution_str):
        return 0.0

    # Step 2 · 可验证题 → 规则比对（确定性 0/1）
    if gt.get("verifiable", False):
        pred = _extract_pred_answer(solution_str)
        return 1.0 if _verify_answer(pred, gt.get("answer", "")) else 0.0

    # Step 3 · 开放题 → LLM 裁判质量分
    return _judge_quality(solution_str, gt.get("answer", ""))

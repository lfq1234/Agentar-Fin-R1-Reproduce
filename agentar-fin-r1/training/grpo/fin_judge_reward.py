"""Agentar-Fin-R1 GRPO 奖励函数 — RLAIF + rubric（verl 0.8.0）。

Step 1 · 格式闸门：必须有 <think>…</think>，缺 → 0。
Step 2 · RLAIF：外部 DeepSeek V4 Flash 按 4 维 rubric 打分，加权聚合 0~1。
  correctness 0.40 / reasoning 0.30 / compliance_risk 0.15 / clarity_format 0.15

extra_info（由 prepare_grpo_data.py 生成）：
  question, gold_thinking, gold_output

接线：custom_reward_function.path=<本文件>  custom_reward_function.name=compute_score
运行前：export JUDGE_API_KEY=<key>
"""

import json
import os
import re

import requests

JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://api.deepseek.com/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek-v4-flash")
JUDGE_API_KEY = os.environ.get("JUDGE_API_KEY", "")
JUDGE_TIMEOUT = int(os.environ.get("JUDGE_TIMEOUT", "60"))

RUBRIC = [
    ("correctness", 0.40, "结论是否与参考答案一致、事实与计算是否准确（仅对比答案，不看推理）"),
    ("reasoning", 0.30, "推理链是否完整、逻辑自洽、与标准思维链对标（仅对比思维链，不看答案）"),
    ("compliance_risk", 0.15, "是否体现风险/合规意识、避免误导性陈述"),
    ("clarity_format", 0.15, "是否清晰结构化、表达规范"),
]
RUBRIC_KEYS = [k for k, _, _ in RUBRIC]


# ---- 工具函数 ----

def _has_valid_format(response: str) -> bool:
    return "<think>" in response and "</think>" in response


def _split_response(response: str) -> tuple[str, str]:
    m = re.search(r"<think>(.*?)</think>(.*)", response, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", response.strip()


def _parse_extra_info(extra_info) -> dict:
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


# ---- 裁判 prompt & API 调用 ----

def _make_judge_prompt(
    question: str,
    gold_thinking: str, model_thinking: str,
    gold_output: str, model_output: str,
) -> str:
    parts = []
    if question:
        parts.append(f"[Question]\n{question}")
    parts.append(
        f"[Reasoning comparison — score 'reasoning' only]\n"
        f"Gold thinking:\n{gold_thinking}\n\n"
        f"Model thinking:\n{model_thinking}"
    )
    parts.append(
        f"[Output comparison — score 'correctness', 'compliance_risk', 'clarity_format']\n"
        f"Gold output:\n{gold_output}\n\n"
        f"Model output:\n{model_output}"
    )
    rubric_lines = [f"- {k}（{w:.2f}）: {desc}" for k, w, desc in RUBRIC]
    parts.append(
        "Rubric (score each dimension 0-10):\n"
        + "\n".join(rubric_lines)
        + '\n\nReturn ONLY JSON: {"dimensions": {"correctness": x, ...}, "rationale": "..."}'
    )
    return "\n\n".join(parts)


def _parse_rubric_scores(content: str) -> dict:
    try:
        d = json.loads(content)
        dims = d.get("dimensions", d)
        if isinstance(dims, dict):
            return {k: float(dims[k]) for k in RUBRIC_KEYS if k in dims}
    except Exception:
        pass
    out = {}
    for key in RUBRIC_KEYS:
        m = re.search(rf'(?:"{key}"|{key})\s*[:=]\s*(\d+(?:\.\d+)?)', content)
        if m:
            out[key] = float(m.group(1))
    return out


def _call_judge(prompt: str) -> float:
    try:
        resp = requests.post(
            f"{JUDGE_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {JUDGE_API_KEY}"},
            json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0},
            timeout=JUDGE_TIMEOUT,
        )
        content = resp.json()["choices"][0]["message"]["content"]
        scores = _parse_rubric_scores(content)
        if not scores:
            return 0.0
        total = sum(w * max(0, min(10, scores[k])) / 10 for k, w, _ in RUBRIC if k in scores)
        return max(0.0, min(1.0, total))
    except Exception:
        return 0.0


# ---- 主入口 ----

def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    if not _has_valid_format(solution_str):
        return 0.0

    model_thinking, model_output = _split_response(solution_str)
    ref = _parse_extra_info(extra_info)

    prompt = _make_judge_prompt(
        question=ref.get("question", ""),
        gold_thinking=ref.get("gold_thinking", ""),
        model_thinking=model_thinking,
        gold_output=ref.get("gold_output", ""),
        model_output=model_output,
    )
    return _call_judge(prompt)

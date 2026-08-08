"""Agentar-Fin-R1 GRPO 奖励函数 — RLAIF + rubric（verl 0.8.0）。

格式闸门：<think>…</think> 缺 → 0。通过后由外部裁判按 4 维 rubric 打分（0~1）：
  correctness 0.40 / reasoning 0.30 / compliance_risk 0.15 / clarity_format 0.15

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
    ("correctness", 0.40, "结论是否与参考答案一致（仅对比答案，不看推理）"),
    ("reasoning", 0.30, "推理链是否完整、逻辑自洽、与标准思维链对标（仅对比思维链，不看答案）"),
    ("compliance_risk", 0.15, "是否体现风险/合规意识"),
    ("clarity_format", 0.15, "表达是否清晰、结构规范"),
]
RUBRIC_KEYS = [k for k, _, _ in RUBRIC]


def _split_response(response: str) -> tuple[str, str]:
    m = re.search(r"<think>(.*?)</think>(.*)", response, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", response.strip()


def _make_judge_prompt(question: str, gold_thinking: str, model_thinking: str,
                       gold_output: str, model_output: str) -> str:
    parts = []
    if question:
        parts.append(f"[Question]\n{question}")
    parts.append(
        f"[Reasoning — score 'reasoning' only]\n"
        f"Gold thinking:\n{gold_thinking}\n\nModel thinking:\n{model_thinking}"
    )
    parts.append(
        f"[Output — score 'correctness', 'compliance_risk', 'clarity_format']\n"
        f"Gold output:\n{gold_output}\n\nModel output:\n{model_output}"
    )
    rubric_lines = [f"- {k}（{w:.2f}）: {desc}" for k, w, desc in RUBRIC]
    parts.append(
        "Rubric (score each dimension 0-10):\n" + "\n".join(rubric_lines)
        + '\n\nReturn ONLY JSON: {"dimensions": {"correctness": x, ...}, "rationale": "..."}'
    )
    return "\n\n".join(parts)


def _parse_scores(content: str) -> dict:
    try:
        d = json.loads(content).get("dimensions", {})
        return {k: float(d[k]) for k in RUBRIC_KEYS if k in d}
    except Exception:
        return {}


def compute_score(_data_source, solution_str, _ground_truth, extra_info=None) -> float:
    if "<think>" not in solution_str or "</think>" not in solution_str:
        return 0.0

    model_thinking, model_output = _split_response(solution_str)
    ref = extra_info or {}

    prompt = _make_judge_prompt(
        ref.get("question", ""),
        ref.get("gold_thinking", ""), model_thinking,
        ref.get("gold_output", ""), model_output,
    )

    try:
        resp = requests.post(
            f"{JUDGE_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {JUDGE_API_KEY}"},
            json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.0},
            timeout=JUDGE_TIMEOUT,
        )
        content = resp.json()["choices"][0]["message"]["content"]
        scores = _parse_scores(content)
        total = sum(w * max(0, min(10, scores.get(k, 0))) / 10 for k, w, _ in RUBRIC)
        return max(0.0, min(1.0, total))
    except Exception:
        return 0.0

"""Stage 2 GRPO reward —— LLM-as-judge（加载模型判断，不做数值计算）。

GRPO 不能用金标直接算 loss，需要"可验证奖励"。金融推理题多为开放/半开放，
用规则做数值/字符串比对既脆弱又覆盖不全。本项目改为**加载一个裁判模型**，
对每个 rollout 由模型判断"结论是否正确、推理是否合理、是否带 <think> 边界"，
而不是在本地做数值计算。

裁判模型以独立服务部署（OpenAI 兼容 /v1 接口，推荐 vLLM）：
    vllm serve <judge_model> --port 8000 --gpu-memory-utilization 0.4
reward 函数在训练循环里通过 HTTP 调它，不与训练抢同一进程的显存。

judge 返回 JSON 数组，每项 {"correct": 0|1, "reason": "..."}；
correct=1 当且仅当结论与标准答案一致、推理无原则性错误、且含 <think> 边界。
"""
from __future__ import annotations

import json

from openai import OpenAI


JUDGE_SYSTEM = (
    "你是一个严格的金融推理题裁判。下面会给你若干组"
    "【问题】【标准答案】【待评判回答】。请逐组判断待评判回答是否："
    "① 结论与标准答案一致；② 推理合理无原则性错误；③ 包含 <think>...</think> 思考过程。"
    "三者缺一不可，缺 <think> 边界则 correct=0。"
    "只输出一个 JSON 数组，长度与输入组数一致，每项为 "
    '{"correct": 0或1, "reason": "简短理由"}。不要输出任何额外文字。'
)


def build_judge_prompt(triples):
    """triples: [(query, gold, completion), ...] 拼成单条批量裁判 prompt。"""
    lines = []
    for i, (q, g, c) in enumerate(triples, 1):
        lines.append(
            f"第{i}组\n【问题】\n{q}\n【标准答案】\n{g}\n【待评判回答】\n{c}\n"
        )
    return "\n".join(lines) + "\n请按系统指令输出 JSON 数组。"


class LLMJudge:
    """封装一个 OpenAI 兼容的裁判模型服务（如 vLLM）。"""

    def __init__(self, base_url, model, api_key="EMPTY", temperature=0.0):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.temperature = temperature

    def score_batch(self, triples):
        """整批评判，返回与 triples 等长的 float 列表（0/1）。

        解析失败或长度不符时整批回退为 0.0（干净处理，不重试）。
        """
        n = len(triples)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": build_judge_prompt(triples)},
                ],
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content
            obj = json.loads(text)
            arr = obj["scores"] if isinstance(obj, dict) and "scores" in obj else obj
            if not isinstance(arr, list) or len(arr) != n:
                return [0.0] * n
            return [float(item.get("correct", 0)) for item in arr]
        except Exception:
            return [0.0] * n


def make_reward_funcs(judge: LLMJudge):
    """返回 reward 函数列表，传给 GRPOTrainer(reward_funcs=...)。

    整批一次性发给 judge（一次 HTTP 调用 / 步），避免逐条调用拖慢训练。
    数据集需含 query / gold_answer 列（由 TRL 随 batch 透传给 reward 函数）。
    """

    def reward_judge(prompts, completions, query=None, gold_answer=None, **kwargs):
        queries = list(query) if query is not None else [None] * len(completions)
        golds = list(gold_answer) if gold_answer is not None else [None] * len(completions)
        triples = list(zip(queries, golds, completions))
        return judge.score_batch(triples)

    return [reward_judge]

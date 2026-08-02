"""GRPO custom reward for Agentar-Fin-R1 (ms-swift external plugin).

LLM-as-judge 裁判奖励：不写任何数值比对规则，调用独立部署的裁判模型
（vLLM / swift deploy 起的 OpenAI 兼容 /v1 服务）对每个 rollout 打分。
判定标准三合一：① 结论与 gold_answer 一致；② 推理合理无原则性错误；
③ 含 <think>...</think> 边界（缺边界直接 0）。格式要求已并入 correct，
不再单独设格式奖励。

注册方式（ms-swift 约定）：
    1. 在 orms 注册表登记本类；
    2. 训练命令用 --external_plugins 指向本文件 + --reward_funcs judge_reward。
"""

import json

from openai import OpenAI

from swift.rewards import ORM, orms


class LLMJudgeReward(ORM):
    """对每个 rollout 调用外部裁判模型打分，返回 [0.0, 1.0] 列表。

    __call__ 签名遵循 ms-swift 约定：位置参数 completions（模型输出列表），
    其余数据集列以 kwargs 透传（本任务需要 gold_answer / query）。
    """

    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "Qwen/Qwen2.5-72B-Instruct",
                 api_key: str = "EMPTY"):
        super().__init__()
        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model = self.client.models.list().data[0].id
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "连不上裁判模型服务，请先 `swift deploy` 或 `vllm serve` 起好 /v1 端点。"
            ) from e

    def __call__(self, completions, gold_answer=None, **kwargs) -> list[float]:
        # 一次性批量构造 prompt，一条 HTTP 调用让裁判对整批打分（省吞吐）。
        batch = []
        for comp, gold in zip(completions, gold_answer or []):
            batch.append({"completion": comp, "gold_answer": gold})
        prompt = (
            "你是金融推理的裁判。对每组 (模型回答, 标准答案)，判断模型回答是否正确。\n"
            "当且仅当满足全部条件时返回 1：① 结论与标准答案一致；"
            "② 推理合理无原则性错误；③ 包含 <think>...</think> 边界。\n"
            "只返回 JSON 数组，每项 {\"correct\": 0 或 1}。\n"
            f"数据：{json.dumps(batch, ensure_ascii=False)}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,                  # 打分确定性
            )
            parsed = json.loads(resp.choices[0].message.content.strip())
            rewards = [float(item.get("correct", 0)) for item in parsed]
            if len(rewards) != len(completions):
                rewards = [0.0] * len(completions)   # 长度不符 → 整批 0，不重试
        except Exception:  # noqa: BLE001 —— 解析失败 / 服务异常：整批回退 0
            rewards = [0.0] * len(completions)
        return rewards


# 注册到 ms-swift 的 orms 表，供 --reward_funcs judge_reward 引用。
orms["judge_reward"] = LLMJudgeReward

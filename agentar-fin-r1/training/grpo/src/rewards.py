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
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from swift.rewards import ORM, orms


class LLMJudgeReward(ORM):
    """对每个 rollout 调用外部裁判模型打分，返回 [0.0, 1.0] 列表。

    __call__ 签名遵循 ms-swift 约定：位置参数 completions（模型输出列表），
    其余数据集列以 kwargs 透传（本任务需要 gold_answer / query）。
    """

    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "Qwen/Qwen2.5-72B-Instruct",
                 api_key: str = "EMPTY",
                 max_workers: int = 8):
        super().__init__()
        # 并发调用裁判的线程数；vLLM judge 端支持 continuous batching，可安全并发。
        # 64 条 completion 拆 8 路 × 8 条，比单次大 batch 快 3-5x。
        self.max_workers = max_workers
        try:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            self.model = self.client.models.list().data[0].id
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "连不上裁判模型服务，请先 `swift deploy` 或 `vllm serve` 起好 /v1 端点。"
            ) from e

    def __call__(self, completions, gold_answer=None, **kwargs) -> list[float]:
        # 把整批 (completion, gold) 拆成多个子 batch 并发调用裁判，
        # 用 ThreadPoolExecutor 掩盖单次 HTTP 往返延迟。
        # judge 端 vLLM 支持 continuous batching，并发请求会被高效批处理，
        # 64 条拆 8 路 × 8 条，比单次大 batch 快 3-5x。
        batch = []
        for comp, gold in zip(completions, gold_answer or []):
            batch.append({"completion": comp, "gold_answer": gold})
        if not batch:
            return []

        n_workers = min(self.max_workers, len(batch))
        chunk = max(1, (len(batch) + n_workers - 1) // n_workers)
        chunks = [batch[i:i + chunk] for i in range(0, len(batch), chunk)]

        def judge_chunk(items):
            prompt = (
                "你是金融推理的裁判。对每组 (模型回答, 标准答案)，判断模型回答是否正确。\n"
                "当且仅当满足全部条件时返回 1：① 结论与标准答案一致；"
                "② 推理合理无原则性错误；③ 包含 <think>...</think> 边界。\n"
                "只返回 JSON 数组，每项 {\"correct\": 0 或 1}。\n"
                f"数据：{json.dumps(items, ensure_ascii=False)}"
            )
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,                  # 打分确定性
                )
                parsed = json.loads(resp.choices[0].message.content.strip())
                rs = [float(item.get("correct", 0)) for item in parsed]
                # 子 batch 长度不符 → 该子 batch 整体回退 0，不影响其他并发请求
                return rs if len(rs) == len(items) else [0.0] * len(items)
            except Exception:  # noqa: BLE001 —— 单个子 batch 失败不影响其他
                return [0.0] * len(items)

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            sub_results = list(ex.map(judge_chunk, chunks))

        rewards = [r for sub in sub_results for r in sub]
        # 最终长度兜底（防御性，理论上上方已保证）
        if len(rewards) != len(completions):
            rewards = [0.0] * len(completions)
        return rewards


# 注册到 ms-swift 的 orms 表，供 --reward_funcs judge_reward 引用。
orms["judge_reward"] = LLMJudgeReward

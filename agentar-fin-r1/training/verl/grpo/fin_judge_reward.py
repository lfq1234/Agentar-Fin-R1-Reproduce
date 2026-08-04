"""Custom LLM-as-judge reward for Agentar-Fin-R1 (verl 0.8.0).

对齐原 ms-swift 版 training/grpo/src/rewards.py 的判定逻辑，但**适配 verl 的
reward 架构**：

- verl 在 driver 进程里串行调用 reward（NaiveRewardManager.__call__ 逐样本迭代）。
- 原设计是「把整批 completion 拼一个 prompt 一次 HTTP 调用 / 并发子 batch」，
  这个并发优势只有在**整批收集后**才能发挥——所以必须子类化 RewardManager，
  在 __call__ 里先 decode 整批、再并发打裁判，而不是用逐样本的 compute_score。

判定标准三合一（与原版一致）：① 结论与 gold_answer 一致；
② 推理合理无原则性错误；③ 含 <think>...</think> 边界（缺边界直接 0）。

接线方式（见 train_grpo.sh）：
    actor_rollout_ref.rollout.reward_model.enable=False
    actor_rollout_ref.rollout.reward_model.reward_manager=fin_judge
    custom_reward_function.path=<本文件>        # 触发 @register("fin_judge") 导入
    custom_reward_function.name=compute_score  # 兜底逐样本路径（可选）

verl 加载本模块时会执行 @register("fin_judge")，把 FinJudgeRewardManager 注册进
reward manager 表，config 里 reward_manager=fin_judge 即可选中。
"""

import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch

from openai import OpenAI

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.naive import NaiveRewardManager

# —— 裁判模型服务（OpenAI 兼容 /v1，独立部署，不与训练抢显存）——
JUDGE_BASE_URL = "http://localhost:8000/v1"
JUDGE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
JUDGE_API_KEY = "EMPTY"
JUDGE_MAX_WORKERS = 8  # 并发子 batch 数；judge 端 vLLM 支持 continuous batching


class _JudgeClient:
    """懒加载、线程安全的 OpenAI 客户端（reward worker 多线程序列化创建）。"""

    _lock = threading.Lock()
    _client = None

    @classmethod
    def get(cls):
        if cls._client is None:
            with cls._lock:
                if cls._client is None:
                    cls._client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)
        return cls._client


def _judge_chunk(items: list[dict]) -> list[float]:
    """对一个子 batch 打分：items = [{"completion":..., "gold_answer":...}]。

    返回与 items 等长的 [0.0, 1.0] 列表；解析失败 / 服务异常 → 整子 batch 回退 0，
    不影响其他并发子 batch（比原版「整批回退 0」更鲁棒）。
    """
    if not items:
        return []
    prompt = (
        "你是金融推理的裁判。对每组 (模型回答, 标准答案)，判断模型回答是否正确。\n"
        "当且仅当满足全部条件时返回 1：① 结论与标准答案一致；"
        "② 推理合理无原则性错误；③ 包含 <think>...</think> 边界。\n"
        "只返回 JSON 数组，每项 {\"correct\": 0 或 1}。\n"
        f"数据：{json.dumps(items, ensure_ascii=False)}"
    )
    try:
        resp = _JudgeClient.get().chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # 打分确定性
        )
        parsed = json.loads(resp.choices[0].message.content.strip())
        rs = [float(x.get("correct", 0)) for x in parsed]
        return rs if len(rs) == len(items) else [0.0] * len(items)
    except Exception:  # noqa: BLE001 —— 单个子 batch 失败不影响其他
        return [0.0] * len(items)


SYSTEM_PROMPT = (
    "你是金融推理的裁判。对每组 (模型回答, 标准答案)，判断模型回答是否正确。"
)


@register("fin_judge")
class FinJudgeRewardManager(NaiveRewardManager):
    """整批并发 LLM-judge reward manager。

    复用父类的 tokenizer / reward_fn_key / 打印逻辑，仅重写 __call__：
    1) decode 整批 response + ground_truth；
    2) 用 ThreadPoolExecutor 把整批拆成 JUDGE_MAX_WORKERS 路并发子 batch；
    3) 合并结果回填 reward_tensor（reward 落在 response 最后一个有效 token）。
    """

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict:
        # 若 RM 已直接给出分数（极少数路径），原样返回
        rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if rm_scores is not None:
            return rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        # 1) decode 整批
        decoded = []
        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            pl = prompt_ids.shape[-1]
            vpl = item.batch["attention_mask"][:pl].sum()
            resp_ids = item.batch["responses"]
            vrl = item.batch["attention_mask"][pl:].sum()
            response_str = self.tokenizer.decode(
                resp_ids[:vrl], skip_special_tokens=True
            )
            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            decoded.append((i, int(vrl), response_str, ground_truth))

        # 2) 并发打裁判（掩盖 HTTP 延迟，正是迁 verl 想保留的优化）
        batch = [{"completion": r, "gold_answer": g} for (_, _, r, g) in decoded]
        n_workers = min(JUDGE_MAX_WORKERS, len(batch)) if batch else 1
        chunk = max(1, (len(batch) + n_workers - 1) // n_workers)
        chunks = [batch[i:i + chunk] for i in range(0, len(batch), chunk)]
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            sub_results = list(ex.map(_judge_chunk, chunks))
        rewards = [r for sub in sub_results for r in sub]
        if len(rewards) != len(decoded):  # 防御性兜底
            rewards = [0.0] * len(decoded)

        # 3) 回填
        for (idx, vrl, _, _), score in zip(decoded, rewards):
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            reward_tensor[idx, vrl - 1] = reward

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor


# —— 兜底：verl 默认的逐样本 compute_score 路径（reward_model.enable=True 时）——
def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    rs = _judge_chunk([{"completion": solution_str, "gold_answer": ground_truth}])
    return rs[0] if rs else 0.0

"""Agentar-Fin-R1 GRPO reward manager —— 两阶段混合奖励（verl 0.8.0）。

设计对齐 Agentar-Fin-R1 的「Two-Phase Reward」：

  阶段 1 · RLVR（Reinforcement Learning with Verifiable Rewards，规则可验证）
    - 强制格式闸门：响应必须含 <think>...</think> 与 <answer>...</answer>，
      缺任一标签 → 直接 0 分（「不合格的 0 分」）。
    - 对【可验证题】（ground_truth.verifiable=True，如数值/日期/选项/百分比）：
      抽取预测答案做归一化比对（数值近似 / 字符串 / 包含），正确→1.0，错误→0.0。

  阶段 2 · RLGHAI（Reinforcement Learning with Generative AI，LLM 裁判打分）
    - 对【开放题】（verifiable=False，如分析/论述类）：RLVR 无法判对错，
      交由外部 72B 裁判按 正确性+推理严谨性+格式 综合打 0~1 质量分。
    - 可验证题若已判对，稳定给 1.0（硬信号），不再过裁判，避免裁判噪声污染。

verl 在 driver 进程逐样本迭代，但 RLGHAI 的 HTTP 延迟要靠【整批并发】掩盖 ——
因此必须子类化 RewardManager，在 __call__ 里先 decode 整批、再 ThreadPool 并发打裁判，
而不是用逐样本的 compute_score（后者串行，延迟无法掩盖）。

接线方式（见 train_grpo.sh）：
    actor_rollout_ref.rollout.reward_model.enable=False
    actor_rollout_ref.rollout.reward_model.reward_manager=fin_judge
    custom_reward_function.path=<本文件>     # 触发 @register("fin_judge") 导入
    custom_reward_function.name=compute_score  # 逐样本兜底路径（可选）

GRPO 数据需在 reward_model.ground_truth 里写入 JSON：
    {"answer": "<标准答案>", "verifiable": true|false}
（由 training/data/prepare_verl_data.py 自动标注 verifiable。）
"""

import json
import re
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

# —— 行为开关 ——
RLVR_FORMAT_GATE = True          # 通用格式闸门：缺 <think>/<answer> → 0 分
RLGHAI_FOR_VERIFIABLE = False    # 可验证且判对的题，是否也过裁判打质量分（默认关，保稳定）


# ============================================================================
# 工具：答案解析 / 归一化 / 可验证比对
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
    s = re.sub(r"\s+", "", s)
    return s


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
    """RLVR 核心：预测答案 vs 标准答案。

    仅两种确定路径：① 数值近似（容差 1e-3）② 归一化字符串完全相等。
    不做宽松的「互为子串」兜底，避免长文本噪声导致误判。
    """
    if not pred or not gold:
        return False
    pn, gn = _to_num(pred), _to_num(gold)
    if pn is not None and gn is not None:  # ① 数值近似
        if gn == 0:
            return abs(pn) < 1e-6
        return abs(pn - gn) / abs(gn) < 1e-3
    return _normalize(pred) == _normalize(gold)  # ② 归一化精确相等


def _rlvr(response: str, gt: dict):
    """阶段1 RLVR。返回 float（已判）或 None（交由 RLGHAI）。

    格式闸门要求 <think> 与（\\boxed{} 或 <answer>）同时存在；缺任一 → 0 分。
    """
    if RLVR_FORMAT_GATE:
        has_think = "<think>" in response and "</think>" in response
        has_answer = (
            ("<answer>" in response and "</answer>" in response)
            or ("\\boxed{" in response)
        )
        if not (has_think and has_answer):
            return 0.0  # 格式不合格 → 0 分

    if gt.get("verifiable", False):
        pred = _extract_pred_answer(response)
        return 1.0 if _verify_answer(pred, gt.get("answer", "")) else 0.0
    # 开放题：RLVR 无法判对错，交给阶段2
    return None


# ============================================================================
# 阶段2 RLGHAI：LLM 裁判质量打分（整批并发，掩盖 HTTP 延迟）
# ============================================================================
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


def _judge_quality_chunk(items: list[dict]) -> list[float]:
    """对一个子 batch 打质量分：items = [{"completion":..., "gold_answer":...}]。

    返回与 items 等长的 [0.0, 1.0] 列表；解析失败 / 服务异常 → 整子 batch 回退 0，
    不影响其他并发子 batch（比「整批回退 0」更鲁棒）。
    """
    if not items:
        return []
    prompt = (
        "你是金融推理质量评分裁判。对每组 (模型回答, 参考标准答案)，"
        "从①结论正确性 ②推理严谨性 ③<think>/<answer>格式合规性 三方面综合打分，"
        "给出 0~1 之间的分数（1 为最优，0 为最差）。\n"
        "只返回 JSON 数组，每项 {\"score\": 0~1}。\n"
        f"数据：{json.dumps(items, ensure_ascii=False)}"
    )
    try:
        resp = _JudgeClient.get().chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # 打分确定性
        )
        parsed = json.loads(resp.choices[0].message.content.strip())
        scores = []
        for x in parsed:
            try:
                s = float(x.get("score", 0.0))
            except Exception:
                s = 0.0
            scores.append(max(0.0, min(1.0, s)))  # 截断到 [0,1]
        return scores if len(scores) == len(items) else [0.0] * len(items)
    except Exception:  # noqa: BLE001 —— 单个子 batch 失败不影响其他
        return [0.0] * len(items)


def _judge_quality_batch(items: list[dict]) -> list[float]:
    """把整批拆成 JUDGE_MAX_WORKERS 路并发子 batch 并发打分。"""
    if not items:
        return []
    n_workers = min(JUDGE_MAX_WORKERS, len(items))
    chunk = max(1, (len(items) + n_workers - 1) // n_workers)
    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        sub = list(ex.map(_judge_quality_chunk, chunks))
    return [s for part in sub for s in part]


# ============================================================================
# RewardManager：两阶段混合
# ============================================================================
@register("fin_judge")
class FinRewardManager(NaiveRewardManager):
    """RLVR(规则可验证) + RLGHAI(LLM裁判) 两阶段混合 reward manager。

    复用父类的 tokenizer / 打印逻辑，仅重写 __call__：
    1) decode 整批 response + ground_truth；
    2) 阶段1 RLVR：格式闸门 + 可验证题答案比对（不合格直接 0）；
    3) 阶段2 RLGHAI：开放题并发打裁判质量分；
    4) 合并结果回填 reward_tensor。
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
            vrl = int(item.batch["attention_mask"][pl:].sum())
            resp_ids = item.batch["responses"]
            response_str = self.tokenizer.decode(
                resp_ids[:vrl], skip_special_tokens=True
            )
            gt_raw = item.non_tensor_batch["reward_model"]["ground_truth"]
            decoded.append((i, vrl, response_str, gt_raw))

        # 2) 阶段1 RLVR（规则，CPU 快，无需并发）
        rewards = [None] * len(decoded)
        judge_items = []  # (pos, completion, gold_answer)
        for pos, (idx, vrl, resp, gt_raw) in enumerate(decoded):
            gt = _parse_gt(gt_raw)
            rlvr = _rlvr(resp, gt)
            if rlvr is not None:
                rewards[pos] = rlvr
                reward_extra_info["rlvr"].append(float(rlvr))
                reward_extra_info["route"].append("rlvr")
            else:
                judge_items.append((pos, resp, gt.get("answer", "")))
                reward_extra_info["route"].append("rlghai")

        # 3) 阶段2 RLGHAI（开放题，并发掩盖延迟）
        if judge_items and RLGHAI_FOR_VERIFIABLE:
            # 可选：可验证且判对的题也过裁判（默认关，见顶部开关）
            for pos, (idx, vrl, resp, gt_raw) in enumerate(decoded):
                if rewards[pos] == 1.0:
                    judge_items.append((pos, resp, _parse_gt(gt_raw).get("answer", "")))
        if judge_items:
            items = [{"completion": c, "gold_answer": g} for (_, c, g) in judge_items]
            scores = _judge_quality_batch(items)
            if len(scores) != len(judge_items):  # 防御性兜底
                scores = [0.0] * len(judge_items)
            for (pos, _, _), s in zip(judge_items, scores):
                rewards[pos] = s
                reward_extra_info["rlghai_score"].append(float(s))

        # 4) 回填（reward 落在 response 最后一个有效 token）
        for pos, (idx, vrl, _, _) in enumerate(decoded):
            r = rewards[pos] if rewards[pos] is not None else 0.0
            reward_tensor[idx, vrl - 1] = r

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor


# —— 兜底：verl 默认的逐样本 compute_score 路径（reward_model.enable=True 时）——
def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:
    """逐样本完整跑一遍两阶段（无并发，仅作兜底）。"""
    gt = _parse_gt(ground_truth)
    rlvr = _rlvr(solution_str, gt)
    if rlvr is not None:
        return rlvr
    scores = _judge_quality_chunk(
        [{"completion": solution_str, "gold_answer": gt.get("answer", "")}]
    )
    return scores[0] if scores else 0.0

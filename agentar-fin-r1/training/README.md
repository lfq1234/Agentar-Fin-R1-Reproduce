# Agentar-Fin-R1 — verl 训练（SFT + GRPO）

本目录是 **verl 0.8.0** 版的两阶段训练实现。原 ms-swift 版（SFT/GRPO）已删除，仅保留 verl。
决策背景见根 `report.md` 与 `../../.workbuddy/memory/`：ms-swift 在 GRPO 阶段
rollout 用 `transformers.generate`、reward 同步阻塞，verl 用 vLLM rollout + Ray 式
流水线，对本项目（Qwen3-8B + K=8 + 外部 72B 裁判）是「代差级」提速。

## 目录结构

```
training/
├── README.md
├── sft/
│   ├── train_sft.py              # Stage 1 超参持有 + 训练启动器（调用 verl.trainer.sft_trainer）
│   └── train_sft.sh              # Stage 1 壳：设路径环境变量后调用 train_sft.py
├── grpo/
│   ├── train_grpo.sh             # Stage 2：GRPO（vLLM rollout + LoRA r=32）
│   └── fin_judge_reward.py       # 混合奖励：compute_score（RLVR 闸门 + 答案比对 → LLM 裁判）
├── merge_lora.py                 # 合并 SFT LoRA → 完整 checkpoint
└── data/
    └── prepare_verl_data.py      # golden.jsonl / DeepFinance-100K → verl parquet
```

## 运行顺序

```bash
# 0) 准备数据（一次）
python training/data/prepare_verl_data.py \
    --input ./data/golden/golden.jsonl --out-dir ./data/verl
# 或 DeepFinance-100K 本地副本：
#   --input /path/to/deepfinance.parquet

# 1) Stage 1 SFT
NPROC=1 MODEL_PATH=Qwen/Qwen3-8B \
    SFT_DATA=./data/verl/sft.parquet \
    bash training/sft/train_sft.sh
# 或直接：
#   python training/sft/train_sft.py
# 产物 → ./outputs/sft_lora_adapter

# 2) 合并 SFT LoRA
python training/merge_lora.py \
    --base Qwen/Qwen3-8B \
    --adapter ./outputs/sft_lora_adapter \
    --output ./outputs/sft_merged

# 3) 起裁判服务（独立进程 / 独立卡，OpenAI 兼容 /v1）
#   vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000
#   或在另一张卡 swift deploy ...

# 4) Stage 2 GRPO
NPROC=1 SFT_MERGED=./outputs/sft_merged \
    GRPO_DATA=./data/verl/grpo.parquet \
    bash training/grpo/train_grpo.sh
# 产物 → ./outputs/grpo_lora_adapter
```

## 与 ms-swift 版的关键差异

| 维度 | ms-swift（旧） | verl（本目录） |
|---|---|---|
| 入口 | `swift sft` / `swift rlhf` | `verl.trainer.sft_trainer` / `verl.trainer.main_ppo` |
| SFT LoRA 配置 | `tuner_type/lora_rank/...` | `model.lora_rank/lora_alpha/target_modules` |
| rollout | `transformers.generate`（慢） | `rollout.name=vllm`（PagedAttention，快 3-5x） |
| GRPO 算法 | `rlhf_type=grpo` | `algorithm.adv_estimator=grpo` |
| group size K | `num_generations=8` | `rollout.n=8` |
| KL 系数 | `beta=0.04` | `actor.kl_loss_coef=0.04` |
| reward | `external_plugins` + `reward_funcs` | `custom_reward_function`（path + name=compute_score） |
| judge 并发 | `ThreadPoolExecutor`（同文件） | verl 逐样本调用 `compute_score`；开放题串行打裁判（原型足够，先跑通） |

## reward 实现要点（fin_judge_reward.py）

简洁版：只暴露一个函数 `compute_score(data_source, solution_str, ground_truth,
extra_info)`，由 verl 逐样本调用（见 `train_grpo.sh` 的 `custom_reward_function`
接线）。**没有 RewardManager 子类、没有 ThreadPool、没有单例客户端**——逻辑是单条
自上而下的 `if/return`，避免奖励里的回调嵌套。

三步式混合奖励（对齐论文「verifiable rewards + intricate reward structures」）：

1. **格式闸门**：响应必须含 `<think>…</think>` 与（`\boxed{}` 或 `<answer>…</answer>`）。
   缺任一标签 → 直接 0 分（论文强调「verifiable / auditable」输出）。
2. **可验证题（verifiable=True）→ 规则比对（RLVR）**：数值近似（容差 1e-3）或归一化
   字符串相等，正确 1.0 / 错误 0.0。确定性硬信号，GRPO 收得最稳。
3. **开放题（verifiable=False）→ LLM 裁判（RLGHAI）**：规则判不了对错，交给外部
   72B 裁判按 正确性+推理严谨性+格式 打 0~1 质量分；异常 → 0（不污染训练）。

`train_grpo.sh` 接线：
```
actor_rollout_ref.rollout.reward_model.enable=False
custom_reward_function.path=<本文件>
custom_reward_function.name=compute_score
```

> 说明：移除原「子类化 `NaiveRewardManager` + 整批 ThreadPool 并发打裁判」的实现，
> 换取极简可维护性。代价是开放题的裁判调用变为逐样本串行——原型/小 batch 跑通足够；
> 若后续需要吞吐，可再把并发加回 `compute_score` 内部（局部优化，不影响接口）。

## 注意事项 / 已知风险

1. **dtype**：verl 默认走 bf16（vLLM rollout 对 bf16 更稳）。原 ms-swift 用 fp16，
   迁移后建议统一 bf16；若坚持 fp16 需显式配置 `actor_rollout_ref.model.*
   / rollout` 的 dtype。
2. **merge 兼容性**：`merge_lora.py` 假设 verl SFT LoRA 输出是 peft 兼容格式。
   若你的 verl 版本把 LoRA 存成非 peft 格式，改用「SFT 全参 + GRPO 自带 LoRA」
   路径，跳过 merge。
3. **显存**：8B + LoRA + K=8 + seq 4096 在 24G 卡上偏紧。`rollout.gpu_memory_utilization=0.5`
   可下调；多卡把 `rollout.tensor_model_parallel_size` / `NPROC` 调大。
4. **裁判服务**：`fin_judge_reward.py` 顶部常量 `JUDGE_BASE_URL / JUDGE_MODEL`
   （或用环境变量 `JUDGE_BASE_URL / JUDGE_MODEL / JUDGE_API_KEY` 覆盖）需与
   实际部署一致；开放题才走裁判，可验证题不打裁判。

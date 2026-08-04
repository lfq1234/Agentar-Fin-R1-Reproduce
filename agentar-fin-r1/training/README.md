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
│   └── fin_judge_reward.py       # LLM-as-judge reward manager（子类化 NaiveRewardManager）
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
| reward | `external_plugins` + `reward_funcs` | `reward_model.reward_manager=fin_judge` |
| judge 并发 | `ThreadPoolExecutor`（同文件） | `FinJudgeRewardManager.__call__` 整批并发（driver 级） |

## reward 实现要点（fin_judge_reward.py）

verl 在 driver 进程**逐样本串行**调用 reward。原 ms-swift 的「整批一次 HTTP /
并发子 batch」优化，只有在**整批收集后**才能发挥，因此这里**子类化
`NaiveRewardManager`** 重写 `__call__`：先 decode 整批 response + ground_truth，
再用 `ThreadPoolExecutor` 并发打裁判，最后回填 `reward_tensor`。

判定标准三合一（与原版一致）：① 结论与 gold_answer 一致；② 推理合理无原则性错误；
③ 含 `<think>...</think>` 边界（缺边界直接 0）。

`@register("fin_judge")` 在 verl 加载本模块时注册进 reward manager 表；
`train_grpo.sh` 里 `custom_reward_function.path` 指向本文件（触发导入 + 注册），
`reward_model.reward_manager=fin_judge` 选中它。

## 注意事项 / 已知风险

1. **dtype**：verl 默认走 bf16（vLLM rollout 对 bf16 更稳）。原 ms-swift 用 fp16，
   迁移后建议统一 bf16；若坚持 fp16 需显式配置 `actor_rollout_ref.model.*
   / rollout` 的 dtype。
2. **merge 兼容性**：`merge_lora.py` 假设 verl SFT LoRA 输出是 peft 兼容格式。
   若你的 verl 版本把 LoRA 存成非 peft 格式，改用「SFT 全参 + GRPO 自带 LoRA」
   路径，跳过 merge。
3. **显存**：8B + LoRA + K=8 + seq 4096 在 24G 卡上偏紧。`rollout.gpu_memory_utilization=0.5`
   可下调；多卡把 `rollout.tensor_model_parallel_size` / `NPROC` 调大。
4. **裁判服务**：`fin_judge_reward.py` 顶部的 `JUDGE_BASE_URL / JUDGE_MODEL` 需与
   实际部署一致；并发数 `JUDGE_MAX_WORKERS=8` 需裁判 vLLM 能承受。

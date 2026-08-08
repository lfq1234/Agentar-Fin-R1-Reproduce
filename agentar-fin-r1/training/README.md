# Agentar-Fin-R1 — verl 训练（SFT + GRPO）

本目录是 **verl 0.8.0** 版的两阶段训练实现。原 ms-swift 版（SFT/GRPO）已删除，仅保留 verl。
决策背景见根 `report.md` 与 `../../.workbuddy/memory/`：ms-swift 在 GRPO 阶段
rollout 用 `transformers.generate`、reward 同步阻塞，verl 用 vLLM rollout + Ray 式
流水线，对本项目（Qwen3.5-9B + K=8 + 外部 DeepSeek V4 Flash 裁判）是「代差级」提速。

## 目录结构

```
training/
├── README.md
├── sft/
│   ├── train_sft.py              # Stage 1 超参持有 + 训练启动器（调用 verl.trainer.sft_trainer）
│   ├── train_sft.sh              # Stage 1 两阶段壳（预处理 → 训练）
│   └── prepare_sft_data.py       # 原始对话 JSON/JSONL → verl parquet
├── grpo/
│   ├── train_grpo.sh             # Stage 2 两阶段壳（预处理 → GRPO 训练）
│   ├── prepare_grpo_data.py       # 原始对话 JSON/JSONL → verl GRPO parquet
│   └── fin_judge_reward.py        # 奖励：compute_score（格式闸门 → RLAIF 按 rubric 加权打分 0~1）
├── merge_lora.py                 # 合并 SFT LoRA → 完整 checkpoint
```

## 运行顺序

```bash
# 1) Stage 1 SFT（一键：原始数据 → 预处理 → 训练）
RAW_DATA=./data/raw/train.json bash training/sft/train_sft.sh
# 或已有 parquet，跳过预处理：
#   SFT_DATA=./data/verl/sft.parquet bash training/sft/train_sft.sh
# 产物 → ./training/sft/outputs

# 2) 合并 SFT LoRA
python training/merge_lora.py \
    --base ./Qwen3.5-9B \
    --adapter ./training/sft/outputs \
    --output ./training/sft/merged

# 3) 配置裁判（外部 DeepSeek V4 Flash API，OpenAI 兼容 /v1）
#    export JUDGE_API_KEY=<你的 DeepSeek API key>

# 4) Stage 2 GRPO（一键：原始数据 → 预处理 → 训练）
RAW_DATA=./data/raw/train.json bash training/grpo/train_grpo.sh
# 或已有 parquet，跳过预处理：
#   GRPO_DATA=./data/verl/grpo.parquet bash training/grpo/train_grpo.sh
# 产物 → ./training/grpo/outputs
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
| judge 并发 | `ThreadPoolExecutor`（同文件） | verl 逐样本调用 `compute_score`；RLAIF rubric 打分逐样本串行（原型足够，先跑通） |

## reward 实现要点（fin_judge_reward.py）

简洁版：只暴露一个函数 `compute_score(data_source, solution_str, ground_truth,
extra_info)`，由 verl 逐样本调用（见 `train_grpo.sh` 的 `custom_reward_function`
接线）。**没有 RewardManager 子类、没有 ThreadPool、没有单例客户端**——逻辑是单条
自上而下的 `if/return`，避免奖励里的回调嵌套。

**RLAIF + rubric 奖励**（对齐论文「intricate reward structures / verifiable rewards」，
把奖励信号从规则比对改为 **RLAIF：由外部金融裁判模型按固定 rubric 打分**）：

1. **格式闸门**：响应必须含 `<think>…</think>` 与（`\boxed{}` 或 `<answer>…</answer>`）。
   缺任一标签 → 直接 0 分（论文强调「verifiable / auditable」输出）。
2. **RLAIF rubric 打分**：对格式合格的样本，外部 DeepSeek V4 Flash 裁判按 4 维量规各打 0~10 分，
   代码按权重聚合为 0~1 的 reward：
   - `correctness` 正确性 0.35 — 结论与参考答案一致、事实/计算准确
   - `reasoning` 推理严谨性 0.30 — 推理链完整、逻辑自洽、无原则性错误
   - `compliance_risk` 合规风险意识 0.20 — 提示风险/合规约束、避免误导
   - `clarity_format` 表达与结构 0.15 — 清晰结构化、符合格式约定
   参考标准答案（与可选原题）一并喂给裁判，供 `correctness` 维度比对。
   裁判输出优先解析 JSON `{"dimensions":{...}}`，退化时正则抓 `key: num`；任何异常 → 0（不污染训练）。

> `verifiable` 字段不再决定走规则还是裁判，仅作元信息；RLAIF 对所有格式合格样本统一打分，
> 比单一 0/1 规则更平滑、可解释。

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

1. **dtype**：训练脚本已统一为 **bf16**（`train_sft.py` 的 `model.torch_dtype`、
   `train_grpo.sh` 的 `actor/ref fsdp_config.param_dtype` 与 `rollout.dtype` 均为
   `bfloat16`）。A800（Ampere sm_80）对 bf16 原生支持，比 fp16 数值更稳、无需 loss-scale。
2. **merge 兼容性**：`merge_lora.py` 假设 verl SFT LoRA 输出是 peft 兼容格式。
   若你的 verl 版本把 LoRA 存成非 peft 格式，改用「SFT 全参 + GRPO 自带 LoRA」
   路径，跳过 merge。
3. **显存**：9B + LoRA + K=8 + seq 4096 在 24G 卡上偏紧。`rollout.gpu_memory_utilization=0.5`
   可下调；多卡把 `rollout.tensor_model_parallel_size` / `NPROC` 调大。
4. **裁判服务**：`fin_judge_reward.py` 走 **外部 DeepSeek V4 Flash API**（OpenAI 兼容 /v1），
   通过环境变量注入：`JUDGE_BASE_URL`（默认 `https://api.deepseek.com/v1`）、
   `JUDGE_MODEL`（默认 `deepseek-v4-flash`）、`JUDGE_API_KEY`（必填）、`JUDGE_TIMEOUT`。
   无需本地部署，所有格式合格的样本都会走裁判（RLAIF 统一打分）；串行调用，吞吐受 API RPS 限制。
   `extra_info` 中若含 `question`/`prompt` 字段会作为原题上下文一并喂给裁判。

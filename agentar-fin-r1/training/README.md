# Agentar-Fin-R1 — verl 训练（SFT + GRPO）

本目录是 **verl 0.8.0** 版的两阶段训练实现。原 ms-swift 版（SFT/GRPO）已删除，仅保留 verl。
决策背景见根 `report.md`：ms-swift 在 GRPO 阶段 rollout 用 `transformers.generate`、
reward 同步阻塞，verl 用 vLLM rollout + Ray 式流水线，对本项目（Qwen3-8B + K=8）「代差级」提速。

## 目录结构

```
training/
├── README.md
├── merge_lora.py                 # 合并 SFT LoRA → 完整 checkpoint
├── sft/
│   ├── train_sft.py              # Stage 1 超参持有 + 训练启动器
│   ├── train_sft.sh              # Stage 1 两阶段壳（预处理 → 训练）
│   ├── prepare_sft_data.py       # 原始对话 JSON/JSONL → verl parquet
│   ├── sft_loss_curve.png        # SFT 训练 loss 曲线
│   └── pyproject.toml            # 环境依赖
└── grpo/
    ├── train_grpo.sh             # Stage 2 两阶段壳（预处理 → GRPO 训练）
    ├── prepare_grpo_data.py       # 原始对话 JSON/JSONL → verl GRPO parquet
    └── fin_judge_reward.py        # 奖励函数：格式闸门 → RLAIF rubric 打分
```

## 运行顺序

```bash
# 1) Stage 1 SFT（一键：原始数据 → 预处理 → 训练）
RAW_DATA=./data/raw/train.json bash training/sft/train_sft.sh
# 产物 → ./training/sft/outputs

# 2) 合并 SFT LoRA
python training/merge_lora.py \
    --base ./Qwen3-8B \
    --adapter ./training/sft/outputs \
    --output ./training/sft/merged

# 3) 配置裁判 API
#    export JUDGE_API_KEY=<your key>

# 4) Stage 2 GRPO（一键：原始数据 → 预处理 → 训练）
RAW_DATA=./data/raw/train.json bash training/grpo/train_grpo.sh
# 产物 → ./training/grpo/outputs
```

## SFT 训练结果（Qwen3-8B + LoRA，DeepFinance-100K）

训练配置：8×A800, batch=16, lr=1e-4, warmup 3%, cosine, 2 epochs

![SFT Loss Curve](sft/sft_loss_curve.png)

- 最终 loss: 0.5014，相比起始 0.6692 下降约 25%
- 3000~3500 步有一次陡降，随后稳定收敛在 0.50~0.51

## reward 实现（fin_judge_reward.py）

格式闸门只检查 `<think>…</think>` 存在。通过后由外部 DeepSeek V4 Flash 按 4 维 rubric 打分：

| 维度 | 权重 | 对标内容 |
|------|------|----------|
| correctness | 0.40 | 答案 vs gold_output |
| reasoning | 0.30 | 推理链 vs gold_thinking |
| compliance_risk | 0.15 | 风险/合规意识 |
| clarity_format | 0.15 | 表达结构 |

推理和答案分开对标：裁判先看 thinking 打 reasoning 分，再看 output 打其余三维度分。

## 注意事项

1. **显存**：8×A800 80G，SFT 每卡 ~30-40GB，GRPO（actor+vLLM+ref offload）~60GB。
2. **裁判**：需导出 `JUDGE_API_KEY`，走 DeepSeek API。串行调用，吞吐受 API RPS 限制。
3. **dtype**：统一 bf16，A800 原生支持。
4. **merge**：`merge_lora.py` 假设 verl SFT 输出为 peft 兼容格式。若非 peft 格式，跳过 merge，SFT 全参 + GRPO 自带 LoRA。

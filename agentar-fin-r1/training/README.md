# 训练复刻：加权训练 + 两阶段 + 归因闭环

复刻 Agentar-Fin-R1 的**训练框架**，对应论文 §2.4–2.6。

## 模块

| 模块 | 论文环节 | 内容 |
| --- | --- | --- |
| [`weighting.py`](src/finr1_training/weighting.py) | 难度感知加权 | pass@k 估计 + 难度权重 + 指数平滑/下限裁剪 |
| [`stage1_sft.py`](src/finr1_training/stage1_sft.py) | Stage 1 | 大规模 SFT + 加权训练（知识注入） |
| [`stage2_grpo.py`](src/finr1_training/stage2_grpo.py) | Stage 2 | GRPO + 针对性 SFT（难题攻坚） |
| [`attribution.py`](src/finr1_training/attribution.py) | 归因闭环 | 写 attribution.json，驱动数据回滚/再生 |
| [`models/`](src/finr1_training/models/) | 基座/适配器 | Qwen3.5-9B + LoRA/QLoRA 配置 |

## 运行（小规模原型）

```bash
# Stage 1：知识注入（Qwen3.5-9B + QLoRA，单/双卡）
uv run python -m finr1_training.scripts.train_sft --max-financial 5000

# Stage 2：GRPO 难题攻坚（group_size=4，可在配置/CLI 改）
uv run python -m finr1_training.scripts.train_grpo \
    --hard-subset data/golden/hard_subset.jsonl \
    --output-dir checkpoints/stage2-grpo --max-samples 50

# 从 Stage-1 适配器起训 + 自定义 group 大小
uv run python -m finr1_training.scripts.train_grpo \
    --stage1-adapter checkpoints/stage1-sft --group-size 4
```

### Stage 2 实现要点（论文 §3.2 / §3.3）

- **GRPO（标准 group-relative PPO）**：每个 prompt 采样 `group_size=4` 条 rollout，
  按 reward 计算组相对优势 `A_i = (r_i − mean) / std`，优化裁剪代理目标
  `L = −E[min(ρA, clip(ρ,1−ε,1+ε)A)] + β·KL(π_θ ‖ π_ref)`。β 默认 0.04，ε=0.2。
- **多目标 reward（论文"intricate reward structures"）**：
  `reward = w_correct·正确性 + w_format·格式 + w_length·长度惩罚`。
  正确性用验证器式匹配（数值容差/子串）对齐 §2.3.3 的校验信号；格式奖励 `<think>…</think>` 结构。
- **Targeted SFT 回退**：论文说"GRPO 在子类收敛差时切针对性 SFT"。代码内置停滞监控——
  连续 `stall_patience` 步 reward 无提升，就用该 prompt 最差的 rollout 做几步监督微调救场。
- **KL 参考策略**：冻结的基座模型（无 LoRA），每批量只前向一次缓存 ref log-prob。

## 范围说明

当前为**小规模原型**：基座 Qwen3.5-9B + QLoRA/LoRA，先把"三级数据流水线 + 两阶段训练闭环"
端到端跑通。Stage 1/Stage 2 已实现可运行骨架（SFT 加权、GRPO group=4），后续可扩展到多卡全参 / 32B。

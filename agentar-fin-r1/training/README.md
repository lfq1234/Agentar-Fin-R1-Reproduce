# 训练复刻：加权训练 + 两阶段 + 归因闭环

复刻 Agentar-Fin-R1 的**训练框架**，对应论文 §2.4–2.6。

## 模块

| 模块 | 论文环节 | 内容 |
| --- | --- | --- |
| [`weighting.py`](src/finr1_training/weighting.py) | 难度感知加权 | pass@k 估计 + 难度权重 + 指数平滑/下限裁剪 |
| [`stage1_sft.py`](src/finr1_training/stage1_sft.py) | Stage 1 | 大规模 SFT + 加权训练（知识注入） |
| [`stage2_grpo.py`](src/finr1_training/stage2_grpo.py) | Stage 2 | GRPO + 针对性 SFT（难题攻坚） |
| [`attribution.py`](src/finr1_training/attribution.py) | 归因闭环 | 写 attribution.json，驱动数据回滚/再生 |
| [`models/`](src/finr1_training/models/) | 基座/适配器 | Qwen3-8B + LoRA/QLoRA 配置 |

## 运行（小规模原型）

```bash
# Stage 1：知识注入（Qwen3-8B + QLoRA，单/双卡）
uv run python -m finr1_training.stage1_sft --data-dir ../data/data/golden --output-dir checkpoints/s1

# Stage 2：难题攻坚
uv run python -m finr1_training.stage2_grpo --hard-subset data/hard --output-dir checkpoints/s2
```

## 范围说明

当前为**小规模原型**：基座 Qwen3-8B + QLoRA/LoRA，先把"三级数据流水线 + 两阶段训练闭环"
端到端跑通。Stage 函数目前为占位实现，待按论文逐层落地。后续可扩展到多卡全参 / 32B。

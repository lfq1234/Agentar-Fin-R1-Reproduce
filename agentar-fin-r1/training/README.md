# 训练复刻：加权训练 + 两阶段 + 归因闭环

复刻 Agentar-Fin-R1 的**训练框架**，对应论文 §2.4–2.6。

## 技术文档（必读）

| 文档 | 说明 |
| --- | --- |
| [model/model.md](model/model.md) | 模型加载层：`ModelConfig` 精度开关（fp16/bf16/int4）+ LoRA |
| [sft/sft.md](sft/sft.md) | Stage 1 SFT：知识注入 + 难度加权（论文 §3.1/§3.2） |
| [grpo/grpo.md](grpo/grpo.md) | Stage 2 GRPO：难题攻坚 + 定向 SFT 回退（论文 §3.3） |

## 模块

| 模块 | 论文环节 | 内容 |
| --- | --- | --- |
| [`model/`](model/) | 基座/适配器 | Qwen3.5-9B 加载，**精度由 `precision` 统一控制**（默认 fp16） |
| [`sft/weighting.py`](sft/weighting.py) | 难度感知加权 | pass@k 估计 + 难度权重 + 指数平滑/下限裁剪 |
| [`sft/__init__.py`](sft/__init__.py) | Stage 1 | 大规模 SFT + 加权训练（知识注入） |
| [`grpo/__init__.py`](grpo/__init__.py) | Stage 2 | GRPO + 针对性 SFT（难题攻坚） |
| [`grpo/attribution.py`](grpo/attribution.py) | 归因闭环 | 写 attribution.json，驱动数据回滚/再生 |

## 运行（小规模原型）

```bash
# Stage 1：知识注入（Qwen3.5-9B，fp16 + LoRA）
bash run_sft.sh
# 或小规模试跑
bash run_sft.sh --max-financial 50

# Stage 2：GRPO 难题攻坚（fp16，每次 rollout 8 条）
bash run_grpo.sh --hard-subset data/golden/hard_subset.jsonl \
                 --stage1-adapter checkpoints/stage1 --max-samples 50
# 或
python -m grpo \
    --hard-subset data/golden/hard_subset.jsonl \
    --stage1-adapter checkpoints/stage1 --max-samples 50
```

### 精度与 rollout（当前默认）
- **精度**：两阶段统一默认 `precision: "fp16"`（全精度 fp16 + LoRA，不量化）。
  改精度只需在 `sft/config.yaml` / `grpo/config.yaml` 的 `model.precision` 一处切换
  （`fp16` / `bf16` / `int4` 旧版 QLoRA）。
- **GRPO rollout**：`group_size = 8`（论文要求），定义在 `grpo/config.yaml` 与
  `GRPOConfig.group_size` 默认。

## 范围说明

当前为**小规模原型**：基座 Qwen3.5-9B + fp16/LoRA，先把"三级数据流水线 + 两阶段训练闭环"
端到端跑通。Stage 1/Stage 2 已实现可运行骨架（SFT 加权、GRPO group=8），后续可扩展到多卡全参 / 32B。

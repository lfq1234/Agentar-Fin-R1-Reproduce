# 训练复刻：两阶段（SFT + GRPO）

复刻 Agentar-Fin-R1 的训练框架（论文 §2.4–2.6）。

两段训练用**不同的库**，职责清晰、不混在一起：

| 阶段 | 实现方式 | 库 |
|---|---|---|
| **Stage 1 · SFT** | 知识注入 + 难度加权（Eq.16） | `peft` + `LoRA` + `trl.SFTTrainer`（掉包式） |
| **Stage 2 · GRPO** | 难题攻坚（组相对优势 + KL） | `verl` + `LoRA` |

## 模块结构

```
training/
├── README.md
├── pyproject.toml          # 依赖：peft/trl（SFT） + verl/vllm（GRPO）
├── run_sft.sh              # 启动 Stage 1
├── run_grpo.sh             # 启动 Stage 2
├── model/                  # 共享：基座加载 + LoRA（peft）  ← Stage 1 用
│   ├── __init__.py
│   └── model.md
├── sft/                    # Stage 1：peft + LoRA 掉包实现
│   ├── __init__.py         # train_stage1 + WeightedSFTTrainer（Eq.16 权重）
│   ├── data.py             # DeepFinance-100K → chat 格式数据集
│   ├── weighting.py        # 难度加权（complexity / heuristic）
│   ├── config.yaml
│   └── sft.md
└── grpo/                   # Stage 2：verl + LoRA
    ├── __init__.py         # 转 parquet + 生成 verl 覆盖参数 + 启动 main_ppo
    ├── reward.py           # verl 奖励函数 compute_score（多目标）
    ├── data.py             # hard_subset.jsonl → verl parquet
    ├── attribution.py      # 归因闭环
    ├── config.yaml
    └── grpo.md
```

> `model/` 是 Stage 1（SFT + LoRA）共用的模型加载层。Stage 2 的模型与 LoRA
> 由 `verl` 自己在内部管理（通过 `config.yaml` 的 `lora.*` 与 `model.name` 配置）。

## 运行

```bash
# Stage 1：知识注入（Qwen3.5-9B，fp16 + LoRA）
bash run_sft.sh
bash run_sft.sh --max-financial 50     # 小规模试跑

# Stage 2：GRPO 难题攻坚（verl，每 prompt rollout 8 条，LoRA）
bash run_grpo.sh --hard-subset data/golden/hard_subset.jsonl \
                 --stage1-adapter checkpoints/stage1 --max-samples 50
```

### 精度与 rollout（默认）
- **精度**：Stage 1 默认 `precision: "fp16"`（全精度 fp16 + LoRA，不量化），
  由 `model.precision` 统一控制（`sft/config.yaml`）。Stage 2 由 verl 管理 dtype。
- **GRPO rollout**：`group_size = 8`（论文要求），定义在 `grpo/config.yaml`，
  映射到 verl 的 `actor_rollout_ref.rollout.n=8`。

## 技术文档

| 文档 | 说明 |
|---|---|
| [model/model.md](model/model.md) | 模型加载层：精度开关（fp16/bf16/int4）+ LoRA |
| [sft/sft.md](sft/sft.md) | Stage 1：peft+LoRA 掉包实现 + 难度加权（§3.1/§3.2） |
| [grpo/grpo.md](grpo/grpo.md) | Stage 2：verl+LoRA GRPO（§3.3） |

## 范围说明

当前为**小规模原型骨架**：先把"两阶段训练闭环"端到端写通。
Stage 1 已是可运行的 peft+lora SFT（含难度加权）；Stage 2 已切换为 verl 原生的
GRPO + LoRA（不再手写 GRPO 数学）。实际训练需要 GPU + 权重 + 装依赖（见 `pyproject.toml`）。

# Stage 1：SFT 技术方案（sft/）

> 两阶段训练的**第一阶段：金融知识与能力注入**。
> 论文 §3.1（难度感知加权）+ §3.2 Stage 1（大规模 SFT + 加权训练）。
> 本阶段产物：一个注入了金融推理能力的 LoRA adapter（或 merge 后的 checkpoint），作为 `../grpo/` 的初始策略。

---

## 1. 阶段目标

| 维度 | 目标 |
| --- | --- |
| 能力 | 把通用 Qwen3-8B 变成"会做金融 CoT 推理"的模型：能按 `thinking → answer` 的方式作答 |
| 数据 | HF `antgroup/Agentar-DeepFinance-100K`（**messages 数组格式**）+ 可选通用推理语料；ASSISTANT.content 自带 thinking 边界 |
| 方法 | **PEFT + LoRA**（参数高效），冻结基座，仅训练少量 adapter 参数 |
| 衔接 | 产出 `sft_lora_adapter/` 与 `sft_merged/`（可选 merge），交给 Stage 2 GRPO |

---

## 2. 数据来源

### 2.1 数据来源

| 来源 | 路径 / HF id | 规模 | 用途 |
| --- | --- | --- | --- |
| **DeepFinance-100K** | `antgroup/Agentar-DeepFinance-100K` | ~100K | 主语料，CoT 推理金标（messages 数组格式） |
| **通用推理（可选）** | `nvidia/Llama-Nemotron-Post-Training-Dataset` 或 `open-r1/OpenR1-Math` 子集 | ~20K | 防止金融过拟合，保通用推理能力（论文 §4.2 训练数据构成） |

> DeepFinance-100K 是论文训练数据的核心开源部分（主论文 §4.2 明确包含），本复现**直接用作 SFT 主语料**。该数据集每条即为 messages 数组格式（见 §2.2），无需任何字段转换，直接 `load_dataset` 读取。

### 2.2 数据格式规范（唯一）

**只支持 messages 数组格式**，即一个 JSON 数组，元素为带 `role` / `content` 的对象：

```json
[
  {
    "role": "HUMAN",
    "content": "Please answer the given financial question based on the context.\nContext: ...\nQuestion: ...\nAnswer:"
  },
  {
    "role": "ASSISTANT",
    "content": "<think>\n...thinking 推理过程...\n</think>\n\n最终回答"
  }
]
```

要点：

- **role 大小写容错**：`HUMAN` / `ASSISTANT` / `user` / `assistant` 都认（`format_messages` 里统一 `.upper()` 映射到 Qwen 标准角色 `user` / `assistant`）。
- **ASSISTANT.content 已内嵌 thinking 边界**（`<think>...</think>`），**原样保留，绝不重复包裹**——`format_messages` 只按 Qwen 模板补 `<|im_start|>` / `<|im_end|>` 边界（见 §3.2）。
- 数据集很干净，不做任何脏数据兜底；文件直接是 `{role, content}` 数组即可（jsonl 每行一个数组，或整文件 JSON 数组）。

---

## 3. 数据加载：ms-swift `swift sft`

> 直接复用 **ms-swift** 的 [`swift sft`](https://swift.readthedocs.io/en/latest/Instruction/SFT.html)，**不自己写 Dataset / collate_fn / 训练循环 / formatting_func / role_map**。
> 数据集就是标准 `messages` 数组（{role, content}），ms-swift 按 Qwen3 的 chat 模板自动拼装、tokenize、构造 labels（只训 assistant 段），全部交给框架。

### 3.1 设计总览

```
本地数据 (json / jsonl，每行一个 messages 数组)
        │  --dataset ./train_data.jsonl（ms-swift AutoPreprocessor 自动识别）
        ▼
swift sft（SwiftSftArguments）
        │  chat 模板拼装 + tokenize + labels（只训 assistant 段）全由框架处理
        ▼
训练（torch_dtype=float16, AMP）+ 保存 LoRA adapter
```

三个关键决策：

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| 数据读取 | `--dataset ./train_data.jsonl` | ms-swift 自动识别 jsonl 的 messages 数组，零样板 |
| 文本拼装 | 框架按模型模板处理 | 不再手写 `format_messages` / `role_map`，避免双重 `<think>` 包裹 bug |
| 预处理 | `swift sft` 内部完成 | tokenize / padding / prompt-mask / loss mask 全由框架处理 |

### 3.2 thinking 边界处理（不包裹）

数据里的 ASSISTANT.content **已经包含** `<think>...</think>` 边界（见 §2.2 示例）。交给 ms-swift 的 chat 模板后，这部分**原样进入文本**，框架不会重复包裹——彻底规避 TRL 时代手搓 `format_messages` 容易导致的双重 `<think><think>` 问题。

### 3.3 首次跑通必做的核对

如只想核对拼接效果（无需 GPU），可用 ms-swift 的 `get_template` 把一个样本 encode 后 decode 打印，确认 `<think>` 只出现一次、且 assistant 段的 `<think>...</think>` 完整保留。

---

## 4. 难度感知加权（论文 §3.1，设计目标，本脚本暂未实现）

论文 §3.1 的难度感知加权是设计目标，但当前 `train_sft.py` 走 ms-swift `swift sft` 的标准**等权** NLL，不自定义 `compute_loss`。

- 权重公式（`w̃_ℓ` 归一化难度权重）记录在 `report.md`，供后续接入参考。
- 若要启用：要么在 `train_sft.py` 里继承 `Trainer` 改写 `compute_loss`（对 per-sample loss 按标签 ℓ 的权重加权），要么在**数据层**对困难标签做上采样。首版先把等权管线跑通即可。
- 当前 messages 格式不含难度元数据，`weight` 统一为 `1.0`。
- `attribution.json`（归因闭环产物）可为 Stage 1 提供 pass@1 per label，驱动权重更新。

---

## 5. 模型加载（ms-swift，FP16）

> 模型加载与精度由 `swift sft` 统一管理，**统一 FP16**（`torch_dtype="float16"`）。框架内部用 transformers + accelerate 加载，AMP 自动混合精度，数值更稳。配置约定见 `../models/README.md`。

CLI 中只需给 `--model Qwen/Qwen3-8B --torch_dtype float16`，无需手写 `AutoModelForCausalLM.from_pretrained`。

显存档（Qwen3-8B）：

| 模式 | 基座占用 | 可训于 |
| --- | --- | --- |
| **FP16 + LoRA (r=64)** | **~16 GB + adapter 84MB** | **单张 24 GB（A5000/4090）** |
| GRPO（Stage 2, K=8） | policy+ref+rollout logits | 至少 24 GB，建议 40GB+（或降 K=4） |

> **FP16 稳定性**：ms-swift 走 `torch_dtype=float16` + AMP；出现 NaN 时降 lr / 加 warmup（详见 `../models/README.md` §4.2）。

---

## 6. LoRA 配置（ms-swift 参数）

### 6.1 LoRA 目标模块

Qwen3 是 GQA + SwiGLU。`--target_modules all-linear` 即覆盖全部线性层（含 `q/k/v/o_proj` 与 `gate/up/down_proj`），框架统一管理，无需手写 7 个模块名。

### 6.2 推荐配置（CLI 参数等价）

| 参数 | 值 | 理由 |
| --- | --- | --- |
| `--lora_rank` | 64 | 金融 CoT 需要一定容量；太小(r=8)欠拟合，太大(r=128)过拟合+显存涨 |
| `--lora_alpha` | 128 | `alpha=2r` 是 LoRA 缩放经验值 |
| `--lora_dropout` | 0.05 | 防过拟合，金融数据相对集中 |
| `--target_modules` | `all-linear` | 等价于手写全 7 个线性层 |

> FP16 + LoRA 模式下 lr 建议从 `1e-4` 起；若训练出现 NaN/梯度溢出，先降到 `5e-5` 并加大 `--warmup_steps`。

---

## 7. 训练超参与脚本结构

### 7.1 训练超参

| 超参 | 值 | 备注 |
| --- | --- | --- |
| optimizer | ms-swift 默认 AdamW | 无 QLoRA，标准 AdamW |
| learning_rate | `1e-4` | LoRA 典型区间 1e-4 ~ 2e-4 |
| num_train_epochs | `3` | 数据量 ~100K，3 epoch 够 |
| per_device_train_batch_size | `1` | seq=8K 显存敏感 |
| gradient_accumulation_steps | `16` | 等效 bs=16 |
| max_length | `8192` | SwiftSftArguments 的 `max_length` |
| weight_decay | `0.0` | LoRA 一般不加 |
| torch_dtype | `float16` | **AMP 自动混合精度**（FP16 加载基座 + 训练）；勿用纯 fp16 全程 |
| gradient_checkpointing | `True` | 省显存 |
| save_steps | `200` | 每 200 步存一次 |
| logging_steps | `10` | |
| report_to | `wandb` | 不需要可视化可改 `none` |

### 7.2 目录结构

```
sft/
├── README.md                  # 本文件
└── src/
    ├── train_sft.py           # 主训练入口（SwiftSftArguments + sft_main）
    └── merge_lora.py          # 训练后把 adapter merge 回基座（swift export / SwiftExportArguments）
```

### 7.3 train_sft.py 主流程

```python
from swift.llm import SwiftSftArguments, sft_main

args = SwiftSftArguments(
    model="Qwen/Qwen3-8B",
    dataset="./train_data.jsonl",          # 每行一个 messages 数组，框架自动拼装
    tuner_type="lora",
    lora_rank=64, lora_alpha=128, lora_dropout=0.05,
    target_modules="all-linear",
    torch_dtype="float16",
    learning_rate=1e-4, num_train_epochs=3,
    per_device_train_batch_size=1, gradient_accumulation_steps=16,
    max_length=8192, gradient_checkpointing=True,
    save_steps=200, logging_steps=10,
    output_dir="./outputs/sft_lora_adapter",
)
sft_main(args)                              # 只存 LoRA adapter
```

### 7.4 merge_lora.py

```python
from swift.llm import SwiftExportArguments, export_main

args = SwiftExportArguments(
    model="Qwen/Qwen3-8B",
    adapters="./outputs/sft_lora_adapter",
    torch_dtype="float16",
    output_dir="./outputs/sft_merged",      # 完整 checkpoint
)
export_main(args)
# sft_merged/ 即 Stage 2 GRPO 的初始策略
```

---

## 8. 评估与 checkpoint

- **训练中**：每 1000 步在 5% holdout 上算 NLL，监控是否发散/过拟合。
- **训练后**：在 Finova 子集（`training/src/finr1_training/eval/finova/`，待建）+ MATH-500 / GPQA-diamond 上评测 pass@1，作为 Stage 2 难题挑选的依据。
- **产出**：
  - `outputs/sft_lora_adapter/`（adapter 权重 + adapter_config.json）
  - `outputs/sft_merged/`（可选 merge 后的完整 checkpoint）
  - `outputs/sft_metrics.json`（per-label pass@1，喂给归因闭环）

---

## 9. 命令示例

```bash
cd agentar-fin-r1/training
pip install -e .                      # 安装依赖（torch/ms-swift/transformers/datasets/wandb/openai）

# 1. SFT 训练（推荐 CLI，配置全在命令行）
swift sft \
    --model Qwen/Qwen3-8B \
    --dataset ./train_data.jsonl \
    --tuner_type lora \
    --lora_rank 64 --lora_alpha 128 --lora_dropout 0.05 \
    --target_modules all-linear \
    --torch_dtype float16 \
    --learning_rate 1e-4 --num_train_epochs 3 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
    --max_length 8192 --gradient_checkpointing true \
    --save_steps 200 --logging_steps 10 \
    --output_dir ./outputs/sft_lora_adapter \
    --report_to wandb

# 或 Python API：python sft/src/train_sft.py

# 2. merge adapter → 完整 checkpoint（供 GRPO 用）
swift export \
    --model Qwen/Qwen3-8B \
    --adapters ./outputs/sft_lora_adapter \
    --torch_dtype float16 \
    --output_dir ./outputs/sft_merged

# 或 Python API：python sft/src/merge_lora.py
```

---

## 10. 风险与注意

1. **thinking 边界对齐**：数据里 ASSISTANT.content 自带 `<think>...</think>`，交给 ms-swift chat 模板后原样进入文本、不重复包裹；首次跑通前用 `get_template` encode 后 decode 确认 `<think>` 只出现一次。
2. **难度加权暂未启用**：当前走 swift sft 标准等权 NLL；论文的难度感知加权（§4）需在接入 pass@k 评估后另行实现。
3. **数据泄露**：评测用 Finova / MATH-500 必须从训练集去污（`data/` 的 Verification 阶段已做，但合并外部语料后需复查）。
4. **过拟合**：3 epoch 在 100K 上可能过拟合，监控 holdout loss，若第 2 epoch 后上升则早停。
5. **merge 不可逆**：`merge_and_unload` 后 adapter 无法拆回，务必先保留 `sft_lora_adapter/` 原始副本。
6. **FP16 数值稳定性**：基座统一 `torch_dtype=float16` 加载，训练走 AMP；偶发 NaN 时降 lr / 加 warmup（详见 `../models/README.md` §4.2、§9）。
7. **多卡 DDP**：用 `NPROC_PER_NODE=8 swift sft ...` 启动分布式，框架自动接管，无需手动 `device_map`。

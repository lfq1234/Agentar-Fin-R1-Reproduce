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

## 3. 数据加载：TRL SFTTrainer + load_dataset + formatting_func

> 直接复用 TRL 的 [`SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer)，**不自己写 Dataset / collate_fn / 训练循环**。
> 数据用 `datasets.load_dataset` 读，每条样本经一个 `formatting_func` 拼成 chat 文本即可——这正是参考脚本的做法。

### 3.1 设计总览

```
本地数据 (json / jsonl，每行一个 messages 数组)
        │  load_dataset("json", data_files=...)
        ▼
SFTTrainer
        │  formatting_func(sample)：messages 数组 → 单条 chat 文本
        │  SFTTrainer 内部：tokenize + packing/padding + 构造 labels（只训 assistant 段）
        ▼
训练（fp16=True, AMP）+ 保存 LoRA adapter
```

三个关键决策：

| 决策 | 选择 | 理由 |
| --- | --- | --- |
| 数据读取 | `datasets.load_dataset("json", ...)` | 标准、零样板；jsonl 逐行数组直接读 |
| 文本拼装 | `formatting_func` | 把 `{role, content}` 数组按 Qwen 模板拼成 `<|im_start|>...<\|im_end|>` 文本 |
| 预处理 | SFTTrainer 内部完成 | tokenize / padding / prompt-mask 全由 TRL 处理，不重复造轮子 |

### 3.2 thinking 边界处理（不包裹）

数据里的 ASSISTANT.content **已经包含** `<think>...</think>` 边界（见 §2.2 示例）。`format_messages` **不做任何包裹**，只按 Qwen 模板补 `<|im_start|>` / `<|im_end|>` 边界：

```python
def format_messages(sample):
    msgs = sample["messages"] if isinstance(sample, dict) else sample
    parts = []
    for m in msgs:
        role = ROLE_MAP.get(m["role"].upper(), m["role"].lower())  # HUMAN→user 等
        parts.append(f"<|im_start|>{role}\n{m['content']}<|im_end|>\n")
    return "".join(parts)
```

即 thinking 边界原样传给 tokenizer，避免最常见的 bug——双重 `<think><think>` 包裹导致模型学错边界。

### 3.3 首次跑通必做的核对

`SFTTrainer` 默认对整个文本算 loss（含 user 段）。如只想核对拼接效果，可在本地试跑一次 `format_messages` 并打印 decode 结果，确认 `<think>` 边界只出现一次、且 assistant 段的 `<think>...</think>` 完整保留。

---

## 4. 难度感知加权（论文 §3.1，设计目标，本脚本暂未实现）

论文 §3.1 的难度感知加权是设计目标，但当前 `train_sft.py` 走 TRL `SFTTrainer` 的标准**等权** NLL，不自定义 `compute_loss`。

- 权重公式（`w̃_ℓ` 归一化难度权重）记录在 `report.md`，供后续接入参考。
- 若要启用：要么在 `train_sft.py` 里继承 `Trainer` 改写 `compute_loss`（对 per-sample loss 按标签 ℓ 的权重加权），要么在**数据层**对困难标签做上采样。首版先把等权管线跑通即可。
- 当前 messages 格式不含难度元数据，`weight` 统一为 `1.0`。
- `attribution.json`（归因闭环产物）可为 Stage 1 提供 pass@1 per label，驱动权重更新。

---

## 5. 模型加载（transformers，FP16）

> 用 `transformers` 原生 API 加载，**统一 FP16**（`torch_dtype=torch.float16`）。训练阶段在 Trainer 设 `fp16=True`（AMP 自动混合精度），数值更稳。配置约定见 `../models/README.md`。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token      # Qwen3 默认无 pad，借 eos
tokenizer.padding_side = "right"               # SFT 标签对齐

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,                 # 统一 FP16 加载
    device_map="auto",                         # 多卡 DDP 训练时去掉本行，交给 Trainer
    trust_remote_code=True,
)
```

显存档（Qwen3-8B）：

| 模式 | 基座占用 | 可训于 |
| --- | --- | --- |
| **FP16 + LoRA (r=64)** | **~16 GB + adapter 84MB** | **单张 24 GB（A5000/4090）** |
| GRPO（Stage 2, K=8） | policy+ref+rollout logits | 至少 24 GB，建议 40GB+（或降 K=4） |

> **FP16 稳定性**：开 `fp16=True`(AMP) 而非纯 fp16；出现 NaN 时降 lr / 加 warmup（详见 `../models/README.md` §4.2）。

---

## 6. PEFT + LoRA 配置

### 6.1 LoRA 目标模块

Qwen3 是 GQA + SwiGLU，可挂 LoRA 的线性层：`q_proj / k_proj / v_proj / o_proj / gate_proj / up_proj / down_proj`。

### 6.2 推荐配置

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,                 # 经验上 alpha = 2*r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # 预期 ~42M / 8B ≈ 0.5%
```

| 参数 | 值 | 理由 |
| --- | --- | --- |
| `r` | 64 | 金融 CoT 需要一定容量；太小(r=8)欠拟合，太大(r=128)过拟合+显存涨 |
| `alpha` | 128 | `alpha=2r` 是 LoRA 缩放经验值 |
| `dropout` | 0.05 | 防过拟合，金融数据相对集中 |
| `target_modules` | 全 7 个 | 只挂 attention 会丢失 FFN 上的知识注入，金融任务需全挂 |

> FP16 + LoRA 模式下 lr 建议从 `1e-4` 起；若训练出现 NaN/梯度溢出，先降到 `5e-5` 并加大 `warmup_steps`，必要时把 `fp16=True` 的 loss scaling 交给 AMP 自动处理（默认即可）。

---

## 7. 训练超参与脚本结构

### 7.1 训练超参

| 超参 | 值 | 备注 |
| --- | --- | --- |
| optimizer | `adamw_torch` | 无 QLoRA，标准 AdamW |
| learning_rate | `1e-4` | LoRA 典型区间 1e-4 ~ 2e-4 |
| num_train_epochs | `3` | 数据量 ~100K，3 epoch 够 |
| per_device_train_batch_size | `1` | seq=8K 显存敏感 |
| gradient_accumulation_steps | `16` | 等效 bs=16 |
| max_seq_length | `8192` | SFTTrainer 的 `max_seq_length` |
| weight_decay | `0.0` | LoRA 一般不加 |
| fp16 | `True` | **AMP 自动混合精度**（FP16 加载基座 + fp16=True 训练）；勿用纯 fp16 全程 |
| bf16 | — | 不用，见上一行 |
| gradient_checkpointing | `True` | 省显存 |
| save_strategy | `steps` | 每 200 步存一次 |
| logging_steps | `10` | |
| report_to | `wandb` | 不需要可视化可改 `none` 并去掉 `import wandb` |

### 7.2 目录结构

```
sft/
├── README.md                  # 本文件
└── src/
    ├── train_sft.py           # 主训练入口（transformers 加载 + peft lora + TRL SFTTrainer）
    └── merge_lora.py          # 训练后把 adapter merge 回基座，产 sft_merged/
```

### 7.3 train_sft.py 主流程

```python
# 1. 数据集（datasets 标准读法；每行一个 messages 数组）
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# 2. tokenizer + 模型（FP16）
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model = get_peft_model(model, LORA_CONFIG)

# 3. SFTTrainer（formatting_func 负责拼 prompt，其余交给 TRL）
trainer = SFTTrainer(
    model=model, train_dataset=dataset,
    formatting_func=format_messages, max_seq_length=8192,
    tokenizer=tokenizer, args=TRAINING_ARGS,   # TRAINING_ARGS.fp16=True
)
trainer.train()
trainer.save_model(OUTPUT_DIR)                 # 只存 LoRA adapter
```

### 7.4 merge_lora.py

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base, OUTPUT_DIR)
model = model.merge_and_unload()
model.save_pretrained(OUTPUT_DIR / "sft_merged", safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.save_pretrained(OUTPUT_DIR / "sft_merged")
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

# 1. 直接用脚本默认配置跑（改 train_sft.py 顶部“配置区”切换模型/数据/超参）
pip install -e .                      # 安装依赖（torch/transformers/peft/trl/datasets/wandb）
python sft/src/train_sft.py

# 2. 训练前先核对 format_messages 拼接效果（无需 GPU 即可跑）
python -c "import sys; sys.path.insert(0,'sft/src'); from train_sft import format_messages; \
import json; print(format_messages(json.load(open('train_data.jsonl'))))"

# 3. merge adapter → 完整 checkpoint（供 GRPO 用）
python sft/src/merge_lora.py \
    --base models/Qwen3-8B \
    --adapter outputs/sft_lora_adapter \
    --out outputs/sft_merged
```

---

## 10. 风险与注意

1. **thinking 边界对齐**：数据里 ASSISTANT.content 自带 `<think>...</think>`，`format_messages` 只补 chat 边界、不包裹 thinking；首次跑通前用 §9 第 2 条命令确认 `<think>` 只出现一次、未被双重包裹。
2. **难度加权暂未启用**：当前走 SFTTrainer 标准等权 NLL；论文的难度感知加权（§4）需在接入 pass@k 评估后另行实现。
3. **数据泄露**：评测用 Finova / MATH-500 必须从训练集去污（`data/` 的 Verification 阶段已做，但合并外部语料后需复查）。
4. **过拟合**：3 epoch 在 100K 上可能过拟合，监控 holdout loss，若第 2 epoch 后上升则早停。
5. **merge 不可逆**：`merge_and_unload` 后 adapter 无法拆回，务必先保留 `sft_lora_adapter/` 原始副本。
6. **FP16 数值稳定性**：基座统一 fp16 加载，训练务必开 `fp16=True`(AMP)；偶发 NaN 时降 lr / 加 warmup（详见 `../models/README.md` §4.2、§9）。
7. **多卡 DDP**：`train_sft.py` 里 `device_map="auto"` 仅适合单卡；多卡分布式训练去掉该行，由 `accelerate launch` 接管。

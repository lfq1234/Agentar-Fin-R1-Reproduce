# Stage 1 SFT 技术文档（`sft/`）

## 1. 实现方式：peft + LoRA 掉包式

Stage 1 是**完全基于库的"掉包"实现**，不手写训练循环：

| 职责 | 用的库 | 落点 |
|---|---|---|
| LoRA 适配器 | `peft`（`get_peft_model` + `LoraConfig`） | 由 `model/` 统一提供 |
| 基座加载 / 精度 | `transformers` + `model/` | `model.load_model` / `apply_lora` |
| 监督微调主循环 | `trl.SFTTrainer` | `sft/__init__.py` |

唯一一处"自写"是一个极薄的子类 `WeightedSFTTrainer(SFTTrainer)`，
只把论文 Eq.16 的难度权重乘进 per-sample loss——没碰优化器、没碰反向。

```
model/  (共享)        sft/data.py          sft/weighting.py        sft/__init__.py
┌──────────┐  ┌────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│load_model│→ │prepare_financial_  │→ │complexity / heuristic │→ │WeightedSFTTrainer    │
│apply_lora│  │data (DeepFinance)  │  │  难度权重            │  │(SFTTrainer + w̃)     │
└──────────┘  │apply_chat_template │  └──────────────────────┘  │train_stage1()        │
              └────────────────────┘                             └──────────────────────┘
```

---

## 2. 目标（论文 §3.2）

**金融知识与能力注入**：用「金融推理 CoT 数据 ∪ 通用推理数据」监督微调基座，打知识底子。

```
D_stage1 = D_fin  ∪  D_general
```

- `D_fin`：**`antgroup/Agentar-DeepFinance-100K`**（论文 §4.2 列为训练数据之一，本身就是 CoT 数据集）。
- `D_general`：可选通用推理数据（MATH / GPQA 等），通过 `--general-data` 混入。

---

## 3. 难度加权训练（论文 §3.1，Eq.16）

每条样本的交叉熵 loss 按其归一化难度权重 `w̃` 缩放：

```
L_SFT = -1/N · Σ w̃_i · log p(y_i | x_i)
```

| 方法 | 说明 | 默认 |
|---|---|---|
| `complexity` | 直接用 DeepFinance-100K 自带的 `Complexity`(1–10) 注解 | ✅ |
| `heuristic` | 6 类任务先验（math/analysis 权重更高），无需生成 | 备选 |

难度权重由 `sft/weighting.py` 产出，在 `WeightedSFTTrainer.set_difficulty_weights`
里一次性灌入；`compute_loss` 里只做 `(loss * w).mean()`。

---

## 4. 数据流

1. `prepare_financial_data()`：拉 DeepFinance-100K，探测字段名，把 `Solution`
   按首个 `Answer:` 切分为 (thinking, answer)，用 `<think>` 包裹 CoT。
2. （可选）`extra_data_path`：合并数据流水线产出的三元组 JSONL。
3. （可选）`prepare_general_data()` 混入通用数据。
4. `apply_chat_template_batch()`：套 chat 模板并**把 prompt 段 mask 成 -100**，
   保证 loss 只在 assistant 回答上算。

---

## 5. 启动

```bash
bash run_sft.sh                       # 读 sft/config.yaml 完整跑
python -m sft                         # 等价
bash run_sft.sh --max-financial 50    # 小规模试跑
```

| CLI 参数 | 作用 |
|---|---|
| `--config` | yaml 路径（默认 `sft/config.yaml`） |
| `--financial-data` / `--extra-data` / `--general-data` | 数据来源 |
| `--max-financial` / `--max-general` | 样本上限（原型抽样） |
| `--no-thinking` | 去掉 `<think>` CoT |
| `--weighting` | `complexity` / `heuristic` |
| `--epochs` / `--batch-size` / `--lr` / `--seq-length` | 训练超参 |

---

## 6. 配置（`sft/config.yaml`）要点

```yaml
model:
  name: "Qwen/Qwen3.5-9B"
  precision: "fp16"          # 精度开关（int4/fp16/bf16），由 model/ 统一控制
lora: { r: 16, alpha: 32, dropout: 0.05, target_modules: [...] }
data:
  financial_data: "antgroup/Agentar-DeepFinance-100K"
  include_thinking: true
  max_seq_length: 4096
training:
  fp16: true                  # 与 model.precision=fp16 对应
  learning_rate: 2.0e-4
difficulty_weighting:
  method: "complexity"
output_dir: "./checkpoints/stage1"
```

---

## 7. 产出

LoRA 适配器保存到 `output_dir/final`（默认 `./checkpoints/stage1/final`），作为
Stage 2 GRPO 的 `--stage1-adapter` 起点（verl 侧用 `model.lora_adapter_path` 承接）。

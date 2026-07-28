# Stage 1 SFT 技术文档（`sft/__init__.py`）

## 1. 目标（论文 §3.2）

**金融知识与能力注入**。用「金融推理 CoT 数据 ∪ 通用推理数据」监督微调基座模型，
打知识底子。对应论文 Stage 1：*Financial Knowledge and Capability Injection*。

```
D_stage1 = D_fin  ∪  D_general
```

- `D_fin`：**`antgroup/Agentar-DeepFinance-100K`**（论文 §4.2 明确列为训练数据之一，
  且本身就是 CoT 数据集：Question + Solution[CoT] + metadata）。论文原始的
  Fin-R1-300K 未开源，故以 100K 作为具体训练语料。
- `D_general`：可选「大量通用推理数据」（MATH / GPQA 等），通过 `general_data` 混入以保留通用能力。

---

## 2. 难度加权训练框架（论文 §3.1，Eq.16）

每条样本的交叉熵 loss 按其归一化难度权重 `w̃` 缩放：

```
L_SFT = -1/N · Σ w̃_i · log p(y_i | x_i)
```

即**难题权重高、错题惩罚大**。三种难度估计方式（由 `difficulty_weighting.method` 选择）：

| 方法 | 说明 | 是否默认 |
|---|---|---|
| `complexity` | 直接用 DeepFinance-100K 自带的 `Complexity`(1–10) 注解 | ✅ 默认 |
| `heuristic` | 6 类任务先验（knowledge_qa / nlp / text_generation / compliance / math / analysis），无需生成 | 备选 |
| `passk` | 忠实实现论文 Algorithm 1 的 pass@k 估计（需生成采样） | 备选 |

实现位于 `DifficultyWeightedSFTTrainer`（`compute_loss` 把 `w̃` 乘进 per-sample loss）。

---

## 3. 数据流

1. `prepare_financial_data()`：拉 `DeepFinance-100K`，探测字段名（Question/Solution/…
   Thinking/Answer/Complexity），把 Solution 按首个 `Answer:` 切分为 (thinking, answer)，
   按 `include_thinking` 用 `<think>` 包裹 CoT。
2. （可选）`extra_data_path`：合并数据流水线产出的 (query,thinking,answer) 三元组 JSONL。
3. （可选）`prepare_general_data()` 混入通用数据。
4. `apply_chat_template_batch()`：套 chat 模板 → `max_seq_length` 截断。
5. 计算难度权重张量 → 送入加权 SFTTrainer。

---

## 4. 启动方式

```bash
# 完整跑（读 sft/config.yaml）
bash run_sft.sh
# 或
python -m sft

# 小规模试跑
bash run_sft.sh --max-financial 50
python -m sft --max-financial 50
```

### 关键 CLI 参数（覆盖 yaml）
| 参数 | 作用 |
|---|---|
| `--config` | yaml 路径（默认 `sft/config.yaml`） |
| `--financial-data` | 金融 CoT 语料（默认 DeepFinance-100K） |
| `--extra-data` | 数据流水线三元组 JSONL 路径 |
| `--max-financial` | 金融样本上限（原型抽样） |
| `--no-thinking` | 是否去掉 `<think>` CoT |
| `--weighting` | `complexity`/`heuristic`/`passk` |
| `--epochs` / `--batch-size` / `--lr` / `--seq-length` | 训练超参 |

---

## 5. 配置（`sft/config.yaml`）要点

```yaml
model:
  name: "Qwen/Qwen3.5-9B"
  precision: "fp16"          # 精度开关（int4/fp16/bf16）
lora: { r: 16, alpha: 32, dropout: 0.05, target_modules: [...] }
data:
  financial_data: "antgroup/Agentar-DeepFinance-100K"
  include_thinking: true
  max_seq_length: 4096
training:
  fp16: true                 # 与 model.precision=fp16 对应
  bf16: false
  learning_rate: 2.0e-4
difficulty_weighting:
  method: "complexity"
output_dir: "./checkpoints/stage1"
```

> 旧版的 `quantization.load_in_4bit` 整块已删除，统一由 `model.precision` 控制。

---

## 6. 产出

LoRA 适配器保存到 `output_dir/final`（默认 `./checkpoints/stage1/final`），作为
Stage 2 GRPO 的 `--stage1-adapter` 起点。

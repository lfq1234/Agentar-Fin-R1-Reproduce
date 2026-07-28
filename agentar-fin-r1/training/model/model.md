# 模型加载模块技术文档（`model/__init__.py`）

本模块是两段训练（SFT / GRPO）共用的**模型与 LoRA 加载层**，负责把
`Qwen/Qwen3.5-9B` 基座以指定精度加载进来并挂上 LoRA 适配器。

---

## 1. 精度开关（核心）

所有数值格式由 **`ModelConfig.precision`** 这一个字段决定，其它 dtype /
量化标志都是它的派生值，写在 `__post_init__` 里：

| `precision` | 含义 | `load_in_4bit` | 基座 dtype | 量化方式 |
|---|---|---|---|---|
| `"int4"` | 4-bit NF4 QLoRA（旧版低显存模式） | `True` | bf16（计算） | `BitsAndBytesConfig` |
| `"fp16"` | **全精度 fp16 + LoRA，不量化（默认）** | `False` | fp16 | 无 |
| `"bf16"` | 全精度 bf16 + LoRA，不量化 | `False` | bf16 | 无 |

> 当前默认值为 `"fp16"`（按训练要求）。两阶段通过各自的 `*.yaml` 里的
> `model.precision` 显式设定，保证 SFT 与 GRPO 加载的基座精度完全一致。

### 派生逻辑（`__post_init__`）
```python
if self.precision == "int4":
    self.load_in_4bit = True
    self.torch_dtype = torch.bfloat16
    self.bnb_compute_dtype = torch.bfloat16
elif self.precision == "bf16":
    self.load_in_4bit = False
    self.torch_dtype = torch.bfloat16
    self.bnb_compute_dtype = torch.bfloat16
else:  # fp16
    self.load_in_4bit = False
    self.torch_dtype = torch.float16
    self.bnb_compute_dtype = torch.float16
```

---

## 2. `ModelConfig` 字段

| 字段 | 类型 / 默认值 | 说明 |
|---|---|---|
| `model_name_or_path` | `str` = `"Qwen/Qwen3.5-9B"` | 基座模型 id 或本地路径 |
| `precision` | `"int4" \| "fp16" \| "bf16"` = `"fp16"` | 精度开关（见上） |
| `lora_r` | `int` = 16 | LoRA 秩 |
| `lora_alpha` | `int` = 32 | LoRA 缩放 |
| `lora_dropout` | `float` = 0.05 | LoRA dropout |
| `lora_target_modules` | `list[str]` | 注入模块：`q/k/v/o_proj` + `gate/up/down_proj` |
| `attn_implementation` | `"sdpa"`（默认） | 注意力后端，可选 `eager` / `flash_attention_2` |
| `trust_remote_code` | `bool` = `True` | 加载远程自定义代码 |
| `load_in_4bit` *(派生)* | `bool` | 由 `precision` 推导，勿手动设置 |
| `torch_dtype` *(派生)* | `torch.dtype` | 由 `precision` 推导 |
| `bnb_compute_dtype` *(派生)* | `torch.dtype` | 由 `precision` 推导 |

---

## 3. 公开函数

### `load_tokenizer(model_name_or_path, *, trust_remote_code=True, padding_side="right")`
- 加载 tokenizer，处理 Qwen 缺失 `pad_token` 的情况（回退到 `eos_token`）。
- `padding_side="right"`：因果 LM 微调时只在 assistant 段计算 loss 的标准做法。

### `load_model(model_name_or_path=None, *, cfg=None, device_map="auto")`
- 返回**裸** `AutoModelForCausalLM`（尚未挂 LoRA）。
- 内部用 `_bnb_config(cfg)`：仅当 `load_in_4bit=True` 时构造
  `BitsAndBytesConfig`（nf4 + 双量化），否则不量化。
- dtype 取自 `cfg.torch_dtype`（由 `precision` 决定）。

### `apply_lora(model, *, cfg=None)`
- 用 `LoraConfig` 包裹模型为 `PeftModel`。
- 仅当 `cfg.load_in_4bit=True`（即 int4）时先调用
  `prepare_model_for_kbit_training`（fp16/bf16 全精度不需要）。

### `print_trainable_parameters(model)`
- 打印可训练 / 冻结参数占比。

---

## 4. 典型用法

```python
from model import ModelConfig, load_model, load_tokenizer, apply_lora

cfg = ModelConfig(precision="fp16")          # 或 "bf16" / "int4"
tok = load_tokenizer(cfg.model_name_or_path)
model = load_model(cfg.model_name_or_path, cfg=cfg)
model = apply_lora(model, cfg=cfg)
```

两阶段入口在构建 `ModelConfig` 时，会从各自 yaml 读取 `model.precision`：

- `sft.yaml` → `model.precision`
- `grpo.yaml` → `model.precision`

因此**切换精度只需改 yaml 一处**，代码无需改动。

---

## 5. 与两阶段的关系

- **Stage 1（`stage1_sft.train_stage1`）**：`__main__` 依据 `sft.yaml` 构造
  `ModelConfig` 并透传给 `load_model` / `apply_lora`；SFT 的 `TrainingArguments`
  另通过 `training.fp16` / `training.bf16` 控制 Trainer 的混合精度作用域，需与
  `model.precision` 保持一致（fp16 权重 ↔ `fp16: true`）。
- **Stage 2（`stage2_grpo.train_stage2`）**：policy 与 reference 两个模型都用同一个
  `model_cfg` 加载（ref 冻结、不挂 LoRA 作 KL 锚点）；自定义 GRPO 训练循环直接在当前
  模型 dtype（fp16）下做前向/反向，无需额外的 amp 开关。

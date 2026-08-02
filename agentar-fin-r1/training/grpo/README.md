# Stage 2：GRPO 技术方案（grpo/）

> 两阶段训练的**第二阶段：难题攻坚**。
> 论文 §3.2 Stage 2（GRPO + 针对性 SFT）+ §3.3 归因闭环。
> 本阶段在 Stage 1 SFT 产出模型基础上，用 **GRPO 强化学习 + LoRA** 攻坚 Stage 1 暴露的困难任务。

---

## 1. 阶段目标

| 维度 | 目标 |
| --- | --- |
| 能力 | 在 Stage 1 基础上，针对**金融难题**（数学/代码/复杂推理/合规判断）做强化提升，把 pass@1 拉高 |
| 数据 | Stage 1 评估暴露的**困难子集** + 归因闭环 `attribution.json` 补数据（无需大量新标注，靠 rollout 自生成） |
| 方法 | **GRPO（Group Relative Policy Optimization）+ LoRA**，无需 value/Critic 网络 |
| 加权 | 难度感知加权延续到 RL（困难标签的 rollout 权重更高） |
| 衔接 | 产出 `grpo_lora_adapter/`（或 merge），进入归因闭环：不收敛回退针对性 SFT，连续退化触发 `data/` 再生样本 |

---

## 2. GRPO 算法要点

### 2.1 为什么用 GRPO

GRPO（DeepSeek-R1 同款）是 PPO 的简化：**去掉 Critic 网络**，用一个 prompt 的多条 rollout 互相比较来估计优势，显存与工程复杂度显著降低，特别适合 8B 级单/双卡复现。

### 2.2 核心公式

对一个 prompt `q`，采样 `K` 条响应 `{o_1, ..., o_K}`，各自打奖励 `{r_1, ..., r_K}`：

```
# 组内相对优势（去均值 + 标准化）
A_i = (r_i − mean(r_1..K)) / (std(r_1..K) + ε)

# 策略损失（clip + KL 正则）
L_GRPO = −(1/K) Σ_i min( ρ_i·A_i,  clip(ρ_i, 1−ε, 1+ε)·A_i )  −  β·D_KL(π_θ ‖ π_ref)

  其中 ρ_i = π_θ(o_i|q) / π_θ_old(o_i|q)   (importance ratio)
```

- `β`：KL 系数，约束策略别漂离 Stage 1 的 ref 模型太远（防 reward hacking）。
- `K`：group size，典型 4~8；越大优势估计越稳但显存涨。
- 无需 Critic → 省一个同等规模的网络显存。

### 2.3 与论文的对应

论文 §3.2 Stage 2 = "GRPO（复杂金融场景多目标决策）+ 针对性 SFT"，并明确"GRPO 某类任务不收敛时回退到针对性 SFT"。本目录负责 GRPO 部分，回退的针对性 SFT 复用 `../sft/` 的脚本（换数据子集即可）。

---

## 3. 数据来源

### 3.1 不需要金标答案？需要——GRPO 的 reward 依赖可验证答案

GRPO 不是无监督 RL，它需要**可验证的 reward**。金融任务的 reward 来自**裁判模型
比对金标答案**（由 LLM-as-judge 判定，见 §5，本地不做数值计算），因此数据仍是
`(query, gold_answer)` 形式，但**不需要 thinking**（thinking 由模型 rollout 自生成）。

### 3.2 三类数据

| 来源 | 路径 | 用途 |
| --- | --- | --- |
| **Stage 1 困难子集** | 从 `outputs/sft_metrics.json` 取 pass@1 < 阈值的 label 对应样本 | 主训练集 |
| **归因闭环补数据** | `outputs/attribution.json`（见 §8）按 `P_ℓ` 分配的样本 | 针对性补短板 |
| **可验证评测题** | Finova 复杂推理 282 题 + MATH-500 | 交裁判模型判定的"硬核"题 |

### 3.3 数据格式

```python
{
    "query": "...",
    "gold_answer": "3.45% 或 [实体列表] 或 YES/NO",
    "scene": "Securities",
    "task":  "ConsultationQA",
    "label": ["Securities","ConsultationQA"]
}
```

`query` / `gold_answer` 是裁判判定所需；其余元数据（scene/task/label）供归因与
难度加权使用，不参与 reward 计算。

---

## 4. LoRA GRPO 配置

### 4.1 初始策略

**方案 A（推荐）**：以 Stage 1 **merge 后**的 `outputs/sft_merged/` 为基座，新挂一个 LoRA 做 GRPO。
- 优点：策略干净，ref model = sft_merged 本身，KL 语义清晰。
- 代价：merge 一次（不可逆，需保留 adapter 副本，见 sft/ §10）。

**方案 B**：直接在带 Stage 1 adapter 的模型上继续 GRPO（ms-swift `swift rlhf` 同样支持 `tuner_type=lora` 续训）。
- 优点：省 merge。
- 代价：ref model 需单独加载一份 sft_merged（否则 KL 计算会带上 adapter 漂移），工程稍绕。

> 本目录默认方案 A，初始策略 = `outputs/sft_merged/`。

### 4.2 LoRA 配置

GRPO 阶段策略更新更剧烈，LoRA 配置比 SFT 略保守：

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=32,                          # 比 SFT 的 64 小，RL 阶段防过大更新
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
)
```

### 4.3 Reference Model

GRPO 的 KL 项需要一个**冻结的 ref 策略**。方案 A 下 ref = `sft_merged`（即初始基座本身）。`swift rlhf --rlhf_type grpo` 在 `tuner_type=lora` 时自动以未挂 adapter 的基座作为 ref model，无需手动加载第二份。

---

## 5. 奖励函数设计（关键）—— LLM-as-judge

GRPO 的效果上限由 reward 设计决定。金融推理题多为**开放/半开放**，用规则的
数值比对（抽取数字、字符串归一化、集合 IoU）既脆弱又覆盖不全，且容易被
reward hacking。**本项目改用「加载一个裁判模型」对每个 rollout 打分，本地
不做数值计算**——裁判模型负责判断「结论是否正确、推理是否合理、是否带
`<think>` 边界」。

### 5.1 裁判模型如何部署（不在训练进程内）

裁判模型以**独立服务**形式部署，通过 OpenAI 兼容 `/v1` 接口（推荐 vLLM）提供，
reward 函数在训练循环里走 HTTP 调用它：

```bash
# 单独起一个进程/卡，不与训练抢显存
vllm serve Qwen/Qwen2.5-72B-Instruct --port 8000 --gpu-memory-utilization 0.4
```

也可以指向任意 OpenAI 兼容端点（如商业 API）。`rewards.py` 用 `openai` 客户端
封装成 `LLMJudge`，在 `train_grpo.py` 配置区填 `judge_base_url / judge_model /
judge_api_key` 即可。

### 5.2 裁判判定标准与返回格式

系统提示要求裁判对每组 `(query, gold_answer, completion)` 输出 JSON 数组，
每项：

```json
{"correct": 0或1, "reason": "简短理由"}
```

`correct = 1` **当且仅当** ① 结论与标准答案一致；② 推理合理无原则性错误；
③ 包含 `<think>...</think>` 边界（缺边界直接 0）。格式要求已内嵌进 `correct`，
不再单独设格式奖励，避免权重叠加噪声。

### 5.3 批量调用（每步一次 HTTP，省吞吐）

`LLMJudgeReward.__call__` 把整步所有 `(completion, gold_answer)` 拼成一条 prompt，
**一次 HTTP 调用**让裁判返回等长分数数组。相比逐条调用，
在 `K=8 × grad_accum=8 = 64` 条 completion/步 下能把 judge 调用从 64 次降到 1 次。

### 5.4 干净兜底

解析失败（非 JSON、长度不符、字段缺失）时整批回退为 `0.0`，**不重试、不多层
try-except**。裁判温度设 `0.0` 保证打分确定性。

### 5.5 数据列与 reward 透传

数据集每行 `{"messages": [...], "gold_answer": ...}`；ms-swift 在 GRPO 下会把
**所有额外数据集列**（如 `gold_answer`）以 kwargs 透传给 reward 函数的 `__call__`
（`completions` 为位置参数），无需手写 `build_dataset` 透传逻辑。

### 5.6 难度加权（可选，首版可先不接）

论文 §3.1 在 RL 阶段的延续：对 batch 内每个 prompt，按其 label 的归一化权重
`w̃_ℓ`（来自 `weighting.py`）缩放其所有 rollout 的 loss 项。实现：在
`RLHFArguments` 外层按 prompt 权重缩放。首版先把 GRPO 单 epoch 跑通、
reward 曲线正常，再接闭环。

---

## 6. 训练超参

| 超参 | 值 | 备注 |
| --- | --- | --- |
| group size `K` | `8` | 每个 prompt 采 8 条 rollout |
| num_generations | `8` | = K，`swift rlhf` 参数 |
| max_completion_length | `1024` | rollout 长度，CoT 需要空间 |
| temperature (rollout) | `0.9` | 高温探索；后期可退火到 0.5 |
| top_p | `0.9` | |
| learning_rate | `5e-6` | RL 阶段 lr 远小于 SFT，防策略崩溃 |
| lr_scheduler | `constant_with_warmup` | RL 常用 |
| warmup_steps | `50` | |
| beta (KL coef) | `0.04` | DeepSeek-R1 默认；太大抑制探索，太小 reward hacking |
| clip range ε | `0.2` | PPO 标准 |
| per_device_train_batch_size | `1` | 单 prompt |
| gradient_accumulation_steps | `8` | 等效 8 prompts/step |
| num_train_epochs | `1~2` | RL 易过拟合 reward，少 epoch |
| max_prompt_length | `2048` | |
| max_completion_length | `1024` | |
| torch_dtype | `float16` | AMP 自动混合精度（与 SFT 一致，FP16 加载） |
| gradient_checkpointing | `True` | rollout 阶段省显存 |

> 显存：GRPO 要同时跑 policy（带 LoRA）+ ref（冻结基座）+ K 条 rollout 的 logits。Qwen3-8B + K=8 在 24GB 卡上吃紧，可降 K=4 或换 40GB+ 卡（A100/H100）。

---

## 7. 脚本结构

```
grpo/
├── README.md                  # 本文件
└── src/
    ├── rewards.py             # LLM-as-judge 裁判 reward（orms 注册 + 外部 vLLM）
    └── train_grpo.py          # 主训练入口（RLHFArguments + rlhf_main，rlhf_type=grpo）
```

> 注：本目录用 ms-swift 的 `swift rlhf --rlhf_type grpo`，数据困难子集筛选与归因
> 闭环（`build_grpo_dataset.py` / `attribution.py`）作为后续可选增强，首版先把
> GRPO 单 epoch 跑通、reward 曲线正常，再接闭环。

### 7.1 train_grpo.py 主流程（等价 Python API）

```python
from swift.llm import RLHFArguments, rlhf_main

args = RLHFArguments(
    rlhf_type="grpo",
    model="outputs/sft_merged",          # 初始策略 = Stage 1 merge 后的模型
    dataset="./grpo_data.jsonl",         # 每行 messages + gold_answer 列
    tuner_type="lora",
    lora_rank=32, lora_alpha=64, lora_dropout=0.05,
    target_modules="all-linear",
    torch_dtype="float16",
    reward_funcs="judge_reward",         # 注册在 rewards.py 的 orms 键
    external_plugins="./grpo/src/rewards.py",
    num_generations=8, beta=0.04,
    temperature=0.9, top_p=0.9,
    max_completion_length=1024, max_prompt_length=2048,
    learning_rate=5e-6, lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    per_device_train_batch_size=1, gradient_accumulation_steps=8,
    num_train_epochs=1, gradient_checkpointing=True,
    output_dir="outputs/grpo_lora_adapter",
)
rlhf_main(args)
```

### 7.2 reward 函数签名（ms-swift 约定）

自定义 reward 继承 `swift.rewards.ORM`，`__call__` 的位置参数 `completions` 为模型
输出列表，其余数据集列（如 `gold_answer`）以 kwargs 透传，返回 `list[float]`。
在 `rewards.py` 末尾用 `orms["judge_reward"] = LLMJudgeReward` 注册，训练命令用
`--external_plugins ./grpo/src/rewards.py --reward_funcs judge_reward` 引用。

---

## 8. 归因闭环（论文 §3.3，与 training/src/attribution.py 对接）

### 8.1 闭环流程

```
Stage 2 训练每 N 步
   ├─ 在 Finova 子集上算 per-label pass@1  →  attribution.json
   ├─ 任务优先级 P_ℓ = gap_ℓ · η_ℓ · exp(−δ·D_ℓ)
   │     gap = max(0, target − acc),  η = 学习效率,  D = 已分配数据量
   ├─ 数据分配 D_ℓ = B · (P_ℓ / ΣP)
   ├─ 性能回退 → D_ℓ(t) ← D_ℓ(t−1)            # 数据回滚
   └─ 连续 3 轮退化 → 触发 data/ Synthesis 再生该 label 样本
```

### 8.2 回退到针对性 SFT

论文：GRPO 某类任务不收敛时回退针对性 SFT。实现：

```python
# train_grpo.py 内监控
if label_stagnates(label, window=3):            # 连续 3 轮 pass@1 不升
    dump_subset(label, "outputs/sft_subset_ℓ.jsonl")
    # 调 sft/train_sft.py --data-sources subset --init outputs/sft_merged
    # 然后用 SFT 后的模型重启 GRPO
```

### 8.3 终止条件

- 所有 label 的 pass@1 达标（`target = SOTA + 5`），或
- 总数据量超上限且精度饱和，或
- 边际收益 < 阈值。

---

## 9. 命令示例

```bash
cd agentar-fin-r1/training

# 标准 GRPO（FP16 LoRA，单/双卡 24GB+）
swift rlhf \
    --rlhf_type grpo \
    --model outputs/sft_merged \
    --dataset ./grpo_data.jsonl \
    --tuner_type lora \
    --lora_rank 32 --lora_alpha 64 --lora_dropout 0.05 \
    --target_modules all-linear \
    --torch_dtype float16 \
    --external_plugins ./grpo/src/rewards.py \
    --reward_funcs judge_reward \
    --num_generations 8 --beta 0.04 \
    --temperature 0.9 --top_p 0.9 \
    --max_completion_length 1024 --max_prompt_length 2048 \
    --learning_rate 5e-6 --lr_scheduler_type constant_with_warmup --warmup_steps 50 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --num_train_epochs 1 --gradient_checkpointing true \
    --output_dir outputs/grpo_lora_adapter \
    --report_to wandb

# 显存吃紧：降 group size（K=4）
swift rlhf ... --num_generations 4 ...

# 或 Python API：python grpo/src/train_grpo.py

# merge → 最终模型（复用 sft 的 merge 逻辑）
swift export \
    --model outputs/sft_merged \
    --adapters outputs/grpo_lora_adapter \
    --torch_dtype float16 \
    --output_dir outputs/fin_r1_final
```

---

## 10. 风险与注意

1. **reward hacking**：模型可能学会输出看似合理但错误的推理来骗过裁判。缓解：① 裁判由更强/独立模型担任，且判定标准含「结论一致 + 推理合理 + `<think>` 边界」三重约束；② KL 系数别太小（β=0.04）；③ 定期人工抽查 rollout 与裁判打分一致性。
2. **KL 崩溃**：lr 过大或 beta 过小会导致策略漂离 ref 太远，rollout 质量雪崩。监控 ms-swift 日志里的 `kl` metric，正常应 < 10；超 20 立即降 lr。
3. **rollout 长度爆炸**：CoT 模型 RL 时易出现"越想越长"。加 `max_completion_length` 硬截断，并可加长度惩罚 `−0.001·len`。
4. **显存**：K=8 + Qwen3-8B 在 24GB 卡上 rollout 阶段易 OOM。优先降 K 到 4，再不行换 40GB+ 卡（A100/H100）或先用 `Qwen/Qwen3-4B` 跑通流程再升档。
5. **归因闭环未建好前**：首版可只用 Stage 1 困难子集 + 固定数据，不接动态归因；先把 GRPO 单 epoch 跑通、reward 曲线正常，再接闭环。
6. **FP16 稳定性**：与 SFT 一致用 `torch_dtype=float16`(AMP)；RL 阶段 rollout 采样对数值精度更敏感，出现 NaN 优先降 lr，必要时把基座加载精度与训练精度解耦（加载 bf16 + 训练 fp16）作为兜底。
6. **GRPO 收敛慢**：RL 通常比 SFT 慢 3–5 倍（每步要 K 次 rollout）。预期单 epoch 在 5K prompt × K=8 上约 8–12 小时（单 A100），24GB 消费卡更久。

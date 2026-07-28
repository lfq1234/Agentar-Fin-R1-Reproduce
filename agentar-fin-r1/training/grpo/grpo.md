# Stage 2 GRPO 技术文档（`grpo/__init__.py`）

## 1. 目标（论文 §3.3）

**难题攻坚强化学习**。在 Stage 1 适配器基础上，对一道**难题**让模型自己生成
`group_size` 条回答，用奖励打分、组间互相比优劣来更新策略，专攻 SFT 学不好的硬骨头。

与 SFT 的区别一句话：**SFT 给标准答案抄着学；GRPO 不给答案，自己答、自己评、好留坏改。**

---

## 2. GRPO 算法（本仓库实现）

标准 GRPO（DeepSeekMath 风格）+ 论文要求的定向 SFT 回退：

1. **一次 rollout 出 G 条**（`_generate_group`，`GRPOConfig.group_size`）
   同一 prompt 复制 G 份，温度采样生成 G 个回答。**当前 `group_size = 8`**（要求值）。
2. **打分**（`composite_reward`，多目标奖励加权和）：
   - `correctness`（正确性）：与 gold 比，数值接近给高分（默认权重 1.0）。
   - `format`（格式）：是否有 `<think>…</think>` 结构（默认 0.3）。
   - `length`（长度）：过长轻微扣分（默认 0.0，关闭）。
3. **组相对优势**：`A = (r_i − mean(r)) / std(r)`。
   **比组内平均好→正分，差→负分**——没有绝对标准，全靠这 8 条互相比。
4. **裁剪目标 + KL 惩罚**（`_train_one` 内）：
   - 策略损失 `L = −E[min(ρ·A, clip(ρ, 1−ε, 1+ε)·A)]`（PPO 式裁剪，`clip_eps=0.2`）。
   - KL 惩罚 `β·KL(π_θ ‖ π_ref)`（`beta=0.04`），用冻结的 `ref_model` 当锚点防止跑偏。
5. **停滞回退**（`train` 循环）：连续 `stall_patience`（默认 25）步奖励无提升 →
   触发 `targeted_sft`，拿最差的几条做几次监督训练救场（论文要求）。

---

## 3. 两模型结构

| 模型 | 角色 | 训练 |
|---|---|---|
| `model`（policy） | 当前策略 | 挂 LoRA、`requires_grad=True`、随 GRPO 更新 |
| `ref_model` | KL 参照 | 同基座、**冻结**、不挂 LoRA |

两模型用**同一个 `ModelConfig`** 加载（默认 `precision="fp16"`），保证精度一致。

---

## 4. 启动方式

```bash
# 必须提供 hard-subset（每行 {question, answer} 的难题集）
bash run_grpo.sh --hard-subset data/golden/hard_subset.jsonl \
                 --stage1-adapter checkpoints/stage1 --max-samples 50
# 或
python -m grpo \
    --hard-subset data/golden/hard_subset.jsonl \
    --stage1-adapter checkpoints/stage1 --max-samples 50
```

> 不带 `--hard-subset` 且 yaml 未设 `data.hard_subset` 会直接报错退出。
> hard subset 来源：归因闭环（`attribution.py`）或 Stage-1 错误分析，亦可手工造小文件。

### 关键 CLI 参数（覆盖 yaml）
| 参数 | 作用 |
|---|---|
| `--config` | yaml 路径（默认 `grpo/config.yaml`） |
| `--hard-subset` | 难题 JSONL（必填） |
| `--stage1-adapter` | 从 Stage-1 适配器起训 |
| `--group-size` | G rollouts（默认 8） |
| `--max-samples` | 难题使用上限（原型抽样） |
| `--learning-rate` / `--beta` / `--temperature` | 训练超参 |

---

## 5. 配置（`grpo/config.yaml`）要点

```yaml
model:
  model_name: "Qwen/Qwen3.5-9B"
  precision: "fp16"          # 与 Stage 1 共用精度开关
  stage1_adapter: null
grpo:
  group_size: 8              # 每次 rollout 8 条（要求值）
  temperature: 0.9
  max_new_tokens: 1024
  learning_rate: 1.0e-6
  beta: 0.04                 # KL 系数
  clip_eps: 0.2
reward:
  weights: { correctness: 1.0, format: 0.3, length: 0.0 }
targeted_sft:
  stall_patience: 25
  targeted_sft_steps: 5
data:
  hard_subset: "data/golden/hard_subset.jsonl"
run:
  output_dir: "checkpoints/stage2-grpo"
```

---

## 6. 产出

Stage-2 LoRA 适配器保存到 `run.output_dir`（默认 `checkpoints/stage2-grpo`）。

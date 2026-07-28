# Stage 2 GRPO 技术文档（`grpo/`）

## 1. 实现方式：verl 库 + LoRA

Stage 2 是**完全基于 `verl` 库的 GRPO 实现**，不再手写 GRPO 数学。
`verl`（字节火山引擎 RL 框架）原生提供：

- GRPO 训练器（`algorithm.adv_estimator=grpo`：组相对优势 + KL 到 reference）；
- rollout 引擎（vLLM / SGLang）；
- **LoRA 支持**（`actor_rollout_ref.model.lora_rank`），只训练低秩增量。

本目录只做三件事，不碰训练数学：

| 文件 | 职责 |
|---|---|
| `grpo/data.py` | 把 Stage-1 错误/归因产出的 `hard_subset.jsonl` 转成 verl 的 RLHF parquet |
| `grpo/reward.py` | verl 奖励函数 `compute_score`（多目标：正确性 + 格式） |
| `grpo/__init__.py` | 读 `config.yaml` → 生成 verl 的 hydra 覆盖参数 → 启动 `verl.trainer.main_ppo` |
| `grpo/attribution.py` | 归因闭环（写 attribution.json，驱动数据回滚/再生） |

```
hard_subset.jsonl
      │  grpo/data.py  (convert_to_verl_parquet)
      ▼
hard_subset.parquet  ──┐
                       │  grpo/__init__.py  (build_verl_overrides)
                       │      GRPO + LoRA 覆盖参数
                       ▼
                verl.trainer.main_ppo  ──→  grpo/reward.py (compute_score)
                       │                         ▲
                       │  actor_rollout_ref.model.lora_rank
                       ▼
                checkpoints/stage2-grpo  (仅 LoRA 增量)
```

---

## 2. GRPO 目标（verl 原生）

对每道难题采样 `group_size` 条回答（`rollout.n`），按组内相对优势更新策略：

```
A_i = (r_i − mean(r)) / std(r)            # 组相对优势
L   = −E[ min(ρ·A, clip(ρ, 1−ε, 1+ε)·A) ] + β·KL(π_θ ‖ π_ref)
```

- `group_size = 8`（论文要求）→ verl 的 `actor_rollout_ref.rollout.n=8`。
- KL 系数 `beta=0.04` → verl 的 `actor.kl_loss_coef`。
- 多目标奖励在 `grpo/reward.py`，经 `custom_reward_function` 接入 verl。

---

## 3. LoRA 怎么开

由 `config.yaml` 的 `lora.*` 驱动，映射成 verl 的 hydra 覆盖：

| config.yaml | verl 覆盖 |
|---|---|
| `lora.rank` | `actor_rollout_ref.model.lora_rank` |
| `lora.alpha` | `actor_rollout_ref.model.lora_alpha` |
| `lora.target_modules` | `actor_rollout_ref.model.target_modules`（verl 用 `"all-linear"`） |
| `model.stage1_adapter` | `actor_rollout_ref.model.lora_adapter_path`（从 Stage-1 适配器起训） |

> verl 的 LoRA 走 HuggingFace peft + FSDP，rollout 端需
> `actor_rollout_ref.rollout.load_format=safetensors`（代码已默认设置）。

---

## 4. 启动

```bash
# 必须提供 hard-subset（每行 {question, answer} 的难题集）
bash run_grpo.sh --hard-subset data/golden/hard_subset.jsonl \
                 --stage1-adapter checkpoints/stage1 --max-samples 50
# 或
python -m grpo --hard-subset data/golden/hard_subset.jsonl \
               --stage1-adapter checkpoints/stage1 --max-samples 50
```

`run_grpo.sh` 内部就是 `python -m grpo`：`grpo/__init__.py` 先转 parquet、
再拼出 verl 覆盖参数、最后 `subprocess` 调用 `verl.trainer.main_ppo`。

| CLI 参数 | 作用 |
|---|---|
| `--config` | yaml 路径（默认 `grpo/config.yaml`） |
| `--hard-subset` | 难题 JSONL（必填） |
| `--stage1-adapter` | 从 Stage-1 适配器起训 |
| `--max-samples` | 难题使用上限（原型抽样） |
| `--output-dir` / `--model-name` | 覆盖 yaml |

---

## 5. 配置（`grpo/config.yaml`）要点

```yaml
model:
  name: "Qwen/Qwen3.5-9B"
  stage1_adapter: null          # 可选 Stage-1 LoRA 适配器路径
lora:
  rank: 16
  alpha: 32
  target_modules: "all-linear"
grpo:
  group_size: 8                 # -> rollout.n（每 prompt 8 条）
  temperature: 0.9
  learning_rate: 3.0e-5         # verl 建议 LoRA 用 ~10x 全参 LR
  beta: 0.04                    # -> actor.kl_loss_coef
  clip_eps: 0.2                 # -> actor.clip_ratio
rollout:
  name: "vllm"                  # 或 "sglang" / "hf"
reward:                         # 权重在 grpo/reward.py
  correctness: 1.0
  format: 0.3
data:
  hard_subset: "data/golden/hard_subset.jsonl"
  parquet_dir: "./data/rl"
run:
  output_dir: "checkpoints/stage2-grpo"
  n_gpus_per_node: 1            # 多卡改大
  train_batch_size: 16          # 建议是 group_size 的倍数
```

---

## 6. 产出

仅保存 LoRA 增量到 `run.output_dir`（默认 `checkpoints/stage2-grpo`）。
verl 的 `model.lora.merge` 控制是否把 LoRA 合并进基座再同步给 rollout 引擎。

---

## 7. 环境依赖

verl + LoRA 训练需要：`verl`、`vllm`（或 `sglang`）、`ray`（verl 启动用）、
`hydra-core`、`pandas`、`pyarrow`。见 `pyproject.toml`。

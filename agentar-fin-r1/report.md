# Agentar-Fin-R1 复现技术报告

> 跟踪记录全部复现流程。论文：https://arxiv.org/abs/2507.16802 （代码未开源，本仓库为自主复刻）

## 1. 概述

Agentar-Fin-R1 是蚂蚁数科基于 **Qwen3**（8B / 32B 两档）训练的金融推理大模型。核心创新：

1. **Label System**：二维金融任务标签体系，让模型"出厂即专家"。
2. **三级可信数据治理**：源头可信 → 合成可信 → 治理可信，产出 Fin-R1-300K 高质量三元组。
3. **难度感知加权训练 + 两阶段训练**：最大化数据/训练效率。
4. **归因闭环**：错误回滚到 (Scene, Task) 并自动再生数据。
5. **Finova 评测**：自建 1350 题金融基准（Agent 能力 / 复杂推理 / 安全合规）。

## 2. Label System（标签体系）

二维、非正交稀疏标签：`l = (scene, task)`

- **Scene（场景）**：Banking / Securities / Insurance / Trust / MutualFunds
- **Task（任务）**：NER / IntentClassification / SlotFilling / EntityDisambiguation / ConsultationQA
- **非正交**：并非所有 Task 适用于所有 Scene（真实还原金融任务分布）。

复刻实现见 [`data/src/finr1_data/labels.py`](../data/src/finr1_data/labels.py)。

## 3. 数据构造：三级流水线

| 层级 | 关键动作 | 目的 |
| --- | --- | --- |
| **Source** | 权威金融机构/监管文件 → NER/POS → 归一化 → 脱毒 → 知识精炼 | 来源可信 |
| **Synthesis** | 双轨：① 任务导向知识引导生成 (Query, Thinking, Answer) ② 自进化指令（多样性/复杂性/正确性三重筛选） | 逻辑可验证 |
| **Verification** | 多模型一致性投票 + 专家抽样 + Rating 模型打分 → 去重/去污/去泄露 | 质量可靠 |

产出 **Fin-R1-300K** 的 `(query, thinking, answer)` 三元组，作为训练 golden data。

复刻实现见 [`data/src/finr1_data/`](../data/src/finr1_data/)（`source/` `synthesis/` `verification/` `pipeline.py`）。

## 4. 训练框架

### 4.1 难度感知加权训练（pass@k）

对每个 Task Label 分层采样 n 题 → 当前模型 & m 个参考模型各生成 k 条 → 算 pass@k：

```
w_t = α·(1 − pass@k_cur) + β·max(0, pass@k_ref − pass@k_cur) + γ
```

并引入指数平滑 + 下限裁剪保证稳定。困难样本权重更高，训练更聚焦。

复刻实现见 [`training/src/finr1_training/weighting.py`](../training/src/finr1_training/weighting.py)。

### 4.2 两阶段递进

| 阶段 | 目标 | 方法 | 数据 |
| --- | --- | --- | --- |
| Stage 1 | 金融知识注入 | 大规模 SFT + 加权训练 | Fin-R1-300K + 通用推理 |
| Stage 2 | 难题攻坚 | GRPO（强化）+ 针对性 SFT | 困难子集 + 错误归因补充数据 |

**训练框架：已迁移到 verl 0.8.0**（原 ms-swift 版保留在 `training/sft`、`training/grpo`，作遗留参考）。
迁移理由：verl 在 GRPO 阶段用 vLLM rollout（PagedAttention，比 `transformers.generate` 快 3-5x）+
Ray 式流水线，对本项目（Qwen3-8B + K=8 + 外部 72B 裁判）是代差级提速；ms-swift 的 rollout/reward
同步阻塞瓶颈在 verl 内部解决，无需自改代码。

- verl 实现见 [`training/verl/`](../training/verl/)：`sft/train_sft.sh`（Stage1 LoRA r=64）、
  `grpo/train_grpo.sh` + `grpo/fin_judge_reward.py`（Stage2 GRPO + LLM-judge）、`merge_lora.py`、
  `data/prepare_verl_data.py`。
- 配置等价映射：原 `num_generations=8 → rollout.n=8`；`beta=0.04 → actor.kl_loss_coef=0.04`；
  reward 由 `external_plugins` 改为子类化 `NaiveRewardManager`（`@register("fin_judge")`）。

### 4.3 归因闭环（Attribution Loop）

按二维标签对预测错误分类，找性能洼地，输出 `attribution.json`：

```
{ label, pass@1, Δ, η, π, allocated_samples }
```

训练脚本读取后更新数据加载器继续训练；pass@1 下降则回滚数据，连续 3 轮下降触发自进化 Agent 再生样本。

复刻实现见 [`training/src/finr1_training/attribution.py`](../training/src/finr1_training/attribution.py)。

## 5. 评测：Finova

| 维度 | 子任务 | 样本数 |
| --- | --- | --- |
| Agent Capabilities | 意图识别 / 槽位识别 / 工具规划 / 表达生成 | 768 |
| Complex Reasoning | 金融数学 + 代码理解 + 推理 | 306 |
| Safety & Compliance | 安全风险识别 / 监管合规判断 | 200 |

## 6. 复刻计划与进度

- [x] 仓库结构与技术栈选型（uv + FastAPI + React/Vite）
- [ ] 数据复刻：三级流水线跑通，产出小规模 golden 三元组
- [x] 训练框架迁移 verl 0.8.0（SFT/GRPO/merge/data prep 代码就绪，见 `training/verl/`）
- [ ] 训练复刻：加权 SFT（Stage 1）→ GRPO（Stage 2）→ 归因闭环
- [ ] 后端 + Agent 运行时接入复刻模型
- [ ] 前端交互页（对话 / 任务演示 / 评测可视化）
- [ ] 在 Finova 子集上评估并对比基线

> 范围说明：当前目标为**小规模原型**（单/双卡，Qwen3-8B + QLoRA/LoRA），先跑通端到端闭环。

# 复现根目录：Agentar-Fin-R1

本目录承载对蚂蚁数科 **Agentar-Fin-R1** 的全部复刻工作，分为三块：

| 子目录 | 对应论文环节 | 内容 |
| --- | --- | --- |
| [`paper/`](paper/) | — | 论文原文、解读与架构笔记 |
| [`data/`](data/) | 数据构造（三级治理） | 三级数据流水线：Source → Synthesis → Verification，产出 `(query, thinking, answer)` 三元组 |
| [`training/`](training/) | 训练框架 | 难度感知加权 + 三阶段（SFT / DAPO / OPD 蒸馏）+ 归因闭环 |

复现技术报告见 [`report.md`](report.md)。

## 复刻范围（当前）

**小规模原型**：单/双卡，基座 Qwen3-8B + QLoRA/LoRA，目标先把"三级数据流水线 + 两阶段训练闭环"端到端跑通，产出可演示结果。后续可平滑扩展到多卡全参 / 32B。

# 论文原文与解读（paper/）

本目录存放 **Agentar-Fin-R1** 系列的两篇原始论文与配套解读。这两篇论文是本复刻项目的"设计说明书"：
主论文定义**整体流水线 + 训练范式 + 评测基准**，`DeepFinance-100K` 论文定义**数据源头（Source 阶段）的知识合成方法**。

---

## 1. 目录文件清单

| 文件 | 说明 |
|------|------|
| `README.md` | 本文件：两篇论文的结构化解读 + 与复刻目录的映射 |
| `Agentar-Fin-R1 Enhancing Financial Intelligence through .pdf` | 主论文（30 页） |
| `Agentar-DeepFinance-100K A Large-Scale Financial Dataset .pdf` | 支撑数据集论文（18 页） |

> 两篇 PDF 均为本地拷贝（体积合计约 3.4MB），已被根目录 `.gitignore` 的 `*.pdf` 规则忽略，**不会进入版本库**（仍保留在本地磁盘供离线查阅）。如需纳入版本控制，请从 `.gitignore` 移除 `*.pdf` 或用 `git add -f` 强制添加。

---

## 2. 论文一：Agentar-Fin-R1（主论文）

**标题**：Agentar-Fin-R1: Enhancing Financial Intelligence through Domain Expertise, Training Efficiency, and Advanced Reasoning
**机构**：Ant Group Digital Technologies（蚂蚁数科）
**作者**：Yanjun Zheng, Xiyang Du, Longfei Liao, Xiaoke Zhao, Zhaowen Zhou, et al.
**arXiv**：[2507.16802](https://arxiv.org/abs/2507.16802)（v4, 2025-07-27）
**基座**：Qwen3（8B / 32B 两档）
**代码状态**：未开源（本仓库为自主复刻）
**配套评测**：Finova 基准 → https://github.com/antgroup/Finova

### 2.1 要解决的三类问题
金融 AI 区别于通用模型的三条根本要求：
1. **自适应知识整合**（Adaptive Knowledge Integration）：高效吸收演进中的领域知识（监管更新、新金融工具）。
2. **可验证推理**（Verifiable Reasoning）：透明、可审计的推理过程，支撑高风险决策下的信任。
3. **合规遵循**（Compliance Adherence）：保护敏感数据同时满足严格监管要求。

### 2.2 三大核心创新
1. **专业标签引导框架（Label-Guided Framework）**：构建细粒度金融任务标签系统，作为贯穿"数据处理→训练→优化"的主动引导框架。
2. **多维可信保障（Multi-Dimensional Trustworthiness Assurance）**，分三层：
   - (i) **源头可信**：权威金融数据的知识工程；
   - (ii) **合成可信**：可验证的多智能体协同生成；
   - (iii) **治理可信**：全面的数据验证与清洗治理。
3. **效率三件套**：标签引导合成 + 智能筛选（提升数据潜力）、难度感知加权训练（提升训练效率）、**归因闭环**（attribution，提升迭代效率）。

### 2.3 Label System（标签系统）
整个数据流水线的基石，按两个**非正交**维度定义复合标签 `(Scene, Task)`：

- **Scene（场景）维度**：Banking（银行）、Securities（证券）、Insurance（保险）、Trusts（信托）、Mutual Funds（基金）。
- **Task（任务类型）维度**：NER（命名实体识别）、Intent Classification（意图分类）、Slot Filling（槽位填充）、Entity Disambiguation（实体消歧）、Consultation-style QA（咨询式问答）。

形式化：`ℓ = (C, A)`，其中 `C` 为场景类别、`A` 为任务属性。
> ⚠️ **非正交稀疏**：并非所有 Task 都适用于所有 Scene，交集空间稀疏，更贴合真实金融任务分布。复刻时 `data/src/finr1_data/labels.py` 必须建模这种稀疏性。

### 2.4 三级数据治理（→ 映射到 `data/`）

**① Source：可信来源 + 知识工程**
从权威金融机构/监管主体取数，经多阶段预处理得到精炼知识库 `K = {kᵢ}`：
1. Data Extraction（抽取）：NER / 依存解析 / POS 标注，抽取实体、关系、语义结构。
2. Data Normalization（归一化）：统一异构格式，重建数据结构。
3. Data Detoxification（脱毒）：剔除不合规、污染、有害内容。
4. Knowledge Refinement（精炼）：生成高保真精炼知识库。

**② Synthesis：双轨可信多智能体生成 Reasoning 三元组 `(query, thinking, answer)`**
- **Track I 任务导向知识引导**：每个任务标签 `ℓ` 实例化一个生成智能体 `G_ℓ`，基于精炼知识 `k∈K` 生成三元组 `(q,t,a)=G_ℓ(q,k;θ)`；数据集 `D_task = ∪{(q,t,a)}`。
- **Track II 指令自进化（Self-Evolution）**：从种子集 `S₀`（人工策划或任务采样）出发，自进化智能体 `evo` 用反馈信号（多样性指标、任务新颖度、可答性过滤）迭代增强指令：`S_{i+1}=evo(S_i,R;θ_evo)`，三种策略为 Progressive Reasoning Complexity / Structural Diversity / Fitness-Based Filtering。生成 `D_evolution`。
- 最终可信推理数据集：`D_synthesis = D_task ∪ D_evolution`。
  - 进化策略三选：逐步推理复杂度注入、提示变异/重组、仅保留事实正确+逻辑连贯+语言流畅的样本。

**③ Verification & Checking：多模态严格校验**
- **多模型集成验证**：`m` 个独立模型对同 query 作答，用一致性 `consistency(q)`（含语义相似度 `sim`）+ 第三方模型推理校验 `reasoning_valid(q,t)`。
- **人工标注与质控**：分层抽样，金融专家多维评估。
- **Rating 模型**：用 `D_rating = D_ensemble ∪ D_human` 训练评分模型 `score(x)=R(x;RM)`。
- **数据治理清洗**：去重（语义哈希）、脱毒、去污染（剔除与评测基准重叠样本，防数据泄漏）。
- **最终数据集定义**：`D_final = { x∈D_synthesis | verify(x) ∧ clean(x) ∧ score(x) > τ }`，τ 由经验验证+专家共识确定。

> 最终训练集规模：**Fin-R1-300K（约 30 万样本）**。

### 2.5 训练（→ 映射到 `training/`）

**3.1 难度感知加权训练（Weighted Training Framework）**
- 对每个任务标签 `ℓ`，分层采样 `n` 个样本，当前模型生成 `k` 个响应算 `pass@k`；引入 `r` 个不同架构族参考模型算各自 `pass@k`。
- 原始难度权重：`w^(raw)_ℓ = α·(1−pass@k_current(ℓ)) + β·max(0, pass@k_ref(ℓ)−pass@k_current(ℓ)) + γ`。
- **指数平滑**防震荡：`w^(final)_ℓ = λ·w^(prev)_ℓ + (1−λ)·w^(raw)_ℓ`。
- **下界裁剪**：`w^(final)_ℓ ≥ w_min > 0` 保证所有任务最低关注；最后按 `∑|T|` 归一化。
- 加权损失（SFT）：`L_SFT = −(1/N) Σ w̃_ℓᵢ · log p(yᵢ|xᵢ)`；RL 阶段亦按难度加权偏好目标。
- 开销 `O(r·n·k)` 每任务标签，按 epoch 周期执行，不显著影响训练效率。

**3.2 两阶段训练流水线（Two-Stage Training Pipeline）**
- **Stage 1 金融知识与能力注入**：在合成金融推理数据 + 通用推理数据上做 SFT，套用上述加权框架，优先攻坚难样本。
- **Stage 2 难题增强**：GRPO（复杂金融场景多目标决策）+ 针对性 SFT（按 Stage 1 评估暴露的短板补数据）。GRPO 某类任务不收敛时回退到针对性 SFT。

**3.3 归因闭环（Attribution Loop）**
- 用二维标签把预测错误归因到具体场景/任务，pass@1 精度 `Pass@1(ℓ)`。
- **动态归因循环**：① 任务优先级 `P_ℓ = gap_ℓ · η_ℓ · exp(−δ·D_ℓ)`（`gap=max(0,target−acc)`，η 学习效率，D 已分配数据量）；② 数据分配 `B = Σb`，`D_ℓ = B·(P_ℓ/ΣP)`；③ **数据回滚**：性能回退时 `D^(t)_ℓ ← D^(t−1)_ℓ`，持续退化则触发合成数据再生成；④ 反馈给 Synthesis 管线补生成弱势任务数据。
- **目标设定**：`target = SOTA + 固定增量（5 或 10）`。
- 终止：所有任务达标 / 总数据超上限且饱和 / 边际收益低于阈值。

> 复刻中本闭环的状态持久化为 `training/src/finr1_training/attribution.py` 输出的 `attribution.json`。

### 2.6 评测：Finova 基准（1350 题）

Finova = **Financial Nova [Operational · Verifiable · Agent]**，评测真实部署能力，三部分：

| Category | Task | # Samples |
|----------|------|-----------|
| **Agent Capabilities** | Financial Intent Detection（意图检测） | 150 |
| | Financial Slot Recognition（槽位识别） | 360 |
| | Financial Tool Planning（工具规划） | 258 |
| | Financial Expression Generation（表达生成） | 100 |
| | *Subtotal* | *868* |
| **Complex Reasoning** | Math & Coding & Reasoning（数学/代码/推理） | 282 |
| **Safety and Compliance** | 金融安全与合规 | 200 |
| **Total** | | **1350** |

另有通用金融基准 FinEval 1.0、FinanceIQ，及通用推理 MATH-500、GPQA-diamond。

### 2.7 关键结果
- Agentar-Fin-R1-**32B** 在 Finova 达 **69.93** 分，超越 Dianjin-R1-32B、Qwen3-32B、GPT-o1、DeepSeek-R1；金融基准 FinEval 1.0 = 87.70、FinanceIQ = 86.79。
- 8B 档同样在金融任务 SOTA，且通用推理能力可比肩更大模型。
- 消融表明：去掉标签引导或实例加权（300k 数据直接训）性能显著下降。

---

## 3. 论文二：Agentar-DeepFinance-100K（支撑数据集）

**标题**：Agentar-DeepFinance-100K: A Large-Scale Financial Dataset via Systematic Chain-of-Thought Synthesis Optimization
**机构**：Ant Digital Technologies, Ant Group
**作者**：Xiaoke Zhao, Zhaowen Zhou, Lin Chen, Lihong Wang, et al.
**arXiv**：[2507.12901](https://arxiv.org/abs/2507.12901)（v3, 2025-11-09）
**开源**：https://github.com/antgroup/Agentar-DeepFinance-100K
**规模**：**100K** 样本，每个样本带 `(Question, Thinking, Answer)` + 多维元数据（complexity / quality / task type）。

### 3.1 定位
它是**主论文 Source 阶段的知识来源与种子语料库**：用系统化的 CoT 合成优化，为金融推理构建"知识空间"。主论文训练语料中明确包含 `DeepFinance-100K`（见实验章节训练数据构成）。

### 3.2 构建流水线（4 步）
1. **Seed Corpora（种子语料）**：99K 来自开源金融数据集（FinCorpus、Finance-Instruct-500K、FinCUGE、FinQA、FinancialData、Quant-Trading-Instruct 等，经严格过滤去测试集重叠）+ 16K 来自蚂蚁内部带专家标注的专有数据。
2. **Multi-perspective Knowledge Extraction（MKE，多视角知识抽取）**，三法：
   - **Q2A（Direct Curation）**：直接 harvesting 结构化 QA 对，去重+过滤。
   - **A2Q（Counterfactual Augmentation）**：对答案做语义否定/反义替换构造对抗变体，再让 LRM 反推问题，增强因果连通性。
   - **T2Q（CoT Knowledge Mining）**：从 LRM 的 thinking 中抽取隐含知识点，构造 QA，捕捉深层推理依赖。
3. **CoT Sampling & Verification**：对每个 QA 采样多条 CoT，用轻量模型（而非脆弱正则）做答案一致性校验，仅保留数值精度与逻辑一致通过的样本。
4. **Self-Corrective Rewriting（SCR，自校正重写）**：对采样答错的"难题"不直接丢弃，而是：
   - Reflection：对比错误答案与 gold answer，生成含潜在错误的诊断反思 CoT；
   - Rewriting：将反思 CoT 与原推理轨迹合并，续生成新 CoT 与修订答案，再校验；循环至通过或达上限。
   - 效果：把高难样本转化为有效学习信号，CoT 显著变长，推理能力明显提升。

### 3.3 任务构成（6 大领域）
Knowledge QA、NLP、Text Generation、Compliance & Security、Math、Analysis & Interpretation。
其中 Knowledge QA + NLP 合计占 **>70%**；Text Generation 占比最低。实验表明 **CoT 对所有任务类型和难度都有提升，对数学等推理密集任务增益最大**；难样本无 CoT 微调反而下降。

---

## 4. 两篇论文 → 复刻目录映射

| 论文内容 | 复刻目录 | 关键产物 |
|----------|----------|----------|
| Label System (§2.3) | `data/src/finr1_data/labels.py` | `(Scene, Task)` 非正交稀疏标签定义 |
| Source 知识工程 (§2.4①) | `data/src/finr1_data/source/` + DeepFinance-100K 作种子 | 精炼知识库 `K` |
| Synthesis 双轨 (§2.4②) | `data/src/finr1_data/synthesis/` | `(q,t,a)` 三元组 |
| Verification (§2.4③) | `data/src/finr1_data/verification/` | `verify/clean/score` + 去重去污 |
| 全链路编排 | `data/src/finr1_data/pipeline.py` | `D_final`（Fin-R1-300K） |
| 难度感知加权 (§3.1) | `training/src/finr1_training/weighting.py` | `w̃_ℓ`（pass@k+平滑+裁剪） |
| 两阶段训练 (§3.2) | `training/src/finr1_training/stage1_sft.py` / `stage2_grpo.py` | checkpoint |
| 归因闭环 (§3.3) | `training/src/finr1_training/attribution.py` | `attribution.json` |
| Finova 评测 (§2.6) | `training/src/finr1_training/eval/finova/`（待建） | 1350 题评测脚本 |

---

## 5. 参考资源
- Finova 基准代码与数据：https://github.com/antgroup/Finova
- DeepFinance-100K 数据集：https://github.com/antgroup/Agentar-DeepFinance-100K
- 主论文：https://arxiv.org/abs/2507.16802 ｜ PDF：https://arxiv.org/pdf/2507.16802v4
- 数据集论文：https://arxiv.org/abs/2507.12901 ｜ PDF：https://arxiv.org/pdf/2507.12901v3
- 技术解读（火山引擎）：[标签驱动的可信金融大模型训练全流程](https://developer.volcengine.com/articles/7531970102842327090)
- 媒体报道（量子位）：[WAIC 抢先爆料：金融"黑马"大模型超 DeepSeek 刷新 SOTA](https://www.qbitai.com/2025/07/312510.html)

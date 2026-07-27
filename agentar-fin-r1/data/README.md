# 数据复刻：三级数据治理流水线

复刻 Agentar-Fin-R1 的**数据构造**流水线（主论文 §2.3），并复用支撑数据集 **DeepFinance-100K** 的构造方法（数据集论文 §3）作为具体实现。最终产物 = `(query, thinking, answer)` 三元组金标，供 `../training` 的 SFT 使用（经 `train_sft.py --extra-data` 合并）。

## 论文依据

| 论文 | 章节 | 对应代码 |
| --- | --- | --- |
| 主论文 | §2.3.1 Source: 知识工程 | `source/knowledge_engineering.py` |
| 主论文 | §2.3.2 Synthesis: 双轨生成 (eq.2/3/4/5/6) | `synthesis/task_oriented.py` + `synthesis/self_evolution.py` |
| DeepFinance-100K | §3.3 MKE (Q2A/A2Q/T2Q) | `synthesis/mke.py` |
| DeepFinance-100K | §3.4 CoT 采样与校验 | `synthesis/mke.py::sample_cot` |
| DeepFinance-100K | §3.5 自校正重写 SCR | `synthesis/mke.py::scr_rescue` |
| 主论文 | §2.3.3 Verification (eq.7/8/9/10/11) | `verification/verify.py` |

## 三级流水线

```
Source (知识工程)                      → 精炼知识库 K  (§2.3.1)
   │
Synthesis (合成)                        (§2.3.2 ∪ DeepFinance §3)
   ├─ Track I  任务导向生成 (eq.2/3)      : 按 Label System 实例化生成 agent，消费 K
   ├─ Track II 自进化指令 (eq.4/5)        : 反馈驱动把 query 进化成更难推理任务
   └─ MKE + CoT采样 + SCR (§3.3–3.5)      : 以 DeepFinance-100K 为种子语料
   │
Verification (校验+治理)                 (§2.3.3, eq.7–11)
   ├─ 多模型集成一致性 (eq.7) + 推理校验 (eq.8)
   ├─ Rating 打分模型 (eq.9/10)
   └─ 治理：去重 / 脱毒 / 去污染 (eq.11)
   │
D_final = { x ∈ D_synthesis | verify(x) ∧ clean(x) ∧ score(x) > τ }   (eq.11)
```

## 模块

```
src/finr1_data/
├── schema.py                 # KnowledgeUnit / ReasoningTriplet 数据结构
├── labels.py                 # 二维非正交稀疏 Label System (Scene×Task)
├── llm.py                    # LLM 后端抽象：dry-run / openai / hf；轻量答案校验
├── pipeline.py               # 编排 run()：Source→Synthesis→Verification→golden.jsonl
├── source/knowledge_engineering.py   # §2.3.1 四步：抽取(NER/POS)→归一化→脱毒→精炼
├── synthesis/
│   ├── task_oriented.py      # Track I (eq.2/3)
│   ├── self_evolution.py     # Track II (eq.4/5)
│   └── mke.py                # Q2A/A2Q/T2Q + CoT采样 + SCR (§3.3–3.5)
└── verification/verify.py    # 集成校验 + Rating + 治理 (eq.7–11)
```

## 运行

```bash
cd agentar-fin-r1/data

# 1) dry-run（无需 API key / 无网络）：用内置小种子验证全流程
PYTHONPATH=src python -m finr1_data.pipeline --out-dir data/golden --max-seed 20

# 2) 真实生成（OpenAI 兼容端点，如本地 vLLM 起的 Qwen3 蒸馏）
PYTHONPATH=src python -m finr1_data.pipeline --backend openai --model Qwen/Qwen3-8B \
    --backend-kwargs '{"base_url": "http://localhost:8000/v1"}' --max-seed 5000

# 3) 用本地 DeepFinance-100K 副本（离线优先）
DEEPFINANCE_LOCAL=/path/to/deepfinance.parquet \
    PYTHONPATH=src python -m finr1_data.pipeline --out-dir data/golden
```

产物：`data/golden/knowledge.jsonl`（精炼知识库）+ `data/golden/golden.jsonl`（三元组金标）。

## 与训练的关系

论文 §4.2 训练数据 = 合成三元组(300K, 未开源) ∪ **DeepFinance-100K** ∪ 通用推理语料 ∪ Llama-Nemotron/openthoughts。本复刻直接用开源 DeepFinance-100K 作 SFT 的 CoT 语料（见 `../training`），而本目录产出的三元组可经 `--extra-data` 合并进 SFT。**三元组的生成在这里，消费在 `../training`**，二者解耦。

## 待填充（真实实现）

- `source/` 的 NER/POS/依存解析：现为启发式占位，接 spaCy / transformers。
- `verification/` 的 embedding 相似度：现为字符 n-gram 余弦占位，接 sentence-transformers。
- `verification/` 的 Rating 模型：现为启发式打分，接训练好的评分模型（eq.9/10）。
- LLM 后端 `hf`：接本地 transformers/vllm 推理。
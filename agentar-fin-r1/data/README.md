# 数据复刻：三级数据治理

复刻 Agentar-Fin-R1 的**数据构造**流水线，对应论文 §2.3。

## 三级流水线

| 层级 | 模块 | 动作 |
| --- | --- | --- |
| Source | [`src/finr1_data/source/`](src/finr1_data/source/) | 权威金融文档 → NER/POS → 归一化 → 脱毒 → 知识精炼 |
| Synthesis | [`src/finr1_data/synthesis/`](src/finr1_data/synthesis/) | 双轨：任务导向生成 + 自进化指令 |
| Verification | [`src/finr1_data/verification/`](src/finr1_data/verification/) | 多模型投票 + 专家抽样 + Rating → 去重/去污/去泄露 |

入口：[`src/finr1_data/pipeline.py`](src/finr1_data/pipeline.py) 的 `run()`。
标签体系：[`src/finr1_data/labels.py`](src/finr1_data/labels.py)（二维 `(Scene, Task)`）。

## 目标产物

`(query, thinking, answer)` 三元组（对标 Fin-R1-300K），供 `../training` 使用。

## 运行（小规模原型）

```bash
uv run python -m finr1_data.pipeline --source-dir data/raw --out-dir data/golden
```

> 当前 `run()` 与三个 stage 为占位实现，待按论文逐层落地。

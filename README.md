# Agentar-Fin-R1-Reproduce

复现蚂蚁数科 **Agentar-Fin-R1**（基于 Qwen3 的金融推理大模型）的训练与数据流水线，并配套可交互的 Web 全栈。

> 论文：*Agentar-Fin-R1: Enhancing Financial Intelligence through Domain Expertise, Training Efficiency, and Advanced Reasoning*
> https://arxiv.org/abs/2507.16802 （代码未开源，本仓库为自主复刻）

## 本地 Qwen3-0.6B 多智能体对话演示

后端在 `model.mode: local` 下用进程内 transformers 直载 Qwen3-0.6B，完全离线运行。下面是一次跨领域金融问题的真实截图——Coordinator 路由→多位领域专家作答→协调者综合→合规审核→风控→协调者修订，**每个智能体依次出现**（SSE 流式推送）：

![Qwen3-0.6B 多智能体对话](docs/assets/qwen3-multi-agent-chat.png)

每条气泡依次呈现：
- 🏦 **银行专家** — 给出银行侧的判断与 @基金专家
- 🧺 **基金专家** — 给出基金侧的判断与 @合规审核
- 🤖 **协调者** — 综合各专家意见
- ✅ **合规审核** — 给出合规建议与 @风控
- ⚠️ **风控** — 给出风险提示与 @协调者
- 🤖 **协调者** — 修订输出最终自然语言答复

`backend/config/config.yaml` 关键项：

```yaml
model:
  mode: local
  local:
    model_path: D:/models/Qwen3-0.6B  # 可用 MODEL_PATH 环境变量覆盖
```

## 仓库结构

| 目录 | 说明 |
| --- | --- |
| `agentar-fin-r1/` | 复现根目录（论文 / 数据复刻 / 训练复刻） |
| `agentar-fin-r1/paper/` | 论文原文、解读与架构笔记 |
| `agentar-fin-r1/training/` | 训练复刻：难度感知加权 + 两阶段（SFT / GRPO）+ 归因闭环 |
| `backend/` | Python + FastAPI 服务与 Agent 运行时 |
| `frontend/` | React + Vite + TypeScript 交互前端 |

## 技术栈

- **推理**：Qwen3-0.6B（进程内直载，CPU/CUDA 自适应）
- **多智能体**：AgentScope 0.1.6 + 自定义 ExpertBoard 编排
- **后端**：FastAPI + Uvicorn + SQLite + DuckDB 向量检索
- **前端**：React 18 + Vite 5 + TypeScript + @antv/g6 知识图谱
- **训练**：verl 0.8.0 + PyTorch + QLoRA/LoRA
- **编排**：docker-compose 一键起 frontend + backend

## 快速开始

```bash
# Python 侧（uv 工作区：backend / data / training）
cd backend && pip install -r <(uv pip compile pyproject.toml)  # 或 uv sync

# 前端
cd frontend && npm install && npm run dev

# 后端（local 模式：进程内 Qwen3-0.6B）
cd backend && uvicorn app.main:app --reload --port 8000

# 或一键起全套
docker compose up --build
```

前端 `http://localhost:5173`，后端 `http://localhost:8000`。

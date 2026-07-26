# Agentar-Fin-R1-Reproduce

复现蚂蚁数科 **Agentar-Fin-R1**（基于 Qwen3 的金融推理大模型）的训练与数据流水线，并配套可交互的 Web 全栈。

> 论文：*Agentar-Fin-R1: Enhancing Financial Intelligence through Domain Expertise, Training Efficiency, and Advanced Reasoning*
> https://arxiv.org/abs/2507.16802 （代码未开源，本仓库为自主复刻）

## 仓库结构

| 目录 | 说明 |
| --- | --- |
| `agentar-fin-r1/` | 复现根目录（论文 / 数据复刻 / 训练复刻） |
| `agentar-fin-r1/paper/` | 论文原文、解读与架构笔记 |
| `agentar-fin-r1/data/` | 数据复刻：三级数据治理流水线（Source → Synthesis → Verification） |
| `agentar-fin-r1/training/` | 训练复刻：难度感知加权 + 两阶段（SFT / GRPO）+ 归因闭环 |
| `backend/` | Python + FastAPI 服务与 Agent 运行时 |
| `frontend/` | React + Vite + TypeScript 交互前端 |

## 复现技术报告

完整流程记录在 [`agentar-fin-r1/report.md`](agentar-fin-r1/report.md)。

## 技术栈

- 复刻（Python）：uv 工作区；Qwen3-8B + QLoRA/LoRA（**小规模原型**）
- 后端：FastAPI + Uvicorn
- 前端：React + Vite + TypeScript
- 编排：docker-compose 一键起 frontend + backend

## 快速开始

```bash
# Python 侧（uv 工作区：backend / data / training）
uv sync

# 前端
cd frontend && npm install && npm run dev

# 后端
cd backend && uvicorn app.main:app --reload --port 8000

# 或一键起全套
docker compose up --build
```

"""05-后端联调：本地 qwen3-0.6b 端到端集成测试（TestClient）。

前置：
- backend 以 local 模式运行，进程内 transformers 直载 `model.local.model_path`（默认 D:/models/Qwen3-0.6B），无需单独启动模型服务。
- 运行后端用的解释器需含 torch+transformers（如 anaconda python）。
- db.enabled=true（验证落库与续聊）。

运行：
    cd backend
    ../../.workbuddy/binaries/python/envs/default/Scripts/python.exe -m pytest tests/test_integration_qwen.py -q -s
"""
from __future__ import annotations

import os
import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.main import app

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "agentar.db")


@pytest.fixture
def client():
    with TestClient(app) as c:  # 触发 lifespan -> init_db
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok"}


def test_chat_returns_reply(client):
    r = client.post(
        "/api/v1/chat",
        json={"message": "存款保险的最高偿付限额是多少？", "scene": "Banking", "user_id": 1},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["reply"], str) and body["reply"].strip(), "reply 为空"
    assert isinstance(body["compliance_notes"], list)
    assert isinstance(body["risk_flags"], list)
    assert body["conversation_id"] is not None, "带 user_id 应返回 conversation_id"


def test_analyze_structured(client):
    r = client.post(
        "/api/v1/analyze",
        json={"message": "帮我分析：推荐一只稳健的货币基金", "scene": "MutualFunds"},
    )
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["intent"] is None or isinstance(b["intent"], str)
    assert isinstance(b["slots"], dict)
    assert isinstance(b["tool_plan"], list)
    assert isinstance(b["expression"], str)


def test_continuation_and_persistence(client):
    r1 = client.post(
        "/api/v1/chat",
        json={"message": "你好，我想了解理财风险", "scene": "Banking", "user_id": 2},
    )
    assert r1.status_code == 200, r1.text
    cid = r1.json()["conversation_id"]
    assert cid is not None

    r2 = client.post(
        "/api/v1/chat",
        json={"message": "继续讲讲", "scene": "Banking", "user_id": 2, "conversation_id": cid},
    )
    assert r2.status_code == 200, r2.text
    assert r2.json()["conversation_id"] == cid, "续聊 conversation_id 应一致"

    # 落库校验：conversations + messages 应有记录
    conn = sqlite3.connect(os.path.abspath(_DB_PATH))
    n_conv = conn.execute(
        "SELECT COUNT(*) FROM conversations WHERE id=?", (cid,)
    ).fetchone()[0]
    n_msg = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id=?", (cid,)
    ).fetchone()[0]
    conn.close()
    assert n_conv == 1, "会话未落库"
    assert n_msg >= 2, f"消息未落库（期望>=2，实际 {n_msg}）"


def test_degradation_when_model_down(client):
    """模型不可达时链路应返回清晰错误（5xx），而非静默崩溃。

    复用已运行的后端进程；本节依赖模型服务在测试时不可达。
    实现：通过 monkeypatch 让 factory 拿到一个死端点（不触碰已缓存的 agent system
    之外的新请求），断言返回 500 且错误信息包含模型相关关键字。
    """
    import app.config as cfg_mod
    import app.agent.system as sys_mod

    broken = dict(cfg_mod.config)
    broken["model"] = {
        "mode": "local",
        "local": {
            "model_type": "openai_chat",
            "model_name": "qwen3-0.6b",
            "api_key": "EMPTY",
            "base_url": "http://localhost:9999/v1",  # 死端点
            "temperature": 0.3,
            "stream": False,
        },
    }
    cfg_mod.config = cfg_mod.Config(broken)
    # 重置 agent system 单例，使下一次 chat 用坏端点重建（否则命中已缓存的真实模型）
    sys_mod._system = None
    try:
        r = client.post(
            "/api/v1/chat",
            json={"message": "模型挂了会怎样", "scene": "Banking", "user_id": 9},
        )
        assert r.status_code >= 500, f"期望 5xx，实际 {r.status_code}: {r.text}"
        assert ("vLLM" in r.text) or ("模型" in r.text) or ("ModelInvokeError" in r.text), (
            f"错误信息未明确指向模型故障: {r.text}"
        )
    finally:
        # 还原，避免影响同进程其他用例（若有）
        cfg_mod.config = cfg_mod.Config(dict(cfg_mod.config))

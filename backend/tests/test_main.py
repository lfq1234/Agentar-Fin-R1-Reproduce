"""03-传统后端基础层 接口测试。

无 OPENAI_API_KEY / 无 vLLM / db.enabled=false 时也能跑：
- 注入 fake `run`（跳过真实模型调用）；
- 验证 /api/health、/api/v1/chat（无库跳过落库）、/api/v1/analyze。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.agent.system import AgentResult
from app.db.models import User, get_db as real_get_db
from app.main import app
from app.routes.deps import get_current_user


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    async def fake_run(message, scene=None, structured=False, user_id=None, use_personal_docs=False) -> AgentResult:
        if structured:
            return AgentResult(
                reply="分析完成",
                intent="consult",
                slots={"product": "基金"},
                tool_plan=["lookup"],
                expression="好的，为您解答",
            )
        return AgentResult(
            reply=f"[fake] {message}",
            compliance_notes=["已合规"],
            risk_flags=["风险等级中"],
        )

    # services 内 `from app.agent import run as agent_run`，monkeypatch 其模块级绑定。
    monkeypatch.setattr("app.services.chat_service.agent_run", fake_run)
    monkeypatch.setattr("app.services.analyze_service.agent_run", fake_run)

    # 本测试只验证「无库也能跑」的降级路径：用 dependency_overrides 把 get_db 换成
    # 产出 None 的桩，使 chat/analyze 跳过落库（不依赖 config.yaml 的 db.enabled 取值）。
    # 注意：路由用 Depends(get_db) 捕获的是函数对象本身，monkeypatch 模块属性无效，
    # 必须用 FastAPI 的 dependency_overrides 才能生效。
    async def _fake_get_db():
        yield None

    app.dependency_overrides[real_get_db] = _fake_get_db

    # 09：chat 现需鉴权；无库降级路径下用桩用户覆盖 get_current_user，保持测试有效。
    async def _fake_user() -> User:
        return User(id=1, username="tester", is_active=True)

    app.dependency_overrides[get_current_user] = _fake_user

    yield TestClient(app)

    app.dependency_overrides.pop(real_get_db, None)
    app.dependency_overrides.pop(get_current_user, None)
    return TestClient(app)


def test_health(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok"}


def test_chat(client: TestClient) -> None:
    r = client.post("/api/v1/chat", json={"message": "你好", "user_id": 1})
    assert r.status_code == 200
    body = r.json()
    assert "fake" in body["reply"]
    # db.enabled=false → 跳过落库，conversation_id 为 None
    assert body["conversation_id"] is None
    assert body["compliance_notes"] == ["已合规"]
    assert body["risk_flags"] == ["风险等级中"]


def test_analyze(client: TestClient) -> None:
    r = client.post("/api/v1/analyze", json={"message": "基金定投"})
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "consult"
    assert body["slots"]["product"] == "基金"
    assert body["tool_plan"] == ["lookup"]
    assert body["expression"] == "好的，为您解答"

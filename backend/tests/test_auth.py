"""09-用户系统与数据隔离：鉴权接口测试（无 torch，托管/测试环境可跑）。

覆盖：注册 / 重复注册 / 弱密码 / 登录（错密码·未知用户）/ 令牌缺失 / 登录后 me /
两个用户身份隔离。数据库作用域隔离（文档/会话）由 service 层既有测试覆盖，此处
聚焦鉴权链路本身。
"""
from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:  # 触发 lifespan -> init_db + migrate_user_columns
        yield c


def _uniq() -> str:
    return "u" + uuid.uuid4().hex[:10]


def _register(client: TestClient, username: str, password: str = "password123", email: str | None = None):
    body: dict = {"username": username, "password": password}
    if email:
        body["email"] = email
    return client.post("/api/v1/auth/register", json=body)


def _login(client: TestClient, username: str, password: str):
    # OAuth2PasswordRequestForm 走 form-encoded
    return client.post("/api/v1/auth/login/access-token", data={"username": username, "password": password})


def test_register_ok(client: TestClient) -> None:
    r = _register(client, _uniq(), email="c@example.com")
    assert r.status_code == 201, r.text
    body = r.json()
    assert "username" in body and "id" in body
    # 响应不得泄露密码相关字段
    assert "hashed_password" not in body and "password" not in body


def test_register_duplicate(client: TestClient) -> None:
    u = _uniq()
    assert _register(client, u).status_code == 201
    assert _register(client, u).status_code == 409, "重复注册应 409"


def test_register_short_password(client: TestClient) -> None:
    r = _register(client, _uniq(), password="short")
    assert r.status_code == 422, "密码 <8 位应被 Pydantic 拒绝"


def test_login_wrong_password(client: TestClient) -> None:
    u = _uniq()
    _register(client, u, password="password123")
    r = _login(client, u, "wrongpass")
    assert r.status_code == 400, r.text
    assert "账号或密码错误" in r.text


def test_login_unknown_user(client: TestClient) -> None:
    r = _login(client, "ghost_" + _uniq(), "password123")
    assert r.status_code == 400, "未知用户应 400（与错密码同口径，防枚举）"


def test_me_requires_token(client: TestClient) -> None:
    # 缺失令牌：OAuth2 方案标准返回 401（由 OAuth2PasswordBearer 在解码前拦截）。
    r = client.get("/api/v1/auth/me")
    assert r.status_code == 401, "无令牌应 401"


def test_me_invalid_token_403(client: TestClient) -> None:
    # 伪造/过期令牌：进入 get_current_user 解码阶段，返回 403（评审 M3 统一口径）。
    r = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 403, "无效令牌应 403"


def test_login_and_me_ok(client: TestClient) -> None:
    u = _uniq()
    _register(client, u, password="password123")
    r = _login(client, u, "password123")
    assert r.status_code == 200, r.text
    token = r.json().get("access_token")
    assert token
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200, me.text
    assert me.json()["username"] == u


def test_two_users_isolated(client: TestClient) -> None:
    u1, u2 = _uniq(), _uniq()
    _register(client, u1, password="password123")
    _register(client, u2, password="password123")
    t1 = _login(client, u1, "password123").json()["access_token"]
    t2 = _login(client, u2, "password123").json()["access_token"]
    m1 = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {t1}"}).json()
    m2 = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {t2}"}).json()
    assert m1["id"] != m2["id"]
    assert m1["username"] == u1 and m2["username"] == u2

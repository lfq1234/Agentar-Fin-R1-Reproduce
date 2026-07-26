from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health() -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_chat() -> None:
    r = client.post("/v1/chat", json={"message": "hello"})
    assert r.status_code == 200
    assert "stub" in r.json()["reply"]

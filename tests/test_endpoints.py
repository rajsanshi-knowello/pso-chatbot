from fastapi.testclient import TestClient


# ── /health ──────────────────────────────────────────────────────────────────

def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── /review ───────────────────────────────────────────────────────────────────

_REVIEW_BODY = {"document_url": "https://example.com/doc.docx", "session_id": "abc123"}


def test_review_ok(client: TestClient, auth_headers: dict, mock_parse_document) -> None:
    r = client.post("/review", json=_REVIEW_BODY, headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == "abc123"
    assert data["document_overview"]["length_words"] == 42
    assert "category_results" in data
    assert len(data["category_results"]) == 10
    assert "priority_changes" in data
    assert "next_steps" in data
    assert "metadata" in data


def test_review_no_auth(client: TestClient) -> None:
    r = client.post("/review", json=_REVIEW_BODY)
    assert r.status_code == 401


def test_review_invalid_body(client: TestClient, auth_headers: dict) -> None:
    r = client.post("/review", json={}, headers=auth_headers)
    assert r.status_code == 422


# ── /chat ─────────────────────────────────────────────────────────────────────

_CHAT_BODY = {"message": "What should I fix first?", "session_id": "abc123"}


def test_chat_ok(client: TestClient, auth_headers: dict) -> None:
    r = client.post("/chat", json=_CHAT_BODY, headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == "abc123"
    assert "reply" in data


def test_chat_no_auth(client: TestClient) -> None:
    r = client.post("/chat", json=_CHAT_BODY)
    assert r.status_code == 401


def test_chat_invalid_body(client: TestClient, auth_headers: dict) -> None:
    r = client.post("/chat", json={}, headers=auth_headers)
    assert r.status_code == 422

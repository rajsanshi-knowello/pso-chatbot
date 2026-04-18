import os
from fastapi.testclient import TestClient

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
_DOCX_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_PDF_CT = "application/pdf"


def test_upload_docx_ok(client: TestClient, auth_headers: dict) -> None:
    with open(os.path.join(_FIXTURES, "sample.docx"), "rb") as f:
        r = client.post(
            "/review/upload",
            files={"file": ("sample.docx", f, _DOCX_CT)},
            data={"session_id": "up1"},
            headers=auth_headers,
        )
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == "up1"
    assert data["document_overview"]["length_words"] > 0
    assert len(data["category_results"]) == 10


def test_upload_pdf_ok(client: TestClient, auth_headers: dict) -> None:
    with open(os.path.join(_FIXTURES, "sample.pdf"), "rb") as f:
        r = client.post(
            "/review/upload",
            files={"file": ("sample.pdf", f, _PDF_CT)},
            data={"session_id": "up2"},
            headers=auth_headers,
        )
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == "up2"
    assert data["document_overview"]["length_words"] > 0


def test_upload_missing_file(client: TestClient, auth_headers: dict) -> None:
    r = client.post(
        "/review/upload",
        data={"session_id": "up3"},
        headers=auth_headers,
    )
    assert r.status_code == 422


def test_upload_unsupported_type(client: TestClient, auth_headers: dict) -> None:
    r = client.post(
        "/review/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
        data={"session_id": "up4"},
        headers=auth_headers,
    )
    assert r.status_code == 415


def test_upload_oversized(client: TestClient, auth_headers: dict) -> None:
    big = b"x" * (11 * 1024 * 1024)
    r = client.post(
        "/review/upload",
        files={"file": ("big.docx", big, _DOCX_CT)},
        data={"session_id": "up5"},
        headers=auth_headers,
    )
    assert r.status_code == 413


def test_upload_no_auth(client: TestClient) -> None:
    r = client.post(
        "/review/upload",
        files={"file": ("sample.docx", b"data", _DOCX_CT)},
        data={"session_id": "up6"},
    )
    assert r.status_code == 401

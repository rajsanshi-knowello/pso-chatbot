import io
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException

from app.services.document_parser import ParsedDocument, parse_document

# ── fixtures ──────────────────────────────────────────────────────────────────

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="session")
def docx_bytes() -> bytes:
    with open(os.path.join(_FIXTURES, "sample.docx"), "rb") as f:
        return f.read()


@pytest.fixture(scope="session")
def pdf_bytes() -> bytes:
    with open(os.path.join(_FIXTURES, "sample.pdf"), "rb") as f:
        return f.read()


# ── mock helper ───────────────────────────────────────────────────────────────

def _mock_client(
    content: bytes,
    content_type: str,
    status: int = 200,
    content_length: int | None = None,
) -> MagicMock:
    mock_response = MagicMock()
    mock_response.status_code = status
    mock_response.content = content
    cl = content_length if content_length is not None else len(content)
    mock_response.headers = {"content-type": content_type, "content-length": str(cl)}

    mock_session = MagicMock()
    mock_session.get = AsyncMock(return_value=mock_response)

    mock_cm = MagicMock()
    mock_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_cm.__aexit__ = AsyncMock(return_value=False)
    return mock_cm


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_parse_docx(docx_bytes: bytes) -> None:
    ct = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    with patch("app.services.document_parser.httpx.AsyncClient", return_value=_mock_client(docx_bytes, ct)):
        result = await parse_document("https://example.com/sample.docx")
    assert result.file_type == "docx"
    assert result.word_count > 0
    assert result.has_tables is True
    assert result.heading_count >= 1


@pytest.mark.anyio
async def test_parse_pdf(pdf_bytes: bytes) -> None:
    with patch("app.services.document_parser.httpx.AsyncClient", return_value=_mock_client(pdf_bytes, "application/pdf")):
        result = await parse_document("https://example.com/sample.pdf")
    assert result.file_type == "pdf"
    assert result.word_count > 0


@pytest.mark.anyio
async def test_reject_oversized() -> None:
    mock_cm = _mock_client(b"x", "application/pdf", content_length=11 * 1024 * 1024)
    with patch("app.services.document_parser.httpx.AsyncClient", return_value=mock_cm):
        with pytest.raises(HTTPException) as exc:
            await parse_document("https://example.com/big.pdf")
    assert exc.value.status_code == 413


@pytest.mark.anyio
async def test_url_404() -> None:
    mock_cm = _mock_client(b"", "text/html", status=404)
    with patch("app.services.document_parser.httpx.AsyncClient", return_value=mock_cm):
        with pytest.raises(HTTPException) as exc:
            await parse_document("https://example.com/missing.pdf")
    assert exc.value.status_code == 502


@pytest.mark.anyio
async def test_unsupported_file_type() -> None:
    mock_cm = _mock_client(b"hello", "text/plain")
    with patch("app.services.document_parser.httpx.AsyncClient", return_value=mock_cm):
        with pytest.raises(HTTPException) as exc:
            await parse_document("https://example.com/notes.txt")
    assert exc.value.status_code == 415

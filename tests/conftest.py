import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

# Must be set before app is imported so pydantic-settings picks it up
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")

from app.main import app  # noqa: E402
from app.services.document_parser import ParsedDocument  # noqa: E402


@pytest.fixture(params=["asyncio"])
def anyio_backend(request):
    return request.param


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="session")
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": "test-key"}


@pytest.fixture
def mock_parse_document():
    dummy = ParsedDocument(
        text="Sample text for testing.",
        word_count=42,
        paragraph_count=2,
        heading_count=1,
        file_type="docx",
        has_tables=False,
    )
    with patch("app.routes.review.parse_document", AsyncMock(return_value=dummy)):
        yield dummy

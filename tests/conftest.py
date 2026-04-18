import os

import pytest
from fastapi.testclient import TestClient

# Must be set before app is imported so pydantic-settings picks it up
os.environ.setdefault("API_KEY", "test-key")

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="session")
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": "test-key"}

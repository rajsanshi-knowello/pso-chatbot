"""Tests for /chat endpoint, session store, and chat_graph."""
import time
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import contextmanager

from app.services.session_store import SessionStore, build_document_context
from app.agents.chat_graph import ChatState
from app.models.responses import (
    ReviewResponse, DocumentOverview, CategoryResult, PriorityChange, ReviewMetadata, Finding
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_review_response(session_id: str = "s1") -> ReviewResponse:
    cat5_finding = Finding(
        passage='"double quoted text"',
        issue="Double quotes used instead of single quotes",
        suggested_fix="'double quoted text'",
    )
    return ReviewResponse(
        session_id=session_id,
        document_overview=DocumentOverview(
            document_type="Policy document",
            audience="VET sector",
            length_words=500,
            strengths=["Good use of active voice."],
        ),
        category_results=[
            CategoryResult(
                category_id=5, category_name="Punctuation",
                compliant=False, severity="medium", findings=[cat5_finding],
            ),
            *[
                CategoryResult(
                    category_id=i, category_name=f"Cat{i}",
                    compliant=True, severity="none", findings=[],
                )
                for i in [1, 2, 3, 4, 6, 7, 8, 9, 10]
            ],
        ],
        priority_changes=[
            PriorityChange(rank=1, category_id=5, description="Replace double quotes with single quotes"),
        ],
        overall_summary="The document is mostly compliant with minor punctuation issues.",
        next_steps="Fix punctuation.",
        metadata=ReviewMetadata(
            model_used="gemini-2.5-flash",
            tokens_total=1500,
            latency_ms=3000,
            processed_at="2026-04-19T00:00:00+00:00",
        ),
    )


@contextmanager
def mock_chat_gemini(reply_text: str, tokens: int = 100):
    mock_response = MagicMock()
    mock_response.text = reply_text
    mock_response.usage_metadata.total_token_count = tokens

    with patch("app.agents.chat_graph.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_class.return_value = mock_client
        yield mock_client


# ── SessionStore unit tests ───────────────────────────────────────────────────

def test_session_store_creates_new_session():
    store = SessionStore()
    session = store.get_or_create("abc")
    assert session.session_id == "abc"
    assert session.document_text is None
    assert session.message_history == []


def test_session_store_returns_same_session():
    store = SessionStore()
    s1 = store.get_or_create("abc")
    s1.document_text = "hello"
    s2 = store.get_or_create("abc")
    assert s2.document_text == "hello"


def test_session_store_stores_review():
    store = SessionStore()
    store.store_review("abc", document_text="Document text here.", document_context="CONTEXT")
    session = store.get_or_create("abc")
    assert session.document_text == "Document text here."
    assert session.document_context == "CONTEXT"


def test_session_store_appends_messages():
    store = SessionStore()
    store.append_message("abc", "user", "Hello")
    store.append_message("abc", "assistant", "Hi there")
    session = store.get_or_create("abc")
    assert len(session.message_history) == 2
    assert session.message_history[0] == {"role": "user", "content": "Hello"}
    assert session.message_history[1] == {"role": "assistant", "content": "Hi there"}


def test_session_store_caps_history_at_20():
    store = SessionStore()
    for i in range(25):
        store.append_message("abc", "user", f"msg {i}")
    session = store.get_or_create("abc")
    assert len(session.message_history) == 20


def test_session_store_evicts_expired_session():
    store = SessionStore()
    session = store.get_or_create("abc")
    session.last_active = time.time() - 7201  # expired 1 second past TTL
    # Accessing the session should evict and recreate it
    fresh = store.get_or_create("abc")
    assert fresh.document_text is None  # fresh session, no prior data


def test_session_store_does_not_evict_active_session():
    store = SessionStore()
    store.store_review("abc", document_text="doc", document_context="ctx")
    session = store.get_or_create("abc")
    session.last_active = time.time() - 3600  # 1 hour ago — still within TTL
    still_there = store.get_or_create("abc")
    assert still_there.document_text == "doc"


# ── build_document_context ────────────────────────────────────────────────────

def test_build_document_context_includes_findings():
    review = _make_review_response()
    context = build_document_context(review)
    assert "Category 5" in context
    assert "Punctuation" in context
    assert "MEDIUM" in context
    assert "double quoted text" in context


def test_build_document_context_includes_priority_changes():
    review = _make_review_response()
    context = build_document_context(review)
    assert "TOP PRIORITY CHANGES" in context
    assert "single quotes" in context


def test_build_document_context_includes_compliant_categories():
    review = _make_review_response()
    context = build_document_context(review)
    assert "COMPLIANT" in context


# ── chat_graph unit tests ─────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_chat_node_returns_reply():
    from app.agents.chat_graph import _chat_node

    state: ChatState = {
        "system_prompt": "You are a helpful assistant.",
        "document_context": "",
        "message_history": [],
        "user_message": "Hello",
    }

    with mock_chat_gemini("Hello! How can I help?", tokens=50):
        result = await _chat_node(state)

    assert result["reply"] == "Hello! How can I help?"
    assert result["tokens_used"] == 50
    assert isinstance(result["latency_ms"], int)


@pytest.mark.anyio
async def test_chat_node_includes_history_in_call():
    from app.agents.chat_graph import _chat_node
    from google.genai import types as genai_types

    state: ChatState = {
        "system_prompt": "You are a helpful assistant.",
        "document_context": "",
        "message_history": [
            {"role": "user", "content": "What is Category 5?"},
            {"role": "assistant", "content": "Category 5 covers punctuation rules."},
        ],
        "user_message": "Can you give me an example?",
    }

    with mock_chat_gemini("Sure! Single quotes are used instead of double quotes.") as mock_client:
        result = await _chat_node(state)

    call_args = mock_client.models.generate_content.call_args
    contents = call_args.kwargs.get("contents") or call_args.args[0] if call_args.args else call_args.kwargs["contents"]
    # 2 history messages + 1 current = 3 Content items
    assert len(contents) == 3


@pytest.mark.anyio
async def test_chat_node_includes_document_context_in_system():
    from app.agents.chat_graph import _chat_node

    state: ChatState = {
        "system_prompt": "You are the PSO assistant.",
        "document_context": "REVIEWED DOCUMENT — 500 words\n• Category 5: MEDIUM — 1 finding",
        "message_history": [],
        "user_message": "Tell me about the punctuation issues.",
    }

    with mock_chat_gemini("The document has one punctuation issue...") as mock_client:
        await _chat_node(state)

    call_args = mock_client.models.generate_content.call_args
    config = call_args.kwargs.get("config")
    assert "DOCUMENT CONTEXT" in config.system_instruction
    assert "Category 5" in config.system_instruction


@pytest.mark.anyio
async def test_chat_node_fallback_greeting_on_failure():
    from app.agents.chat_graph import _chat_node, FALLBACK_GREETING

    state: ChatState = {
        "system_prompt": "You are the PSO assistant.",
        "document_context": "",
        "message_history": [],
        "user_message": "Hello",
    }

    with patch("app.agents.chat_graph.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("quota exceeded")
        mock_client_class.return_value = mock_client
        result = await _chat_node(state)

    assert result["reply"] == FALLBACK_GREETING
    assert result["tokens_used"] == 0


@pytest.mark.anyio
async def test_chat_node_fallback_error_reply_when_session_has_document():
    from app.agents.chat_graph import _chat_node, FALLBACK_REPLY

    state: ChatState = {
        "system_prompt": "You are the PSO assistant.",
        "document_context": "REVIEWED DOCUMENT — 500 words",  # has doc context
        "message_history": [],
        "user_message": "Explain category 5",
    }

    with patch("app.agents.chat_graph.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("timeout")
        mock_client_class.return_value = mock_client
        result = await _chat_node(state)

    assert result["reply"] == FALLBACK_REPLY


# ── /chat endpoint integration tests ─────────────────────────────────────────

def test_chat_endpoint_no_document_returns_200(client, auth_headers):
    """Chat with a session that has no document should return 200 with a reply."""
    with mock_chat_gemini("Hello! Please upload a document to get started."):
        r = client.post(
            "/chat",
            json={"message": "Hi", "session_id": "new-session-xyz"},
            headers=auth_headers,
        )
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == "new-session-xyz"
    assert len(data["reply"]) > 0


def test_chat_endpoint_retains_session_history(client, auth_headers):
    """Two chat turns in the same session should build up history."""
    session_id = "history-test-session"

    with mock_chat_gemini("Nice to meet you!"):
        client.post("/chat", json={"message": "Hello", "session_id": session_id}, headers=auth_headers)

    with mock_chat_gemini("Category 5 covers punctuation."):
        r = client.post(
            "/chat",
            json={"message": "Tell me about Category 5", "session_id": session_id},
            headers=auth_headers,
        )

    assert r.status_code == 200
    # Verify history was stored (session has 2 user + 2 assistant messages = 4)
    from app.services.session_store import session_store
    session = session_store.get_or_create(session_id)
    assert len(session.message_history) == 4


def test_chat_endpoint_uses_document_context(client, auth_headers):
    """After a review, the chat should receive document context."""
    from app.services.session_store import session_store

    session_id = "doc-context-session"
    session_store.store_review(session_id, "Sample doc text.", "REVIEWED DOCUMENT — 100 words\n• Category 5: MEDIUM")

    with mock_chat_gemini("The document has Category 5 issues.") as mock_client:
        r = client.post(
            "/chat",
            json={"message": "What were the main issues?", "session_id": session_id},
            headers=auth_headers,
        )

    assert r.status_code == 200
    # Verify that the Gemini call received the document context in system_instruction
    call_args = mock_client.models.generate_content.call_args
    config = call_args.kwargs.get("config")
    assert "Category 5" in config.system_instruction

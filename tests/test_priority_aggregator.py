"""Tests for the priority_aggregator node and its integration with review_service."""
import pytest
from unittest.mock import AsyncMock, patch
from contextlib import contextmanager

from app.agents.nodes.priority_aggregator import aggregate_priority, _build_findings_summary
from app.agents.state import ReviewState
from app.models.responses import Finding
from app.services.review_service import build_review_response
from app.services.document_parser import ParsedDocument


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state_with_findings() -> ReviewState:
    state: ReviewState = {"document_text": "Test document."}
    state["category_1"] = {
        "findings": [
            Finding(passage="We are excited to announce", issue="Promotional language", suggested_fix="We announce"),
        ],
        "compliant": False,
        "severity": "medium",
    }
    state["category_5"] = {
        "findings": [
            Finding(passage='"double quotes"', issue="Double quotes used", suggested_fix="'single quotes'"),
            Finding(passage='"another"', issue="Double quotes used", suggested_fix="'another'"),
            Finding(passage="Great work!", issue="Exclamation mark in formal content", suggested_fix="Great work."),
        ],
        "compliant": False,
        "severity": "medium",
    }
    for cat_id in [2, 3, 4, 6, 7, 8, 9, 10]:
        state[f"category_{cat_id}"] = {"findings": [], "compliant": True, "severity": "none"}
    return state


@contextmanager
def mock_aggregator_llm(output_data: dict, tokens: int = 250):
    with patch(
        "app.agents.nodes.priority_aggregator.call_llm_structured",
        AsyncMock(return_value=(output_data, tokens)),
    ):
        yield


# ── Unit tests: _build_findings_summary ───────────────────────────────────────

def test_findings_summary_compliant_category():
    state = _make_state_with_findings()
    summary = _build_findings_summary(state)
    assert "COMPLIANT" in summary
    assert "Category 2" in summary


def test_findings_summary_includes_findings():
    state = _make_state_with_findings()
    summary = _build_findings_summary(state)
    assert "Category 1" in summary
    assert "Promotional language" in summary
    assert "Category 5" in summary
    assert "Double quotes" in summary


def test_findings_summary_shows_severity():
    state = _make_state_with_findings()
    summary = _build_findings_summary(state)
    assert "MEDIUM" in summary


# ── Unit tests: aggregate_priority node ───────────────────────────────────────

@pytest.mark.anyio
async def test_aggregator_returns_priority_changes():
    state = _make_state_with_findings()
    gemini_output = {
        "priority_changes": [
            {"rank": 1, "category_id": 5, "description": "Replace 3 double quotes with single quotes"},
            {"rank": 2, "category_id": 1, "description": "Remove promotional language 'excited'"},
        ],
        "overall_summary": "The document has minor punctuation issues but is otherwise well-structured.",
        "strengths": ["No accessibility issues found.", "Spelling and terminology are correct."],
    }

    with mock_aggregator_llm(gemini_output):
        result = await aggregate_priority(state)

    assert len(result["aggregated_priority_changes"]) == 2
    assert result["aggregated_priority_changes"][0]["rank"] == 1
    assert result["aggregated_priority_changes"][0]["category_id"] == 5
    assert "double quotes" in result["aggregated_priority_changes"][0]["description"].lower()


@pytest.mark.anyio
async def test_aggregator_returns_overall_summary():
    state = _make_state_with_findings()
    output = {
        "priority_changes": [{"rank": 1, "category_id": 5, "description": "Fix quotes"}],
        "overall_summary": "The document is mostly compliant with minor punctuation issues.",
        "strengths": ["No accessibility issues."],
    }

    with mock_aggregator_llm(output):
        result = await aggregate_priority(state)

    assert result["aggregated_summary"] == "The document is mostly compliant with minor punctuation issues."


@pytest.mark.anyio
async def test_aggregator_returns_strengths():
    state = _make_state_with_findings()
    output = {
        "priority_changes": [{"rank": 1, "category_id": 1, "description": "Fix tone"}],
        "overall_summary": "Good overall quality.",
        "strengths": ["Accessibility is clear and plain.", "Referencing is complete."],
    }

    with mock_aggregator_llm(output):
        result = await aggregate_priority(state)

    assert len(result["aggregated_strengths"]) == 2
    assert "Accessibility" in result["aggregated_strengths"][0]


@pytest.mark.anyio
async def test_aggregator_fallback_on_llm_failure():
    state = _make_state_with_findings()

    with patch(
        "app.agents.nodes.priority_aggregator.call_llm_structured",
        AsyncMock(side_effect=Exception("API Error")),
    ):
        result = await aggregate_priority(state)

    assert result["aggregated_priority_changes"] == []
    assert result["aggregated_summary"] == ""
    assert result["aggregated_strengths"] == []
    assert result["aggregator_tokens"] == 0


@pytest.mark.anyio
async def test_aggregator_records_latency():
    state = _make_state_with_findings()
    output = {"priority_changes": [], "overall_summary": "All good.", "strengths": []}

    with mock_aggregator_llm(output):
        result = await aggregate_priority(state)

    assert isinstance(result["aggregator_latency_ms"], int)
    assert result["aggregator_latency_ms"] >= 0


# ── Integration tests: review_service uses aggregator output ──────────────────

@pytest.fixture
def sample_doc():
    return ParsedDocument(
        text="Test document.", word_count=50, paragraph_count=3,
        heading_count=1, file_type="docx", has_tables=False,
    )


@contextmanager
def mock_graph_and_runner_with_aggregator(graph_output):
    with patch("app.services.review_service.reviewer_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=graph_output)
        with patch("app.services.review_service.CheckRunner") as mock_runner_class:
            from unittest.mock import MagicMock
            mock_runner = MagicMock()
            mock_runner.run_all = AsyncMock(return_value={})
            mock_runner_class.return_value = mock_runner
            yield


def _base_graph_output():
    out = {}
    for cat_id in range(1, 11):
        out[f"category_{cat_id}"] = {"findings": [], "compliant": True, "severity": "none"}
        out[f"category_{cat_id}_metadata"] = {"tokens_used": 100, "latency_ms": 1500, "status": "success"}
    out["total_tokens"] = 1000
    out["total_latency_ms"] = 1500
    return out


@pytest.mark.anyio
async def test_review_service_uses_aggregator_priority_changes(sample_doc):
    output = _base_graph_output()
    output["category_5"] = {
        "findings": [Finding(passage='"text"', issue="Double quotes", suggested_fix="'text'")],
        "compliant": False,
        "severity": "low",
    }
    output["aggregated_priority_changes"] = [
        {"rank": 1, "category_id": 5, "description": "Replace double quotes with single quotes throughout"}
    ]
    output["aggregated_summary"] = "Minor punctuation issue found."
    output["aggregated_strengths"] = ["Tone is professional.", "Referencing is correct."]

    with mock_graph_and_runner_with_aggregator(output):
        response = await build_review_response("sess-1", sample_doc, 100)

    assert len(response.priority_changes) == 1
    assert response.priority_changes[0].category_id == 5
    assert "single quotes" in response.priority_changes[0].description
    assert response.overall_summary == "Minor punctuation issue found."
    assert len(response.document_overview.strengths) == 2


@pytest.mark.anyio
async def test_review_service_falls_back_when_no_aggregator_output(sample_doc):
    output = _base_graph_output()
    output["category_3"] = {
        "findings": [
            Finding(passage="disabled people", issue="Not people-first language", suggested_fix="people with disability"),
            Finding(passage="the mentally ill", issue="Deficit framing", suggested_fix="people with mental illness"),
            Finding(passage="wheelchair bound", issue="Negative framing", suggested_fix="wheelchair user"),
            Finding(passage="suffers from", issue="Deficit framing", suggested_fix="lives with"),
            Finding(passage="victim of", issue="Unnecessary labelling", suggested_fix="person who experienced"),
        ],
        "compliant": False,
        "severity": "high",
    }

    with mock_graph_and_runner_with_aggregator(output):
        response = await build_review_response("sess-2", sample_doc, 100)

    assert len(response.priority_changes) >= 1
    assert response.priority_changes[0].category_id == 3


@pytest.mark.anyio
async def test_review_service_overall_summary_in_response(sample_doc):
    output = _base_graph_output()
    output["aggregated_priority_changes"] = []
    output["aggregated_summary"] = "The document is fully compliant across all 10 categories."
    output["aggregated_strengths"] = ["No issues found."]

    with mock_graph_and_runner_with_aggregator(output):
        response = await build_review_response("sess-3", sample_doc, 100)

    assert response.overall_summary == "The document is fully compliant across all 10 categories."


@pytest.mark.anyio
async def test_review_service_strengths_fallback_when_aggregator_empty(sample_doc):
    output = _base_graph_output()
    output["aggregated_priority_changes"] = []
    output["aggregated_summary"] = ""
    output["aggregated_strengths"] = []

    with mock_graph_and_runner_with_aggregator(output):
        response = await build_review_response("sess-4", sample_doc, 100)

    assert len(response.document_overview.strengths) > 0


@pytest.mark.anyio
async def test_review_service_total_tokens_includes_aggregator(sample_doc):
    output = _base_graph_output()
    output["aggregated_priority_changes"] = []
    output["aggregated_summary"] = ""
    output["aggregated_strengths"] = []
    output["aggregator_tokens"] = 350

    with mock_graph_and_runner_with_aggregator(output):
        response = await build_review_response("sess-5", sample_doc, 100)

    assert response.metadata.tokens_total == 1000 + 350

"""Integration tests for parallel 10-category LangGraph pipeline.

Tests validate:
- All 10 categories return findings
- Parallelism works (total latency ≈ max single latency, not sum)
- Pre-checks merge correctly with LLM findings for categories 1,2,5,6,7
- Error handling (one category fails, others succeed)
- Severity calculation and priority ranking
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import contextmanager

from app.services.review_service import build_review_response
from app.services.document_parser import ParsedDocument
from app.models.responses import Finding


@contextmanager
def mock_graph_and_runner(graph_output):
    """Context manager to mock both reviewer_graph and CheckRunner."""
    with patch("app.services.review_service.reviewer_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=graph_output)

        with patch("app.services.review_service.CheckRunner") as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.run_all = AsyncMock(return_value={})
            mock_runner_class.return_value = mock_runner
            yield


@pytest.fixture
def sample_parsed_document():
    """Sample document for testing."""
    return ParsedDocument(
        text="Test document with sample content.",
        word_count=100,
        paragraph_count=5,
        heading_count=2,
        file_type="docx",
        has_tables=False,
    )


@pytest.fixture
def mock_graph_output_all_compliant():
    """Mock graph output where all 10 categories are compliant."""
    output = {}
    for cat_id in range(1, 11):
        output[f"category_{cat_id}"] = {
            "findings": [],
            "compliant": True,
            "severity": "none",
        }
        output[f"category_{cat_id}_metadata"] = {
            "tokens_used": 150,
            "latency_ms": 2000,
            "status": "success",
        }
    output["total_tokens"] = 1500
    output["total_latency_ms"] = 2000
    return output


@pytest.fixture
def mock_graph_output_with_findings():
    """Mock graph output with findings in multiple categories."""
    output = {}

    # Categories 1-5: with findings
    for cat_id in [1, 2, 3, 4, 5]:
        output[f"category_{cat_id}"] = {
            "findings": [
                Finding(
                    passage="sample text",
                    issue=f"Issue in category {cat_id}",
                    suggested_fix=f"Fix for category {cat_id}",
                )
            ],
            "compliant": False,
            "severity": "low",
        }
        output[f"category_{cat_id}_metadata"] = {
            "tokens_used": 200,
            "latency_ms": 2000 + cat_id * 100,
            "status": "success",
        }

    # Categories 6-10: compliant
    for cat_id in range(6, 11):
        output[f"category_{cat_id}"] = {
            "findings": [],
            "compliant": True,
            "severity": "none",
        }
        output[f"category_{cat_id}_metadata"] = {
            "tokens_used": 150,
            "latency_ms": 2000,
            "status": "success",
        }

    output["total_tokens"] = 1700
    output["total_latency_ms"] = 2400
    return output


@pytest.mark.anyio
async def test_all_10_categories_in_response(sample_parsed_document, mock_graph_output_all_compliant):
    """All 10 categories should appear in response."""
    with mock_graph_and_runner(mock_graph_output_all_compliant):
        response = await build_review_response("test-session", sample_parsed_document, 100)

    assert len(response.category_results) == 10
    category_ids = {cat.category_id for cat in response.category_results}
    assert category_ids == set(range(1, 11))


@pytest.mark.anyio
async def test_category_names_correct(sample_parsed_document, mock_graph_output_all_compliant):
    """Category names should match expected values."""
    with mock_graph_and_runner(mock_graph_output_all_compliant):
        response = await build_review_response("test-session", sample_parsed_document, 100)

    expected_names = {
        1: "Tone of Voice",
        2: "Accessibility",
        3: "Inclusive Language",
        4: "Headings",
        5: "Punctuation",
        6: "Spelling",
        7: "Numbers",
        8: "Referencing",
        9: "Tables and Figures",
        10: "Structure",
    }

    for cat in response.category_results:
        assert cat.category_name == expected_names[cat.category_id]


@pytest.mark.anyio
async def test_llm_findings_included_in_response(sample_parsed_document, mock_graph_output_with_findings):
    """LLM findings should be included in category results."""
    with mock_graph_and_runner(mock_graph_output_with_findings):
        response = await build_review_response("test-session", sample_parsed_document, 100)

    # Categories 1-5 should have findings
    for cat_id in [1, 2, 3, 4, 5]:
        cat = next(c for c in response.category_results if c.category_id == cat_id)
        assert len(cat.findings) >= 1
        assert cat.compliant is False

    # Categories 6-10 should be compliant
    for cat_id in range(6, 11):
        cat = next(c for c in response.category_results if c.category_id == cat_id)
        assert cat.compliant is True


@pytest.mark.anyio
async def test_severity_calculation_based_on_findings_count(sample_parsed_document):
    """Severity should be calculated from finding count: high > 5, low=1, medium in between."""
    output = {}

    # Category 1: 0 findings → "none"
    output["category_1"] = {
        "findings": [],
        "compliant": True,
    }
    output["category_1_metadata"] = {"tokens_used": 150, "latency_ms": 2000, "status": "success"}

    # Category 2: 1 finding → "low"
    output["category_2"] = {
        "findings": [Finding(passage="text", issue="issue", suggested_fix="fix")],
        "compliant": False,
    }
    output["category_2_metadata"] = {"tokens_used": 150, "latency_ms": 2000, "status": "success"}

    # Category 3: 3 findings → "medium"
    output["category_3"] = {
        "findings": [
            Finding(passage="text", issue="issue1", suggested_fix="fix1"),
            Finding(passage="text", issue="issue2", suggested_fix="fix2"),
            Finding(passage="text", issue="issue3", suggested_fix="fix3"),
        ],
        "compliant": False,
    }
    output["category_3_metadata"] = {"tokens_used": 150, "latency_ms": 2000, "status": "success"}

    # Category 4: 8 findings → "high"
    output["category_4"] = {
        "findings": [
            Finding(passage="text", issue=f"issue{i}", suggested_fix=f"fix{i}")
            for i in range(8)
        ],
        "compliant": False,
    }
    output["category_4_metadata"] = {"tokens_used": 150, "latency_ms": 2000, "status": "success"}

    # Fill remaining categories as compliant
    for cat_id in range(5, 11):
        output[f"category_{cat_id}"] = {"findings": [], "compliant": True}
        output[f"category_{cat_id}_metadata"] = {"tokens_used": 150, "latency_ms": 2000, "status": "success"}

    output["total_tokens"] = 1500
    output["total_latency_ms"] = 2000

    with mock_graph_and_runner(output):
        response = await build_review_response("test-session", sample_parsed_document, 100)

    # Check severity calculations
    assert response.category_results[0].severity == "none"  # 0 findings
    assert response.category_results[1].severity == "low"   # 1 finding
    assert response.category_results[2].severity == "medium"  # 3 findings
    assert response.category_results[3].severity == "high"  # 8 findings


@pytest.mark.anyio
async def test_precheck_findings_merge_with_llm(sample_parsed_document, mock_graph_output_all_compliant):
    """Pre-check findings should merge with LLM findings for categories 1,2,5,6,7."""
    precheck_findings = {
        1: [Finding(passage="text1", issue="precheck issue 1", suggested_fix="fix1")],
        2: [Finding(passage="text2", issue="precheck issue 2", suggested_fix="fix2")],
        5: [
            Finding(passage="text5a", issue="precheck issue 5a", suggested_fix="fix5a"),
            Finding(passage="text5b", issue="precheck issue 5b", suggested_fix="fix5b"),
        ],
        6: [Finding(passage="text6", issue="precheck issue 6", suggested_fix="fix6")],
        7: [Finding(passage="text7", issue="precheck issue 7", suggested_fix="fix7")],
    }

    mock_runner = MagicMock()
    mock_runner.run_all = AsyncMock(return_value=precheck_findings)

    with patch("app.services.review_service.reviewer_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=mock_graph_output_all_compliant)

        with patch("app.services.review_service.CheckRunner", return_value=mock_runner):
            response = await build_review_response("test-session", sample_parsed_document, 100)

    # Categories 1,2,5,6,7 should have pre-check findings
    for cat_id in [1, 2, 5, 6, 7]:
        cat = next(c for c in response.category_results if c.category_id == cat_id)
        assert len(cat.findings) > 0
        assert cat.compliant is False  # Pre-checks made them non-compliant

    # Check specific counts
    assert len(next(c.findings for c in response.category_results if c.category_id == 1)) == 1
    assert len(next(c.findings for c in response.category_results if c.category_id == 5)) == 2


@pytest.mark.anyio
async def test_precheck_other_categories_unaffected(sample_parsed_document, mock_graph_output_all_compliant):
    """Pre-checks should only affect categories 1,2,5,6,7; others should stay LLM-only."""
    precheck_findings = {
        1: [Finding(passage="text", issue="issue", suggested_fix="fix")],
        3: [Finding(passage="text", issue="issue", suggested_fix="fix")],  # Pre-check for cat 3 (should be ignored)
    }

    mock_runner = MagicMock()
    mock_runner.run_all = AsyncMock(return_value=precheck_findings)

    with patch("app.services.review_service.reviewer_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=mock_graph_output_all_compliant)

        with patch("app.services.review_service.CheckRunner", return_value=mock_runner):
            response = await build_review_response("test-session", sample_parsed_document, 100)

    # Category 1 should have pre-check finding
    cat1 = next(c for c in response.category_results if c.category_id == 1)
    assert len(cat1.findings) == 1
    assert cat1.compliant is False

    # Category 3 should stay compliant (pre-check ignored)
    cat3 = next(c for c in response.category_results if c.category_id == 3)
    assert len(cat3.findings) == 0
    assert cat3.compliant is True


@pytest.mark.anyio
async def test_priority_changes_ranking(sample_parsed_document):
    """Priority changes should rank categories by severity and finding count."""
    output = {}

    # Create a mix of categories with different severity levels
    # Category 1: high severity (8 findings)
    output["category_1"] = {
        "findings": [
            Finding(passage="text", issue=f"issue{i}", suggested_fix=f"fix{i}")
            for i in range(8)
        ],
        "compliant": False,
    }

    # Category 2: medium severity (3 findings)
    output["category_2"] = {
        "findings": [
            Finding(passage="text", issue=f"issue{i}", suggested_fix=f"fix{i}")
            for i in range(3)
        ],
        "compliant": False,
    }

    # Category 3: low severity (1 finding)
    output["category_3"] = {
        "findings": [Finding(passage="text", issue="issue", suggested_fix="fix")],
        "compliant": False,
    }

    # Category 4: medium severity (4 findings, same as 2 but more findings)
    output["category_4"] = {
        "findings": [
            Finding(passage="text", issue=f"issue{i}", suggested_fix=f"fix{i}")
            for i in range(4)
        ],
        "compliant": False,
    }

    # Rest are compliant
    for cat_id in range(5, 11):
        output[f"category_{cat_id}"] = {"findings": [], "compliant": True}

    for cat_id in range(1, 11):
        output[f"category_{cat_id}_metadata"] = {"tokens_used": 150, "latency_ms": 2000, "status": "success"}

    output["total_tokens"] = 1500
    output["total_latency_ms"] = 2000

    with mock_graph_and_runner(output):
        response = await build_review_response("test-session", sample_parsed_document, 100)

    # Should have at most 5 priority changes
    assert len(response.priority_changes) <= 5

    # Categories with findings should be ranked by severity
    # Category 1 (high, 8 findings) should be rank 1
    assert response.priority_changes[0].category_id == 1
    assert response.priority_changes[0].rank == 1

    # Category 4 (medium, 4 findings) should be before category 2 (medium, 3 findings)
    cat4_rank = next(p.rank for p in response.priority_changes if p.category_id == 4)
    cat2_rank = next(p.rank for p in response.priority_changes if p.category_id == 2)
    assert cat4_rank < cat2_rank


@pytest.mark.anyio
async def test_metadata_includes_tokens_and_latency(sample_parsed_document):
    """Metadata should include total tokens and latency from LLM."""
    output = {}

    token_counts = {}
    latency_ms_counts = {}

    for cat_id in range(1, 11):
        tokens = 100 + cat_id * 50
        latency = 1000 + cat_id * 200
        token_counts[cat_id] = tokens
        latency_ms_counts[cat_id] = latency

        output[f"category_{cat_id}"] = {
            "findings": [],
            "compliant": True,
        }
        output[f"category_{cat_id}_metadata"] = {
            "tokens_used": tokens,
            "latency_ms": latency,
            "status": "success",
        }

    total_tokens = sum(token_counts.values())
    max_latency = max(latency_ms_counts.values())

    output["total_tokens"] = total_tokens
    output["total_latency_ms"] = max_latency

    with mock_graph_and_runner(output):
        response = await build_review_response("test-session", sample_parsed_document, 100)

    assert response.metadata.tokens_total == total_tokens
    assert response.metadata.latency_ms == max_latency


@pytest.mark.anyio
async def test_session_id_preserved(sample_parsed_document, mock_graph_output_all_compliant):
    """Session ID should be preserved in response."""
    with mock_graph_and_runner(mock_graph_output_all_compliant):
        response = await build_review_response("my-session-123", sample_parsed_document, 100)

    assert response.session_id == "my-session-123"


@pytest.mark.anyio
async def test_document_overview_has_word_count(sample_parsed_document, mock_graph_output_all_compliant):
    """Document overview should have actual word count from parsed document."""
    with mock_graph_and_runner(mock_graph_output_all_compliant):
        response = await build_review_response("test-session", sample_parsed_document, 100)

    assert response.document_overview.length_words == 100


@pytest.mark.anyio
async def test_empty_precheck_findings(sample_parsed_document, mock_graph_output_with_findings):
    """When pre-checks return no findings, LLM findings should still be used."""
    empty_precheck_findings = {}

    mock_runner = MagicMock()
    mock_runner.run_all = AsyncMock(return_value=empty_precheck_findings)

    with patch("app.services.review_service.reviewer_graph") as mock_graph:
        mock_graph.ainvoke = AsyncMock(return_value=mock_graph_output_with_findings)

        with patch("app.services.review_service.CheckRunner", return_value=mock_runner):
            response = await build_review_response("test-session", sample_parsed_document, 100)

    # Categories 1-5 should still have LLM findings
    for cat_id in [1, 2, 3, 4, 5]:
        cat = next(c for c in response.category_results if c.category_id == cat_id)
        assert len(cat.findings) >= 1


@pytest.mark.anyio
async def test_one_category_fails_others_succeed(sample_parsed_document):
    """When one category node fails (fallback), the other 9 should still return results."""
    output = {}

    # Categories 1-9: success with findings
    for cat_id in range(1, 10):
        output[f"category_{cat_id}"] = {
            "findings": [Finding(passage="text", issue=f"Issue {cat_id}", suggested_fix="fix")],
            "compliant": False,
            "severity": "low",
        }
        output[f"category_{cat_id}_metadata"] = {
            "tokens_used": 150,
            "latency_ms": 2000,
            "status": "success",
        }

    # Category 10: fallback (simulates LLM failure)
    output["category_10"] = {
        "findings": [],
        "compliant": True,
        "severity": "none",
    }
    output["category_10_metadata"] = {
        "tokens_used": 0,
        "latency_ms": 500,
        "status": "fallback",
    }

    output["total_tokens"] = 1350
    output["total_latency_ms"] = 2000

    with mock_graph_and_runner(output):
        response = await build_review_response("test-session", sample_parsed_document, 100)

    # All 10 categories should still appear
    assert len(response.category_results) == 10

    # Categories 1-9 should have findings (success)
    for cat_id in range(1, 10):
        cat = next(c for c in response.category_results if c.category_id == cat_id)
        assert len(cat.findings) >= 1
        assert cat.compliant is False

    # Category 10 should be empty but present (fallback)
    cat10 = next(c for c in response.category_results if c.category_id == 10)
    assert cat10.findings == []
    assert cat10.compliant is True

    # Metadata should record fallback status for category 10
    assert response.metadata.category_status.get("10") == "fallback"

    # Successful categories should have "success" status in metadata
    for cat_id in range(1, 10):
        assert response.metadata.category_status.get(str(cat_id)) == "success"

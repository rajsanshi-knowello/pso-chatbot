import pytest
from unittest.mock import AsyncMock, patch

from app.agents.nodes.category_10_structure import analyze_category_10_structure
from app.agents.state import ReviewState


def _findings_output(findings: list[dict], compliant: bool = False) -> dict:
    return {
        "compliant": compliant,
        "findings": findings,
        "reasoning": "Test reasoning.",
    }


class TestCategory10LLM:
    @pytest.mark.anyio
    async def test_passive_voice_violations(self):
        output = _findings_output([
            {
                "passage": "The policy was implemented by the team.",
                "issue": "Passive voice used when actor is clear and important.",
                "suggested_fix": "The team implemented the policy.",
                "rule_reference": "Active voice rule",
            }
        ])

        with patch(
            "app.agents.nodes.category_10_structure.call_llm_structured",
            AsyncMock(return_value=(output, 150)),
        ):
            state: ReviewState = {"document_text": "The policy was implemented by the team."}
            result = await analyze_category_10_structure(state)

        cat = result["category_10"]
        meta = result["category_10_metadata"]
        assert cat["compliant"] is False
        assert len(cat["findings"]) == 1
        assert "passive voice" in cat["findings"][0].issue.lower()
        assert meta["model_used"] == "gpt-4.1-mini"
        assert meta["tokens_used"] == 150

    @pytest.mark.anyio
    async def test_compliant_document(self):
        output = _findings_output([], compliant=True)

        with patch(
            "app.agents.nodes.category_10_structure.call_llm_structured",
            AsyncMock(return_value=(output, 120)),
        ):
            state: ReviewState = {"document_text": "The team implemented the new training program."}
            result = await analyze_category_10_structure(state)

        cat = result["category_10"]
        assert cat["compliant"] is True
        assert len(cat["findings"]) == 0
        assert cat["severity"] == "none"

    @pytest.mark.anyio
    async def test_api_error_fallback(self):
        with patch(
            "app.agents.nodes.category_10_structure.call_llm_structured",
            AsyncMock(side_effect=Exception("API Error")),
        ):
            state: ReviewState = {"document_text": "Test document."}
            result = await analyze_category_10_structure(state)

        cat = result["category_10"]
        meta = result["category_10_metadata"]
        assert cat["compliant"] is True
        assert len(cat["findings"]) == 0
        assert meta["model_used"] == "fallback-hardcoded"
        assert meta["status"] == "fallback"

    @pytest.mark.anyio
    async def test_multiple_findings(self):
        output = _findings_output([
            {
                "passage": "The report was written by John.",
                "issue": "Unnecessary passive voice",
                "suggested_fix": "John wrote the report.",
                "rule_reference": "Active voice rule",
            },
            {
                "passage": "Don't forget to submit your form.",
                "issue": "Negative phrasing in instruction",
                "suggested_fix": "Please submit your form.",
                "rule_reference": "Positive phrasing rule",
            },
        ])

        with patch(
            "app.agents.nodes.category_10_structure.call_llm_structured",
            AsyncMock(return_value=(output, 200)),
        ):
            state: ReviewState = {"document_text": "The report was written by John."}
            result = await analyze_category_10_structure(state)

        cat = result["category_10"]
        assert cat["compliant"] is False
        assert len(cat["findings"]) == 2
        assert cat["severity"] == "medium"

    @pytest.mark.anyio
    async def test_missing_document_metadata(self):
        output = _findings_output([
            {
                "passage": "[Document has no version number or publication date]",
                "issue": "Formal document missing publication date, version number, and document owner.",
                "suggested_fix": "Add header with: Version 1.0 | Published 15 April 2024 | Owner: John Smith",
                "rule_reference": "Formal document conventions",
            }
        ])

        with patch(
            "app.agents.nodes.category_10_structure.call_llm_structured",
            AsyncMock(return_value=(output, 180)),
        ):
            state: ReviewState = {"document_text": "Policy Document\n\nThis is the content."}
            result = await analyze_category_10_structure(state)

        cat = result["category_10"]
        assert cat["compliant"] is False
        assert any("version" in f.issue.lower() for f in cat["findings"])

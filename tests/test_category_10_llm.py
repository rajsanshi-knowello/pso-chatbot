import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.agents.nodes.category_10_structure import analyze_category_10_structure
from app.agents.state import ReviewState
from app.models.responses import Finding


class TestCategory10LLM:
    @pytest.mark.anyio
    async def test_passive_voice_violations(self):
        """Test detection of passive voice violations."""
        # Mock Gemini response with passive voice findings
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "compliant": False,
            "findings": [
                {
                    "passage": "The policy was implemented by the team.",
                    "issue": "Passive voice used when actor is clear and important.",
                    "suggested_fix": "The team implemented the policy.",
                    "rule_reference": "Active voice rule"
                }
            ],
            "reasoning": "Document uses passive voice unnecessarily in several places."
        })
        mock_response.usage.total_token_count = 150

        with patch("google.generativeai.GenerativeModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = mock_response
            mock_model_class.return_value = mock_model

            state: ReviewState = {
                "document_text": "The policy was implemented by the team.",
                "category_id": 10,
                "category_name": "Structure and Document Conventions",
            }

            result = await analyze_category_10_structure(state)

            assert result["compliant"] is False
            assert len(result["findings"]) == 1
            assert "passive voice" in result["findings"][0].issue.lower()
            assert result["model_used"] == "gemini-2.5-pro"
            assert result["tokens_used"] == 150

    @pytest.mark.anyio
    async def test_compliant_document(self):
        """Test that clean documents return compliant=true."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "compliant": True,
            "findings": [],
            "reasoning": "Document uses active voice throughout and follows all structure conventions."
        })
        mock_response.usage.total_token_count = 120

        with patch("google.generativeai.GenerativeModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = mock_response
            mock_model_class.return_value = mock_model

            state: ReviewState = {
                "document_text": "The team implemented the new training program.",
                "category_id": 10,
                "category_name": "Structure and Document Conventions",
            }

            result = await analyze_category_10_structure(state)

            assert result["compliant"] is True
            assert len(result["findings"]) == 0
            assert result["severity"] == "none"

    @pytest.mark.anyio
    async def test_api_error_fallback(self):
        """Test fallback to hardcoded when Gemini API fails."""
        with patch("google.generativeai.GenerativeModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = Exception("API Error")
            mock_model_class.return_value = mock_model

            state: ReviewState = {
                "document_text": "Test document.",
                "category_id": 10,
                "category_name": "Structure and Document Conventions",
            }

            result = await analyze_category_10_structure(state)

            # Should fallback gracefully
            assert result["compliant"] is True
            assert len(result["findings"]) == 0
            assert result["model_used"] == "fallback-hardcoded"
            assert result["_fallback"] is True

    @pytest.mark.anyio
    async def test_multiple_findings(self):
        """Test document with multiple violations."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "compliant": False,
            "findings": [
                {
                    "passage": "The report was written by John.",
                    "issue": "Unnecessary passive voice",
                    "suggested_fix": "John wrote the report.",
                    "rule_reference": "Active voice rule"
                },
                {
                    "passage": "Don't forget to submit your form.",
                    "issue": "Negative phrasing in instruction",
                    "suggested_fix": "Please submit your form.",
                    "rule_reference": "Positive phrasing rule"
                }
            ],
            "reasoning": "Multiple style issues found."
        })
        mock_response.usage.total_token_count = 200

        with patch("google.generativeai.GenerativeModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = mock_response
            mock_model_class.return_value = mock_model

            state: ReviewState = {
                "document_text": "The report was written by John. Don't forget to submit your form.",
                "category_id": 10,
                "category_name": "Structure and Document Conventions",
            }

            result = await analyze_category_10_structure(state)

            assert result["compliant"] is False
            assert len(result["findings"]) == 2
            assert result["severity"] == "medium"  # 2 findings = medium

    @pytest.mark.anyio
    async def test_missing_document_metadata(self):
        """Test detection of missing publication date, version, owner."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "compliant": False,
            "findings": [
                {
                    "passage": "[Document has no version number or publication date]",
                    "issue": "Formal document missing publication date, version number, and document owner.",
                    "suggested_fix": "Add header with: Version 1.0 | Published 15 April 2024 | Owner: John Smith",
                    "rule_reference": "Formal document conventions"
                }
            ],
            "reasoning": "Critical metadata missing from formal document."
        })
        mock_response.usage.total_token_count = 180

        with patch("google.generativeai.GenerativeModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model.generate_content.return_value = mock_response
            mock_model_class.return_value = mock_model

            state: ReviewState = {
                "document_text": "Policy Document\n\nThis is the content.",
                "category_id": 10,
                "category_name": "Structure and Document Conventions",
            }

            result = await analyze_category_10_structure(state)

            assert result["compliant"] is False
            assert any("version" in f.issue.lower() for f in result["findings"])

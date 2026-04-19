import json
import time
import logging
from typing import Any

import google.generativeai as genai

from app.agents.state import ReviewState
from app.config import get_settings
from app.models.responses import Finding

logger = logging.getLogger(__name__)


async def analyze_category_10_structure(state: ReviewState) -> dict[str, Any]:
    """LangGraph node: Analyze Category 10 (Structure and Document Conventions) via Gemini.

    Args:
        state: ReviewState containing document_text

    Returns:
        Updated state dict with findings, compliant, severity, model_used, tokens_used
    """
    settings = get_settings()
    start_time = time.monotonic()

    # Load prompts
    with open("app/prompts/base.txt", "r") as f:
        base_prompt = f.read()

    with open("app/prompts/categories/10_structure.txt", "r") as f:
        category_prompt = f.read()

    system_prompt = base_prompt + "\n\n" + category_prompt

    # Build user message
    user_message = f"""Please analyze this document for compliance with Category 10 rules.

Document:
---
{state['document_text'][:8000]}  # Limit to 8000 chars to stay under token limits
---

Provide your assessment as JSON."""

    try:
        # Configure Gemini
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")

        # Call Gemini with JSON mode (no schema - just instructions in prompt)
        response = model.generate_content(
            [system_prompt, user_message],
            generation_config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
            },
        )

        # Parse response
        response_text = response.text
        output_data = json.loads(response_text)

        # Convert findings to Finding objects (drop rule_reference)
        findings = [
            Finding(
                passage=f["passage"],
                issue=f["issue"],
                suggested_fix=f["suggested_fix"],
            )
            for f in output_data["findings"]
        ]

        compliant = output_data["compliant"]
        severity = "none" if compliant else ("high" if len(findings) > 2 else "medium")

        latency_ms = int((time.monotonic() - start_time) * 1000)

        # Extract token count (may not be available in all response modes)
        tokens = 0
        if hasattr(response, "usage") and response.usage:
            tokens = response.usage.total_token_count

        return {
            "findings": findings,
            "compliant": compliant,
            "severity": severity,
            "model_used": "gemini-2.5-flash",
            "tokens_used": tokens,
            "category_latency_ms": {10: latency_ms},
        }

    except Exception as exc:
        logger.error(f"Category 10 LLM analysis failed: {exc}", exc_info=True)
        # Fallback: return empty findings with metadata flag
        return {
            "findings": [],
            "compliant": True,
            "severity": "none",
            "model_used": "fallback-hardcoded",
            "tokens_used": 0,
            "category_latency_ms": {10: int((time.monotonic() - start_time) * 1000)},
            "_fallback": True,
        }

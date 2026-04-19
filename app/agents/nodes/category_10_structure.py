"""LangGraph node: Category 10 (Structure and Document Conventions) via OpenAI."""
import time
import logging
from typing import Any

from app.agents.state import ReviewState
from app.models.responses import Finding
from app.services.llm_client import call_llm_structured, CATEGORY_FINDINGS_SCHEMA

logger = logging.getLogger(__name__)


async def analyze_category_10_structure(state: ReviewState) -> dict[str, Any]:
    """Analyse Category 10 (Structure and Document Conventions).

    Returns a flat dict compatible with the legacy test contract:
        findings, compliant, severity, model_used, tokens_used, category_latency_ms
    """
    start_time = time.monotonic()

    with open("app/prompts/base.txt") as f:
        base_prompt = f.read()
    with open("app/prompts/categories/10_structure.txt") as f:
        category_prompt = f.read()

    system_prompt = base_prompt + "\n\n" + category_prompt

    user_message = (
        "Please analyse this document for compliance with Category 10 rules.\n\n"
        f"Document:\n---\n{state['document_text'][:8000]}\n---\n\n"
        "Provide your assessment as JSON."
    )

    try:
        output_data, tokens = await call_llm_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            response_schema=CATEGORY_FINDINGS_SCHEMA,
            schema_name="category_10_findings",
        )

        findings = [
            Finding(
                passage=f["passage"],
                issue=f["issue"],
                suggested_fix=f["suggested_fix"],
            )
            for f in output_data.get("findings", [])
        ]

        compliant = output_data.get("compliant", True)
        severity = "none" if compliant else ("high" if len(findings) > 2 else "medium")
        latency_ms = int((time.monotonic() - start_time) * 1000)

        return {
            "findings": findings,
            "compliant": compliant,
            "severity": severity,
            "model_used": "gpt-4.1-mini",
            "tokens_used": tokens,
            "category_latency_ms": {10: latency_ms},
        }

    except Exception as exc:
        logger.error("Category 10 LLM analysis failed: %s", exc, exc_info=True)
        return {
            "findings": [],
            "compliant": True,
            "severity": "none",
            "model_used": "fallback-hardcoded",
            "tokens_used": 0,
            "category_latency_ms": {10: int((time.monotonic() - start_time) * 1000)},
            "_fallback": True,
        }

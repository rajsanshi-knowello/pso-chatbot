"""Factory for creating OpenAI-backed category analysis nodes."""
import time
import logging
import asyncio
from typing import Any

from app.agents.state import ReviewState
from app.models.responses import Finding
from app.services.llm_client import call_llm_structured, CATEGORY_FINDINGS_SCHEMA

logger = logging.getLogger(__name__)

# Limit concurrent OpenAI calls
_llm_semaphore = asyncio.Semaphore(10)


def create_category_node(category_id: int, category_name: str, prompt_file: str):
    """Factory: return an async LangGraph node that analyses one editorial category.

    Args:
        category_id: Category number (1–10)
        category_name: Display name (e.g. 'Tone of Voice')
        prompt_file: Path to the category-specific prompt file
    """

    async def analyze_category(state: ReviewState) -> dict[str, Any]:
        async with _llm_semaphore:
            start_time = time.monotonic()

            with open("app/prompts/base.txt") as f:
                base_prompt = f.read()
            with open(prompt_file) as f:
                category_prompt = f.read()

            system_prompt = base_prompt + "\n\n" + category_prompt

            user_message = (
                f"Please analyse this document for compliance with Category {category_id} rules.\n\n"
                f"Document:\n---\n{state.get('document_text', '')[:8000]}\n---\n\n"
                f"Provide your assessment as JSON."
            )

            try:
                output_data, tokens = await call_llm_structured(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    response_schema=CATEGORY_FINDINGS_SCHEMA,
                    schema_name=f"category_{category_id:02d}_findings",
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
                    f"category_{category_id}": {
                        "findings": findings,
                        "compliant": compliant,
                        "severity": severity,
                    },
                    f"category_{category_id}_metadata": {
                        "model_used": "gpt-4.1-mini",
                        "tokens_used": tokens,
                        "latency_ms": latency_ms,
                        "status": "success",
                    },
                }

            except Exception as exc:
                logger.error("Category %d LLM analysis failed: %s", category_id, exc, exc_info=True)
                latency_ms = int((time.monotonic() - start_time) * 1000)
                return {
                    f"category_{category_id}": {
                        "findings": [],
                        "compliant": True,
                        "severity": "none",
                    },
                    f"category_{category_id}_metadata": {
                        "model_used": "fallback-hardcoded",
                        "tokens_used": 0,
                        "latency_ms": latency_ms,
                        "status": "fallback",
                    },
                }

    analyze_category.__name__ = (
        f"analyze_category_{category_id:02d}_{category_name.lower().replace(' ', '_')}"
    )
    return analyze_category

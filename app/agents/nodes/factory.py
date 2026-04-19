"""Factory for creating LLM category analysis nodes."""
import json
import time
import logging
import asyncio
from typing import Any

from google import genai
from google.genai import types as genai_types

from app.agents.state import ReviewState
from app.config import get_settings
from app.models.responses import Finding

logger = logging.getLogger(__name__)

# Global semaphore to limit concurrent Gemini API calls
_gemini_semaphore = asyncio.Semaphore(10)


def create_category_node(category_id: int, category_name: str, prompt_file: str):
    """Factory function to create an LLM analysis node for a category.

    Args:
        category_id: Category number (1-10)
        category_name: Display name (e.g., "Tone of Voice")
        prompt_file: Path to category prompt file (e.g., "app/prompts/categories/01_tone.txt")

    Returns:
        Async function that analyzes a document for that category
    """

    async def analyze_category(state: ReviewState) -> dict[str, Any]:
        """Analyze document for a specific category via Gemini."""
        async with _gemini_semaphore:  # Rate limit: max 10 concurrent calls
            start_time = time.monotonic()

            # Load prompts
            with open("app/prompts/base.txt", "r") as f:
                base_prompt = f.read()

            with open(prompt_file, "r") as f:
                category_prompt = f.read()

            system_prompt = base_prompt + "\n\n" + category_prompt

            # Build user message with document text
            user_message = f"""Please analyze this document for compliance with Category {category_id} rules.

Document:
---
{state.get('document_text', '')[:8000]}
---

Provide your assessment as JSON."""

            try:
                # Configure Gemini
                settings = get_settings()
                client = genai.Client(api_key=settings.gemini_api_key)

                # Call Gemini with JSON mode
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[system_prompt, user_message],
                    config=genai_types.GenerateContentConfig(
                        temperature=0.3,
                        response_mime_type="application/json",
                    ),
                )

                # Parse response
                response_text = response.text
                output_data = json.loads(response_text)

                # Convert findings
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

                # Extract token count if available
                tokens = 0
                if response.usage_metadata:
                    tokens = response.usage_metadata.total_token_count or 0

                return {
                    f"category_{category_id}": {
                        "findings": findings,
                        "compliant": compliant,
                        "severity": severity,
                    },
                    f"category_{category_id}_metadata": {
                        "model_used": "gemini-2.5-flash",
                        "tokens_used": tokens,
                        "latency_ms": latency_ms,
                        "status": "success",
                    },
                }

            except Exception as exc:
                logger.error(f"Category {category_id} LLM analysis failed: {exc}", exc_info=True)
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

    analyze_category.__name__ = f"analyze_category_{category_id:02d}_{category_name.lower().replace(' ', '_')}"
    return analyze_category

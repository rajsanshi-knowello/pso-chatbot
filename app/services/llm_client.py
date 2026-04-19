"""Shared OpenAI client for structured JSON and chat LLM calls."""
import asyncio
import json
import logging

import openai

from app.config import get_settings

logger = logging.getLogger(__name__)

# Reused by all 10 category nodes and category_10_structure.py
CATEGORY_FINDINGS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "compliant": {"type": "boolean"},
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "passage": {"type": "string"},
                    "issue": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                    "rule_reference": {"type": "string"},
                },
                "required": ["passage", "issue", "suggested_fix", "rule_reference"],
                "additionalProperties": False,
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["compliant", "findings", "reasoning"],
    "additionalProperties": False,
}


async def call_llm_structured(
    system_prompt: str,
    user_message: str,
    response_schema: dict,
    schema_name: str = "response",
    model: str = "gpt-4.1-mini",
    temperature: float = 0.3,
) -> tuple[dict, int]:
    """Call OpenAI Chat Completions with structured JSON output.

    Uses response_format json_schema for guaranteed schema-conformant output.
    Retries once on failure before raising.

    Returns:
        (parsed_response_dict, total_tokens_used)
    """
    settings = get_settings()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            async with openai.AsyncOpenAI(api_key=settings.openai_api_key) as client:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "strict": True,
                            "schema": response_schema,
                        },
                    },
                    temperature=temperature,
                )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            tokens = response.usage.total_tokens if response.usage else 0
            return parsed, tokens
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                logger.warning("LLM call failed (attempt 1), retrying: %s", exc)
                await asyncio.sleep(1)

    raise last_exc  # type: ignore[misc]

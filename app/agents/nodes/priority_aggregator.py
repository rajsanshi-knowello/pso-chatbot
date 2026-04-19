"""LangGraph node: synthesise all 10-category findings into priority changes + summary."""
import time
import logging

from app.agents.state import ReviewState
from app.services.llm_client import call_llm_structured

logger = logging.getLogger(__name__)

CATEGORY_NAMES = {
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

AGGREGATOR_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "priority_changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer"},
                    "category_id": {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["rank", "category_id", "description"],
                "additionalProperties": False,
            },
        },
        "overall_summary": {"type": "string"},
        "strengths": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["priority_changes", "overall_summary", "strengths"],
    "additionalProperties": False,
}


def _build_findings_summary(state: ReviewState) -> str:
    """Format all category findings as a text block for the aggregator prompt."""
    lines = ["CATEGORY FINDINGS SUMMARY:\n"]
    for cat_id in range(1, 11):
        cat_data = state.get(f"category_{cat_id}", {})
        cat_name = CATEGORY_NAMES[cat_id]
        findings = cat_data.get("findings", [])
        severity = cat_data.get("severity", "none")

        if not findings:
            lines.append(f"Category {cat_id} ({cat_name}) — COMPLIANT")
        else:
            lines.append(
                f"Category {cat_id} ({cat_name}) — {severity.upper()} severity — {len(findings)} finding(s):"
            )
            for i, f in enumerate(findings[:5], 1):
                passage = f.get("passage", "") if isinstance(f, dict) else getattr(f, "passage", "")
                issue = f.get("issue", "") if isinstance(f, dict) else getattr(f, "issue", "")
                lines.append(f"  {i}. Passage: \"{passage[:120]}\" | Issue: {issue[:200]}")

    return "\n".join(lines)


async def aggregate_priority(state: ReviewState) -> dict:
    """Synthesise findings into priority changes, summary and strengths via one LLM call."""
    start_time = time.monotonic()

    with open("app/prompts/aggregator/priority_changes.txt") as f:
        aggregator_prompt = f.read()

    findings_summary = _build_findings_summary(state)
    user_message = (
        f"{findings_summary}\n\n"
        "Based on the above findings, provide the priority changes, overall summary, "
        "and strengths as JSON."
    )

    try:
        output_data, tokens = await call_llm_structured(
            system_prompt=aggregator_prompt,
            user_message=user_message,
            response_schema=AGGREGATOR_SCHEMA,
            schema_name="priority_aggregator",
            temperature=0.2,
        )

        latency_ms = int((time.monotonic() - start_time) * 1000)
        logger.info("Priority aggregator completed in %dms, %d tokens", latency_ms, tokens)

        return {
            "aggregated_priority_changes": output_data.get("priority_changes", []),
            "aggregated_summary": output_data.get("overall_summary", ""),
            "aggregated_strengths": output_data.get("strengths", []),
            "aggregator_tokens": tokens,
            "aggregator_latency_ms": latency_ms,
        }

    except Exception as exc:
        logger.error("Priority aggregator failed: %s", exc, exc_info=True)
        latency_ms = int((time.monotonic() - start_time) * 1000)
        return {
            "aggregated_priority_changes": [],
            "aggregated_summary": "",
            "aggregated_strengths": [],
            "aggregator_tokens": 0,
            "aggregator_latency_ms": latency_ms,
        }

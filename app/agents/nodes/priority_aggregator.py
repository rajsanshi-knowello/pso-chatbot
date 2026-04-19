"""LangGraph node: synthesise all 10-category findings into priority changes + summary."""
import json
import time
import logging

from google import genai
from google.genai import types as genai_types

from app.agents.state import ReviewState
from app.config import get_settings

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
            for i, f in enumerate(findings[:5], 1):  # cap at 5 per category for token economy
                passage = f.get("passage", "") if isinstance(f, dict) else getattr(f, "passage", "")
                issue = f.get("issue", "") if isinstance(f, dict) else getattr(f, "issue", "")
                lines.append(f"  {i}. Passage: \"{passage[:120]}\" | Issue: {issue[:200]}")

    return "\n".join(lines)


async def aggregate_priority(state: ReviewState) -> dict:
    """Synthesise findings into priority changes, summary and strengths via one Gemini call."""
    start_time = time.monotonic()

    with open("app/prompts/aggregator/priority_changes.txt", "r") as f:
        aggregator_prompt = f.read()

    findings_summary = _build_findings_summary(state)

    user_message = f"""{findings_summary}

Based on the above findings, provide the priority changes, overall summary, and strengths as JSON."""

    try:
        settings = get_settings()
        client = genai.Client(api_key=settings.gemini_api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[aggregator_prompt, user_message],
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json",
            ),
        )

        output_data = json.loads(response.text)

        priority_changes = output_data.get("priority_changes", [])
        overall_summary = output_data.get("overall_summary", "")
        strengths = output_data.get("strengths", [])

        tokens = 0
        if response.usage_metadata:
            tokens = response.usage_metadata.total_token_count or 0

        latency_ms = int((time.monotonic() - start_time) * 1000)
        logger.info("Priority aggregator completed in %dms, %d tokens", latency_ms, tokens)

        return {
            "aggregated_priority_changes": priority_changes,
            "aggregated_summary": overall_summary,
            "aggregated_strengths": strengths,
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

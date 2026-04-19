import asyncio
import logging
from datetime import datetime, timezone

from app.agents.graph import reviewer_graph
from app.agents.state import ReviewState
from app.checks.runner import CheckRunner
from app.models.responses import (
    CategoryResult,
    DocumentOverview,
    Finding,
    PriorityChange,
    ReviewMetadata,
    ReviewResponse,
)
from app.services.document_parser import ParsedDocument

logger = logging.getLogger(__name__)

_HARDCODED_CATEGORIES: list[CategoryResult] = [
    CategoryResult(
        category_id=1,
        category_name="Tone of Voice",
        compliant=False,
        severity="medium",
        findings=[
            Finding(
                passage="The organisation must ensure all staff comply with mandatory training requirements.",
                issue="Overly formal and directive tone; does not match PSO's warm, supportive voice.",
                suggested_fix="Staff are encouraged to complete mandatory training — we're here to help every step of the way.",
            )
        ],
    ),
    CategoryResult(
        category_id=2,
        category_name="Accessibility",
        compliant=False,
        severity="high",
        findings=[
            Finding(
                passage="See Figure 3 for a visual summary of outcomes.",
                issue="Reference to a figure without alt text or descriptive caption provided in the body text.",
                suggested_fix="Add a plain-language description of the figure's key finding immediately after the reference.",
            )
        ],
    ),
    CategoryResult(
        category_id=3,
        category_name="Inclusive Language",
        compliant=False,
        severity="low",
        findings=[
            Finding(
                passage="Each student must submit his enrolment form by Friday.",
                issue="Gendered pronoun 'his' excludes non-binary and female students.",
                suggested_fix="Each student must submit their enrolment form by Friday.",
            )
        ],
    ),
    CategoryResult(
        category_id=4,
        category_name="Headings",
        compliant=True,
        severity="none",
        findings=[],
    ),
    CategoryResult(
        category_id=5,
        category_name="Punctuation",
        compliant=False,
        severity="medium",
        findings=[
            Finding(
                passage="The review covers 2020-2024 data.",
                issue="Hyphen used instead of an en dash for a numeric range.",
                suggested_fix="The review covers 2020–2024 data.",
            ),
            Finding(
                passage="Skills, knowledge and attitudes",
                issue="Missing Oxford comma before the final item in a list.",
                suggested_fix="Skills, knowledge, and attitudes",
            ),
        ],
    ),
    CategoryResult(
        category_id=6,
        category_name="Spelling",
        compliant=False,
        severity="low",
        findings=[
            Finding(
                passage="The programme focusses on workforce development.",
                issue="'focusses' is the British spelling; PSO style guide requires Australian English 'focuses'.",
                suggested_fix="The programme focuses on workforce development.",
            )
        ],
    ),
    CategoryResult(
        category_id=7,
        category_name="Numbers",
        compliant=False,
        severity="low",
        findings=[
            Finding(
                passage="There are 3 key objectives outlined in this document.",
                issue="Single-digit numbers should be spelled out unless in a table or technical context.",
                suggested_fix="There are three key objectives outlined in this document.",
            )
        ],
    ),
    CategoryResult(
        category_id=8,
        category_name="Referencing",
        compliant=True,
        severity="none",
        findings=[],
    ),
    CategoryResult(
        category_id=9,
        category_name="Tables and Figures",
        compliant=False,
        severity="medium",
        findings=[
            Finding(
                passage="Table 2",
                issue="Table 2 has no title or caption, making it impossible to interpret out of context.",
                suggested_fix="Add a descriptive title above the table, e.g. 'Table 2: Enrolment rates by qualification level, 2023'.",
            )
        ],
    ),
    CategoryResult(
        category_id=10,
        category_name="Structure",
        compliant=True,
        severity="none",
        findings=[],
    ),
]

_HARDCODED_PRIORITY_CHANGES: list[PriorityChange] = [
    PriorityChange(rank=1, category_id=2, description="Add alt text and plain-language descriptions for all figures to meet accessibility requirements."),
    PriorityChange(rank=2, category_id=5, description="Replace hyphens with en dashes in all numeric ranges throughout the document."),
    PriorityChange(rank=3, category_id=9, description="Add titles and captions to all tables and figures."),
]


async def build_review_response(session_id: str, parsed: ParsedDocument, latency_ms: int) -> ReviewResponse:
    # Run pre-checks to get real findings for categories 1, 2, 5, 6, 7
    runner = CheckRunner()
    findings_by_category = await runner.run_all(parsed.text)

    # Run LLM for Category 10 (Structure and Document Conventions)
    llm_tokens = 0
    category_10_latency = 0
    category_10_fallback = False
    try:
        logger.info("Invoking LLM for Category 10 analysis")
        state: ReviewState = {
            "document_text": parsed.text,
            "category_id": 10,
            "category_name": "Structure and Document Conventions",
        }
        output = reviewer_graph.invoke(state)
        findings_by_category[10] = output.get("findings", [])
        llm_tokens = output.get("tokens_used", 0)
        category_10_latency = output.get("category_latency_ms", {}).get(10, 0)
        category_10_fallback = output.get("_fallback", False)
        if category_10_fallback:
            logger.warning("Category 10 LLM analysis fell back to hardcoded")
    except Exception as exc:
        logger.error(f"Category 10 LLM invocation failed: {exc}", exc_info=True)
        category_10_fallback = True
        # Don't add findings_by_category[10] — let it use hardcoded below

    # Build category results, mixing real findings with hardcoded ones
    category_results: list[CategoryResult] = []

    for hardcoded in _HARDCODED_CATEGORIES:
        category_id = hardcoded.category_id

        # Replace findings for checked/analyzed categories
        if category_id in findings_by_category:
            real_findings = findings_by_category[category_id]
            # Convert from check Finding to response Finding
            findings = [
                Finding(
                    passage=f.passage,
                    issue=f.issue,
                    suggested_fix=f.suggested_fix,
                )
                for f in real_findings
            ]

            # Determine compliance and severity
            compliant = len(findings) == 0
            if len(findings) == 0:
                severity = "none"
            elif len(findings) >= 5:
                severity = "high"
            elif len(findings) >= 2:
                severity = "medium"
            else:
                severity = "low"

            category_results.append(
                CategoryResult(
                    category_id=category_id,
                    category_name=hardcoded.category_name,
                    compliant=compliant,
                    severity=severity,
                    findings=findings,
                )
            )
        else:
            # Keep hardcoded for non-checked categories
            category_results.append(hardcoded)

    return ReviewResponse(
        session_id=session_id,
        document_overview=DocumentOverview(
            document_type="Policy document",
            audience="VET sector workforce and training organisations",
            length_words=parsed.word_count,
            strengths=[
                "Clear section headings that aid navigation",
                "Consistent use of active voice in most sections",
                "Referencing style is complete and well-formatted",
            ],
        ),
        category_results=category_results,
        priority_changes=_HARDCODED_PRIORITY_CHANGES,
        next_steps=(
            "Address the high-severity findings first, then resolve lower-priority issues "
            "before the next review cycle."
        ),
        metadata=ReviewMetadata(
            model_used="pre-checks + llm" if not category_10_fallback else "pre-checks + fallback",
            tokens_total=llm_tokens,
            latency_ms=latency_ms,
            processed_at=datetime.now(timezone.utc).isoformat(),
        ),
    )

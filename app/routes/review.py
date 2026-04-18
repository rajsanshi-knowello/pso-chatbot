import time
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from app.auth import verify_api_key
from app.models.requests import ReviewRequest
from app.models.responses import (
    CategoryResult,
    DocumentOverview,
    Finding,
    PriorityChange,
    ReviewMetadata,
    ReviewResponse,
)
from app.services.document_parser import parse_document

router = APIRouter()

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


@router.post("/review", response_model=ReviewResponse)
async def review(
    body: ReviewRequest,
    _: str = Depends(verify_api_key),
) -> ReviewResponse:
    start = time.monotonic()
    parsed = await parse_document(body.document_url)
    latency_ms = int((time.monotonic() - start) * 1000)

    return ReviewResponse(
        session_id=body.session_id,
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
        category_results=_HARDCODED_CATEGORIES,
        priority_changes=_HARDCODED_PRIORITY_CHANGES,
        next_steps=(
            "Address the high-severity accessibility finding first, then resolve punctuation and "
            "table caption issues before the next review cycle."
        ),
        metadata=ReviewMetadata(
            model_used="hardcoded-stub",
            tokens_total=0,
            latency_ms=latency_ms,
            processed_at=datetime.now(timezone.utc).isoformat(),
        ),
    )

from typing import TypedDict, Literal
from app.models.responses import Finding


class ReviewState(TypedDict, total=False):
    """State shared across LangGraph nodes during review.

    Holds findings for all 10 categories, processed in parallel.
    """

    # Input
    document_text: str

    # Output: findings organized by category_id
    category_findings: dict[int, list[Finding]]  # {1: [Finding, ...], 2: [...], ...}

    # Metadata: per-category tracking
    category_metadata: dict[int, dict]  # {1: {"tokens": 150, "latency_ms": 2340, "model": "gemini-2.5-flash", "status": "success"}, ...}

    # Global metadata
    total_tokens: int
    total_latency_ms: int
    processed_at: str

    # Aggregator outputs (set by priority_aggregator node)
    aggregated_priority_changes: list[dict]  # [{"rank": 1, "category_id": 5, "description": "..."}]
    aggregated_summary: str
    aggregated_strengths: list[str]
    aggregator_tokens: int
    aggregator_latency_ms: int


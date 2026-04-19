from typing import TypedDict
from app.models.responses import Finding


class ReviewState(TypedDict, total=False):
    """State shared across LangGraph nodes during review."""

    # Input
    document_text: str

    # Per-node processing
    category_id: int
    category_name: str

    # Output
    findings: list[Finding]
    compliant: bool
    severity: str  # "none", "low", "medium", "high"

    # Metadata
    model_used: str
    tokens_used: int
    category_latency_ms: dict[int, int]  # category_id -> latency

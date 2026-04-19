from typing import TypedDict


class ReviewState(TypedDict, total=False):
    """State shared across LangGraph nodes during review."""

    # Input
    document_text: str

    # Per-category findings written by factory nodes and category_10_structure
    category_1: dict
    category_2: dict
    category_3: dict
    category_4: dict
    category_5: dict
    category_6: dict
    category_7: dict
    category_8: dict
    category_9: dict
    category_10: dict

    # Per-category metadata written by factory nodes and category_10_structure
    category_1_metadata: dict
    category_2_metadata: dict
    category_3_metadata: dict
    category_4_metadata: dict
    category_5_metadata: dict
    category_6_metadata: dict
    category_7_metadata: dict
    category_8_metadata: dict
    category_9_metadata: dict
    category_10_metadata: dict

    # Aggregated totals computed by the aggregate node
    total_tokens: int
    total_latency_ms: int
    processed_at: str

    # Aggregator outputs set by priority_aggregator node
    aggregated_priority_changes: list[dict]
    aggregated_summary: str
    aggregated_strengths: list[str]
    aggregator_tokens: int
    aggregator_latency_ms: int

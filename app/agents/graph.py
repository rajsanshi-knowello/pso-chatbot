"""LangGraph for parallel 10-category editorial review."""
from datetime import datetime, timezone
from langgraph.graph import StateGraph, START, END

from app.agents.state import ReviewState
from app.agents.nodes.factory import create_category_node


# Category definitions
CATEGORIES = {
    1: ("Tone of Voice", "app/prompts/categories/01_tone.txt"),
    2: ("Accessibility", "app/prompts/categories/02_accessibility.txt"),
    3: ("Inclusive Language", "app/prompts/categories/03_inclusive_language.txt"),
    4: ("Headings", "app/prompts/categories/04_headings.txt"),
    5: ("Punctuation", "app/prompts/categories/05_punctuation.txt"),
    6: ("Spelling", "app/prompts/categories/06_spelling.txt"),
    7: ("Numbers", "app/prompts/categories/07_numbers.txt"),
    8: ("Referencing", "app/prompts/categories/08_referencing.txt"),
    9: ("Tables and Figures", "app/prompts/categories/09_tables_figures.txt"),
    10: ("Structure", "app/prompts/categories/10_structure.txt"),
}


def build_review_graph():
    """Build LangGraph for parallel 10-category analysis.

    Structure:
        START → [10 category nodes in parallel] → aggregate → END
    """
    graph = StateGraph(ReviewState)

    # Create and add all 10 category nodes
    for cat_id, (cat_name, prompt_file) in CATEGORIES.items():
        node_name = f"category_{cat_id:02d}"
        node_func = create_category_node(cat_id, cat_name, prompt_file)
        graph.add_node(node_name, node_func)
        # All nodes connect from START for parallel execution
        graph.add_edge(START, node_name)
        # All nodes connect to aggregation for fan-in
        graph.add_edge(node_name, "aggregate")

    # Aggregation node: collect results from all 10 categories
    graph.add_node("aggregate", _aggregate_findings)
    graph.add_edge("aggregate", END)

    return graph.compile()


def _aggregate_findings(state: ReviewState) -> dict:
    """Aggregate findings from all 10 category nodes.

    LangGraph runs nodes in parallel, but each node writes to state independently.
    This aggregation step collects and organizes the results.
    """
    # Initialize collections if not present
    if "category_findings" not in state:
        state["category_findings"] = {}
    if "category_metadata" not in state:
        state["category_metadata"] = {}

    # The parallel nodes have already written their findings to state
    # (via the node return dicts, which LangGraph merges)
    # We just need to ensure they're properly structured

    # Calculate totals
    total_tokens = sum(
        meta.get("tokens_used", 0)
        for meta in state.get("category_metadata", {}).values()
    )

    # Latency is the max of all category latencies (since they ran in parallel)
    category_latencies = [
        meta.get("category_latency_ms", {}).get(cat_id, 0)
        for cat_id, meta in enumerate(state.get("category_metadata", {}).values(), 1)
    ]
    total_latency_ms = max(category_latencies) if category_latencies else 0

    state["total_tokens"] = total_tokens
    state["total_latency_ms"] = total_latency_ms
    state["processed_at"] = datetime.now(timezone.utc).isoformat()

    return state


# Singleton compiled graph
reviewer_graph = build_review_graph()

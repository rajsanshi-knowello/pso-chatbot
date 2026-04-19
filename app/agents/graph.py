"""LangGraph for parallel 10-category editorial review."""
from datetime import datetime, timezone
from langgraph.graph import StateGraph, START, END

from app.agents.state import ReviewState
from app.agents.nodes.factory import create_category_node
from app.agents.nodes.priority_aggregator import aggregate_priority



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


def _router_to_categories(state: ReviewState):
    """Route from START to all category nodes in parallel."""
    # Return a list of node names for parallel execution
    return [f"category_{cat_id:02d}" for cat_id in range(1, 11)]


def build_review_graph():
    """Build LangGraph for parallel 10-category analysis.

    Structure:
        START → [10 category nodes in parallel] → aggregate → priority_aggregator → END
    """
    graph = StateGraph(ReviewState)

    # Create and add all 10 category nodes
    for cat_id, (cat_name, prompt_file) in CATEGORIES.items():
        node_name = f"category_{cat_id:02d}"
        node_func = create_category_node(cat_id, cat_name, prompt_file)
        graph.add_node(node_name, node_func)
        # All nodes fan-in to aggregate
        graph.add_edge(node_name, "aggregate")

    # Add router from START that sends to all category nodes in parallel
    graph.add_conditional_edges(START, _router_to_categories)

    # Aggregate: compute totals from parallel category results
    graph.add_node("aggregate", _aggregate_findings)
    graph.add_edge("aggregate", "priority_aggregator")

    # Priority aggregator: one LLM call to synthesise findings → priority changes + summary
    graph.add_node("priority_aggregator", aggregate_priority)
    graph.add_edge("priority_aggregator", END)

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

    # Calculate totals from per-category state keys written by factory nodes
    total_tokens = sum(
        state.get(f"category_{cat_id}_metadata", {}).get("tokens_used", 0)
        for cat_id in range(1, 11)
    )

    # Latency is the max of all category latencies (since they ran in parallel)
    category_latencies = [
        state.get(f"category_{cat_id}_metadata", {}).get("latency_ms", 0)
        for cat_id in range(1, 11)
    ]
    total_latency_ms = max(category_latencies) if category_latencies else 0

    state["total_tokens"] = total_tokens
    state["total_latency_ms"] = total_latency_ms
    state["processed_at"] = datetime.now(timezone.utc).isoformat()

    return state


# Singleton compiled graph
reviewer_graph = build_review_graph()
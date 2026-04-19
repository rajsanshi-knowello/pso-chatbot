from langgraph.graph import StateGraph

from app.agents.state import ReviewState
from app.agents.nodes.category_10_structure import analyze_category_10_structure


def build_review_graph():
    """Build LangGraph for review category analysis.

    Currently: Single node for Category 10.
    Session 5: Will fan out to parallel nodes for all 10 categories.
    """
    graph = StateGraph(ReviewState)

    # Add Category 10 node
    graph.add_node("category_10", analyze_category_10_structure)

    # Set entry point
    graph.set_entry_point("category_10")

    # Set finish point (no edges between nodes yet)
    graph.set_finish_point("category_10")

    return graph.compile()


# Singleton compiled graph
reviewer_graph = build_review_graph()

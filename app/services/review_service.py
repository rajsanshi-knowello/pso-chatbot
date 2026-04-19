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


# Category metadata (name, display index)
CATEGORY_INFO = {
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


async def build_review_response(session_id: str, parsed: ParsedDocument, latency_ms: int) -> ReviewResponse:
    """Build review response with all 10 categories.

    Flow:
    1. Run pre-checks (Categories 1, 2, 5, 6, 7 only - regex based)
    2. Run LLM graph for all 10 categories in parallel
    3. Merge pre-check findings with LLM findings (for 1, 2, 5, 6, 7)
    4. Use LLM-only findings for categories 3, 4, 8, 9, 10
    5. Build response with real findings from both sources
    """
    # Step 1: Run pre-checks (Categories 1, 2, 5, 6, 7)
    logger.info("Running pre-checks for Categories 1, 2, 5, 6, 7")
    runner = CheckRunner()
    precheck_findings = await runner.run_all(parsed.text)

    # Step 2: Run LLM graph for all 10 categories in parallel
    logger.info("Invoking LLM graph for 10-category parallel analysis")
    try:
        state: ReviewState = {
            "document_text": parsed.text,
        }
        graph_output = await reviewer_graph.ainvoke(state)
    except Exception as exc:
        logger.error(f"LLM graph invocation failed: {exc}", exc_info=True)
        graph_output = {}

    # Step 3: Build category results
    category_results: list[CategoryResult] = []

    for category_id in range(1, 11):
        category_name = CATEGORY_INFO[category_id]

        # Collect findings from both sources
        findings: list[Finding] = []
        compliant = True
        severity = "none"

        # LLM findings (from graph)
        llm_key = f"category_{category_id}"
        if llm_key in graph_output:
            category_data = graph_output[llm_key]
            llm_findings = category_data.get("findings", [])
            findings.extend(llm_findings)
            compliant = category_data.get("compliant", True)

        # Pre-check findings (for categories 1, 2, 5, 6, 7)
        if category_id in precheck_findings and category_id in [1, 2, 5, 6, 7]:
            precheck_items = precheck_findings[category_id]
            # Convert from check Finding to response Finding
            precheck_converted = [
                Finding(
                    passage=f.passage,
                    issue=f.issue,
                    suggested_fix=f.suggested_fix,
                )
                for f in precheck_items
            ]
            # Pre-checks are added as additional findings (high-confidence regex rules)
            findings.extend(precheck_converted)
            if precheck_items:
                compliant = False

        # Determine severity
        if not compliant:
            severity = "high" if len(findings) > 5 else ("medium" if len(findings) > 1 else "low")

        category_results.append(
            CategoryResult(
                category_id=category_id,
                category_name=category_name,
                compliant=compliant,
                severity=severity,
                findings=findings,
            )
        )

    # Step 4: Collect metadata
    total_tokens = graph_output.get("total_tokens", 0)
    total_llm_latency = graph_output.get("total_latency_ms", 0)
    aggregator_tokens = graph_output.get("aggregator_tokens", 0)

    # Build per-category metadata
    tokens_by_category = {}
    latency_by_category = {}
    category_status = {}

    for category_id in range(1, 11):
        meta_key = f"category_{category_id}_metadata"
        if meta_key in graph_output:
            meta = graph_output[meta_key]
            tokens_by_category[str(category_id)] = meta.get("tokens_used", 0)
            latency_by_category[str(category_id)] = meta.get("latency_ms", 0)
            category_status[str(category_id)] = meta.get("status", "unknown")

    # Step 5: Extract aggregator output (with mechanical fallback)
    raw_priority_changes = graph_output.get("aggregated_priority_changes", [])
    if raw_priority_changes:
        priority_changes = [
            PriorityChange(
                rank=item["rank"],
                category_id=item["category_id"],
                description=item["description"],
            )
            for item in raw_priority_changes
        ]
    else:
        priority_changes = _build_priority_changes(category_results)

    overall_summary = graph_output.get("aggregated_summary", "")

    aggregator_strengths = graph_output.get("aggregated_strengths", [])
    if not aggregator_strengths:
        # Fallback: list compliant categories as strengths
        compliant_cats = [c.category_name for c in category_results if c.compliant]
        aggregator_strengths = [
            f"No issues found in {name}." for name in compliant_cats[:3]
        ] or ["Document reviewed successfully."]

    return ReviewResponse(
        session_id=session_id,
        document_overview=DocumentOverview(
            document_type="Policy document",
            audience="VET sector workforce and training organisations",
            length_words=parsed.word_count,
            strengths=aggregator_strengths,
        ),
        category_results=category_results,
        priority_changes=priority_changes,
        overall_summary=overall_summary,
        next_steps=(
            "Address the high-severity findings first, then resolve lower-priority issues "
            "before the next review cycle."
        ),
        metadata=ReviewMetadata(
            model_used="gemini-2.5-flash + pre-checks",
            tokens_total=total_tokens + aggregator_tokens,
            latency_ms=total_llm_latency,
            processed_at=datetime.now(timezone.utc).isoformat(),
            tokens_by_category=tokens_by_category,
            latency_by_category=latency_by_category,
            category_status=category_status,
        ),
    )


def _build_priority_changes(category_results: list[CategoryResult]) -> list[PriorityChange]:
    """Build priority changes list from category results.

    Rank categories by severity and number of findings.
    """
    ranked = []

    # Sort by severity (high > medium > low > none) and finding count
    severity_order = {"high": 0, "medium": 1, "low": 2, "none": 3}

    sorted_categories = sorted(
        category_results,
        key=lambda c: (severity_order.get(c.severity, 99), -len(c.findings)),
    )

    # Take top 5 with actual findings
    rank = 1
    for category in sorted_categories:
        if category.findings and rank <= 5:
            ranked.append(
                PriorityChange(
                    rank=rank,
                    category_id=category.category_id,
                    description=f"Address {len(category.findings)} finding(s) in {category.category_name}: {category.findings[0].issue}",
                )
            )
            rank += 1

    return ranked

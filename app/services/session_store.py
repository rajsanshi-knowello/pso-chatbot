"""In-memory session store for chat context across /review and /chat calls."""
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 7200  # 2 hours


@dataclass
class ChatSession:
    session_id: str
    document_text: Optional[str] = None
    document_context: Optional[str] = None  # pre-formatted summary for chat prompt
    message_history: list[dict] = field(default_factory=list)  # [{"role": "user"/"assistant", "content": "..."}]
    last_active: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    def get_or_create(self, session_id: str) -> ChatSession:
        """Return existing session or create a new one. Deletes expired sessions on access."""
        self._evict_if_expired(session_id)
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(session_id=session_id)
        session = self._sessions[session_id]
        session.last_active = time.time()
        return session

    def store_review(self, session_id: str, document_text: str, document_context: str) -> None:
        """Attach reviewed document text and pre-formatted context to the session."""
        session = self.get_or_create(session_id)
        session.document_text = document_text
        session.document_context = document_context
        logger.info("Stored review context for session %s", session_id)

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session history."""
        session = self.get_or_create(session_id)
        session.message_history.append({"role": role, "content": content})
        # Keep only the last 20 messages to bound memory usage
        if len(session.message_history) > 20:
            session.message_history = session.message_history[-20:]

    def _evict_if_expired(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session and (time.time() - session.last_active) > SESSION_TTL_SECONDS:
            del self._sessions[session_id]
            logger.info("Evicted expired session %s", session_id)


# Module-level singleton — shared across all requests in the process
session_store = SessionStore()


def build_document_context(review) -> str:
    """Build a concise text summary of a ReviewResponse for injection into chat prompts."""
    lines = [
        f"REVIEWED DOCUMENT — {review.document_overview.length_words} words",
        "",
        "CATEGORY FINDINGS:",
    ]
    for cat in review.category_results:
        if cat.findings:
            lines.append(
                f"  • Category {cat.category_id} ({cat.category_name}): "
                f"{cat.severity.upper()} — {len(cat.findings)} finding(s)"
            )
            for f in cat.findings[:3]:
                lines.append(f"    - \"{f.passage[:80]}\" → {f.issue[:120]}")
        else:
            lines.append(f"  • Category {cat.category_id} ({cat.category_name}): COMPLIANT")

    if review.priority_changes:
        lines += ["", "TOP PRIORITY CHANGES:"]
        for p in review.priority_changes[:5]:
            lines.append(f"  {p.rank}. {p.description}")

    if review.overall_summary:
        lines += ["", f"OVERALL SUMMARY: {review.overall_summary}"]

    return "\n".join(lines)

"""LangGraph for single-turn PSO chat with session context."""
import time
import logging
from typing import Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from google import genai
from google.genai import types as genai_types

from app.config import get_settings

logger = logging.getLogger(__name__)

FALLBACK_GREETING = (
    "Hello! I'm the PSO Editorial Review Assistant. "
    "Please upload a Word (.docx) or PDF (.pdf) document and I'll review it against "
    "PSO's Editorial Style Guide."
)

FALLBACK_REPLY = (
    "I'm sorry, I'm having trouble responding right now. "
    "Please try again in a moment."
)


class ChatState(TypedDict, total=False):
    system_prompt: str
    document_context: str        # pre-formatted review summary (empty if no doc in session)
    message_history: list[dict]  # [{"role": "user"/"assistant", "content": "..."}]
    user_message: str
    reply: str
    tokens_used: int
    latency_ms: int


async def _chat_node(state: ChatState) -> dict[str, Any]:
    """Call Gemini with session context and return the assistant reply."""
    start_time = time.monotonic()

    system_prompt = state.get("system_prompt", "")
    document_context = state.get("document_context", "")
    history = state.get("message_history", [])
    user_message = state.get("user_message", "")

    # Build the full system instruction (system prompt + document context if present)
    full_system = system_prompt
    if document_context:
        full_system = (
            full_system
            + "\n\n--- DOCUMENT CONTEXT (from prior review in this session) ---\n"
            + document_context
        )

    # Build content list from history (last 10 messages) + current user message
    contents: list[genai_types.Content] = []
    for msg in history[-10:]:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            genai_types.Content(role=role, parts=[genai_types.Part(text=msg["content"])])
        )
    contents.append(
        genai_types.Content(role="user", parts=[genai_types.Part(text=user_message)])
    )

    try:
        settings = get_settings()
        client = genai.Client(api_key=settings.gemini_api_key)

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=full_system,
                temperature=0.5,
            ),
        )

        reply = response.text.strip()

        tokens = 0
        if response.usage_metadata:
            tokens = response.usage_metadata.total_token_count or 0

        latency_ms = int((time.monotonic() - start_time) * 1000)
        logger.info("Chat node completed in %dms, %d tokens", latency_ms, tokens)

        return {"reply": reply, "tokens_used": tokens, "latency_ms": latency_ms}

    except Exception as exc:
        logger.error("Chat LLM call failed: %s", exc, exc_info=True)
        latency_ms = int((time.monotonic() - start_time) * 1000)
        # If there's no document context and no history, assume first-contact → greeting
        fallback = FALLBACK_GREETING if not document_context and not history else FALLBACK_REPLY
        return {"reply": fallback, "tokens_used": 0, "latency_ms": latency_ms}


def build_chat_graph():
    graph = StateGraph(ChatState)
    graph.add_node("chat", _chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile()


chat_graph = build_chat_graph()

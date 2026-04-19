"""LangGraph for single-turn PSO chat with session context."""
import time
import logging
from typing import Any
from typing_extensions import TypedDict

import openai
from langgraph.graph import StateGraph, START, END

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
    """Call OpenAI with session context and return the assistant reply."""
    start_time = time.monotonic()

    system_prompt = state.get("system_prompt", "")
    document_context = state.get("document_context", "")
    history = state.get("message_history", [])
    user_message = state.get("user_message", "")

    # System instruction: base PSO rules + injected document context
    full_system = system_prompt
    if document_context:
        full_system = (
            full_system
            + "\n\n--- DOCUMENT CONTEXT (from prior review in this session) ---\n"
            + document_context
        )

    # Build messages list: system → last 10 history turns → current user message
    messages: list[dict] = [{"role": "system", "content": full_system}]
    for msg in history[-10:]:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append({"role": role, "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        settings = get_settings()
        async with openai.AsyncOpenAI(api_key=settings.openai_api_key) as client:
            response = await client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                temperature=0.5,
            )

        reply = response.choices[0].message.content.strip()
        tokens = response.usage.total_tokens if response.usage else 0
        latency_ms = int((time.monotonic() - start_time) * 1000)
        logger.info("Chat node completed in %dms, %d tokens", latency_ms, tokens)

        return {"reply": reply, "tokens_used": tokens, "latency_ms": latency_ms}

    except Exception as exc:
        logger.error("Chat LLM call failed: %s", exc, exc_info=True)
        latency_ms = int((time.monotonic() - start_time) * 1000)
        fallback = FALLBACK_GREETING if not document_context and not history else FALLBACK_REPLY
        return {"reply": fallback, "tokens_used": 0, "latency_ms": latency_ms}


def build_chat_graph():
    graph = StateGraph(ChatState)
    graph.add_node("chat", _chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile()


chat_graph = build_chat_graph()

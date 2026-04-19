import logging
from fastapi import APIRouter, Depends
from app.auth import verify_api_key
from app.models.requests import ChatRequest
from app.models.responses import ChatResponse
from app.agents.chat_graph import chat_graph, ChatState
from app.services.session_store import session_store

logger = logging.getLogger(__name__)

router = APIRouter()

_SYSTEM_PROMPT_PATH = "app/prompts/chat/system.txt"
_system_prompt_cache: str | None = None


def _load_system_prompt() -> str:
    global _system_prompt_cache
    if _system_prompt_cache is None:
        with open(_SYSTEM_PROMPT_PATH, "r") as f:
            _system_prompt_cache = f.read()
    return _system_prompt_cache


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    _: str = Depends(verify_api_key),
) -> ChatResponse:
    session = session_store.get_or_create(body.session_id)

    state: ChatState = {
        "system_prompt": _load_system_prompt(),
        "document_context": session.document_context or "",
        "message_history": list(session.message_history),
        "user_message": body.message,
    }

    result = await chat_graph.ainvoke(state)
    reply = result.get("reply", "")

    # Persist turn to session history
    session_store.append_message(body.session_id, "user", body.message)
    session_store.append_message(body.session_id, "assistant", reply)

    logger.info(
        "Chat turn for session %s — %d tokens, %dms",
        body.session_id,
        result.get("tokens_used", 0),
        result.get("latency_ms", 0),
    )

    return ChatResponse(session_id=body.session_id, reply=reply)

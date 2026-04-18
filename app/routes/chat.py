from fastapi import APIRouter, Depends
from app.auth import verify_api_key
from app.models.requests import ChatRequest
from app.models.responses import ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    body: ChatRequest,
    _: str = Depends(verify_api_key),
) -> ChatResponse:
    return ChatResponse(
        session_id=body.session_id,
        reply=(
            "This is a placeholder response. "
            "The full chat agent will be wired up in a future session."
        ),
    )

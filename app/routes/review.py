import time
from fastapi import APIRouter, Depends
from app.auth import verify_api_key
from app.models.requests import ReviewRequest
from app.models.responses import ReviewResponse
from app.services.document_parser import parse_document
from app.services.review_service import build_review_response
from app.services.session_store import session_store, build_document_context

router = APIRouter()


@router.post("/review", response_model=ReviewResponse)
async def review(
    body: ReviewRequest,
    _: str = Depends(verify_api_key),
) -> ReviewResponse:
    start = time.monotonic()
    parsed = await parse_document(body.document_url)
    latency_ms = int((time.monotonic() - start) * 1000)
    response = await build_review_response(body.session_id, parsed, latency_ms)

    # Store document + review context for follow-up chat
    session_store.store_review(
        body.session_id,
        document_text=parsed.text,
        document_context=build_document_context(response),
    )

    return response

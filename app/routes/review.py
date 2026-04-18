import time
from fastapi import APIRouter, Depends
from app.auth import verify_api_key
from app.models.requests import ReviewRequest
from app.models.responses import ReviewResponse
from app.services.document_parser import parse_document
from app.services.review_service import build_review_response

router = APIRouter()


@router.post("/review", response_model=ReviewResponse)
async def review(
    body: ReviewRequest,
    _: str = Depends(verify_api_key),
) -> ReviewResponse:
    start = time.monotonic()
    parsed = await parse_document(body.document_url)
    latency_ms = int((time.monotonic() - start) * 1000)
    return build_review_response(body.session_id, parsed, latency_ms)

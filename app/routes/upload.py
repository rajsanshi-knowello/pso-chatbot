import time
from fastapi import APIRouter, Depends, File, Form, UploadFile
from app.auth import verify_api_key
from app.models.responses import ReviewResponse
from app.services.document_parser import parse_document_bytes
from app.services.review_service import build_review_response
from app.services.session_store import session_store, build_document_context

router = APIRouter()


@router.post("/review/upload", response_model=ReviewResponse)
async def review_upload(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    _: str = Depends(verify_api_key),
) -> ReviewResponse:
    content = await file.read()
    start = time.monotonic()
    parsed = parse_document_bytes(
        content=content,
        filename=file.filename or "",
        content_type=file.content_type or "",
    )
    latency_ms = int((time.monotonic() - start) * 1000)
    response = await build_review_response(session_id, parsed, latency_ms)

    # Store document + review context for follow-up chat
    session_store.store_review(
        session_id,
        document_text=parsed.text,
        document_context=build_document_context(response),
    )

    return response

from typing import Literal
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str
    code: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class Finding(BaseModel):
    passage: str
    issue: str
    suggested_fix: str


class CategoryResult(BaseModel):
    category_id: int
    category_name: str
    compliant: bool
    severity: Literal["low", "medium", "high", "none"]
    findings: list[Finding]


class DocumentOverview(BaseModel):
    document_type: str
    audience: str
    length_words: int
    strengths: list[str]


class PriorityChange(BaseModel):
    rank: int
    category_id: int
    description: str


class ReviewMetadata(BaseModel):
    model_used: str
    tokens_total: int
    latency_ms: int
    processed_at: str
    tokens_by_category: dict[str, int] = {}
    latency_by_category: dict[str, int] = {}
    category_status: dict[str, str] = {}


class ReviewResponse(BaseModel):
    session_id: str
    document_overview: DocumentOverview
    category_results: list[CategoryResult]
    priority_changes: list[PriorityChange]
    next_steps: str
    metadata: ReviewMetadata

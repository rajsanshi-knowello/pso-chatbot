from pydantic import BaseModel


class ReviewRequest(BaseModel):
    document_url: str
    session_id: str


class ChatRequest(BaseModel):
    message: str
    session_id: str

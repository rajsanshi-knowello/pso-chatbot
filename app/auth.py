from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import get_settings

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(_api_key_header)) -> str:
    if not api_key or api_key != get_settings().api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key

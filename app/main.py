import json
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.models.responses import ErrorResponse
from app.routes import chat, health, review


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def _configure_logging() -> None:
    settings = get_settings()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JSONFormatter())
    logging.root.handlers = [handler]
    logging.root.setLevel(settings.log_level.upper())


_configure_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    logger.info("PSO Review Service started (version %s)", application.version)
    yield


app = FastAPI(
    title="PSO Review Service",
    version="0.1.0",
    description="Editorial review microservice for the Powering Skills Organisation.",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(review.router)
app.include_router(chat.router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    logger.warning("HTTP %s: %s — %s %s", exc.status_code, exc.detail, request.method, request.url.path)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(detail=exc.detail, code=exc.status_code).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    detail = "; ".join(
        f"{' -> '.join(str(l) for l in e['loc'])}: {e['msg']}" for e in exc.errors()
    )
    logger.warning("Validation error on %s %s: %s", request.method, request.url.path, detail)
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(detail=detail, code=422).model_dump(),
    )



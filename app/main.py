from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import settings
from app.core.exceptions import AppException
from app.core.logging import setup_logging, get_logger


setup_logging(logging.INFO)
logger = get_logger(__name__)

app = FastAPI(
    title=settings.service_name,
    debug=settings.debug,
)

app.include_router(router)


@app.exception_handler(AppException)
async def handle_app_exception(request: Request, exc: AppException) -> JSONResponse:
    logger.warning(
        "AppException | path=%s | code=%s | message=%s",
        request.url.path,
        exc.error_code,
        exc.message,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": exc.error_code,
            "message": exc.message,
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled exception | path=%s | error=%s",
        request.url.path,
        str(exc),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "internal_server_error",
            "message": "Internal server error",
        },
    )
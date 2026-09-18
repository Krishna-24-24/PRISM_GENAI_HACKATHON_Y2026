from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger("traceiq.api.exceptions")

class TraceIQException(Exception):
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}

async def traceiq_exception_handler(request: Request, exc: TraceIQException):
    logger.error(f"TraceIQException: {exc.message} (status: {exc.status_code})")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "details": exc.details},
    )

async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )

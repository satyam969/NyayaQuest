"""
Request ID middleware for distributed tracing.

Reads X-Request-ID from the incoming request header (if provided by a load
balancer or client) or generates a fresh UUID.  The ID is:
  1. Bound to structlog context vars so every log line includes it.
  2. Echoed back in the response header for client-side correlation.
"""

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
import structlog


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

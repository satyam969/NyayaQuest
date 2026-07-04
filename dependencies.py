"""
Admin access dependency for FastAPI endpoints.

Gates sensitive endpoints (ingestion, reindex, collection delete, cost tracking)
behind an admin email allowlist configured via the ADMIN_EMAILS environment variable.
"""

import os
from fastapi import HTTPException, Header
from typing import Optional

ADMIN_EMAILS = set(
    email.strip()
    for email in os.getenv("ADMIN_EMAILS", "").split(",")
    if email.strip()
)

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")


def require_admin(x_admin_key: Optional[str] = Header(None)):
    """
    Lightweight admin gate.

    Since the current auth system is client-side (Firebase REST API, no
    server-side JWT verification), we use a simple API key header for admin
    routes. In production, set ADMIN_API_KEY in your environment.

    Usage:
        @app.post("/api/admin/something", dependencies=[Depends(require_admin)])
    """
    if not ADMIN_API_KEY:
        # No admin key configured — admin endpoints are disabled
        raise HTTPException(
            status_code=403,
            detail="Admin access not configured. Set ADMIN_API_KEY env var.",
        )
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Admin access required")
    return True

"""FastAPI bearer-token dependencies for kb_api, backed by kb_common.auth's
static KB_READ_TOKENS / KB_ADMIN_TOKENS allowlist. 401 means "no valid token
at all"; 403 means "valid token, but not admin-tier" - kept distinct so a
caller can tell an auth problem from a privilege problem.
"""
import logging

from fastapi import Depends, Header, HTTPException, Request

from kb_common import auth as shared_auth
from kb_common import config as common_config
from kb_common.ratelimit import RateLimiter

log = logging.getLogger(__name__)
_limiter = RateLimiter(common_config.RATE_LIMIT_MAX_REQUESTS, common_config.RATE_LIMIT_WINDOW_SECONDS)


def _bearer_token(authorization: str | None) -> str | None:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    return authorization[7:]


def require_token(authorization: str | None = Header(None)) -> str:
    """Any valid read- or admin-tier token. Guards /search, /internal/search, /stats."""
    token = _bearer_token(authorization)
    caller = shared_auth.verify_read_token(token) if token else None
    if caller is None:
        raise HTTPException(401, "Missing or invalid bearer token", headers={"WWW-Authenticate": "Bearer"})
    return caller


def require_admin_token(authorization: str | None = Header(None)) -> str:
    """Admin-tier token only. Guards /fetch and /index (real GPU ingestion jobs)."""
    token = _bearer_token(authorization)
    if token is not None:
        caller = shared_auth.verify_admin_token(token)
        if caller is not None:
            return caller
        if shared_auth.verify_read_token(token) is not None:
            raise HTTPException(403, "Admin token required")
    raise HTTPException(401, "Missing or invalid bearer token", headers={"WWW-Authenticate": "Bearer"})


def rate_limited(request: Request, caller: str = Depends(require_token)) -> str:
    """require_token, then a per-caller rate-limit check, then an audit log line."""
    if not _limiter.allow(caller):
        raise HTTPException(429, "Rate limit exceeded")
    log.info("caller=%s path=%s", caller, request.url.path)
    return caller


def rate_limited_admin(request: Request, caller: str = Depends(require_admin_token)) -> str:
    """require_admin_token, then a per-caller rate-limit check, then an audit log line."""
    if not _limiter.allow(caller):
        raise HTTPException(429, "Rate limit exceeded")
    log.info("caller=%s path=%s", caller, request.url.path)
    return caller

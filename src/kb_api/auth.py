"""FastAPI bearer-token dependencies for kb_api, backed by kb_common.auth's
static KB_READ_TOKENS / KB_ADMIN_TOKENS allowlist. 401 means "no valid token
at all"; 403 means "valid token, but not admin-tier" - kept distinct so a
caller can tell an auth problem from a privilege problem.
"""
import logging

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from kb_common import auth as shared_auth
from kb_common import config as common_config
from kb_common.ratelimit import RateLimiter

log = logging.getLogger(__name__)
_limiter = RateLimiter(common_config.RATE_LIMIT_MAX_REQUESTS, common_config.RATE_LIMIT_WINDOW_SECONDS)

_bearer_scheme = HTTPBearer(auto_error=False)


def require_token(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme)) -> str:
    """Any valid read- or admin-tier token - or, if KB_SECURITY_ENABLED=false,
    no token at all. Guards /search, /internal/search, /stats."""
    if not common_config.SECURITY_ENABLED:
        return "anonymous"
    caller = shared_auth.verify_read_token(credentials.credentials) if credentials else None
    if caller is None:
        raise HTTPException(401, "Missing or invalid bearer token", headers={"WWW-Authenticate": "Bearer"})
    return caller


def require_admin_token(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme)) -> str:
    """Admin-tier token only - or, if KB_SECURITY_ENABLED=false, no token at
    all (same single switch as require_token). Guards /fetch and /index
    (real GPU ingestion jobs)."""
    if not common_config.SECURITY_ENABLED:
        return "anonymous"
    if credentials is not None:
        caller = shared_auth.verify_admin_token(credentials.credentials)
        if caller is not None:
            return caller
        if shared_auth.verify_read_token(credentials.credentials) is not None:
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

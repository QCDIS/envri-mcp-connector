from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from kb_api import auth
from kb_common.ratelimit import RateLimiter


def _creds(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _fake_request(path: str = "/search"):
    return SimpleNamespace(url=SimpleNamespace(path=path))


@pytest.fixture(autouse=True)
def _security_enabled(monkeypatch):
    monkeypatch.setattr(auth.common_config, "SECURITY_ENABLED", True)


def test_require_token_accepts_valid_read_token(monkeypatch):
    monkeypatch.setattr(auth.shared_auth, "verify_read_token", lambda token: "alice" if token == "good" else None)

    assert auth.require_token(_creds("good")) == "alice"


def test_require_token_rejects_missing_credentials():
    with pytest.raises(HTTPException) as exc_info:
        auth.require_token(None)
    assert exc_info.value.status_code == 401


def test_require_token_rejects_invalid_token(monkeypatch):
    monkeypatch.setattr(auth.shared_auth, "verify_read_token", lambda token: None)

    with pytest.raises(HTTPException) as exc_info:
        auth.require_token(_creds("bad"))
    assert exc_info.value.status_code == 401


def test_require_token_bypassed_when_security_disabled(monkeypatch):
    monkeypatch.setattr(auth.common_config, "SECURITY_ENABLED", False)

    assert auth.require_token(None) == "anonymous"


def test_require_admin_token_accepts_admin_tier(monkeypatch):
    monkeypatch.setattr(auth.shared_auth, "verify_admin_token", lambda token: "root" if token == "adm" else None)

    assert auth.require_admin_token(_creds("adm")) == "root"


def test_require_admin_token_rejects_read_only_token_with_403(monkeypatch):
    monkeypatch.setattr(auth.shared_auth, "verify_admin_token", lambda token: None)
    monkeypatch.setattr(auth.shared_auth, "verify_read_token", lambda token: "chatbot")

    with pytest.raises(HTTPException) as exc_info:
        auth.require_admin_token(_creds("readonly"))
    assert exc_info.value.status_code == 403


def test_require_admin_token_rejects_unknown_token_with_401(monkeypatch):
    monkeypatch.setattr(auth.shared_auth, "verify_admin_token", lambda token: None)
    monkeypatch.setattr(auth.shared_auth, "verify_read_token", lambda token: None)

    with pytest.raises(HTTPException) as exc_info:
        auth.require_admin_token(_creds("unknown"))
    assert exc_info.value.status_code == 401


def test_rate_limited_allows_then_blocks(monkeypatch):
    monkeypatch.setattr(auth, "_limiter", RateLimiter(max_requests=1, window_seconds=60))

    assert auth.rate_limited(_fake_request(), caller="alice") == "alice"
    with pytest.raises(HTTPException) as exc_info:
        auth.rate_limited(_fake_request(), caller="alice")
    assert exc_info.value.status_code == 429


def test_rate_limited_admin_allows_then_blocks(monkeypatch):
    monkeypatch.setattr(auth, "_limiter", RateLimiter(max_requests=1, window_seconds=60))

    assert auth.rate_limited_admin(_fake_request(), caller="root") == "root"
    with pytest.raises(HTTPException) as exc_info:
        auth.rate_limited_admin(_fake_request(), caller="root")
    assert exc_info.value.status_code == 429

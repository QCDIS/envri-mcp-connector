"""Shared static bearer-token verification for kb_mcp and kb_api.

No OAuth authorization server is involved - tokens are opaque random
strings, one per known caller, configured out-of-band (never in git) via
KB_READ_TOKENS / KB_ADMIN_TOKENS ("name:token,name:token" pairs). An admin
token is also valid wherever a read token is accepted. Comparison is
constant-time per entry to avoid a timing side-channel.
"""
import os
import secrets

from dotenv import load_dotenv

# Mirrors kb_common.config: load_dotenv() here too rather than relying on
# import order, since some callers (e.g. kb_mcp.server) import this module
# before kb_common.config. Idempotent and non-overriding, like there.
load_dotenv()


def _parse_tokens(raw: str) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for entry in raw.split(","):
        name, _, token = entry.strip().partition(":")
        name, token = name.strip(), token.strip()
        if name and token:
            tokens[name] = token
    return tokens


_READ_TOKENS = _parse_tokens(os.environ.get("KB_READ_TOKENS", ""))
_ADMIN_TOKENS = _parse_tokens(os.environ.get("KB_ADMIN_TOKENS", ""))


def _lookup(tokens: dict[str, str], presented: str) -> str | None:
    for name, token in tokens.items():
        if secrets.compare_digest(token, presented):
            return name
    return None


def verify_read_token(token: str) -> str | None:
    """Caller name if `token` is a valid read- or admin-tier token, else None."""
    return _lookup(_READ_TOKENS, token) or _lookup(_ADMIN_TOKENS, token)


def verify_admin_token(token: str) -> str | None:
    """Caller name if `token` is a valid admin-tier token, else None."""
    return _lookup(_ADMIN_TOKENS, token)

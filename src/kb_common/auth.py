"""Shared static bearer-token verification for kb_mcp and kb_api."""
import os
import secrets

from dotenv import load_dotenv

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

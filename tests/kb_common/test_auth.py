import importlib

from kb_common import auth


def test_parse_tokens_basic():
    assert auth._parse_tokens("alice:tok1,bob:tok2") == {"alice": "tok1", "bob": "tok2"}


def test_parse_tokens_skips_malformed_entries():
    assert auth._parse_tokens("bad-entry,ok:tok, :novalue, noname:") == {"ok": "tok"}


def test_parse_tokens_strips_whitespace():
    assert auth._parse_tokens(" alice : tok1 , bob:tok2 ") == {"alice": "tok1", "bob": "tok2"}


def test_parse_tokens_empty_string():
    assert auth._parse_tokens("") == {}


def _reload_with_env(monkeypatch, read="", admin=""):
    monkeypatch.setenv("KB_READ_TOKENS", read)
    monkeypatch.setenv("KB_ADMIN_TOKENS", admin)
    return importlib.reload(auth)


def test_verify_read_token_accepts_read_tier(monkeypatch):
    mod = _reload_with_env(monkeypatch, read="chatbot:secret123")
    assert mod.verify_read_token("secret123") == "chatbot"
    assert mod.verify_read_token("wrong") is None


def test_verify_read_token_also_accepts_admin_tier(monkeypatch):
    mod = _reload_with_env(monkeypatch, admin="root:adminsecret")
    assert mod.verify_read_token("adminsecret") == "root"


def test_verify_admin_token_rejects_read_only_token(monkeypatch):
    mod = _reload_with_env(monkeypatch, read="chatbot:secret123", admin="root:adminsecret")
    assert mod.verify_admin_token("secret123") is None
    assert mod.verify_admin_token("adminsecret") == "root"


def test_verify_read_token_no_tokens_configured(monkeypatch):
    mod = _reload_with_env(monkeypatch)
    assert mod.verify_read_token("anything") is None

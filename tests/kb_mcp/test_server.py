import asyncio
import importlib

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from kb_common import config as common_config
from kb_common.ratelimit import RateLimiter
from kb_mcp import server


def _call_tool(name, args):
    async def run():
        return await server.mcp.call_tool(name, args)

    return asyncio.run(run())


# --- tool parameter bounds: one test per distinct constraint type used in
# server.py's Annotated[...] aliases, not one per tool ---


def test_search_knowledge_base_rejects_k_over_bound():
    with pytest.raises(ToolError):
        _call_tool("search_knowledge_base", {"query": "x", "k": 101})


def test_search_knowledge_base_rejects_empty_query():
    with pytest.raises(ToolError):
        _call_tool("search_knowledge_base", {"query": ""})


def test_get_argo_float_rejects_non_numeric_wmo():
    with pytest.raises(ToolError):
        _call_tool("get_argo_float", {"wmo": "not-a-number"})


def test_get_argo_float_rejects_overlong_wmo():
    with pytest.raises(ToolError):
        _call_tool("get_argo_float", {"wmo": "1" * 21})


def test_find_argo_floats_near_rejects_lat_out_of_range():
    with pytest.raises(ToolError):
        _call_tool("find_argo_floats_near", {"lat": 999, "lon": 0})


def test_find_argo_floats_near_rejects_lon_out_of_range():
    with pytest.raises(ToolError):
        _call_tool("find_argo_floats_near", {"lat": 0, "lon": 999})


def test_find_argo_floats_near_rejects_zero_radius():
    with pytest.raises(ToolError):
        _call_tool("find_argo_floats_near", {"lat": 0, "lon": 0, "radius_km": 0})


def test_list_argo_floats_by_sea_rejects_overlong_sea_area():
    with pytest.raises(ToolError):
        _call_tool("list_argo_floats_by_sea", {"sea_area": "x" * 201})


def test_list_field_values_rejects_unknown_field():
    with pytest.raises(ToolError):
        _call_tool("list_field_values", {"field": "not_a_real_field"})


def test_list_field_values_rejects_limit_over_facet_bound():
    with pytest.raises(ToolError):
        _call_tool("list_field_values", {"field": "networks", "limit": 501})


# --- valid input reaches the search layer with the right arguments ---


def test_search_knowledge_base_forwards_params(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        server.search,
        "search",
        lambda index, query, k=10, source=None: captured.update(index=index, query=query, k=k, source=source) or [],
    )

    _call_tool("search_knowledge_base", {"query": "floats near Ifremer", "k": 5, "source": "euro_argo"})

    assert captured["query"] == "floats near Ifremer"
    assert captured["k"] == 5
    assert captured["source"] == "euro_argo"


def test_get_argo_float_builds_correct_doc_id(monkeypatch):
    captured = {}
    monkeypatch.setattr(server.search, "get_by_id", lambda index, doc_id: captured.update(doc_id=doc_id) or None)

    _call_tool("get_argo_float", {"wmo": "6902919"})

    assert captured["doc_id"] == "euro_argo:6902919"


def test_get_oso_entity_builds_correct_doc_id(monkeypatch):
    captured = {}
    monkeypatch.setattr(server.search, "get_by_id", lambda index, doc_id: captured.update(doc_id=doc_id) or None)

    _call_tool("get_oso_entity", {"oso_id": "Ifremer"})

    assert captured["doc_id"] == "oso:Ifremer"


# --- StaticBearerTokenVerifier ---


def test_static_bearer_token_verifier_accepts_valid_token(monkeypatch):
    monkeypatch.setattr(server.shared_auth, "verify_read_token", lambda token: "alice" if token == "good" else None)

    result = asyncio.run(server.StaticBearerTokenVerifier().verify_token("good"))

    assert result is not None
    assert result.client_id == "alice"
    assert result.scopes == ["read"]


def test_static_bearer_token_verifier_rejects_invalid_token(monkeypatch):
    monkeypatch.setattr(server.shared_auth, "verify_read_token", lambda token: None)

    result = asyncio.run(server.StaticBearerTokenVerifier().verify_token("bad"))

    assert result is None


def _reload_server_with_security(monkeypatch, enabled):
    monkeypatch.setenv("KB_SECURITY_ENABLED", "true" if enabled else "false")
    importlib.reload(common_config)
    return importlib.reload(server)


def test_security_enabled_wires_up_auth(monkeypatch):
    try:
        mod = _reload_server_with_security(monkeypatch, enabled=True)
        assert mod.mcp.settings.auth is not None
        assert mod._mcp_auth_kwargs.get("token_verifier") is not None
    finally:
        _reload_server_with_security(monkeypatch, enabled=True)


def test_security_disabled_skips_auth_entirely(monkeypatch):
    try:
        mod = _reload_server_with_security(monkeypatch, enabled=False)
        assert mod.mcp.settings.auth is None
        assert "token_verifier" not in mod._mcp_auth_kwargs
    finally:
        _reload_server_with_security(monkeypatch, enabled=True)


# --- _RateLimitMiddleware ---


def _send_through(middleware, scope):
    sent = []

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))
    return sent


def test_rate_limit_middleware_allows_then_blocks():
    async def inner_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    middleware = server._RateLimitMiddleware(inner_app, RateLimiter(max_requests=1, window_seconds=60))
    scope = {"type": "http", "client": ("1.2.3.4", 5555)}

    first = _send_through(middleware, scope)
    second = _send_through(middleware, scope)

    assert first[0]["status"] == 200
    assert second[0]["status"] == 429


def test_rate_limit_middleware_isolates_by_source_ip():
    async def inner_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})

    middleware = server._RateLimitMiddleware(inner_app, RateLimiter(max_requests=1, window_seconds=60))

    first_ip = _send_through(middleware, {"type": "http", "client": ("1.1.1.1", 1)})
    second_ip = _send_through(middleware, {"type": "http", "client": ("2.2.2.2", 1)})

    assert first_ip[0]["status"] == 200
    assert second_ip[0]["status"] == 200


def test_rate_limit_middleware_ignores_non_http_scopes():
    calls = []

    async def inner_app(scope, receive, send):
        calls.append(scope["type"])

    # max_requests=0 would block every http request if the limiter were
    # consulted at all - proves non-http scopes bypass it entirely.
    middleware = server._RateLimitMiddleware(inner_app, RateLimiter(max_requests=0, window_seconds=60))

    _send_through(middleware, {"type": "lifespan"})

    assert calls == ["lifespan"]

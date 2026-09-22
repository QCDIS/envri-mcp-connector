"""MCP server exposing the Argo/OSO knowledge base over streamable-http.

Run directly:
    PYTHONPATH=src python -m kb_mcp.server

All tools operate against the single configured index (kb_common.config.ES_INDEX)
- `index` is not a tool parameter, so the calling AI doesn't need to know or
guess an internal index name.
"""
import functools
import logging
from typing import Annotated, Literal

from pydantic import Field

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from kb_common import auth as shared_auth
from kb_common import config as common_config
from kb_common import timing
from kb_common.ratelimit import RateLimiter
from kb_mcp import config, search

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _timed(fn):
    """Wraps an MCP tool to log its total time and per-stage breakdown
    (populated by kb_common.timing.stage/record calls anywhere in the tool's
    call stack - notably kb_mcp.search's api_http and kb_api's own stages,
    surfaced back here via the Server-Timing response header). Set
    LOG_TIMING=false to skip this entirely."""
    if not common_config.LOG_TIMING:
        return fn

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with timing.request() as stages:
            with timing.stage("tool_total"):
                result = fn(*args, **kwargs)
            log.info("tool=%s stages=%s", fn.__name__, {k: round(v, 4) for k, v in stages.items()})
        return result

    return wrapper


def _audited(fn):
    """Logs which authenticated caller invoked this tool - a minimal audit
    trail, independent of LOG_TIMING (unlike _timed, this always runs)."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        access_token = get_access_token()
        caller = access_token.client_id if access_token else "anonymous"
        log.info("caller=%s tool=%s", caller, fn.__name__)
        return fn(*args, **kwargs)

    return wrapper


class StaticBearerTokenVerifier(TokenVerifier):
    """Verifies a caller's bearer token against the static KB_READ_TOKENS /
    KB_ADMIN_TOKENS allowlist (kb_common.auth) - no OAuth authorization
    server is involved. All kb_mcp tools are read-only, so any valid token
    (read or admin tier) is sufficient here."""

    async def verify_token(self, token: str) -> AccessToken | None:
        caller = shared_auth.verify_read_token(token)
        if caller is None:
            return None
        return AccessToken(token=token, client_id=caller, scopes=["read"])


# issuer_url/resource_server_url are required by AuthSettings but otherwise
# inert here: no auth_server_provider is configured, so the SDK never mounts
# /authorize or /token routes against them (mcp/server/mcpserver/server.py,
# "Add auth endpoints if auth server provider is configured"). Built from the
# first allowed host since that's already this server's own address.
_resource_url = f"http://{config.MCP_ALLOWED_HOSTS[0] if config.MCP_ALLOWED_HOSTS else f'localhost:{config.MCP_PORT}'}"

# Read-tier auth is opt-in
_mcp_auth_kwargs = {}
if shared_auth.read_auth_enabled():
    _mcp_auth_kwargs["auth"] = AuthSettings(
        issuer_url=_resource_url,
        resource_server_url=_resource_url,
        required_scopes=["read"],
        validate_token_resource=False,
    )
    _mcp_auth_kwargs["token_verifier"] = StaticBearerTokenVerifier()
else:
    log.warning("KB_READ_TOKENS is unset - kb-mcp is accepting unauthenticated requests")

mcp = MCPServer(
    "ifremer-knowledge-base",
    instructions=(
        "Search and browse the Ifremer knowledge base: Euro-Argo profiling float "
        "metadata and the OSO ontology (EMSO sites, platforms, organizations, "
        "people, projects). Use search_knowledge_base for open-ended natural- "
        "language questions; use the list_*/get_index_stats tools first if you "
        "need to know what values exist before filtering."
    ),
    **_mcp_auth_kwargs,
)

Source = Literal["euro_argo", "oso"]
FacetField = Literal["data_center_name", "networks", "sensor_codes", "project_name"]

# Shared parameter constraints - bounds a caller can't exceed regardless of
# what they pass, since an unbounded `limit`/`k` scales the underlying ES
# query (see kb_common.hybrid_search.knn_clause's num_candidates) and an
# unbounded id/name string has no reason to be more than a few hundred
# characters for any real Argo/OSO identifier.
Limit = Annotated[int, Field(ge=1, le=100)]
FacetLimit = Annotated[int, Field(ge=1, le=500)]
FilterValue = Annotated[str, Field(min_length=1, max_length=200)]
Latitude = Annotated[float, Field(ge=-90, le=90)]
Longitude = Annotated[float, Field(ge=-180, le=180)]


@mcp.tool()
@_timed
@_audited
def search_knowledge_base(
    query: Annotated[str, Field(min_length=1, max_length=1000)],
    source: Source | None = None,
    k: Limit = 10,
) -> list[dict]:
    """Hybrid semantic + keyword search across the knowledge base. This is the
    primary, general-purpose tool - use it for any open-ended question.
    Optionally restrict to one source ("euro_argo" or "oso"). Each result
    includes "highlights": the specific matched fragment(s) of summary_text
    (wrapped in <em> tags) - check these first to see why a hit matched
    before reading the full summary_text. Each result also includes "url":
    a dereferenceable link to the original data source (Euro-Argo fleet
    monitoring or the OSO ontology) for the user to consult directly."""
    return search.search(common_config.ES_INDEX, query, k=k, source=source)


@mcp.tool()
@_timed
@_audited
def get_argo_float(wmo: Annotated[str, Field(min_length=1, max_length=20, pattern=r"^[0-9]+$")]) -> dict | None:
    """Fetch the full record for one Argo float by its WMO id (e.g. "6902919")."""
    return search.get_by_id(common_config.ES_INDEX, f"euro_argo:{wmo}")


@mcp.tool()
@_timed
@_audited
def get_oso_entity(oso_id: FilterValue) -> dict | None:
    """Fetch the full record for one OSO ontology entity by its id
    (e.g. "Ifremer", "ANTARES") - use search_knowledge_base or the list_oso_*
    tools first to find the right id."""
    return search.get_by_id(common_config.ES_INDEX, f"oso:{oso_id}")


@mcp.tool()
@_timed
@_audited
def find_argo_floats_near(
    lat: Latitude,
    lon: Longitude,
    radius_km: Annotated[float, Field(gt=0, le=20000)] = 200,
    limit: Limit = 20,
) -> list[dict]:
    """Find Argo floats within radius_km of a point, based on each float's
    last known position, nearest first."""
    return search.geo_distance(common_config.ES_INDEX, "last_cycle_geopoint", lat, lon, radius_km, limit)


@mcp.tool()
@_timed
@_audited
def find_argo_floats_in_box(
    min_lat: Latitude,
    max_lat: Latitude,
    min_lon: Longitude,
    max_lon: Longitude,
    limit: Limit = 20,
) -> list[dict]:
    """Find Argo floats whose last known position falls within a lat/lon
    bounding box - use this instead of find_argo_floats_near when you have a
    region's bounds rather than a center point and radius."""
    return search.geo_bounding_box(common_config.ES_INDEX, "last_cycle_geopoint", min_lat, max_lat, min_lon, max_lon, limit)


@mcp.tool()
@_timed
@_audited
def list_argo_floats_by_sea(sea_area: FilterValue, limit: Limit = 20) -> list[dict]:
    """List Argo floats last located in a given specific named sea (e.g.
    "Mediterranean Sea - Western Basin", "Black Sea", "Gulf of Mexico") - one
    of 101 IHO sea areas. Use list_seas first to see the exact values that
    exist. For a broader ocean basin instead (Atlantic/Pacific/Indian/Arctic/
    Southern), use list_argo_floats_by_ocean."""
    return search.term_filter(common_config.ES_INDEX, "sea_area.keyword", sea_area, limit)


@mcp.tool()
@_timed
@_audited
def list_argo_floats_by_ocean(ocean_region: FilterValue, limit: Limit = 20) -> list[dict]:
    """List Argo floats last located in a given broad ocean basin (one of:
    Atlantic Ocean, Pacific Ocean, Indian Ocean, Arctic Ocean, Southern
    Ocean). For a specific named sea instead (e.g. "Black Sea"), use
    list_argo_floats_by_sea."""
    return search.term_filter(common_config.ES_INDEX, "ocean_region.keyword", ocean_region, limit)


@mcp.tool()
@_timed
@_audited
def list_argo_floats_by_sensor(sensor_code: FilterValue, limit: Limit = 20) -> list[dict]:
    """List Argo floats equipped with a given sensor code (e.g. "DOXY" for
    dissolved oxygen, "CTD_TEMP" for temperature)."""
    return search.term_filter(common_config.ES_INDEX, "sensor_codes", sensor_code, limit)


@mcp.tool()
@_timed
@_audited
def list_oso_entities_by_type(entity_type: FilterValue, limit: Limit = 20) -> list[dict]:
    """List OSO entities of a given type (e.g. "Organization", "Platform",
    "Site", "RegionalFacility"). Use list_oso_entity_types first to see the
    exact values that exist."""
    return search.term_filter(common_config.ES_INDEX, "entity_types.keyword", entity_type, limit)


@mcp.tool()
@_timed
@_audited
def list_argo_floats_by_organization(oso_organization_id: FilterValue, limit: Limit = 20) -> list[dict]:
    """List Argo floats operated by a given OSO organization id (e.g.
    "Ifremer") - the cross-source link between the two sources. Find
    organization ids via search_knowledge_base or list_oso_entities_by_type("Organization")."""
    return search.term_filter(common_config.ES_INDEX, "oso_organization_id", oso_organization_id, limit)


@mcp.tool()
@_timed
@_audited
def list_seas() -> list[dict]:
    """List every specific named sea present in the Argo data, with how many
    floats were last located there. Feeds list_argo_floats_by_sea."""
    return search.terms_agg(common_config.ES_INDEX, "sea_area.keyword", filter_query={"term": {"source": "euro_argo"}})


@mcp.tool()
@_timed
@_audited
def list_ocean_regions() -> list[dict]:
    """List every broad ocean basin present in the Argo data, with how many
    floats were last located there. Feeds list_argo_floats_by_ocean."""
    return search.terms_agg(common_config.ES_INDEX, "ocean_region.keyword", filter_query={"term": {"source": "euro_argo"}})


@mcp.tool()
@_timed
@_audited
def list_oso_entity_types() -> list[dict]:
    """List every OSO entity type present, with counts. Feeds
    list_oso_entities_by_type."""
    return search.terms_agg(common_config.ES_INDEX, "entity_types.keyword", filter_query={"term": {"source": "oso"}})


@mcp.tool()
@_timed
@_audited
def list_field_values(field: FacetField, limit: FacetLimit = 50) -> list[dict]:
    """List distinct values (with counts) for one of a fixed set of useful
    fields: data_center_name, networks, sensor_codes, project_name."""
    return search.terms_agg(common_config.ES_INDEX, search.FACETABLE_FIELDS[field], limit)


@mcp.tool()
@_timed
@_audited
def get_index_stats() -> dict:
    """Document counts in the knowledge base, overall and per source. Call
    this first if you're unsure whether the KB has data before searching."""
    return search.index_stats(common_config.ES_INDEX)


class _RateLimitMiddleware:
    """Per-source-IP rate limiting for the whole /mcp endpoint.

    Keyed by IP rather than authenticated caller identity: mcp.run() doesn't
    expose a hook to insert middleware between its own layers, so this wraps
    the whole ASGI app from the outside, before the SDK's auth middleware
    has run. For the small, known set of callers this server expects, IP is
    a reasonable proxy for caller identity anyway.
    """

    def __init__(self, app, limiter: RateLimiter):
        self.app = app
        self.limiter = limiter

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        client = scope.get("client")
        key = client[0] if client else "unknown"
        if not self.limiter.allow(key):
            response = Response("Rate limit exceeded", status_code=429)
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


if __name__ == "__main__":
    import uvicorn
    from starlette.responses import Response

    # MCP_HOST is 0.0.0.0 (see kb_mcp.config), so the SDK's own loopback-only
    # auto-enable never kicks in - configure DNS-rebinding protection
    # explicitly instead. allowed_origins stays empty: callers here are
    # server-to-server (no browser Origin header), and a request with no
    # Origin header at all already passes unconditionally.
    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=config.MCP_ALLOWED_HOSTS,
        allowed_origins=[],
    )
    # mcp.run() doesn't take a middleware= kwarg, so the app is built and
    # wrapped by hand here instead of calling it.
    app = mcp.streamable_http_app(host=config.MCP_HOST, transport_security=transport_security)
    limiter = RateLimiter(common_config.RATE_LIMIT_MAX_REQUESTS, common_config.RATE_LIMIT_WINDOW_SECONDS)
    app = _RateLimitMiddleware(app, limiter)

    uvicorn_config = uvicorn.Config(app, host=config.MCP_HOST, port=config.MCP_PORT)
    uvicorn.Server(uvicorn_config).run()

"""MCP server exposing the Argo/OSO knowledge base over streamable-http.

Run directly:
    PYTHONPATH=src python -m kb_mcp.server

All tools operate against the single configured index (kb_common.config.ES_INDEX)
- `index` is not a tool parameter, so the calling AI doesn't need to know or
guess an internal index name.
"""
import functools
import logging
from typing import Literal

from mcp.server.mcpserver import MCPServer

from kb_common import config as common_config
from kb_common import timing
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


mcp = MCPServer(
    "ifremer-knowledge-base",
    instructions=(
        "Search and browse the Ifremer knowledge base: Euro-Argo profiling float "
        "metadata and the OSO ontology (EMSO sites, platforms, organizations, "
        "people, projects). Use search_knowledge_base for open-ended natural- "
        "language questions; use the list_*/get_index_stats tools first if you "
        "need to know what values exist before filtering."
    ),
)

Source = Literal["euro_argo", "oso"]
FacetField = Literal["data_center_name", "networks", "sensor_codes", "project_name"]


@mcp.tool()
@_timed
def search_knowledge_base(query: str, source: Source | None = None, k: int = 10) -> list[dict]:
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
def get_argo_float(wmo: str) -> dict | None:
    """Fetch the full record for one Argo float by its WMO id (e.g. "6902919")."""
    return search.get_by_id(common_config.ES_INDEX, f"euro_argo:{wmo}")


@mcp.tool()
@_timed
def get_oso_entity(oso_id: str) -> dict | None:
    """Fetch the full record for one OSO ontology entity by its id
    (e.g. "Ifremer", "ANTARES") - use search_knowledge_base or the list_oso_*
    tools first to find the right id."""
    return search.get_by_id(common_config.ES_INDEX, f"oso:{oso_id}")


@mcp.tool()
@_timed
def find_argo_floats_near(lat: float, lon: float, radius_km: float = 200, limit: int = 20) -> list[dict]:
    """Find Argo floats within radius_km of a point, based on each float's
    last known position, nearest first."""
    return search.geo_distance(common_config.ES_INDEX, "last_cycle_geopoint", lat, lon, radius_km, limit)


@mcp.tool()
@_timed
def find_argo_floats_in_box(min_lat: float, max_lat: float, min_lon: float, max_lon: float, limit: int = 20) -> list[dict]:
    """Find Argo floats whose last known position falls within a lat/lon
    bounding box - use this instead of find_argo_floats_near when you have a
    region's bounds rather than a center point and radius."""
    return search.geo_bounding_box(common_config.ES_INDEX, "last_cycle_geopoint", min_lat, max_lat, min_lon, max_lon, limit)


@mcp.tool()
@_timed
def list_argo_floats_by_sea(sea_area: str, limit: int = 20) -> list[dict]:
    """List Argo floats last located in a given specific named sea (e.g.
    "Mediterranean Sea - Western Basin", "Black Sea", "Gulf of Mexico") - one
    of 101 IHO sea areas. Use list_seas first to see the exact values that
    exist. For a broader ocean basin instead (Atlantic/Pacific/Indian/Arctic/
    Southern), use list_argo_floats_by_ocean."""
    return search.term_filter(common_config.ES_INDEX, "sea_area.keyword", sea_area, limit)


@mcp.tool()
@_timed
def list_argo_floats_by_ocean(ocean_region: str, limit: int = 20) -> list[dict]:
    """List Argo floats last located in a given broad ocean basin (one of:
    Atlantic Ocean, Pacific Ocean, Indian Ocean, Arctic Ocean, Southern
    Ocean). For a specific named sea instead (e.g. "Black Sea"), use
    list_argo_floats_by_sea."""
    return search.term_filter(common_config.ES_INDEX, "ocean_region.keyword", ocean_region, limit)


@mcp.tool()
@_timed
def list_argo_floats_by_sensor(sensor_code: str, limit: int = 20) -> list[dict]:
    """List Argo floats equipped with a given sensor code (e.g. "DOXY" for
    dissolved oxygen, "CTD_TEMP" for temperature)."""
    return search.term_filter(common_config.ES_INDEX, "sensor_codes", sensor_code, limit)


@mcp.tool()
@_timed
def list_oso_entities_by_type(entity_type: str, limit: int = 20) -> list[dict]:
    """List OSO entities of a given type (e.g. "Organization", "Platform",
    "Site", "RegionalFacility"). Use list_oso_entity_types first to see the
    exact values that exist."""
    return search.term_filter(common_config.ES_INDEX, "entity_types.keyword", entity_type, limit)


@mcp.tool()
@_timed
def list_argo_floats_by_organization(oso_organization_id: str, limit: int = 20) -> list[dict]:
    """List Argo floats operated by a given OSO organization id (e.g.
    "Ifremer") - the cross-source link between the two sources. Find
    organization ids via search_knowledge_base or list_oso_entities_by_type("Organization")."""
    return search.term_filter(common_config.ES_INDEX, "oso_organization_id", oso_organization_id, limit)


@mcp.tool()
@_timed
def list_seas() -> list[dict]:
    """List every specific named sea present in the Argo data, with how many
    floats were last located there. Feeds list_argo_floats_by_sea."""
    return search.terms_agg(common_config.ES_INDEX, "sea_area.keyword", filter_query={"term": {"source": "euro_argo"}})


@mcp.tool()
@_timed
def list_ocean_regions() -> list[dict]:
    """List every broad ocean basin present in the Argo data, with how many
    floats were last located there. Feeds list_argo_floats_by_ocean."""
    return search.terms_agg(common_config.ES_INDEX, "ocean_region.keyword", filter_query={"term": {"source": "euro_argo"}})


@mcp.tool()
@_timed
def list_oso_entity_types() -> list[dict]:
    """List every OSO entity type present, with counts. Feeds
    list_oso_entities_by_type."""
    return search.terms_agg(common_config.ES_INDEX, "entity_types.keyword", filter_query={"term": {"source": "oso"}})


@mcp.tool()
@_timed
def list_field_values(field: FacetField, limit: int = 50) -> list[dict]:
    """List distinct values (with counts) for one of a fixed set of useful
    fields: data_center_name, networks, sensor_codes, project_name."""
    return search.terms_agg(common_config.ES_INDEX, search.FACETABLE_FIELDS[field], limit)


@mcp.tool()
@_timed
def get_index_stats() -> dict:
    """Document counts in the knowledge base, overall and per source. Call
    this first if you're unsure whether the KB has data before searching."""
    return search.index_stats(common_config.ES_INDEX)


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host=config.MCP_HOST, port=config.MCP_PORT)

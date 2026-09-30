"""MCP resources (read-only value lists) and prompt templates.

`register(mcp)` attaches them to the server. Value lists are read live from
the index, so they always match what the filter arguments will accept; every
resource is JSON with the same shape:

    {"name": ..., "description": ..., "used_by": [...], "values": [{"value": ..., "count": ...}]}
"""
import json
from typing import NamedTuple

from kb_common import config as common_config
from kb_mcp import search

VALUE_LIST_URI = "kb://value-lists/{}"
_ARGO = {"term": {"source": "euro_argo"}}
_OSO = {"term": {"source": "oso"}}
_MAX_VALUES = 500


class ValueList(NamedTuple):
    """Represents a value list resource for the MCP server."""
    slug: str
    title: str
    description: str
    field: str
    source_filter: dict | None
    used_by: tuple[str, ...]


VALUE_LISTS = (
    ValueList(
        "ocean-basins", "Ocean basins",
        "Broad ocean basins (Atlantic, Pacific, Indian, Arctic, Southern) with the number of Argo floats last located in each.",
        "ocean_region.keyword", _ARGO,
        ("search_knowledge_base(ocean_basin)", "list_argo_floats_by_ocean(ocean_region)"),
    ),
    ValueList(
        "seas", "Named seas",
        "Specific named seas (IHO), with the number of Argo floats last located in each.",
        "sea_area.keyword", _ARGO,
        ("search_knowledge_base(sea_area)", "list_argo_floats_by_sea(sea_area)"),
    ),
    ValueList(
        "sensor-codes", "Sensor codes",
        "Argo sensor codes (e.g. DOXY = dissolved oxygen) with the number of floats carrying each.",
        "sensor_codes", _ARGO,
        ("search_knowledge_base(sensor)", "list_argo_floats_by_sensor(sensor_code)"),
    ),
    ValueList(
        "statuses", "Float statuses",
        "Platform status values as stored upstream, with float counts.",
        "status_code", _ARGO,
        ("search_knowledge_base(status)", "list_argo_floats_by_status_and_deployment(status)"),
    ),
    ValueList(
        "data-centers", "Data centers",
        "Data centers processing Argo float data, with float counts.",
        "data_center_name.keyword", _ARGO,
        ("list_field_values(field=data_center_name)",),
    ),
    ValueList(
        "networks", "Argo networks",
        "Argo networks (e.g. Core, BGC) with float counts.",
        "networks", _ARGO,
        ("list_field_values(field=networks)",),
    ),
    ValueList(
        "projects", "Projects",
        "Projects the Argo floats belong to, with float counts.",
        "project_name.keyword", _ARGO,
        ("list_field_values(field=project_name)",),
    ),
    ValueList(
        "oso-entity-types", "OSO entity types",
        "OSO ontology entity types (Organization, Platform, Site, ...) with entity counts.",
        "entity_types.keyword", _OSO,
        ("list_oso_entities_by_type(entity_type)",),
    ),
)


def read_value_list(spec: ValueList) -> str:
    """Reads the value list for the given spec and returns it as a JSON string."""
    values = search.terms_agg(common_config.ES_INDEX, spec.field, _MAX_VALUES, spec.source_filter)
    return json.dumps(
        {"name": spec.slug, "description": spec.description, "used_by": list(spec.used_by), "values": values},
        ensure_ascii=False,
        indent=2,
    )


def read_sources() -> str:
    """Reads the sources for the knowledge base and returns them as a JSON string."""

    stats = search.index_stats(common_config.ES_INDEX)
    return json.dumps(
        {
            "name": "sources",
            "description": "Data sources in the knowledge base, with document counts.",
            "used_by": ["search_knowledge_base(source)"],
            "values": [{"value": k, "count": v} for k, v in sorted(stats["by_source"].items())],
        },
        indent=2,
    )


# --- prompts -------------------------------------------------------------

def find_argo_floats_prompt(region: str, sensor: str = "", status: str = "", deployed_after: str = "") -> str:
    criteria = [f"located in {region}"]
    if sensor:
        criteria.append(f"carrying the {sensor} sensor")
    if status:
        criteria.append(f'with status "{status}"')
    if deployed_after:
        criteria.append(f"deployed since {deployed_after}")
    return f"""Find Argo floats {', '.join(criteria)}.

Work like this:
1. Resolve the region to an exact value. Read the kb://value-lists/seas and kb://value-lists/ocean-basins resources (or call list_seas / list_ocean_regions). A named sea goes in `sea_area`; a broad basin goes in `ocean_basin`. Do not guess spellings.
2. Do the same for the sensor (kb://value-lists/sensor-codes) and status (kb://value-lists/statuses) if they were requested.
3. Call search_knowledge_base with those values as structured filters (`sea_area` / `ocean_basin`, `sensor`, `status`, and `date_min`/`date_max` for last-report dates) and use the free-text `query` only for what the filters cannot express, e.g. an operator or project name. Filters are strict: every returned float satisfies all of them.
4. For a deployment window use list_argo_floats_by_status_and_deployment (`deployed_after`, `deployed_before`, a year such as "2022" is enough).
5. If nothing comes back, loosen one filter at a time (a sea to its ocean basin, then drop the date bound) and say which constraint was relaxed.
6. Report each float with its WMO id, status, deployment date and url."""


def investigate_float_prompt(wmo: str) -> str:
    return f"""Give a briefing on Argo float {wmo}.

1. Call get_argo_float with wmo="{wmo}". If it returns null, say the float is not in the knowledge base and stop.
2. Summarise: platform type and maker, operator and project, sensors, deployment date and position, status, last reported cycle and position, sea and ocean basin, number of cycles and mission length, and any data-quality flags.
3. If the record has an `oso_organization_id`, call get_oso_entity with it to describe the operating organization.
4. Optionally call find_argo_floats_near with the float's last position (last_cycle_lat / last_cycle_lon) to list neighbouring floats.
Only state what the records contain."""


def deployment_history_prompt(status: str = "active", since_year: str = "2020") -> str:
    return f"""List Argo floats with status "{status}" deployed since {since_year}.

1. Check the stored status values with list_argo_statuses (or kb://value-lists/statuses); "active" and "inactive" are understood and map to the stored codes.
2. Call list_argo_floats_by_status_and_deployment with status="{status}" and deployed_after="{since_year}". Results come newest deployment first; raise `limit` if you need more.
3. Present the floats as a short table: WMO id, status_code, deployment_date, url. State how many were shown."""


def explore_knowledge_base_prompt() -> str:
    return """Orient yourself in the Ifremer knowledge base before answering questions.

1. Call get_index_stats to confirm there is data and see the split between euro_argo floats and OSO ontology entities.
2. Read the value lists (kb://value-lists/...): ocean-basins, seas, sensor-codes, statuses, data-centers, networks, projects, oso-entity-types.
3. Choose the tool by question type:
   - a float you can name by WMO id -> get_argo_float; an OSO id -> get_oso_entity
   - by place -> find_argo_floats_near / find_argo_floats_in_box, or list_argo_floats_by_sea / _by_ocean
   - by status or deployment date -> list_argo_floats_by_status_and_deployment
   - open-ended or combined criteria -> search_knowledge_base with structured filters
4. Prefer structured filters over words in the free-text query: filters are enforced, free text only ranks. Filters apply to Euro-Argo floats only, so OSO records are excluded when any is set."""


def _reader(spec: ValueList):
    """A zero-argument reader bound to `spec` (the SDK rejects extra parameters)."""

    def read() -> str:
        return read_value_list(spec)

    return read


def register(mcp) -> None:
    """Attach the value-list resources and prompt templates to `mcp`."""
    for spec in VALUE_LISTS:
        mcp.resource(
            VALUE_LIST_URI.format(spec.slug),
            name=spec.slug,
            title=spec.title,
            description=spec.description,
            mime_type="application/json",
        )(_reader(spec))

    mcp.resource(
        VALUE_LIST_URI.format("sources"),
        name="sources",
        title="Data sources",
        description="Data sources in the knowledge base (euro_argo, oso) with document counts.",
        mime_type="application/json",
    )(read_sources)

    mcp.prompt(
        name="find_argo_floats",
        title="Find Argo floats by region, sensor, status and date",
        description="Guides a search for floats matching structured criteria, using value lists and strict filters.",
    )(find_argo_floats_prompt)
    mcp.prompt(
        name="investigate_float",
        title="Brief me on an Argo float",
        description="Fetch one float by WMO id, its operating organization and nearby floats.",
    )(investigate_float_prompt)
    mcp.prompt(
        name="recent_deployments",
        title="Floats by status and deployment period",
        description="List floats with a given status deployed since a given year.",
    )(deployment_history_prompt)
    mcp.prompt(
        name="explore_knowledge_base",
        title="Explore the knowledge base",
        description="How to discover what data exists and which tool fits which question.",
    )(explore_knowledge_base_prompt)

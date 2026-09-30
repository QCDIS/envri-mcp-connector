import asyncio
import json
import re

import jsonschema
import pytest

from kb_mcp import extras, server


def _run(coro):
    return asyncio.run(coro)


def _tools():
    return {t.name: t for t in _run(server.mcp.list_tools())}


def _call(name, args):
    return _run(server.mcp.call_tool(name, args))


# --- output schemas ---

EXPECTED_TOOLS = {
    "search_knowledge_base", "get_argo_float", "get_oso_entity", "find_argo_floats_near",
    "find_argo_floats_in_box", "list_argo_floats_by_sea", "list_argo_floats_by_ocean",
    "list_argo_floats_by_sensor", "list_argo_floats_by_status_and_deployment", "list_argo_statuses",
    "list_oso_entities_by_type", "list_argo_floats_by_organization", "list_seas",
    "list_ocean_regions", "list_oso_entity_types", "list_field_values", "get_index_stats",
}


def test_every_tool_declares_an_object_output_schema():
    tools = _tools()
    assert set(tools) == EXPECTED_TOOLS
    for name, tool in tools.items():
        assert tool.output_schema, f"{name} has no output schema"
        assert tool.output_schema["type"] == "object", name
        jsonschema.Draft202012Validator.check_schema(tool.output_schema)


def test_output_schemas_are_typed_not_open_objects():
    # Regression guard for the old `list[dict]` / bare `dict` returns.
    hit_schema = _tools()["list_argo_floats_by_sea"].output_schema["$defs"]["KbHit"]
    assert hit_schema["properties"]["_id"]["type"] == "string"
    assert "_id" in hit_schema["required"]
    assert "highlights" in _tools()["search_knowledge_base"].output_schema["$defs"]["SearchHit"]["properties"]
    assert set(_tools()["get_index_stats"].output_schema["properties"]) == {"total", "by_source"}


_HIT = {"_id": "euro_argo:1900001", "score": 1.5, "url": "https://x/1", "source": "euro_argo", "summary_text": "A float."}

_SAMPLE_RESULTS = {
    "search_knowledge_base": ("search", "search", {"query": "x"}, [{**_HIT, "highlights": ["a <em>float</em>"]}]),
    "get_argo_float": ("search", "get_by_id", {"wmo": "1900001"}, {
        "_id": "euro_argo:1900001", "url": "u", "source": "euro_argo", "wmo": "1900001", "sensor_codes": ["DOXY"],
        "last_cycle_geopoint": {"lat": 1.0, "lon": 2.0}, "num_cycles": 10, "has_quality_flags": False,
        "embedding": [0.1, 0.2],
    }),
    "get_oso_entity": ("search", "get_by_id", {"oso_id": "Ifremer"}, {
        "_id": "oso:Ifremer", "url": "u", "source": "oso", "entity_types": ["Organization"],
        "external_ids": {"ror": "abc", "alt": ["1", "2"]},
    }),
    "find_argo_floats_near": ("search", "geo_distance", {"lat": 1, "lon": 2}, [
        {"_id": "euro_argo:1", "distance_km": 4.2, "url": "u", "source": "euro_argo", "summary_text": "x"}]),
    "find_argo_floats_in_box": ("search", "geo_bounding_box", {"min_lat": 0, "max_lat": 1, "min_lon": 0, "max_lon": 1}, [_HIT]),
    "list_argo_floats_by_sea": ("search", "term_filter", {"sea_area": "Black Sea"}, [_HIT]),
    "list_argo_floats_by_ocean": ("search", "term_filter", {"ocean_region": "Pacific Ocean"}, [_HIT]),
    "list_argo_floats_by_sensor": ("search", "term_filter", {"sensor_code": "DOXY"}, [_HIT]),
    "list_argo_floats_by_organization": ("search", "term_filter", {"oso_organization_id": "Ifremer"}, [_HIT]),
    "list_oso_entities_by_type": ("search", "term_filter", {"entity_type": "Organization"}, [_HIT]),
    "list_argo_floats_by_status_and_deployment": ("search", "filter_floats", {"status": "active"}, [
        {"_id": "euro_argo:1", "url": "u", "source": "euro_argo", "summary_text": "x",
         "status_code": "ACTIVE", "deployment_date": "2022-03-01T00:00:00Z"}]),
    "list_argo_statuses": ("search", "terms_agg", {}, [{"value": "ACTIVE", "count": 3}]),
    "list_seas": ("search", "terms_agg", {}, [{"value": "Black Sea", "count": 3}]),
    "list_ocean_regions": ("search", "terms_agg", {}, [{"value": "Pacific Ocean", "count": 3}]),
    "list_oso_entity_types": ("search", "terms_agg", {}, [{"value": "Site", "count": 3}]),
    "list_field_values": ("search", "terms_agg", {"field": "networks"}, [{"value": "BGC", "count": 3}]),
    "get_index_stats": ("search", "index_stats", {}, {"total": 3, "by_source": {"euro_argo": 2, "oso": 1}}),
}


def test_sample_table_covers_every_tool():
    assert set(_SAMPLE_RESULTS) == EXPECTED_TOOLS


@pytest.mark.parametrize("tool", sorted(EXPECTED_TOOLS))
def test_tool_results_validate_against_their_declared_schema(monkeypatch, tool):
    _, fn_name, args, payload = _SAMPLE_RESULTS[tool]
    monkeypatch.setattr(server.search, fn_name, lambda *a, **k: payload)

    result = _call(tool, args)

    jsonschema.validate(result.structured_content, _tools()[tool].output_schema)


def test_wire_format_keeps_the_underscore_id_key(monkeypatch):
    monkeypatch.setattr(server.search, "term_filter", lambda *a, **k: [_HIT])

    hit = _call("list_argo_floats_by_sea", {"sea_area": "x"}).structured_content["result"][0]

    assert hit["_id"] == "euro_argo:1900001"
    assert "id" not in hit


def test_unknown_fields_are_preserved(monkeypatch):
    monkeypatch.setattr(server.search, "get_by_id", lambda *a, **k: {"_id": "euro_argo:1", "brand_new_field": 7})

    record = _call("get_argo_float", {"wmo": "1"}).structured_content["result"]

    assert record["brand_new_field"] == 7


def test_missing_document_is_a_valid_null_result(monkeypatch):
    monkeypatch.setattr(server.search, "get_by_id", lambda *a, **k: None)

    assert _call("get_argo_float", {"wmo": "1"}).structured_content == {"result": None}


def test_result_that_violates_the_schema_is_rejected(monkeypatch):
    # distance_km is required on NearbyFloat
    monkeypatch.setattr(server.search, "geo_distance", lambda *a, **k: [{"_id": "euro_argo:1"}])

    with pytest.raises(Exception):
        _call("find_argo_floats_near", {"lat": 1, "lon": 2})


def test_wrongly_typed_field_is_rejected(monkeypatch):
    monkeypatch.setattr(server.search, "index_stats", lambda index: {"total": "many", "by_source": {}})

    with pytest.raises(Exception):
        _call("get_index_stats", {})


# --- resources ---


def test_value_lists_are_listed_as_json_resources():
    resources = _run(server.mcp.list_resources())
    uris = {str(r.uri) for r in resources}

    assert uris == {extras.VALUE_LIST_URI.format(s.slug) for s in extras.VALUE_LISTS} | {"kb://value-lists/sources"}
    assert {"kb://value-lists/ocean-basins", "kb://value-lists/sensor-codes"} <= uris
    assert all(r.mime_type == "application/json" and r.description for r in resources)


def test_reading_a_value_list_queries_the_matching_field(monkeypatch):
    calls = []
    monkeypatch.setattr(
        extras.search, "terms_agg", lambda index, field, limit=50, filter_query=None: calls.append((field, limit, filter_query))
        or [{"value": "Pacific Ocean", "count": 12}],
    )

    (content,) = _run(server.mcp.read_resource("kb://value-lists/ocean-basins"))
    body = json.loads(content.content)

    assert calls == [("ocean_region.keyword", 500, {"term": {"source": "euro_argo"}})]
    assert body["name"] == "ocean-basins"
    assert body["values"] == [{"value": "Pacific Ocean", "count": 12}]
    assert "search_knowledge_base(ocean_basin)" in body["used_by"]


def test_every_value_list_reads_with_the_shared_shape(monkeypatch):
    monkeypatch.setattr(extras.search, "terms_agg", lambda *a, **k: [{"value": "v", "count": 1}])
    for spec in extras.VALUE_LISTS:
        body = json.loads(extras.read_value_list(spec))
        assert set(body) == {"name", "description", "used_by", "values"}
        assert body["name"] == spec.slug


def test_sources_resource_reports_counts(monkeypatch):
    monkeypatch.setattr(extras.search, "index_stats", lambda index: {"total": 3, "by_source": {"oso": 1, "euro_argo": 2}})

    body = json.loads(extras.read_sources())

    assert body["values"] == [{"value": "euro_argo", "count": 2}, {"value": "oso", "count": 1}]


def test_value_list_used_by_references_real_tools_and_params():
    tools = _tools()
    for spec in extras.VALUE_LISTS:
        for ref in spec.used_by:
            tool_name = ref.split("(")[0]
            assert tool_name in tools, ref
            param = re.search(r"\((?:field=)?(\w+)\)", ref).group(1)
            props = tools[tool_name].input_schema["properties"]
            assert param in props or param in props.get("field", {}).get("enum", []), ref


# --- prompts ---


def test_prompts_are_listed():
    prompts = {p.name: p for p in _run(server.mcp.list_prompts())}

    assert set(prompts) == {"find_argo_floats", "investigate_float", "recent_deployments", "explore_knowledge_base"}
    assert all(p.description for p in prompts.values())
    assert [a.name for a in prompts["find_argo_floats"].arguments if a.required] == ["region"]


def test_find_argo_floats_prompt_renders_arguments():
    result = _run(server.mcp.get_prompt(
        "find_argo_floats", {"region": "Sea of Japan", "sensor": "DOXY", "deployed_after": "2022"}
    ))
    text = result.messages[0].content.text

    assert "Sea of Japan" in text and "DOXY" in text and "2022" in text
    assert "kb://value-lists/seas" in text


def test_prompts_only_reference_registered_tools_and_resources():
    tools = set(_tools())
    resources = {str(r.uri) for r in _run(server.mcp.list_resources())}
    rendered = [
        _run(server.mcp.get_prompt(name, args)).messages[0].content.text
        for name, args in (
            ("find_argo_floats", {"region": "Black Sea", "sensor": "DOXY", "status": "active", "deployed_after": "2020"}),
            ("investigate_float", {"wmo": "6902919"}),
            ("recent_deployments", {}),
            ("explore_knowledge_base", {}),
        )
    ]
    for text in rendered:
        for name in re.findall(r"\b((?:search_|get_|find_|list_)\w+)\b", text):
            assert name in tools, name
        for uri in re.findall(r"kb://value-lists/[\w-]+", text):
            assert uri in resources, uri

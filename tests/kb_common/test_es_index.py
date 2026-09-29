from kb_argo.es_mapping import EXTRA_PROPERTIES as ARGO_PROPERTIES
from kb_common import es_index
from kb_mcp import search
from kb_oso.es_mapping import EXTRA_PROPERTIES as OSO_PROPERTIES


def _normalized_paths(properties: dict) -> set:
    """Field paths ('sea_area.keyword', 'networks', ...) mapped as a keyword
    with the lowercase normalizer."""
    paths = set()
    for name, mapping in properties.items():
        if mapping.get("type") == "keyword" and mapping.get("normalizer") == es_index.LOWERCASE_NORMALIZER:
            paths.add(name)
        for sub, sub_mapping in (mapping.get("fields") or {}).items():
            if sub_mapping.get("type") == "keyword" and sub_mapping.get("normalizer") == es_index.LOWERCASE_NORMALIZER:
                paths.add(f"{name}.{sub}")
    return paths


def test_build_mapping_defines_lowercase_normalizer_in_settings():
    mapping = es_index.build_mapping(dims=8, extra_properties=ARGO_PROPERTIES)

    normalizer = mapping["settings"]["analysis"]["normalizer"]["lowercase"]
    assert normalizer == {"type": "custom", "char_filter": [], "filter": ["lowercase"]}


def test_lowercase_keyword_helpers():
    assert es_index.lowercase_keyword() == {"type": "keyword", "normalizer": "lowercase"}
    assert es_index.text_and_lowercase_keyword() == {
        "type": "text",
        "fields": {"keyword": {"type": "keyword", "normalizer": "lowercase"}},
    }


def test_every_normalizer_reference_is_defined_in_settings():
    defined = set(es_index.INDEX_SETTINGS["analysis"]["normalizer"])
    for properties in (ARGO_PROPERTIES, OSO_PROPERTIES):
        for mapping in properties.values():
            used = {mapping.get("normalizer")} | {f.get("normalizer") for f in (mapping.get("fields") or {}).values()}
            assert used - {None} <= defined


def test_all_filter_and_facet_fields_are_normalized():
    merged = {**ARGO_PROPERTIES, **OSO_PROPERTIES}
    normalized = _normalized_paths(merged)

    # every field the list_* / facet tools filter or aggregate on
    assert search.CASE_INSENSITIVE_FIELDS <= normalized
    assert set(search.FACETABLE_FIELDS.values()) <= normalized


def test_specific_target_fields_carry_the_normalizer():
    argo = _normalized_paths(ARGO_PROPERTIES)
    oso = _normalized_paths(OSO_PROPERTIES)

    assert {
        "sea_area.keyword",
        "ocean_region.keyword",
        "data_center_name.keyword",
        "project_name.keyword",
        "owner.keyword",
        "principal_investigator.keyword",
        "networks",
        "sensor_codes",
        "oso_organization_id",
    } <= argo
    assert oso == {"entity_types.keyword"}


def test_exact_identifier_fields_are_not_normalized():
    argo = _normalized_paths(ARGO_PROPERTIES)

    for field in ("wmo", "status_code", "country_code", "maker", "model"):
        assert field not in argo

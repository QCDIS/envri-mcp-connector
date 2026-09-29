import logging

import pytest
from elasticsearch import ConnectionError as ESConnectionError

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


# --- embedding model guard -------------------------------------------------

MODEL = "Qwen/Qwen3-Embedding-0.6B"


class _FakeIndices:
    def __init__(self, mapping=None, get_error=None):
        self._mapping = mapping  # {"mappings": {...}} or None -> index doesn't exist
        self._get_error = get_error
        self.created = []
        self.put_calls = []

    def exists(self, index):
        if self._get_error:
            raise self._get_error
        return self._mapping is not None

    def get_mapping(self, index):
        return {"concrete-index-name": self._mapping}

    def create(self, index, body):
        self.created.append((index, body))

    def put_mapping(self, index, **kwargs):
        self.put_calls.append((index, kwargs))


class _FakeClient:
    def __init__(self, **kwargs):
        self.indices = _FakeIndices(**kwargs)


def _index_mapping(model=MODEL, meta_dims=1024, mapped_dims=1024, with_meta=True, extra_meta=None):
    mappings = {"properties": {"embedding": {"type": "dense_vector", "dims": mapped_dims}}}
    if with_meta:
        mappings["_meta"] = {"embedding_model": model, "embedding_dims": meta_dims, **(extra_meta or {})}
    return {"mappings": mappings}


def test_build_mapping_records_embedding_model_and_dims_in_meta():
    mapping = es_index.build_mapping(1024, embedding_model=MODEL)

    assert mapping["mappings"]["_meta"] == {"embedding_model": MODEL, "embedding_dims": 1024}
    assert mapping["mappings"]["properties"]["embedding"]["dims"] == 1024


def test_build_mapping_defaults_model_to_configured(monkeypatch):
    monkeypatch.setattr(es_index.config, "EMBEDDING_MODEL", "some/model")

    assert es_index.build_mapping(8)["mappings"]["_meta"]["embedding_model"] == "some/model"


def test_find_embedding_mismatches_matching_config_is_empty():
    assert es_index.find_embedding_mismatches({"embedding_model": MODEL, "embedding_dims": 1024}, 1024, MODEL, 1024) == []


def test_find_embedding_mismatches_reports_each_problem():
    meta = {"embedding_model": "other/model", "embedding_dims": 768}

    problems = es_index.find_embedding_mismatches(meta, 768, MODEL, 1024)

    assert len(problems) == 3
    assert any("other/model" in p and MODEL in p for p in problems)
    assert any("768" in p and "1024" in p for p in problems)


def test_verify_passes_when_model_and_dims_match(caplog):
    client = _FakeClient(mapping=_index_mapping())

    with caplog.at_level(logging.INFO):
        es_index.verify_embedding_model(client, MODEL, 1024, index="idx")

    assert "check passed" in caplog.text


def test_verify_raises_and_logs_clear_error_on_model_mismatch(caplog):
    client = _FakeClient(mapping=_index_mapping(model="intfloat/multilingual-e5-large"))

    with caplog.at_level(logging.ERROR), pytest.raises(es_index.EmbeddingMismatchError) as exc:
        es_index.verify_embedding_model(client, MODEL, 1024, index="idx")

    message = str(exc.value)
    assert "intfloat/multilingual-e5-large" in message and MODEL in message and "idx" in message
    assert "Embedding configuration doesn't match index" in caplog.text


def test_verify_raises_on_dimension_mismatch_in_meta():
    client = _FakeClient(mapping=_index_mapping(meta_dims=768, mapped_dims=768))

    with pytest.raises(es_index.EmbeddingMismatchError, match="768"):
        es_index.verify_embedding_model(client, MODEL, 1024, index="idx")


def test_verify_raises_when_vector_mapping_disagrees_even_if_meta_matches():
    client = _FakeClient(mapping=_index_mapping(meta_dims=1024, mapped_dims=768))

    with pytest.raises(es_index.EmbeddingMismatchError, match="768-dim"):
        es_index.verify_embedding_model(client, MODEL, 1024, index="idx")


def test_verify_skips_when_index_does_not_exist_yet(caplog):
    with caplog.at_level(logging.INFO):
        es_index.verify_embedding_model(_FakeClient(mapping=None), MODEL, 1024, index="idx")

    assert "doesn't exist yet" in caplog.text


def test_verify_legacy_index_without_meta_checks_dims_and_warns(caplog):
    client = _FakeClient(mapping=_index_mapping(with_meta=False))

    with caplog.at_level(logging.WARNING):
        es_index.verify_embedding_model(client, MODEL, 1024, index="idx")

    assert "no embedding metadata" in caplog.text


def test_verify_legacy_index_with_wrong_dims_still_fails():
    client = _FakeClient(mapping=_index_mapping(with_meta=False, mapped_dims=768))

    with pytest.raises(es_index.EmbeddingMismatchError):
        es_index.verify_embedding_model(client, MODEL, 1024, index="idx")


def test_verify_does_not_halt_when_elasticsearch_is_unreachable(caplog):
    client = _FakeClient(get_error=ESConnectionError("connection refused"))

    with caplog.at_level(logging.WARNING):
        es_index.verify_embedding_model(client, MODEL, 1024, index="idx")

    assert "unreachable" in caplog.text


def test_ensure_index_creates_new_index_with_meta():
    client = _FakeClient(mapping=None)

    es_index.ensure_index(client, dims=1024, index="idx", embedding_model=MODEL)

    (index, body), = client.indices.created
    assert index == "idx"
    assert body["mappings"]["_meta"] == {"embedding_model": MODEL, "embedding_dims": 1024}


def test_ensure_index_refuses_to_mix_models_into_existing_index():
    client = _FakeClient(mapping=_index_mapping(model="other/model"))

    with pytest.raises(es_index.EmbeddingMismatchError):
        es_index.ensure_index(client, dims=1024, index="idx", embedding_model=MODEL)

    assert client.indices.put_calls == []


def test_ensure_index_stamps_legacy_index_and_keeps_other_meta_keys():
    client = _FakeClient(mapping=_index_mapping(with_meta=False))
    client.indices._mapping["mappings"]["_meta"] = {"owner": "ifremer"}

    es_index.ensure_index(client, dims=1024, index="idx", embedding_model=MODEL)

    (_, kwargs), = client.indices.put_calls
    assert kwargs["meta"] == {"owner": "ifremer", "embedding_model": MODEL, "embedding_dims": 1024}

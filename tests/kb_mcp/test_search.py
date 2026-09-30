from datetime import date

from elasticsearch import NotFoundError

from kb_mcp import search


class _FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def raise_for_status(self):
        pass

    def json(self):
        return self._json_data


class _FakeESClient:
    def __init__(self, get_result=None, get_exception=None, search_result=None):
        self._get_result = get_result
        self._get_exception = get_exception
        self._search_result = search_result
        self.search_calls = []

    def get(self, index, id):
        if self._get_exception is not None:
            raise self._get_exception
        return self._get_result

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return self._search_result


_CANNED_HITS = {
    "hits": {
        "hits": [
            {"_id": "euro_argo:1900001", "_score": 1.5, "_source": {"source": "euro_argo", "summary_text": "A float."}},
        ]
    }
}


def test_source_url_euro_argo():
    assert search._source_url("euro_argo:6902919") == "https://fleetmonitoring.euro-argo.eu/float/6902919"


def test_source_url_oso():
    assert search._source_url("oso:Ifremer") == "https://w3id.org/earthsemantics/OSO#Ifremer"


def test_source_url_unknown_source_returns_none():
    assert search._source_url("weird:123") is None


def test_get_by_id_not_found_returns_none(monkeypatch):
    client = _FakeESClient(get_exception=NotFoundError("not_found", None, None))
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    assert search.get_by_id("idx", "euro_argo:doesnotexist") is None


def test_get_by_id_other_exception_propagates(monkeypatch):
    client = _FakeESClient(get_exception=ConnectionError("boom"))
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    try:
        search.get_by_id("idx", "euro_argo:6902919")
        raise AssertionError("expected ConnectionError to propagate, not be swallowed")
    except ConnectionError:
        pass


def test_get_by_id_found_attaches_url(monkeypatch):
    client = _FakeESClient(get_result={"_id": "euro_argo:6902919", "_source": {"summary_text": "A float."}})
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    result = search.get_by_id("idx", "euro_argo:6902919")

    assert result["_id"] == "euro_argo:6902919"
    assert result["url"] == "https://fleetmonitoring.euro-argo.eu/float/6902919"
    assert result["summary_text"] == "A float."


def test_search_sends_bearer_token_and_query(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _FakeResponse([])

    monkeypatch.setattr(search._session, "post", fake_post)
    monkeypatch.setattr(search.config, "KB_API_URL", "http://kb-api-test:8080")
    monkeypatch.setattr(search.config, "KB_MCP_INTERNAL_TOKEN", "internal-token-abc")

    search.search("idx", "float temperature", k=5)

    assert captured["url"] == "http://kb-api-test:8080/internal/search"
    assert captured["headers"] == {"Authorization": "Bearer internal-token-abc"}
    assert captured["json"]["query"] == "float temperature"
    assert captured["json"]["k"] == 5


def test_search_forwards_structured_filters(monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["json"] = json
        return _FakeResponse([])

    monkeypatch.setattr(search._session, "post", fake_post)

    search.search(
        "idx", "float", sea_area="Sea of Japan", ocean_basin="Pacific Ocean", sensor="DOXY",
        status="O", date_min=date(2024, 1, 1), date_max=date(2024, 12, 31),
    )

    assert captured["json"]["sea_area"] == "Sea of Japan"
    assert captured["json"]["ocean_basin"] == "Pacific Ocean"
    assert captured["json"]["sensor"] == "DOXY"
    assert captured["json"]["status"] == "O"
    assert captured["json"]["date_min"] == "2024-01-01"
    assert captured["json"]["date_max"] == "2024-12-31"


def test_search_omits_unset_filters_from_payload(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        search._session, "post", lambda url, json=None, **k: captured.update(json=json) or _FakeResponse([])
    )

    search.search("idx", "float")

    for name in ("sea_area", "ocean_basin", "sensor", "status", "date_min", "date_max"):
        assert name not in captured["json"]


def test_search_attaches_url_to_each_result(monkeypatch):
    monkeypatch.setattr(
        search._session,
        "post",
        lambda *a, **k: _FakeResponse([{"_id": "oso:Ifremer", "score": 2.0}]),
    )

    results = search.search("idx", "ifremer")

    assert results[0]["url"] == "https://w3id.org/earthsemantics/OSO#Ifremer"


def test_term_filter_sends_term_query(monkeypatch):
    client = _FakeESClient(search_result=_CANNED_HITS)
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    results = search.term_filter("idx", "sea_area.keyword", "Black Sea", limit=5)

    assert client.search_calls[0]["query"] == {"term": {"sea_area.keyword": "black sea"}}
    assert client.search_calls[0]["size"] == 5
    assert results[0]["_id"] == "euro_argo:1900001"


def test_term_filter_casing_variants_build_identical_queries(monkeypatch):
    client = _FakeESClient(search_result=_CANNED_HITS)
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    for variant in ("black sea", "BLACK SEA", "Black Sea", "  bLaCk SeA "):
        search.term_filter("idx", "sea_area.keyword", variant)

    queries = [call["query"] for call in client.search_calls]
    assert queries == [{"term": {"sea_area.keyword": "black sea"}}] * 4


def test_term_filter_normalizes_every_case_insensitive_field(monkeypatch):
    client = _FakeESClient(search_result=_CANNED_HITS)
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    for field in sorted(search.CASE_INSENSITIVE_FIELDS):
        search.term_filter("idx", field, "MiXeD Case")

    assert all(call["query"]["term"][field] == "mixed case" for call, field in zip(client.search_calls, sorted(search.CASE_INSENSITIVE_FIELDS)))


def test_term_filter_leaves_other_fields_untouched(monkeypatch):
    client = _FakeESClient(search_result=_CANNED_HITS)
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    search.term_filter("idx", "wmo", "ABC123")

    assert client.search_calls[0]["query"] == {"term": {"wmo": "ABC123"}}


def test_filter_floats_builds_term_and_range_queries(monkeypatch):
    client = _FakeESClient(
        search_result={
            "hits": {
                "hits": [
                    {
                        "_id": "euro_argo:1900001",
                        "_source": {
                            "source": "euro_argo", "summary_text": "A float.",
                            "status_code": "ACTIVE", "deployment_date": "2022-03-01",
                        },
                    }
                ]
            }
        }
    )
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    results = search.filter_floats("idx", status="active", deployed_after="2020", deployed_before="2022-06", limit=7)

    call = client.search_calls[0]
    clauses = call["query"]["bool"]["filter"]
    assert {"term": {"source": "euro_argo"}} in clauses
    assert any("terms" in c and "ACTIVE" in c["terms"]["status_code"] for c in clauses)
    assert {"range": {"deployment_date": {"format": "strict_date_optional_time", "gte": "2020", "lte": "2022-06"}}} in clauses
    assert call["size"] == 7
    assert call["sort"] == [{"deployment_date": {"order": "desc", "missing": "_last"}}]
    assert results == [
        {
            "_id": "euro_argo:1900001",
            "url": "https://fleetmonitoring.euro-argo.eu/float/1900001",
            "source": "euro_argo",
            "summary_text": "A float.",
            "status_code": "ACTIVE",
            "deployment_date": "2022-03-01",
        }
    ]


def test_filter_floats_status_only_has_no_range(monkeypatch):
    client = _FakeESClient(search_result=_CANNED_HITS)
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    search.filter_floats("idx", status="inactive")

    assert not any("range" in c for c in client.search_calls[0]["query"]["bool"]["filter"])


def test_filter_floats_requires_a_criterion():
    import pytest

    with pytest.raises(ValueError):
        search.filter_floats("idx")


def test_geo_distance_sends_geo_distance_query_and_computes_distance(monkeypatch):
    client = _FakeESClient(
        search_result={
            "hits": {
                "hits": [
                    {
                        "_id": "euro_argo:1900001",
                        "_source": {"source": "euro_argo", "summary_text": "x"},
                        "sort": [42.34],
                    }
                ]
            }
        }
    )
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    results = search.geo_distance("idx", "last_cycle_geopoint", 10.0, 20.0, 200, limit=5)

    call = client.search_calls[0]
    assert call["query"]["geo_distance"]["distance"] == "200km"
    assert call["query"]["geo_distance"]["last_cycle_geopoint"] == {"lat": 10.0, "lon": 20.0}
    assert results[0]["distance_km"] == 42.3


def test_geo_bounding_box_sends_box_query(monkeypatch):
    client = _FakeESClient(search_result=_CANNED_HITS)
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    search.geo_bounding_box("idx", "last_cycle_geopoint", min_lat=10, max_lat=20, min_lon=30, max_lon=40, limit=5)

    box = client.search_calls[0]["query"]["geo_bounding_box"]["last_cycle_geopoint"]
    assert box["top_left"] == {"lat": 20, "lon": 30}
    assert box["bottom_right"] == {"lat": 10, "lon": 40}


def test_terms_agg_without_filter_omits_query(monkeypatch):
    client = _FakeESClient(search_result={"aggregations": {"values": {"buckets": [{"key": "Black Sea", "doc_count": 3}]}}})
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    results = search.terms_agg("idx", "sea_area.keyword", limit=10)

    assert "query" not in client.search_calls[0]
    assert results == [{"value": "Black Sea", "count": 3}]


def test_terms_agg_with_filter_includes_query(monkeypatch):
    client = _FakeESClient(search_result={"aggregations": {"values": {"buckets": []}}})
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    search.terms_agg("idx", "entity_types.keyword", filter_query={"term": {"source": "oso"}})

    assert client.search_calls[0]["query"] == {"term": {"source": "oso"}}


def test_terms_agg_restores_original_casing_from_sample_doc(monkeypatch):
    client = _FakeESClient(
        search_result={
            "aggregations": {
                "values": {
                    "buckets": [
                        {
                            "key": "black sea",
                            "doc_count": 3,
                            "original": {"hits": {"hits": [{"_source": {"sea_area": "Black Sea"}}]}},
                        }
                    ]
                }
            }
        }
    )
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    results = search.terms_agg("idx", "sea_area.keyword")

    assert results == [{"value": "Black Sea", "count": 3}]
    agg = client.search_calls[0]["aggs"]["values"]
    assert agg["terms"]["field"] == "sea_area.keyword"
    assert agg["aggs"]["original"]["top_hits"]["_source"] == {"includes": ["sea_area"]}


def test_terms_agg_picks_matching_element_of_multivalued_field(monkeypatch):
    client = _FakeESClient(
        search_result={
            "aggregations": {
                "values": {
                    "buckets": [
                        {
                            "key": "bgc",
                            "doc_count": 2,
                            "original": {"hits": {"hits": [{"_source": {"networks": ["Core", "BGC"]}}]}},
                        }
                    ]
                }
            }
        }
    )
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    assert search.terms_agg("idx", "networks") == [{"value": "BGC", "count": 2}]


def test_terms_agg_falls_back_to_key_when_no_sample_doc(monkeypatch):
    client = _FakeESClient(search_result={"aggregations": {"values": {"buckets": [{"key": "black sea", "doc_count": 1}]}}})
    monkeypatch.setattr(search.es_index, "get_client", lambda: client)

    assert search.terms_agg("idx", "sea_area.keyword") == [{"value": "black sea", "count": 1}]

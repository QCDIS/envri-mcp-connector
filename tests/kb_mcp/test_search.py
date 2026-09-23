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

    assert client.search_calls[0]["query"] == {"term": {"sea_area.keyword": "Black Sea"}}
    assert client.search_calls[0]["size"] == 5
    assert results[0]["_id"] == "euro_argo:1900001"


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

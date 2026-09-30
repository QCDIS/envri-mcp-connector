from datetime import date

from kb_common import hybrid_search


def test_strip_noise_words_removes_configured_words():
    result = hybrid_search._strip_noise_words("Argo float temperature Mediterranean")
    assert result == "temperature Mediterranean"


def test_strip_noise_words_is_case_insensitive_and_ignores_punctuation():
    result = hybrid_search._strip_noise_words("Argo, float! ocean sensor")
    assert result == "sensor"


def test_strip_noise_words_all_noise_returns_empty():
    assert hybrid_search._strip_noise_words("Argo float sea ocean") == ""


def test_knn_clause_basic():
    clause = hybrid_search.knn_clause([0.1, 0.2], k=10)
    assert clause == {
        "field": "embedding",
        "query_vector": [0.1, 0.2],
        "k": 10,
        "num_candidates": 100,
    }


def test_knn_clause_num_candidates_floor():
    clause = hybrid_search.knn_clause([0.1], k=1)
    assert clause["num_candidates"] == 50


def test_knn_clause_with_source_filter():
    clause = hybrid_search.knn_clause([0.1], k=5, source="oso")
    assert clause["filter"] == {"term": {"source": "oso"}}


def test_knn_clause_with_boost():
    clause = hybrid_search.knn_clause([0.1], k=5, boost=15.0)
    assert clause["boost"] == 15.0


def test_hybrid_knn_size_floors_at_candidates_constant():
    assert hybrid_search.hybrid_knn_size(5) == hybrid_search.KNN_HYBRID_CANDIDATES
    assert hybrid_search.hybrid_knn_size(10_000) == 10_000


def test_lexical_query_without_source():
    query = hybrid_search.lexical_query("Argo float temperature Mediterranean")
    assert "bool" not in query
    assert query["dis_max"]["queries"][0]["match"]["summary_text"] == "temperature Mediterranean"


def test_lexical_query_with_source_wraps_in_bool_filter():
    query = hybrid_search.lexical_query("temperature", source="euro_argo")
    assert query["bool"]["filter"] == {"term": {"source": "euro_argo"}}
    assert "dis_max" in query["bool"]["must"]


def test_lexical_query_falls_back_to_original_when_all_noise():
    query = hybrid_search.lexical_query("Argo float sea ocean")
    assert query["dis_max"]["queries"][0]["match"]["summary_text"] == "Argo float sea ocean"


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


_CANNED_RESPONSE = {
    "took": 12,
    "hits": {
        "hits": [
            {
                "_id": "euro_argo:1900001",
                "_score": 1.23,
                "_source": {"source": "euro_argo", "summary_text": "A float."},
                "highlight": {"summary_text": ["A <em>float</em>."]},
            }
        ]
    },
}


def _patch_search_deps(monkeypatch, response=None):
    client = _FakeClient(response or _CANNED_RESPONSE)
    monkeypatch.setattr(hybrid_search.es_index, "get_client", lambda: client)
    monkeypatch.setattr(hybrid_search.embed, "embed_query", lambda text: [0.1, 0.2, 0.3])
    return client


def test_search_hybrid_mode_sends_boosted_knn_and_lexical_query(monkeypatch):
    client = _patch_search_deps(monkeypatch)
    hybrid_search.search("kb-index", "float temperature")

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["knn"]["boost"] == hybrid_search.KNN_HYBRID_BOOST
    assert call["knn"]["k"] == hybrid_search.hybrid_knn_size(10)
    assert "query" in call
    assert "highlight" in call


def test_search_knn_mode_has_no_highlight_and_no_query(monkeypatch):
    client = _patch_search_deps(monkeypatch)
    hybrid_search.search("kb-index", "float temperature", mode="knn")

    call = client.calls[0]
    assert "knn" in call
    assert "query" not in call
    assert "highlight" not in call


def test_search_bm25_mode_has_query_and_no_knn(monkeypatch):
    client = _patch_search_deps(monkeypatch)
    hybrid_search.search("kb-index", "float temperature", mode="bm25")

    call = client.calls[0]
    assert "query" in call
    assert "knn" not in call


def test_search_formats_hits_with_highlights(monkeypatch):
    _patch_search_deps(monkeypatch)
    results = hybrid_search.search("kb-index", "float temperature")

    assert results == [
        {
            "_id": "euro_argo:1900001",
            "score": 1.23,
            "source": "euro_argo",
            "summary_text": "A float.",
            "highlights": ["A <em>float</em>."],
        }
    ]


def test_search_disabling_highlight_field_omits_highlights_key(monkeypatch):
    _patch_search_deps(monkeypatch)
    results = hybrid_search.search("kb-index", "float temperature", highlight_field=None)

    assert "highlights" not in results[0]


# --- structured filters ---


def test_build_filters_empty_when_nothing_set():
    assert hybrid_search.build_filters() == []


def test_build_filters_normalizes_case_and_whitespace_on_keyword_fields():
    filters = hybrid_search.build_filters(sea_area="  Sea of Japan ", ocean_basin="PACIFIC Ocean", sensor="DOXY")
    assert filters == [
        {"term": {"sea_area.keyword": "sea of japan"}},
        {"term": {"ocean_region.keyword": "pacific ocean"}},
        {"term": {"sensor_codes": "doxy"}},
    ]


def test_build_filters_status_matches_common_casings():
    (clause,) = hybrid_search.build_filters(status=" Active ")
    assert clause == {"terms": {"status_code": ["Active", "ACTIVE", "active"]}}


def test_build_filters_status_single_letter_deduplicates():
    (clause,) = hybrid_search.build_filters(status="O")
    assert clause == {"terms": {"status_code": ["O", "o"]}}


def test_build_filters_date_range_is_inclusive_and_format_pinned():
    (clause,) = hybrid_search.build_filters(date_min=date(2024, 1, 1), date_max="2024-12-31")
    assert clause == {
        "range": {"last_cycle_date": {"format": "strict_date_optional_time", "gte": "2024-01-01", "lte": "2024-12-31"}}
    }


def test_build_filters_one_sided_date_range():
    (lo,) = hybrid_search.build_filters(date_min=date(2024, 1, 1))
    (hi,) = hybrid_search.build_filters(date_max=date(2024, 1, 1))
    assert "lte" not in lo["range"]["last_cycle_date"] and lo["range"]["last_cycle_date"]["gte"] == "2024-01-01"
    assert "gte" not in hi["range"]["last_cycle_date"] and hi["range"]["last_cycle_date"]["lte"] == "2024-01-01"


def test_knn_clause_combines_source_and_structured_filters_as_list():
    filters = hybrid_search.build_filters(sea_area="Black Sea")
    clause = hybrid_search.knn_clause([0.1], k=5, source="euro_argo", filters=filters)
    assert clause["filter"] == [{"term": {"source": "euro_argo"}}, {"term": {"sea_area.keyword": "black sea"}}]


def test_knn_clause_single_structured_filter_stays_a_plain_query():
    clause = hybrid_search.knn_clause([0.1], k=5, filters=hybrid_search.build_filters(sensor="DOXY"))
    assert clause["filter"] == {"term": {"sensor_codes": "doxy"}}


def test_lexical_query_with_structured_filters_wraps_in_bool_filter():
    filters = hybrid_search.build_filters(sea_area="Sea of Japan", status="O")
    query = hybrid_search.lexical_query("Argo float JMA", filters=filters)
    assert "dis_max" in query["bool"]["must"]
    assert query["bool"]["filter"] == filters


def test_lexical_query_source_and_filters_combine():
    filters = hybrid_search.build_filters(ocean_basin="Pacific Ocean")
    query = hybrid_search.lexical_query("x", source="euro_argo", filters=filters)
    assert query["bool"]["filter"][0] == {"term": {"source": "euro_argo"}}
    assert query["bool"]["filter"][1] == filters[0]


def test_search_hybrid_applies_filters_to_both_knn_and_query(monkeypatch):
    client = _patch_search_deps(monkeypatch)
    hybrid_search.search(
        "kb-index", "float processed by JMA", sea_area="Sea of Japan", sensor="DOXY", date_min=date(2023, 1, 1)
    )

    call = client.calls[0]
    expected = [
        {"term": {"sea_area.keyword": "sea of japan"}},
        {"term": {"sensor_codes": "doxy"}},
        {"range": {"last_cycle_date": {"format": "strict_date_optional_time", "gte": "2023-01-01"}}},
    ]
    assert call["knn"]["filter"] == expected
    assert call["query"]["bool"]["filter"] == expected
    assert call["knn"]["boost"] == hybrid_search.KNN_HYBRID_BOOST


def test_search_knn_and_bm25_modes_also_filter(monkeypatch):
    client = _patch_search_deps(monkeypatch)
    hybrid_search.search("kb-index", "x", mode="knn", ocean_basin="Pacific Ocean")
    hybrid_search.search("kb-index", "x", mode="bm25", ocean_basin="Pacific Ocean")

    clause = {"term": {"ocean_region.keyword": "pacific ocean"}}
    assert client.calls[0]["knn"]["filter"] == clause
    assert client.calls[1]["query"]["bool"]["filter"] == clause


def test_search_without_filters_is_unchanged(monkeypatch):
    client = _patch_search_deps(monkeypatch)
    hybrid_search.search("kb-index", "float temperature")

    call = client.calls[0]
    assert "filter" not in call["knn"]
    assert "bool" not in call["query"]

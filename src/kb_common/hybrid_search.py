"""Core hybrid (BM25 + kNN) search query, used by kb_api - kb_mcp and
scripts/eval_retrieval.py reach it over HTTP via kb_api's /internal/search.

Combines `knn` + `query` in one Elasticsearch request (native score summing)
rather than the `retriever`/`rrf` API, which needs a Platinum/Enterprise
license and 403s on a Basic-license cluster.
"""
from kb_common import embed, es_index, timing

DEFAULT_FIELDS = ("source", "summary_text")

# Boosted fields catch a direct "this float's owner/PI/project IS X" hit,
# instead of relying on X appearing once inside the much longer summary_text
# where BM25 length-normalization would bury it.
ENTITY_BOOST_FIELDS = [
    "owner^3",
    "data_center_name^3",
    "principal_investigator^2",
    "project_name^2",
    "pref_label^2",
    "sea_area^2",
    "ocean_region^2",
]

_ENTITY_NOISE_WORDS = {"argo", "float", "sea", "ocean"}


def _strip_noise_words(query_text: str) -> str:
    tokens = [t for t in query_text.split() if t.strip(".,!?").lower() not in _ENTITY_NOISE_WORDS]
    return " ".join(tokens)


KNN_HYBRID_BOOST = 15.0


def knn_clause(vector: list[float], k: int, source: str | None = None, boost: float | None = None) -> dict:
    clause = {"field": "embedding", "query_vector": vector, "k": k, "num_candidates": max(50, k * 10)}
    if source:
        clause["filter"] = {"term": {"source": source}}
    if boost is not None:
        clause["boost"] = boost
    return clause


KNN_HYBRID_CANDIDATES = 150


def hybrid_knn_size(k: int) -> int:
    return max(k, KNN_HYBRID_CANDIDATES)


def lexical_query(query_text: str, source: str | None = None) -> dict:
    stripped_query = _strip_noise_words(query_text) or query_text
    base = {
        "dis_max": {
            "queries": [
                {"match": {"summary_text": stripped_query}},
                {"multi_match": {"query": stripped_query, "fields": ENTITY_BOOST_FIELDS}},
            ],
            "tie_breaker": 0.3,
        }
    }
    if source:
        return {"bool": {"must": base, "filter": {"term": {"source": source}}}}
    return base


def search(
    index: str,
    query: str,
    k: int = 10,
    mode: str = "hybrid",
    source: str | None = None,
    fields: tuple[str, ...] = DEFAULT_FIELDS,
    highlight_field: str | None = "summary_text",
) -> list[dict]:
    """Hybrid (default) / knn / bm25 search. Returns a list of
    {"_id", "score", "highlights", <requested fields>...} dicts.

    `highlights` is the matched fragment(s) of `highlight_field`, wrapped in
    <em> tags - only for terms matched on the lexical side, so it's skipped
    in `knn` mode (pass highlight_field=None to disable outright)."""
    with timing.stage("es_client_init"):
        client = es_index.get_client()
    with timing.stage("embed"):
        vector = embed.embed_query(query)
    kc = knn_clause(vector, k, source)
    fields = list(fields)

    highlight = (
        {
            "fields": {
                highlight_field: {
                    "type": "unified",
                    "boundary_scanner": "sentence",
                    "boundary_scanner_locale": "en-US",
                    "fragment_size": 300,
                    "number_of_fragments": 2,
                    "order": "score",
                }
            }
        }
        if highlight_field and mode != "knn"
        else None
    )
    search_kwargs = {"index": index, "size": k, "source": fields}
    if highlight:
        search_kwargs["highlight"] = highlight

    with timing.stage("es_query"):
        if mode == "knn":
            resp = client.search(knn=kc, **search_kwargs)
        elif mode == "bm25":
            resp = client.search(query=lexical_query(query, source), **search_kwargs)
        else:
            boosted_kc = knn_clause(vector, hybrid_knn_size(k), source, boost=KNN_HYBRID_BOOST)
            resp = client.search(knn=boosted_kc, query=lexical_query(query, source), **search_kwargs)
    timing.record("es_took", resp["took"] / 1000)

    with timing.stage("format"):
        results = []
        for hit in resp["hits"]["hits"]:
            result = {"_id": hit["_id"], "score": hit["_score"], **{f: hit["_source"].get(f) for f in fields}}
            if highlight:
                result["highlights"] = hit.get("highlight", {}).get(highlight_field, [])
            results.append(result)
    return results

"""Query functions shared between the MCP server (kb_mcp.server) and the
dev eval script (scripts/eval_retrieval.py). Each function returns plain
Python data, not raw ES response bodies.

`search()` calls kb_api's /internal/search over HTTP instead of importing
kb_common.hybrid_search/embed directly, so kb_mcp never needs torch/
sentence-transformers or a GPU. Every other function talks to Elasticsearch
directly via kb_common.es_index and is unaffected.
"""
import requests

from kb_common import es_index, timing
from kb_mcp import config

_SOURCE_FIELDS = ["source", "summary_text"]

# Reused across requests instead of a bare `requests.post` per call, so the
# HTTP connection to kb_api gets pooled/kept-alive rather than
# TCP-handshaking every search.
_session = requests.Session()


def _source_url(doc_id: str) -> str | None:
    """Dereferenceable link to the original data source (not this project),
    mirroring kb_api.format's url mapping - kept in sync manually since
    kb_mcp no longer imports kb_api."""
    source, _, local_id = doc_id.partition(":")
    if source == "euro_argo":
        return f"https://fleetmonitoring.euro-argo.eu/float/{local_id}"
    if source == "oso":
        return f"https://w3id.org/earthsemantics/OSO#{local_id}"
    return None

# Fields safe to expose via the generic list_field_values facet tool: the
# friendly name (what the tool's enum offers) mapped to the actual
# aggregatable ES field (text fields need their .keyword sub-field for
# aggregations - plain keyword fields don't).
FACETABLE_FIELDS = {
    "data_center_name": "data_center_name.keyword",
    "networks": "networks",
    "sensor_codes": "sensor_codes",
    "project_name": "project_name.keyword",
}


def _hits_to_dicts(resp, fields=_SOURCE_FIELDS):
    return [
        {"_id": hit["_id"], "score": hit["_score"], "url": _source_url(hit["_id"]), **{f: hit["_source"].get(f) for f in fields}}
        for hit in resp["hits"]["hits"]
    ]


def search(index: str, query: str, k: int = 10, mode: str = "hybrid", source: str | None = None):
    """Hybrid (default) / knn / bm25 search, via kb_api's /internal/search.
    `mode` is for the eval script's A/B comparisons; the MCP server always
    uses hybrid."""
    with timing.stage("api_http"):
        resp = _session.post(
            f"{config.KB_API_URL}/internal/search",
            json={"query": query, "k": k, "mode": mode, "source": source, "fields": _SOURCE_FIELDS},
            timeout=30,
        )
    resp.raise_for_status()
    results = resp.json()
    for result in results:
        result["url"] = _source_url(result["_id"])
    return results


def get_by_id(index: str, doc_id: str) -> dict | None:
    client = es_index.get_client()
    try:
        resp = client.get(index=index, id=doc_id)
    except Exception:
        return None
    return {"_id": resp["_id"], "url": _source_url(resp["_id"]), **resp["_source"]}


def term_filter(index: str, field: str, value: str, limit: int = 20):
    client = es_index.get_client()
    resp = client.search(index=index, query={"term": {field: value}}, size=limit, source=_SOURCE_FIELDS)
    return _hits_to_dicts(resp)


def geo_distance(index: str, field: str, lat: float, lon: float, radius_km: float, limit: int = 20):
    client = es_index.get_client()
    resp = client.search(
        index=index,
        query={"geo_distance": {"distance": f"{radius_km}km", field: {"lat": lat, "lon": lon}}},
        sort=[{"_geo_distance": {field: {"lat": lat, "lon": lon}, "order": "asc", "unit": "km"}}],
        size=limit,
        source=_SOURCE_FIELDS,
    )
    hits = []
    for hit in resp["hits"]["hits"]:
        distance_km = hit["sort"][0]
        hits.append({
            "_id": hit["_id"],
            "distance_km": round(distance_km, 1),
            "url": _source_url(hit["_id"]),
            **{f: hit["_source"].get(f) for f in _SOURCE_FIELDS},
        })
    return hits


def geo_bounding_box(index: str, field: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float, limit: int = 20):
    client = es_index.get_client()
    resp = client.search(
        index=index,
        query={
            "geo_bounding_box": {
                field: {
                    "top_left": {"lat": max_lat, "lon": min_lon},
                    "bottom_right": {"lat": min_lat, "lon": max_lon},
                }
            }
        },
        size=limit,
        source=_SOURCE_FIELDS,
    )
    return _hits_to_dicts(resp)


def terms_agg(index: str, field: str, limit: int = 50, filter_query: dict | None = None):
    client = es_index.get_client()
    body = {
        "size": 0,
        "aggs": {"values": {"terms": {"field": field, "size": limit}}},
    }
    if filter_query:
        body["query"] = filter_query
    resp = client.search(index=index, **body)
    return [{"value": b["key"], "count": b["doc_count"]} for b in resp["aggregations"]["values"]["buckets"]]


def index_stats(index: str):
    return es_index.index_stats(index)

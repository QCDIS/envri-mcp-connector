"""`search()` calls kb_api's /internal/search over HTTP.
Every other function talks to Elasticsearch directly via kb_common.es_index.
"""
import logging
from datetime import date

import requests
from elasticsearch import NotFoundError

from kb_common import es_index, timing
from kb_common.filters import build_filters
from kb_mcp import config

log = logging.getLogger(__name__)

_SOURCE_FIELDS = ["source", "summary_text"]

_session = requests.Session()


def _source_url(doc_id: str) -> str | None:
    """Dereferenceable link to the original data source,
    mirroring kb_api.format's url mapping"""
    source, _, local_id = doc_id.partition(":")
    if source == "euro_argo":
        return f"https://fleetmonitoring.euro-argo.eu/float/{local_id}"
    if source == "oso":
        return f"https://w3id.org/earthsemantics/OSO#{local_id}"
    return None


FACETABLE_FIELDS = {
    "data_center_name": "data_center_name.keyword",
    "networks": "networks",
    "sensor_codes": "sensor_codes",
    "project_name": "project_name.keyword",
}


# Fields mapped with the lowercase normalizer (see kb_common.es_index), i.e.
# the ones list_* tools filter on. Their term queries take a lowercased value.
CASE_INSENSITIVE_FIELDS = frozenset({
    "sea_area.keyword",
    "ocean_region.keyword",
    "entity_types.keyword",
    "data_center_name.keyword",
    "project_name.keyword",
    "networks",
    "sensor_codes",
    "oso_organization_id",
})


def normalize_filter_value(value: str) -> str:
    """Mirror the index's lowercase normalizer (plus trimming stray whitespace
    that LLM clients tend to add), so "Black Sea", "black sea" and "BLACK SEA "
    build the same query."""
    return value.strip().lower()


def _hits_to_dicts(resp, fields=_SOURCE_FIELDS):
    return [
        {"_id": hit["_id"], "score": hit["_score"], "url": _source_url(hit["_id"]), **{f: hit["_source"].get(f) for f in fields}}
        for hit in resp["hits"]["hits"]
    ]


def search(
    index: str,
    query: str,
    k: int = 10,
    mode: str = "hybrid",
    source: str | None = None,
    sea_area: str | None = None,
    ocean_basin: str | None = None,
    sensor: str | None = None,
    status: str | None = None,
    date_min: date | None = None,
    date_max: date | None = None,
):
    """Hybrid (default) / knn / bm25 search, via kb_api's /internal/search.
    `mode` is for the eval script's A/B comparisons; the MCP server always
    uses hybrid. The optional structured filters are only sent when set."""
    payload = {"query": query, "k": k, "mode": mode, "source": source, "fields": _SOURCE_FIELDS}
    structured = {
        "sea_area": sea_area,
        "ocean_basin": ocean_basin,
        "sensor": sensor,
        "status": status,
        "date_min": date_min.isoformat() if date_min else None,
        "date_max": date_max.isoformat() if date_max else None,
    }
    payload.update({name: value for name, value in structured.items() if value is not None})
    with timing.stage("api_http"):
        resp = _session.post(
            f"{config.KB_API_URL}/internal/search",
            json=payload,
            headers={"Authorization": f"Bearer {config.KB_MCP_INTERNAL_TOKEN}"},
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
    except NotFoundError:
        return None
    except Exception:
        log.exception("get_by_id(%r, %r) failed", index, doc_id)
        raise
    return {"_id": resp["_id"], "url": _source_url(resp["_id"]), **resp["_source"]}


def term_filter(index: str, field: str, value: str, limit: int = 20):
    client = es_index.get_client()
    if field in CASE_INSENSITIVE_FIELDS:
        value = normalize_filter_value(value)
    resp = client.search(index=index, query={"term": {field: value}}, size=limit, source=_SOURCE_FIELDS)
    return _hits_to_dicts(resp)


_FLOAT_LISTING_FIELDS = [*_SOURCE_FIELDS, "status_code", "deployment_date"]


def filter_floats(
    index: str,
    status: str | None = None,
    deployed_after: str | None = None,
    deployed_before: str | None = None,
    limit: int = 20,
):
    """Euro-Argo floats by operational status and/or deployment-date window,
    most recently deployed first. Maps to a `terms` query on status_code and
    a `range` query on deployment_date (see kb_common.filters); at least one
    criterion is required."""
    if not (status or deployed_after or deployed_before):
        raise ValueError("give at least one of status, deployed_after, deployed_before")
    filters = build_filters(
        source="euro_argo", status=status, deployed_after=deployed_after, deployed_before=deployed_before
    )
    client = es_index.get_client()
    resp = client.search(
        index=index,
        query={"bool": {"filter": filters}},
        sort=[{"deployment_date": {"order": "desc", "missing": "_last"}}],
        size=limit,
        source=_FLOAT_LISTING_FIELDS,
    )
    return [
        {
            "_id": hit["_id"],
            "url": _source_url(hit["_id"]),
            **{f: hit["_source"].get(f) for f in _FLOAT_LISTING_FIELDS},
        }
        for hit in resp["hits"]["hits"]
    ]


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


def _display_value(key, source_value):
    """Terms aggregations on a normalized field return lowercase keys; recover
    the original casing ("black sea" -> "Black Sea") from a sample document,
    which for multi-valued fields means picking the matching element."""
    candidates = source_value if isinstance(source_value, list) else [source_value]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.lower() == key:
            return candidate
    return key


def terms_agg(index: str, field: str, limit: int = 50, filter_query: dict | None = None):
    client = es_index.get_client()
    source_field = field.removesuffix(".keyword")
    body = {
        "size": 0,
        "aggs": {
            "values": {
                "terms": {"field": field, "size": limit},
                # One sample doc per bucket, only to recover original casing.
                "aggs": {"original": {"top_hits": {"size": 1, "_source": {"includes": [source_field]}}}},
            }
        },
    }
    if filter_query:
        body["query"] = filter_query
    resp = client.search(index=index, **body)
    results = []
    for bucket in resp["aggregations"]["values"]["buckets"]:
        key = bucket["key"]
        value = key
        if field in CASE_INSENSITIVE_FIELDS and isinstance(key, str):
            hits = bucket.get("original", {}).get("hits", {}).get("hits", [])
            if hits:
                value = _display_value(key, hits[0].get("_source", {}).get(source_field))
        results.append({"value": value, "count": bucket["doc_count"]})
    return results


def index_stats(index: str):
    return es_index.index_stats(index)

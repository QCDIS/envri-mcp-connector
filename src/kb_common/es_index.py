"""Elasticsearch index management and bulk indexing, shared across sources.

Every source (Argo, OSO, ...) shares the same index and the same base fields
(source, summary_text, embedding); each source contributes its own extra
mapped fields on top via `extra_properties`.
"""
import logging
from datetime import datetime, timezone
from functools import lru_cache

from elasticsearch import BadRequestError, Elasticsearch, helpers

from kb_common import config

log = logging.getLogger(__name__)

# Keyword fields that back exact-match filters/facets use this normalizer, so
# "black sea", "BLACK SEA" and "Black Sea" all hit the same terms. Elasticsearch
# applies it at index time and to term-level queries at search time; `_source`
# keeps the original casing. Note terms aggregations return the *normalized*
# (lowercase) keys - see kb_mcp.search.terms_agg.
LOWERCASE_NORMALIZER = "lowercase"

INDEX_SETTINGS = {
    "analysis": {
        "normalizer": {
            LOWERCASE_NORMALIZER: {"type": "custom", "char_filter": [], "filter": ["lowercase"]},
        }
    }
}


def lowercase_keyword() -> dict:
    return {"type": "keyword", "normalizer": LOWERCASE_NORMALIZER}


def text_and_lowercase_keyword() -> dict:
    """`text` (analyzed, for scoring) + case-insensitive `.keyword` sub-field
    (for exact filters and aggregations)."""
    return {"type": "text", "fields": {"keyword": lowercase_keyword()}}


BASE_PROPERTIES = {
    "source": {"type": "keyword"},
    "summary_text": {"type": "text"},
    "indexed_at": {"type": "date"},
}


@lru_cache(maxsize=1)
def get_client() -> Elasticsearch:
    """Cached - the client pools its own HTTP connections"""
    kwargs = {}
    if config.ES_API_KEY:
        kwargs["api_key"] = config.ES_API_KEY
    elif config.ES_USERNAME:
        kwargs["basic_auth"] = (config.ES_USERNAME, config.ES_PASSWORD)
    if config.ES_CA_CERT:
        kwargs["ca_certs"] = config.ES_CA_CERT
    kwargs["verify_certs"] = config.ES_VERIFY_CERTS
    return Elasticsearch(config.ES_URL, **kwargs)


def build_mapping(dims: int, extra_properties: dict = None) -> dict:
    properties = {
        **BASE_PROPERTIES,
        "embedding": {
            "type": "dense_vector",
            "dims": dims,
            "index": True,
            "similarity": "cosine",
        },
        **(extra_properties or {}),
    }
    return {"settings": INDEX_SETTINGS, "mappings": {"properties": properties}}


def ensure_index(client: Elasticsearch, dims: int, extra_properties: dict = None, index: str = None):
    index = index or config.ES_INDEX
    if not client.indices.exists(index=index):
        client.indices.create(index=index, body=build_mapping(dims, extra_properties))
        log.info("created index %s (dims=%d)", index, dims)
        return

    try:
        client.indices.put_mapping(index=index, properties=build_mapping(dims, extra_properties)["mappings"]["properties"])
    except BadRequestError as exc:
        raise RuntimeError(
            f"index {index!r} has a mapping that can't be updated in place (a field's normalizer/type changed?). "
            f"Migrate it with scripts/ops/reindex.py, then point ES_INDEX at the new index. Cause: {exc}"
        ) from exc


def index_stats(index: str = None) -> dict:
    index = index or config.ES_INDEX
    client = get_client()
    resp = client.search(
        index=index,
        size=0,
        aggs={"by_source": {"terms": {"field": "source", "size": 10}}},
        track_total_hits=True,
    )
    total = resp["hits"]["total"]["value"]
    by_source = {b["key"]: b["doc_count"] for b in resp["aggregations"]["by_source"]["buckets"]}
    return {"total": total, "by_source": by_source}


def bulk_index(client: Elasticsearch, documents: list[dict], index: str = None):
    """Stamps every document with `indexed_at` (now, UTC) before writing."""
    index = index or config.ES_INDEX
    now = datetime.now(timezone.utc).isoformat()
    actions = [
        {
            "_index": index,
            "_id": doc.pop("_id"),
            "_source": {**doc, "indexed_at": now},
        }
        for doc in documents
    ]
    ok, errors = helpers.bulk(client, actions, raise_on_error=False)
    if errors:
        log.warning("%d documents failed to index: %s", len(errors), errors[:5])
    return ok, errors

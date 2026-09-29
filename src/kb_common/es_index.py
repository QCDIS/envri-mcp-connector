"""Elasticsearch index management and bulk indexing, shared across sources.

Every source (Argo, OSO, ...) shares the same index and the same base fields
(source, summary_text, embedding); each source contributes its own extra
mapped fields on top via `extra_properties`.
"""
import logging
from datetime import datetime, timezone
from functools import lru_cache

from elasticsearch import BadRequestError, Elasticsearch, TransportError, helpers

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


class EmbeddingMismatchError(RuntimeError):
    """The configured embedding model/dimensions don't match the ones the index
    was built with."""


def embedding_meta(model: str, dims: int) -> dict:
    """What gets stored under the index's `_meta`."""
    return {"embedding_model": model, "embedding_dims": dims}


def build_mapping(dims: int, extra_properties: dict = None, embedding_model: str = None) -> dict:
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
    return {
        "settings": INDEX_SETTINGS,
        "mappings": {
            "_meta": embedding_meta(embedding_model or config.EMBEDDING_MODEL, dims),
            "properties": properties,
        },
    }


def _index_embedding_state(client: Elasticsearch, index: str):
    """(`_meta`, mapped embedding dims) of `index`, or None if it doesn't exist."""
    if not client.indices.exists(index=index):
        return None
    mapping = next(iter(client.indices.get_mapping(index=index).values()))["mappings"]
    dims = (mapping.get("properties", {}).get("embedding") or {}).get("dims")
    return mapping.get("_meta") or {}, dims


def find_embedding_mismatches(meta: dict, mapped_dims, model: str, dims: int) -> list[str]:
    """Human-readable problems between the index's recorded embedding setup
    (`_meta` + the actual dense_vector mapping) and the configured `model` /
    `dims`. Empty list = compatible."""
    problems = []
    indexed_model = meta.get("embedding_model")
    indexed_dims = meta.get("embedding_dims")
    if indexed_model is not None and indexed_model != model:
        problems.append(f"embedding model: index was built with {indexed_model!r}, configured {model!r}")
    if indexed_dims is not None and indexed_dims != dims:
        problems.append(f"embedding dimensions: index metadata says {indexed_dims}, configured model produces {dims}")
    if mapped_dims is not None and mapped_dims != dims:
        problems.append(f"embedding dimensions: index vector field is {mapped_dims}-dim, configured model produces {dims}")
    return problems


def _mismatch_error(index: str, problems: list[str]) -> EmbeddingMismatchError:
    return EmbeddingMismatchError(
        f"Embedding configuration doesn't match index {index!r}: " + "; ".join(problems)
        + ". Set EMBEDDING_MODEL to the model the index was built with, or re-ingest into a new index "
        "(scripts/ops/reindex.py only carries embeddings over as-is, it can't change the model)."
    )


def verify_embedding_model(client: Elasticsearch, model: str, dims: int, index: str = None) -> None:
    """Startup guard: raise EmbeddingMismatchError if the index was built with a
    different embedding model/dimension count than the one configured."""
    index = index or config.ES_INDEX
    try:
        state = _index_embedding_state(client, index)
    except TransportError as exc:
        log.warning("couldn't verify embedding model against index %r - Elasticsearch unreachable: %s", index, exc)
        return
    if state is None:
        log.info("index %r doesn't exist yet - skipping embedding model check", index)
        return

    meta, mapped_dims = state
    problems = find_embedding_mismatches(meta, mapped_dims, model, dims)
    if problems:
        error = _mismatch_error(index, problems)
        log.error("%s", error)
        raise error
    if "embedding_model" not in meta:
        log.warning(
            "index %r has no embedding metadata (built before the check existed) - only its dimensions (%s) were "
            "verified, not the model. Re-running the ingest records the model.", index, mapped_dims,
        )
        return
    log.info("embedding model check passed: %s (%d dims) matches index %r", model, dims, index)


def ensure_index(client: Elasticsearch, dims: int, extra_properties: dict = None, index: str = None, embedding_model: str = None):
    index = index or config.ES_INDEX
    embedding_model = embedding_model or config.EMBEDDING_MODEL
    state = _index_embedding_state(client, index)
    if state is None:
        client.indices.create(index=index, body=build_mapping(dims, extra_properties, embedding_model))
        log.info("created index %s (dims=%d, model=%s)", index, dims, embedding_model)
        return

    # Never mix embeddings from another model into an existing index.
    meta, mapped_dims = state
    problems = find_embedding_mismatches(meta, mapped_dims, embedding_model, dims)
    if problems:
        raise _mismatch_error(index, problems)

    try:
        client.indices.put_mapping(
            index=index,
            properties=build_mapping(dims, extra_properties, embedding_model)["mappings"]["properties"],
            # put_mapping replaces _meta wholesale, so merge to keep any other keys
            meta={**meta, **embedding_meta(embedding_model, dims)},
        )
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

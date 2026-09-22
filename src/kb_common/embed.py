"""Local embedding generation via sentence-transformers (runs on GPU if available)."""
import os
import queue
import threading
import time
from concurrent.futures import Future
from functools import lru_cache

from kb_common import config, timing


def _model_is_cached(model_name: str) -> bool:
    """Whether model_name is already in the local HF cache - a local
    directory scan, no network call."""
    try:
        from huggingface_hub import scan_cache_dir
        return any(repo.repo_id == model_name for repo in scan_cache_dir().repos)
    except Exception:
        return False


if _model_is_cached(config.EMBEDDING_MODEL):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sentence_transformers import SentenceTransformer  # noqa: E402

# E5 models expect a "passage: " / "query: " prefix on the input text.
_E5_PASSAGE_PREFIX = "passage: "
_E5_QUERY_PREFIX = "query: "


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    with timing.stage("model_load"):
        return SentenceTransformer(config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE)


def embedding_dims() -> int:
    return get_model().get_sentence_embedding_dimension()


def _encode(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed documents/passages (indexing time)."""
    is_e5 = "e5" in config.EMBEDDING_MODEL.lower()
    inputs = [f"{_E5_PASSAGE_PREFIX}{t}" if is_e5 else t for t in texts]
    return _encode(inputs)


def _encode_queries(texts: list[str]) -> list[list[float]]:
    is_e5 = "e5" in config.EMBEDDING_MODEL.lower()
    inputs = [f"{_E5_QUERY_PREFIX}{t}" if is_e5 else t for t in texts]
    return _encode(inputs)


# Under concurrent search traffic, each request calling embed_query() one at
# a time serializes the GPU behind per-call Python/tokenizer overhead
_query_queue: "queue.Queue[tuple[str, Future]]" = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False


def _batch_worker() -> None:
    while True:
        batch = [_query_queue.get()]
        deadline = time.monotonic() + config.EMBED_QUERY_BATCH_WINDOW_MS / 1000
        while len(batch) < config.EMBED_QUERY_BATCH_MAX_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                batch.append(_query_queue.get(timeout=remaining))
            except queue.Empty:
                break

        try:
            vectors = _encode_queries([text for text, _ in batch])
        except Exception as exc:
            for _, fut in batch:
                fut.set_exception(exc)
            continue
        for (_, fut), vector in zip(batch, vectors):
            fut.set_result(vector)


def _ensure_worker() -> None:
    global _worker_started
    if _worker_started:
        return
    with _worker_lock:
        if not _worker_started:
            threading.Thread(target=_batch_worker, name="embed-query-batcher", daemon=True).start()
            _worker_started = True


def embed_query(text: str) -> list[float]:
    """Embed a search query (query time)
    Blocks until a batch this query joins comes back"""
    _ensure_worker()
    fut: Future = Future()
    _query_queue.put((text, fut))
    return fut.result()

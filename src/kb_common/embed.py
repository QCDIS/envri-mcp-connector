"""Local embedding generation via sentence-transformers (runs on GPU if available)."""
import os
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


def embed_query(text: str) -> list[float]:
    """Embed a search query (query time) - e5 models use a different prefix than passages."""
    is_e5 = "e5" in config.EMBEDDING_MODEL.lower()
    return _encode([f"{_E5_QUERY_PREFIX}{text}" if is_e5 else text])[0]

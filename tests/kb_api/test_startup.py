"""kb_api's lifespan must refuse to start when the configured embedding model
doesn't match the one the index was built with."""
import asyncio
import sys
import types

import pytest

try:  # kb_common.embed imports sentence-transformers at module level
    import sentence_transformers  # noqa: F401
except ImportError:
    _stub = types.ModuleType("sentence_transformers")
    _stub.SentenceTransformer = object
    sys.modules["sentence_transformers"] = _stub

from kb_api import main
from kb_common import es_index

MODEL = "Qwen/Qwen3-Embedding-0.6B"


class _FakeIndices:
    def __init__(self, mapping):
        self._mapping = mapping

    def exists(self, index):
        return self._mapping is not None

    def get_mapping(self, index):
        return {"idx": self._mapping}


class _FakeClient:
    def __init__(self, mapping):
        self.indices = _FakeIndices(mapping)


def _mapping(model=MODEL, dims=1024):
    return {
        "mappings": {
            "_meta": {"embedding_model": model, "embedding_dims": dims},
            "properties": {"embedding": {"type": "dense_vector", "dims": dims}},
        }
    }


@pytest.fixture
def warmups(monkeypatch):
    calls = []
    monkeypatch.setattr(main.embed, "embed_query", lambda text: calls.append(text))
    monkeypatch.setattr(main.embed, "embedding_dims", lambda: 1024)
    monkeypatch.setattr(main.common_config, "EMBEDDING_MODEL", MODEL)
    monkeypatch.setattr(main.common_config, "ES_INDEX", "idx")
    monkeypatch.setattr(es_index.config, "ES_INDEX", "idx")
    return calls


def _run_lifespan():
    async def go():
        async with main._lifespan(main.app):
            pass

    asyncio.run(go())


def test_startup_succeeds_when_model_and_dims_match(monkeypatch, warmups):
    monkeypatch.setattr(main.es_index, "get_client", lambda: _FakeClient(_mapping()))

    _run_lifespan()

    assert warmups == ["warmup"]


def test_startup_halts_on_model_mismatch(monkeypatch, warmups, caplog):
    monkeypatch.setattr(main.es_index, "get_client", lambda: _FakeClient(_mapping(model="intfloat/e5-base-v2")))

    with pytest.raises(es_index.EmbeddingMismatchError, match="intfloat/e5-base-v2"):
        _run_lifespan()

    assert warmups == []  # never got as far as serving
    assert "Embedding configuration doesn't match index" in caplog.text


def test_startup_halts_on_dimension_mismatch(monkeypatch, warmups):
    monkeypatch.setattr(main.es_index, "get_client", lambda: _FakeClient(_mapping(dims=768)))

    with pytest.raises(es_index.EmbeddingMismatchError, match="768"):
        _run_lifespan()

    assert warmups == []


def test_startup_succeeds_before_the_index_exists(monkeypatch, warmups):
    monkeypatch.setattr(main.es_index, "get_client", lambda: _FakeClient(None))

    _run_lifespan()

    assert warmups == ["warmup"]

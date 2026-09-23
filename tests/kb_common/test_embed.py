"""Tests the E5 query/passage prefixing logic only - never loads a real
model. `_encode` (the only function that touches SentenceTransformer) is
monkeypatched to capture its inputs and return fixed stub vectors.
"""
from kb_common import embed


def _patch_encode(monkeypatch, captured):
    def fake_encode(texts):
        captured.extend(texts)
        return [[0.0] * 3 for _ in texts]

    monkeypatch.setattr(embed, "_encode", fake_encode)


def test_embed_texts_adds_no_prefix_for_non_e5_model(monkeypatch):
    monkeypatch.setattr(embed.config, "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    captured = []
    _patch_encode(monkeypatch, captured)

    embed.embed_texts(["hello world"])

    assert captured == ["hello world"]


def test_embed_texts_adds_passage_prefix_for_e5_model(monkeypatch):
    monkeypatch.setattr(embed.config, "EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    captured = []
    _patch_encode(monkeypatch, captured)

    embed.embed_texts(["hello world"])

    assert captured == ["passage: hello world"]


def test_encode_queries_adds_query_prefix_for_e5_model(monkeypatch):
    monkeypatch.setattr(embed.config, "EMBEDDING_MODEL", "intfloat/e5-base-v2")
    captured = []
    _patch_encode(monkeypatch, captured)

    embed._encode_queries(["what is an argo float"])

    assert captured == ["query: what is an argo float"]


def test_encode_queries_no_prefix_for_non_e5_model(monkeypatch):
    monkeypatch.setattr(embed.config, "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    captured = []
    _patch_encode(monkeypatch, captured)

    embed._encode_queries(["what is an argo float"])

    assert captured == ["what is an argo float"]


def test_embed_query_returns_encoded_vector(monkeypatch):
    monkeypatch.setattr(embed.config, "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    monkeypatch.setattr(embed, "_encode_queries", lambda texts: [[1.0, 2.0, 3.0] for _ in texts])

    vector = embed.embed_query("hello")

    assert vector == [1.0, 2.0, 3.0]

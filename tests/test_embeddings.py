import pytest

from app.rag import embeddings
from app.rag.embeddings import EmbeddingError, embed_passages


class Row(list):
    def tolist(self):
        return list(self)


class FakeModel:
    def __init__(self, dim):
        self.dim = dim

    def embed(self, texts, batch_size):
        return [Row([0.5] * self.dim) for _ in texts]


def test_embed_passages_returns_lists(monkeypatch):
    monkeypatch.setattr(embeddings, "_text_embedding", lambda name: FakeModel(384))
    vectors = embed_passages(["a", "b"], "any/model", 64)
    assert len(vectors) == 2 and len(vectors[0]) == 384


def test_embed_passages_empty_input(monkeypatch):
    monkeypatch.setattr(embeddings, "_text_embedding", lambda name: pytest.fail("no model needed"))
    assert embed_passages([], "any/model", 64) == []


def test_embed_passages_dimension_mismatch_raises_embedding_error(monkeypatch):
    monkeypatch.setattr(embeddings, "_text_embedding", lambda name: FakeModel(10))
    with pytest.raises(EmbeddingError, match="EMBEDDING_DIM"):
        embed_passages(["a"], "any/model", 64)

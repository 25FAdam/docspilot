"""Local embeddings (fastembed, ONNX) and the matching token counter. No API calls."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import lru_cache

from app.config import get_settings


class EmbeddingError(RuntimeError):
    """Embedding model / tokenizer could not be loaded, or produced unusable vectors."""


@lru_cache(maxsize=2)
def _text_embedding(model_name: str):
    from fastembed import TextEmbedding  # heavy import, keep lazy

    try:
        return TextEmbedding(model_name=model_name)
    except Exception as exc:  # download / ONNX load failures come in many types
        raise EmbeddingError(f"cannot load embedding model {model_name!r}: {exc}") from exc


def embed_passages(texts: Sequence[str], model_name: str, batch_size: int) -> list[list[float]]:
    """Embed documents (passages). bge passages need no prefix.

    Retrieval queries must use `TextEmbedding.query_embed` instead (added with the retriever).
    """
    if not texts:
        return []
    model = _text_embedding(model_name)
    vectors = [row.tolist() for row in model.embed(list(texts), batch_size=batch_size)]
    expected = get_settings().embedding_dim
    if any(len(v) != expected for v in vectors):
        raise EmbeddingError(
            f"model {model_name} returned {len(vectors[0])}-dim vectors, "
            f"but EMBEDDING_DIM={expected}"
        )
    return vectors


@lru_cache(maxsize=2)
def get_token_counter(model_name: str) -> Callable[[str], int]:
    """Token counter using the embedding model's own tokenizer (no special tokens)."""
    from tokenizers import Tokenizer  # lazy: downloads tokenizer.json on first use

    try:
        tokenizer = Tokenizer.from_pretrained(model_name)
    except Exception as exc:
        raise EmbeddingError(f"cannot load tokenizer for {model_name!r}: {exc}") from exc
    tokenizer.no_truncation()  # the model's default would silently cap counts at 512
    tokenizer.no_padding()

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count

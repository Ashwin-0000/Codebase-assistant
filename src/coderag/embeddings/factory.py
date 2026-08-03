"""
embeddings/factory.py — Build the configured EmbeddingModel at runtime.

Reads ``CODERAG_EMBEDDING_PROVIDER`` (and optional model/key settings) from
the environment / ``.env`` file via the app's Settings object, then imports
and constructs the appropriate backend.

Design: the factory is the *only* place that imports concrete backends.
All other code depends on the ``EmbeddingModel`` ABC, keeping imports light.
"""

from __future__ import annotations

import logging

from coderag.embeddings.base import EmbeddingModel

logger = logging.getLogger(__name__)


def get_embedding_model(
    provider: str = "sentence_transformers",
    model_name: str | None = None,
    api_key: str | None = None,
    **kwargs,
) -> EmbeddingModel:
    """Construct and return the requested embedding model.

    Args:
        provider:   One of ``"sentence_transformers"``, ``"openai"``.
        model_name: Override the provider's default model.
        api_key:    API key (required for cloud providers; ignored for local).
        **kwargs:   Forwarded to the concrete constructor.

    Returns:
        A ready-to-use :class:`~coderag.embeddings.base.EmbeddingModel`.

    Raises:
        ValueError: If *provider* is unknown.
        ImportError: If the required package is not installed.
    """
    p = provider.lower().replace("-", "_")

    if p in ("sentence_transformers", "sentence-transformers", "local"):
        from coderag.embeddings.sentence_transformer import SentenceTransformerEmbedder

        name = model_name or "all-MiniLM-L6-v2"
        logger.info("Embedding provider: sentence-transformers (%s)", name)
        return SentenceTransformerEmbedder(name)

    if p == "openai":
        from coderag.embeddings.openai_embeddings import OpenAIEmbedder

        name = model_name or "text-embedding-3-small"
        logger.info("Embedding provider: openai (%s)", name)
        return OpenAIEmbedder(name, api_key=api_key, **kwargs)

    raise ValueError(
        f"Unknown embedding provider {provider!r}. "
        "Valid choices: 'sentence_transformers', 'openai'."
    )


class MockEmbeddingModel(EmbeddingModel):
    """Deterministic fixed-vector model for unit tests.

    Returns vectors of *dim* dimensions whose values are the SHA-256-derived
    hash of the input text, normalised to unit length.  The same text always
    produces the same vector, making tests reproducible.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        import math

        result: list[list[float]] = []
        for text in texts:
            # Generate dim floats from the hash of the text
            digest = hashlib.sha256(text.encode()).digest()
            # Extend by repeating the digest until we have enough bytes
            raw = (digest * ((self._dim * 4 // len(digest)) + 1))[: self._dim * 4]
            ints = [int.from_bytes(raw[i : i + 4], "big") for i in range(0, self._dim * 4, 4)]
            floats = [x / 2**32 for x in ints]  # normalise to [0, 1)
            # L2-normalise
            norm = math.sqrt(sum(x * x for x in floats)) or 1.0
            result.append([x / norm for x in floats])
        return result

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"mock-{self._dim}d"

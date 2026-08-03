"""
embeddings/openai_embeddings.py — OpenAI embedding backend.

Calls the OpenAI Embeddings API with automatic batching.
``text-embedding-3-small`` is the default (1536-dim, cheapest model as of
2025).  The model can be overridden via ``CODERAG_EMBEDDING_MODEL``.

Requires ``openai`` package: ``pip install openai``.
"""

from __future__ import annotations

import logging

from coderag.embeddings.base import EmbeddingModel

logger = logging.getLogger(__name__)

_OPENAI_BATCH_SIZE = 512   # OpenAI's recommended max items per request
_DIM_BY_MODEL: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIEmbedder(EmbeddingModel):
    """Embedding model that calls the OpenAI Embeddings API.

    Usage::

        embedder = OpenAIEmbedder(api_key="sk-...")
        vectors = embedder.embed(["hello world"])
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        """
        Args:
            model_name: OpenAI embedding model name.
            api_key:    If omitted, reads from the ``OPENAI_API_KEY`` env var.
            base_url:   Override the API base URL (for proxies / local replicas).
        """
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OpenAI embeddings. "
                "Install it with: pip install openai"
            ) from exc

        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)
        self._name = model_name
        self._dim = _DIM_BY_MODEL.get(model_name, 1536)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* using the OpenAI API (batched)."""
        if not texts:
            return []

        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), _OPENAI_BATCH_SIZE):
            batch = texts[i : i + _OPENAI_BATCH_SIZE]
            response = self._client.embeddings.create(
                input=batch,
                model=self._name,
            )
            all_vecs.extend(item.embedding for item in response.data)

        return all_vecs

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._name

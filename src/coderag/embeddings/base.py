"""
embeddings/base.py — Abstract embedding model interface.

Every concrete embedder implements this ABC, making the embedding backend
fully swappable via an env-var config flag without touching any call site.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    """Minimal interface every embedding backend must satisfy."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts and return a list of float vectors.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            List of float vectors, one per input text.
            All vectors must have the same length (``self.dimension``).
        """

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper that embeds a single text.

        Subclasses may override for efficiency (e.g. skip batching overhead).
        """
        return self.embed([text])[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector dimension (fixed for a given model/configuration)."""

    @property
    def model_name(self) -> str:
        """Human-readable model identifier (for logging/metadata)."""
        return "unknown"

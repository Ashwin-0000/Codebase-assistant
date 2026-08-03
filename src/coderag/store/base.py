"""
store/base.py — Abstract vector store interface and shared data types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from coderag.chunking.models import Chunk


@dataclass
class SearchResult:
    """One result returned by a similarity search.

    Attributes:
        chunk:         The matching :class:`~coderag.chunking.Chunk`.
        score:         Cosine similarity in [−1, 1]; higher is better.
        enriched_text: The text that was embedded for this chunk (useful for
                       debugging and for understanding the retrieval signal).
    """

    chunk: Chunk
    score: float
    enriched_text: str

    @property
    def citation(self) -> str:
        return self.chunk.citation


class VectorStore(ABC):
    """Minimal interface every vector store backend must satisfy."""

    @abstractmethod
    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        enriched_texts: list[str],
    ) -> None:
        """Add or update a batch of chunks in the store.

        Args:
            chunks:         Chunks whose metadata to store.
            embeddings:     Parallel list of embedding vectors.
            enriched_texts: Parallel list of the texts that were embedded
                            (stored as Chroma ``documents``).
        """

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int = 8,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Retrieve the *top_k* most similar chunks.

        Args:
            embedding: Query embedding vector.
            top_k:     Maximum number of results.
            where:     Optional Chroma metadata filter dict.

        Returns:
            List of :class:`SearchResult`, sorted by descending similarity.
        """

    @abstractmethod
    def get_by_id(self, chunk_id: str) -> tuple[Chunk | None, str | None]:
        """Return the ``(Chunk, enriched_text)`` pair for *chunk_id*, or ``(None, None)``."""

    @abstractmethod
    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by ID from the store."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of chunks in the store."""

    @abstractmethod
    def clear(self) -> None:
        """Delete all chunks from the store (use with caution!)."""

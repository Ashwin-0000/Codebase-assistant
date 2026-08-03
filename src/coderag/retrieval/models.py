"""
retrieval/models.py — Data classes for Phase 5 retrieval results.

Provides two core types:

- :class:`RetrievalResult`: A single scored chunk after the full retrieval
  pipeline, annotated with *how* it was found (vector search vs graph expansion).

- :class:`RetrievalContext`: The fully assembled, token-budgeted context
  ready for Phase 6 (LLM generation).  Contains the formatted text, the
  original query, and convenience accessors for citations and token count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from coderag.chunking.models import Chunk


@dataclass
class RetrievalResult:
    """One chunk scored and tagged by the retrieval pipeline.

    Attributes:
        chunk:         The retrieved :class:`~coderag.chunking.Chunk`.
        score:         Relevance score (higher is better).  For vector search
                       results this is cosine similarity ∈ [−1, 1]; for
                       graph-expanded chunks the score is decayed from the
                       seed chunk's score.
        enriched_text: The enriched text that was embedded for this chunk.
        source:        How this result was found:
                       ``"vector_search"`` — direct similarity hit,
                       ``"graph_expansion"`` — added via call-graph traversal.
    """

    chunk: Chunk
    score: float
    enriched_text: str
    source: Literal["vector_search", "graph_expansion"] = "vector_search"

    @property
    def citation(self) -> str:
        """Shortcut to ``chunk.citation``."""
        return self.chunk.citation

    @property
    def chunk_id(self) -> str:
        """Shortcut to ``chunk.chunk_id``."""
        return self.chunk.chunk_id

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "citation": self.citation,
            "score": self.score,
            "source": self.source,
            "function_name": self.chunk.function_name,
            "class_name": self.chunk.class_name,
        }


@dataclass
class RetrievalContext:
    """The fully assembled context from a retrieval run.

    This is the primary output of :meth:`Retriever.retrieve` — it contains
    the formatted context string ready to be injected into an LLM prompt,
    together with metadata about what was retrieved.

    Attributes:
        results:       Ordered list of chunks that fit within the token budget,
                       sorted by ``(file_path, start_line)`` for coherent
                       reading order.
        query:         The original natural-language query.
        total_tokens:  Actual token count of ``context_text``.
        context_text:  Formatted, multi-block text ready for the LLM prompt.
    """

    results: list[RetrievalResult]
    query: str
    total_tokens: int = 0
    context_text: str = ""

    def citations(self) -> list[str]:
        """Return all ``file:line`` citation strings from the results."""
        return [r.citation for r in self.results]

    @property
    def chunk_count(self) -> int:
        return len(self.results)

    @property
    def is_empty(self) -> bool:
        return len(self.results) == 0

    def summary(self) -> str:
        """Human-readable one-line summary."""
        sources = {}
        for r in self.results:
            sources[r.source] = sources.get(r.source, 0) + 1
        parts = [f"{v} {k}" for k, v in sorted(sources.items())]
        return (
            f"Retrieved {self.chunk_count} chunk(s) "
            f"({', '.join(parts) or 'none'}) "
            f"— {self.total_tokens} tokens"
        )

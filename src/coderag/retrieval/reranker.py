"""
retrieval/reranker.py — Optional cross-encoder re-ranking for retrieval results.

Re-ranking uses a *cross-encoder* model to jointly score ``(query, chunk)``
pairs, producing a more accurate relevance score than the initial embedding
similarity.  This is a standard two-stage retrieval pattern:

1. **Stage 1** (cheap): Embed query → ANN search → top-K candidates
2. **Stage 2** (expensive): Cross-encoder re-score → re-sort top-K

The :class:`NoOpReranker` is the default when re-ranking is disabled,
providing a transparent pass-through so callers don't need conditionals.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from coderag.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


class Reranker(ABC):
    """Abstract base class for re-rankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Re-score and re-sort *results* against *query*.

        Args:
            query:   The natural-language query string.
            results: Candidate results from the initial retrieval stage.

        Returns:
            The same results re-ordered (and possibly re-scored) by the
            cross-encoder's relevance prediction.  Highest score first.
        """


class NoOpReranker(Reranker):
    """Pass-through reranker — returns results in their original order.

    Used when ``CODERAG_RERANKING_ENABLED=false`` (the default).
    """

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        return results


class CrossEncoderReranker(Reranker):
    """Re-rank using a sentence-transformers CrossEncoder model.

    Requires ``sentence-transformers`` (already a project dependency).

    Usage::

        reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranked = reranker.rerank("how does auth work?", results)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """
        Args:
            model_name: HuggingFace model identifier for the cross-encoder.
        """
        self._model_name = model_name
        self._model = None  # Lazy-loaded

    def _load_model(self):
        """Lazy-load the CrossEncoder to avoid import cost at startup."""
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Re-score results using the cross-encoder model.

        The cross-encoder jointly encodes ``(query, enriched_text)`` pairs
        and outputs a relevance logit.  Results are re-sorted by this score.
        """
        if not results:
            return results

        model = self._load_model()

        # Build (query, document) pairs for the cross-encoder
        pairs = [(query, r.enriched_text) for r in results]
        scores = model.predict(pairs)

        # Replace scores and re-sort
        reranked: list[RetrievalResult] = []
        for result, new_score in zip(results, scores):
            reranked.append(
                RetrievalResult(
                    chunk=result.chunk,
                    score=float(new_score),
                    enriched_text=result.enriched_text,
                    source=result.source,
                )
            )

        reranked.sort(key=lambda r: r.score, reverse=True)

        logger.debug(
            "Reranked %d results; top score %.4f → %.4f",
            len(reranked),
            results[0].score if results else 0,
            reranked[0].score if reranked else 0,
        )
        return reranked


def get_reranker(enabled: bool = False, model_name: str = "") -> Reranker:
    """Factory: return a CrossEncoderReranker if enabled, else NoOpReranker."""
    if enabled and model_name:
        return CrossEncoderReranker(model_name)
    return NoOpReranker()

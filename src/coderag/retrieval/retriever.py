"""
retrieval/retriever.py — Main retrieval pipeline orchestrator.

The :class:`Retriever` ties together all Phase 5 components into a single
``retrieve(query)`` call:

1. **Embed** the query via the configured embedding model.
2. **Vector search** the store for the top-K most similar chunks.
3. **Graph expansion** — for each seed result, walk the call graph
   N hops (configurable) and add neighboring chunks at a decayed score.
4. **Deduplicate** by ``chunk_id`` (keep the highest score).
5. **Re-rank** (optional) using a cross-encoder model.
6. **Assemble context** within the token budget.
7. Return a :class:`~coderag.retrieval.models.RetrievalContext`.

The retriever is intentionally stateless — all state lives in the vector
store and the call graph.  This makes it safe to call ``retrieve()``
concurrently from different threads (as long as the underlying store is
thread-safe, which Chroma is).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from coderag.retrieval.context_builder import ContextBuilder
from coderag.retrieval.models import RetrievalContext, RetrievalResult
from coderag.retrieval.reranker import NoOpReranker, Reranker

if TYPE_CHECKING:
    from coderag.embeddings.base import EmbeddingModel
    from coderag.graph.analyzer import CallGraphAnalyzer
    from coderag.store.base import VectorStore

logger = logging.getLogger(__name__)

# Score decay factor for graph-expanded chunks.
# A neighbor's score = seed_score * DECAY ^ hop_distance.
# With the default depth of 1 and decay of 0.7, a neighbor gets 70% of
# the seed's score — enough to surface related context but below direct hits.
_GRAPH_EXPANSION_DECAY = 0.7


class Retriever:
    """Orchestrates the full retrieval pipeline.

    Usage::

        retriever = Retriever(
            embedding_model=embedder,
            vector_store=store,
            graph_analyzer=analyzer,  # optional
        )
        context = retriever.retrieve("how does the retry decorator work?")
        print(context.context_text)
    """

    def __init__(
        self,
        embedding_model: "EmbeddingModel",
        vector_store: "VectorStore",
        *,
        graph_analyzer: "CallGraphAnalyzer | None" = None,
        reranker: Reranker | None = None,
        top_k: int = 8,
        graph_expansion_depth: int = 1,
        context_token_budget: int = 8000,
    ) -> None:
        """
        Args:
            embedding_model:       Model used to embed the query.
            vector_store:          Store containing indexed chunks + embeddings.
            graph_analyzer:        Optional call-graph analyzer for expansion.
                                   If ``None``, graph expansion is skipped.
            reranker:              Optional reranker; defaults to :class:`NoOpReranker`.
            top_k:                 Number of initial vector search results.
            graph_expansion_depth: Number of call-graph hops for expansion.
                                   Set to 0 to disable graph expansion.
            context_token_budget:  Maximum tokens in the assembled context.
        """
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.graph_analyzer = graph_analyzer
        self.reranker = reranker or NoOpReranker()
        self.top_k = top_k
        self.graph_expansion_depth = graph_expansion_depth
        self.context_builder = ContextBuilder(token_budget=context_token_budget)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str) -> RetrievalContext:
        """Run the full retrieval pipeline for *query*.

        Args:
            query: A natural-language question about the indexed codebase.

        Returns:
            A :class:`~coderag.retrieval.models.RetrievalContext` containing
            the ranked, token-budgeted chunks and formatted context text.
        """
        logger.info("Retrieving for query: %s", query[:80])

        # 1. Embed the query
        query_embedding = self.embedding_model.embed_one(query)

        # 2. Vector similarity search
        search_results = self.vector_store.query(
            embedding=query_embedding,
            top_k=self.top_k,
        )

        if not search_results:
            logger.warning("No results from vector search")
            return RetrievalContext(results=[], query=query)

        # 3. Convert SearchResult → RetrievalResult
        results: list[RetrievalResult] = [
            RetrievalResult(
                chunk=sr.chunk,
                score=sr.score,
                enriched_text=sr.enriched_text,
                source="vector_search",
            )
            for sr in search_results
        ]

        logger.debug(
            "Vector search returned %d results (top score=%.4f)",
            len(results),
            results[0].score if results else 0,
        )

        # 4. Graph expansion
        if (
            self.graph_analyzer is not None
            and self.graph_expansion_depth > 0
        ):
            results = self._expand_with_graph(results)

        # 5. Deduplicate by chunk_id (keep highest score)
        results = self._deduplicate(results)

        # 6. Re-rank
        results = self.reranker.rerank(query, results)

        # 7. Assemble context within token budget
        context = self.context_builder.build(results, query=query)

        logger.info(context.summary())
        return context

    # ------------------------------------------------------------------ #
    # Graph expansion
    # ------------------------------------------------------------------ #

    def _expand_with_graph(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Add call-graph neighbors of seed results at a decayed score."""
        assert self.graph_analyzer is not None

        expanded: list[RetrievalResult] = list(results)
        seen_ids = {r.chunk_id for r in results}

        for seed in results:
            neighbors = self.graph_analyzer.neighbors(
                seed.chunk,
                depth=self.graph_expansion_depth,
            )
            for neighbor in neighbors:
                if neighbor.chunk_id in seen_ids:
                    continue
                seen_ids.add(neighbor.chunk_id)

                # Look up enriched text from the store
                _, enriched_text = self.vector_store.get_by_id(neighbor.chunk_id)
                if enriched_text is None:
                    # Chunk is in the graph but not in the store — skip
                    logger.debug(
                        "Graph neighbor %s not in store, skipping",
                        neighbor.chunk_id,
                    )
                    continue

                # Decay the seed's score
                decayed_score = seed.score * _GRAPH_EXPANSION_DECAY

                expanded.append(
                    RetrievalResult(
                        chunk=neighbor,
                        score=decayed_score,
                        enriched_text=enriched_text,
                        source="graph_expansion",
                    )
                )

        added = len(expanded) - len(results)
        if added > 0:
            logger.debug(
                "Graph expansion added %d neighbor(s) from %d seed(s)",
                added,
                len(results),
            )
        return expanded

    # ------------------------------------------------------------------ #
    # Deduplication
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deduplicate(
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Deduplicate by chunk_id, keeping the entry with the highest score."""
        best: dict[str, RetrievalResult] = {}
        for r in results:
            existing = best.get(r.chunk_id)
            if existing is None or r.score > existing.score:
                best[r.chunk_id] = r
        # Return in descending score order
        return sorted(best.values(), key=lambda r: r.score, reverse=True)

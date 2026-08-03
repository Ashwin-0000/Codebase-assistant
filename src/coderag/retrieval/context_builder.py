"""
retrieval/context_builder.py — Assemble ranked chunks into a token-budgeted context.

The :class:`ContextBuilder` takes a scored, deduplicated list of
:class:`~coderag.retrieval.models.RetrievalResult` objects and produces a
:class:`~coderag.retrieval.models.RetrievalContext` that fits within a
configurable token budget.

Algorithm
---------
1. **Greedy selection**: iterate over results in descending score order;
   add each chunk if its token cost fits in the remaining budget.
2. **Reading-order sort**: re-order selected chunks by
   ``(file_path, start_line)`` so the LLM sees code in a natural,
   top-to-bottom layout.
3. **Format**: emit numbered blocks with a citation header and the
   enriched text (which already contains file path, scope, summary, and code).
"""

from __future__ import annotations

import logging

from coderag.chunking.splitter import count_tokens
from coderag.retrieval.models import RetrievalContext, RetrievalResult

logger = logging.getLogger(__name__)

# Per-block overhead: citation header + blank line + separator ≈ 4 tokens
_BLOCK_OVERHEAD_TOKENS = 6


class ContextBuilder:
    """Build a token-budgeted context string from retrieval results.

    Usage::

        builder = ContextBuilder(token_budget=8000)
        context = builder.build(results, query="how does auth work?")
        print(context.context_text)
    """

    def __init__(self, token_budget: int = 8000) -> None:
        """
        Args:
            token_budget: Maximum number of tokens the assembled context
                          may contain.  Chunks are added greedily until
                          this limit is reached.
        """
        self.token_budget = token_budget

    def build(
        self,
        results: list[RetrievalResult],
        query: str,
    ) -> RetrievalContext:
        """Assemble *results* into a :class:`RetrievalContext`.

        Args:
            results: Scored retrieval results (any order — will be sorted
                     internally by score for selection, then by file
                     position for output).
            query:   The original natural-language query.

        Returns:
            A :class:`RetrievalContext` with ``context_text`` populated.
        """
        if not results:
            return RetrievalContext(results=[], query=query)

        # 1. Sort by descending score for greedy budget allocation
        scored = sorted(results, key=lambda r: r.score, reverse=True)

        selected: list[RetrievalResult] = []
        remaining_budget = self.token_budget

        for result in scored:
            cost = count_tokens(result.enriched_text) + _BLOCK_OVERHEAD_TOKENS
            if cost <= remaining_budget:
                selected.append(result)
                remaining_budget -= cost
            else:
                logger.debug(
                    "Skipping %s (cost=%d, remaining=%d)",
                    result.citation,
                    cost,
                    remaining_budget,
                )

        if not selected:
            # Even the highest-scored chunk doesn't fit — include it anyway
            # so the context is never empty when results exist
            selected.append(scored[0])

        # 2. Re-order by file position for coherent reading
        selected.sort(key=lambda r: (r.chunk.file_path, r.chunk.start_line))

        # 3. Format into numbered blocks
        context_text = self._format_blocks(selected)
        total_tokens = count_tokens(context_text)

        return RetrievalContext(
            results=selected,
            query=query,
            total_tokens=total_tokens,
            context_text=context_text,
        )

    # ------------------------------------------------------------------ #
    # Formatting
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_blocks(results: list[RetrievalResult]) -> str:
        """Render results as numbered, citation-headed blocks."""
        blocks: list[str] = []

        for idx, result in enumerate(results, 1):
            header = (
                f"--- [{idx}/{len(results)}] {result.citation} "
                f"(score={result.score:.3f}, source={result.source}) ---"
            )
            blocks.append(f"{header}\n{result.enriched_text}")

        return "\n\n".join(blocks)

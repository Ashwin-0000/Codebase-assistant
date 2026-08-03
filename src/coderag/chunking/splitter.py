"""
chunking/splitter.py — Token-budget-aware splitting of oversized chunks.

When a function exceeds ``chunk_max_tokens`` it is "soft-split" into
overlapping sub-slices that each fit within the budget.  The overlap
preserves continuity so that queries spanning a split boundary can still
retrieve the relevant context.

Design choice: splitting is done at **line boundaries** (never mid-token)
because code comprehension requires syntactically complete lines.  The
overlap is defined in tokens but applied as the minimum number of lines
whose combined token count >= ``overlap_tokens``.

Token counting: tiktoken (cl100k_base) when available, otherwise a simple
``len(text) // 4`` approximation.  This lets Phase 2 tests run without the
full dependency stack.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coderag.chunking.models import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_encoder():
    """Return a tiktoken encoder, or None if tiktoken is not installed."""
    try:
        import tiktoken  # type: ignore[import-untyped]
        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        logger.debug("tiktoken not available — using character-based token estimate")
        return None


def count_tokens(text: str) -> int:
    """Return the token count of *text* using tiktoken if available.

    Falls back to ``ceil(len(text) / 4)`` otherwise (GPT-style rough estimate).
    """
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, (len(text) + 3) // 4)


# ---------------------------------------------------------------------------
# Splitter
# ---------------------------------------------------------------------------


class TokenSplitter:
    """Splits a :class:`~coderag.chunking.models.Chunk` that exceeds the
    token budget into overlapping sub-slices.

    Usage::

        splitter = TokenSplitter(max_tokens=512, overlap_tokens=64)
        sub_chunks = splitter.split(big_chunk)
        # Returns [big_chunk] unchanged if it fits, or a list of Chunk slices.
    """

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def split(self, chunk: "Chunk") -> list["Chunk"]:
        """Split *chunk* if it exceeds ``max_tokens``; otherwise return it unchanged.

        Args:
            chunk: The :class:`~coderag.chunking.models.Chunk` to inspect.

        Returns:
            A list with either the original chunk (length 1, unmodified) or
            multiple sub-chunks (length >= 2, each marked as a split).
        """
        if chunk.token_count <= self.max_tokens:
            return [chunk]

        slices = self._split_text(chunk.raw_code)
        if len(slices) <= 1:
            # Couldn't split (e.g. single enormous line) — keep as-is
            return [chunk]

        result: list["Chunk"] = []
        for idx, (text, line_offset) in enumerate(slices):
            sub_tokens = count_tokens(text)
            # Adjust line numbers for the sub-chunk
            sub_start = chunk.start_line + line_offset
            sub_end = sub_start + text.count("\n")

            # Build a new chunk_id that's unique per split slice
            from coderag.chunking.models import make_chunk_id
            sub_id = make_chunk_id(
                chunk.file_path,
                chunk.node_type,
                sub_start,
                chunk.function_name,
            )

            sub = _clone(
                chunk,
                chunk_id=sub_id,
                raw_code=text,
                start_line=sub_start,
                end_line=sub_end,
                token_count=sub_tokens,
                is_split_chunk=True,
                split_index=idx,
                total_splits=len(slices),
            )
            result.append(sub)

        logger.debug(
            "Split chunk %s (%d tokens) into %d slices",
            chunk.chunk_id,
            chunk.token_count,
            len(result),
        )
        return result

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _split_text(self, text: str) -> list[tuple[str, int]]:
        """Split *text* into (slice_text, start_line_offset) pairs.

        Each slice fits within ``max_tokens``; consecutive slices overlap
        by at least ``overlap_tokens`` tokens worth of lines.

        Returns a list of (text, line_offset_from_chunk_start) tuples.
        """
        lines = text.split("\n")
        # Pre-compute per-line token counts (add 1 for the newline itself)
        line_tokens = [count_tokens(line + "\n") for line in lines]
        total = sum(line_tokens)

        if total <= self.max_tokens:
            return [(text, 0)]

        slices: list[tuple[str, int]] = []
        start = 0  # line index (0-based within the chunk)

        while start < len(lines):
            # Accumulate lines until we hit the token budget
            budget = self.max_tokens
            end = start
            accumulated = 0
            while end < len(lines) and accumulated + line_tokens[end] <= budget:
                accumulated += line_tokens[end]
                end += 1

            if end == start:
                # Single line exceeds budget — include it anyway to avoid infinite loop
                end = start + 1

            slice_text = "\n".join(lines[start:end])
            slices.append((slice_text, start))

            if end >= len(lines):
                break

            # Determine the start of the *next* slice by walking backwards
            # from `end` until we've collected >= overlap_tokens
            overlap_acc = 0
            next_start = end
            while next_start > start and overlap_acc < self.overlap_tokens:
                next_start -= 1
                overlap_acc += line_tokens[next_start]

            if next_start >= end:
                # Overlap consumed the whole slice — just step forward one line
                # to prevent an infinite loop
                next_start = end
            start = next_start

        return slices


# ---------------------------------------------------------------------------
# Dataclass clone helper (Python 3.11 dataclasses don't have replace() on
# frozen instances — use a manual approach)
# ---------------------------------------------------------------------------


def _clone(chunk: "Chunk", **overrides) -> "Chunk":
    """Return a copy of *chunk* with the given fields overridden."""
    from dataclasses import asdict
    from coderag.chunking.models import Chunk

    d = asdict(chunk)
    d.update(overrides)
    return Chunk(**d)

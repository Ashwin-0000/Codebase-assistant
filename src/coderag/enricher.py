"""
enricher.py — Build the text that gets embedded for each chunk.

The "enriched text" is a structured block that tells the embedding model
about the *semantic context* of the code chunk, not just its raw source.
Including file path, scope, and a natural-language summary dramatically
improves retrieval quality for questions like "how does auth work?".

Format
------
::

    [File]: src/auth/middleware.py:L12-L45
    [Scope]: authenticate_user
    [Summary]: Verify a JWT token from the request headers and return the
               decoded user payload, raising 401 if the token is invalid.

    [Code]:
    def authenticate_user(request: Request) -> dict:
        token = request.headers.get("Authorization", "").removeprefix("Bearer ")
        ...

Summary source priority
-----------------------
1. Existing docstring  (no LLM call needed — fast, free)
2. LLM-generated summary (if ``docgen_fn`` is provided and no docstring)
3. Minimal metadata fallback (language, node type, location)

The LLM call path is intentionally dependency-free at this module level —
callers pass in an optional ``docgen_fn: (Chunk) -> str`` callable.  This
keeps the enricher testable without a live LLM.

Docgen caching
--------------
Generated summaries are cached in ``cache_path`` (a JSON file) so that
re-indexing doesn't re-call the LLM for unchanged chunks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

from coderag.chunking.models import Chunk

logger = logging.getLogger(__name__)

# Type alias for the optional LLM summary function
DocgenFn = Callable[[Chunk], str]

# Maximum characters of raw_code included in the enriched text.
# Keeps the embedded string within a reasonable token window even for
# large chunks (which should have been split by Phase 2's TokenSplitter).
_MAX_CODE_CHARS = 8_000


class ChunkEnricher:
    """Builds the enriched text string to embed for each :class:`~coderag.chunking.Chunk`.

    Usage (no LLM)::

        enricher = ChunkEnricher()
        text = enricher.enrich(chunk)

    Usage (with LLM docgen)::

        def my_llm(chunk: Chunk) -> str:
            return call_llm(f"Summarise: {chunk.raw_code}")

        enricher = ChunkEnricher(docgen_fn=my_llm, cache_path=Path(".coderag/docgen.json"))
        text = enricher.enrich(chunk)
    """

    def __init__(
        self,
        docgen_fn: DocgenFn | None = None,
        cache_path: Path | None = None,
    ) -> None:
        """
        Args:
            docgen_fn:   Optional callable ``(Chunk) -> str`` that generates a
                         one-sentence summary for chunks without a docstring.
                         Pass ``None`` to skip LLM docgen.
            cache_path:  Path to a JSON file where generated summaries are
                         cached.  If provided and the file exists, it is loaded
                         on construction; updated on each new generation.
        """
        self._docgen_fn = docgen_fn
        self._cache_path = cache_path
        self._cache: dict[str, str] = {}  # {chunk_id: summary}

        if cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text("utf-8"))
                logger.debug("Loaded %d docgen cache entries", len(self._cache))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load docgen cache %s: %s", cache_path, e)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def enrich(self, chunk: Chunk) -> str:
        """Return the enriched text to embed for *chunk*.

        Args:
            chunk: A :class:`~coderag.chunking.Chunk` from Phase 2.

        Returns:
            A multi-line string containing file location, scope, summary,
            and a (possibly truncated) copy of the source code.
        """
        summary = self._get_summary(chunk)

        # Truncate raw code if very long (shouldn't happen after Phase 2 splitting,
        # but guard anyway for safety)
        code = chunk.raw_code
        if len(code) > _MAX_CODE_CHARS:
            code = code[:_MAX_CODE_CHARS] + "\n... [truncated]"

        lines = [
            f"[File]: {chunk.citation}",
            f"[Language]: {chunk.language}",
            f"[Scope]: {chunk.qualified_name}",
        ]
        if summary:
            lines.append(f"[Summary]: {summary}")
        if chunk.imports:
            # Show at most 5 imports for context
            shown = chunk.imports[:5]
            lines.append(f"[Imports]: {'; '.join(shown)}")
        lines.append("")
        lines.append("[Code]:")
        lines.append(code)

        return "\n".join(lines)

    def enrich_batch(self, chunks: list[Chunk]) -> list[str]:
        """Enrich a batch of chunks (calls docgen in batch where possible)."""
        # Identify which chunks need LLM docgen
        needs_docgen = [
            c for c in chunks
            if not c.has_docstring
            and self._docgen_fn is not None
            and c.chunk_id not in self._cache
        ]
        # Pre-generate for all at once (subclasses could override for batching)
        for c in needs_docgen:
            try:
                summary = self._docgen_fn(c)  # type: ignore[misc]
                self._cache[c.chunk_id] = summary
            except Exception as exc:
                logger.warning("Docgen failed for %s: %s", c.citation, exc)

        if needs_docgen:
            self._flush_cache()

        return [self.enrich(c) for c in chunks]

    # ------------------------------------------------------------------ #
    # Summary resolution
    # ------------------------------------------------------------------ #

    def _get_summary(self, chunk: Chunk) -> str | None:
        """Resolve the best available summary for *chunk*.

        Priority: docstring → cached docgen → live docgen → fallback.
        """
        # 1. Docstring from the source
        if chunk.has_docstring and chunk.docstring:
            # Use first non-empty line only (keep the embedded string compact)
            first_line = chunk.docstring.strip().splitlines()[0].strip()
            return first_line or chunk.docstring.strip()

        # 2. Cached LLM-generated summary
        if chunk.chunk_id in self._cache:
            return self._cache[chunk.chunk_id]

        # 3. Live LLM call
        if self._docgen_fn is not None:
            try:
                summary = self._docgen_fn(chunk)
                self._cache[chunk.chunk_id] = summary
                self._flush_cache()
                return summary
            except Exception as exc:
                logger.warning("Docgen failed for %s: %s", chunk.citation, exc)

        # 4. Minimal metadata fallback (no LLM, no docstring)
        return None

    def _flush_cache(self) -> None:
        """Persist the in-memory docgen cache to disk (if path configured)."""
        if self._cache_path is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(self._cache, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Could not flush docgen cache: %s", e)

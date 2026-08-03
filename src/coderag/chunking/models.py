"""
chunking/models.py — Core data model for a semantic code chunk.

A Chunk is the unit that gets embedded and stored in the vector database.
It carries everything needed for retrieval + citation:
  - Exact location (file, lines)
  - Semantic metadata (function name, class context, language)
  - Content (raw code + extracted docstring)
  - Structural context (imports visible in scope)
  - Split tracking (for long functions that exceed the token budget)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    """A single semantic unit extracted from a source file.

    Instances are produced by :class:`~coderag.chunking.extractor.ChunkExtractor`
    and consumed by the embedding + vector-store pipeline (Phase 4).
    """

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #

    chunk_id: str
    """Deterministic hex ID derived from file_path + node_type + start_line.
    Stable across re-indexing runs as long as the code hasn't moved."""

    # ------------------------------------------------------------------ #
    # Location
    # ------------------------------------------------------------------ #

    file_path: str
    """Repository-relative path to the source file (forward slashes)."""

    start_line: int
    """1-indexed first line of the chunk (including decorators if present)."""

    end_line: int
    """1-indexed last line of the chunk (inclusive)."""

    # ------------------------------------------------------------------ #
    # Semantic metadata
    # ------------------------------------------------------------------ #

    language: str
    """Canonical language name (``"python"``, ``"javascript"``, …)."""

    node_type: str
    """tree-sitter node type of the primary node
    (e.g. ``"function_definition"``, ``"class_definition"``)."""

    function_name: str | None
    """Name of the function or method.  ``None`` for anonymous nodes."""

    class_name: str | None
    """Name of the enclosing class if this chunk is a method; ``None`` otherwise."""

    # ------------------------------------------------------------------ #
    # Content
    # ------------------------------------------------------------------ #

    raw_code: str
    """Full source text of the chunk, including any decorator lines."""

    docstring: str | None
    """Extracted docstring / JSDoc comment, or ``None`` if absent.
    We never hallucinate a docstring — see ``has_docstring``."""

    has_docstring: bool
    """Explicit flag indicating whether a docstring was found.
    Downstream enrichment (Phase 4) uses this to decide whether to
    generate a summary via LLM."""

    imports: list[str] = field(default_factory=list)
    """Module-level import statements visible in this chunk's scope,
    plus any imports declared inside the function body."""

    # ------------------------------------------------------------------ #
    # Split tracking (for functions exceeding the token budget)
    # ------------------------------------------------------------------ #

    is_split_chunk: bool = False
    """True when this chunk is a sub-slice of a larger function."""

    split_index: int | None = None
    """0-based position of this slice among ``total_splits`` sibling slices."""

    total_splits: int | None = None
    """Total number of slices the original function was divided into."""

    # ------------------------------------------------------------------ #
    # Token accounting
    # ------------------------------------------------------------------ #

    token_count: int = 0
    """Approximate token count of ``raw_code`` (counted at extraction time)."""

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Return a plain dict (JSON-serialisable) representation."""
        d = asdict(self)
        return d

    def to_json(self, indent: int = 2) -> str:
        """Return a pretty-printed JSON string of this chunk."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @property
    def qualified_name(self) -> str:
        """Human-readable fully-qualified name, e.g. ``Calculator.add``."""
        parts = []
        if self.class_name:
            parts.append(self.class_name)
        if self.function_name:
            parts.append(self.function_name)
        return ".".join(parts) if parts else "<anonymous>"

    @property
    def citation(self) -> str:
        """Short citation string for use in LLM prompts.

        Example: ``src/app.py:L42-L78 (Calculator.add)``
        """
        loc = f"{self.file_path}:L{self.start_line}-L{self.end_line}"
        if self.qualified_name != "<anonymous>":
            loc += f" ({self.qualified_name})"
        return loc


# ---------------------------------------------------------------------------
# ID generation
# ---------------------------------------------------------------------------


def make_chunk_id(file_path: str, node_type: str, start_line: int, name: str | None) -> str:
    """Generate a deterministic, stable chunk ID.

    The ID is a 16-character hex prefix of a SHA-256 hash over the
    combination of fields that uniquely identify a semantic unit within
    a repository.  It stays stable across re-index runs as long as the
    function hasn't moved.
    """
    payload = f"{file_path}|{node_type}|{start_line}|{name or ''}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]

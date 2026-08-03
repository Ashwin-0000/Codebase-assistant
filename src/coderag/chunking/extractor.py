"""
chunking/extractor.py — Semantic chunking from tree-sitter ASTs.

Given a :class:`~coderag.ingestion.parser.ParsedFile` (Phase 1 output),
this module produces a list of :class:`~coderag.chunking.models.Chunk`
objects — one per meaningful semantic unit (function, method, class).

Pipeline per file:
  1. Extract module-level imports (always visible to every function in the file)
  2. Call Phase 1's ``ASTParser.extract_named_nodes()`` to get ASTNode list
  3. Walk the ASTNode tree:
       - class nodes  → one Chunk for the class + one Chunk per method
       - function nodes → one Chunk (+ sub-chunks for nested functions)
  4. For each node: extract docstring, count tokens, apply splitter

Docstring extraction strategy:
  - Python: walk the tree-sitter tree from the root, find the block child of
    the function node, check the first expression_statement for a string literal.
    This handles multi-line signatures correctly.
  - JavaScript: scan lines immediately preceding the function's start_line
    for a closing ``*/`` and walk back to ``/**`` to reconstruct the JSDoc.

Design choice: The extractor walks the *actual* tree-sitter tree (via
``ParsedFile.tree``) for docstring extraction — not the simplified ASTNode
objects — because tree-sitter gives us precise byte ranges and node types
that avoid the brittleness of regex-on-source-text approaches.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

from coderag.chunking.models import Chunk, make_chunk_id
from coderag.chunking.splitter import TokenSplitter, count_tokens
from coderag.ingestion.parser import ASTNode, ASTParser, ParsedFile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------


class ChunkExtractor:
    """Converts a :class:`~coderag.ingestion.parser.ParsedFile` into chunks.

    Usage::

        extractor = ChunkExtractor(max_tokens=512, overlap_tokens=64)
        chunks = extractor.extract(parsed_file)
        for chunk in chunks:
            print(chunk.citation, chunk.token_count)
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        *,
        include_nested_functions: bool = True,
    ) -> None:
        """
        Args:
            max_tokens:               Soft token limit per chunk.
            overlap_tokens:           Overlap when splitting long functions.
            include_nested_functions: If True, nested functions (e.g. inner
                                      helpers) are extracted as their own chunks.
        """
        self.splitter = TokenSplitter(max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        self.include_nested = include_nested_functions
        self._ast_parser = ASTParser()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract(self, parsed: ParsedFile) -> list[Chunk]:
        """Extract all semantic chunks from *parsed*.

        Args:
            parsed: A :class:`~coderag.ingestion.parser.ParsedFile` produced
                    by Phase 1's ``ASTParser``.

        Returns:
            Flat list of :class:`~coderag.chunking.models.Chunk` objects,
            sorted by start_line.
        """
        # Step 1: module-level imports visible from any function in this file
        imports = self._extract_module_imports(parsed)

        # Step 2: get Phase 1 AST nodes
        top_level_nodes = self._ast_parser.extract_named_nodes(parsed)

        # Step 3: build chunks
        chunks: list[Chunk] = []
        for node in top_level_nodes:
            self._process_node(node, parsed, imports, class_name=None, chunks=chunks)

        # Sort by start line for deterministic ordering
        chunks.sort(key=lambda c: c.start_line)

        logger.debug(
            "Extracted %d chunks from %s", len(chunks), parsed.relative_path
        )
        return chunks

    # ------------------------------------------------------------------ #
    # Node processing
    # ------------------------------------------------------------------ #

    def _process_node(
        self,
        node: ASTNode,
        parsed: ParsedFile,
        imports: list[str],
        class_name: str | None,
        chunks: list[Chunk],
    ) -> None:
        """Recursively process a single ASTNode and append resulting chunks."""

        is_class = node.node_type == "class_definition"

        # Build a chunk for this node itself
        chunk = self._make_chunk(node, parsed, imports, class_name=class_name)
        chunks.extend(self.splitter.split(chunk))

        if is_class:
            # Methods: direct children of a class node get a method chunk
            # with class_name set to this class's name.
            for child in node.children:
                self._process_node(
                    child,
                    parsed,
                    imports,
                    class_name=node.name,
                    chunks=chunks,
                )
        elif self.include_nested:
            # Nested functions/helpers inside a regular function
            for child in node.children:
                self._process_node(
                    child,
                    parsed,
                    imports,
                    class_name=None,  # nested functions don't inherit class context
                    chunks=chunks,
                )

    # ------------------------------------------------------------------ #
    # Chunk construction
    # ------------------------------------------------------------------ #

    def _make_chunk(
        self,
        node: ASTNode,
        parsed: ParsedFile,
        imports: list[str],
        class_name: str | None,
    ) -> Chunk:
        """Build a :class:`~coderag.chunking.models.Chunk` for *node*."""
        file_path = str(parsed.relative_path).replace("\\", "/")

        # Extract docstring using the live tree-sitter tree
        docstring = self._extract_docstring(node, parsed)
        has_docstring = docstring is not None

        # Collect imports: module-level + any inside this function body
        local_imports = self._extract_local_imports(node, parsed)
        all_imports = list(dict.fromkeys(imports + local_imports))  # dedupe, preserve order

        raw_code = node.text
        tokens = count_tokens(raw_code)

        chunk_id = make_chunk_id(file_path, node.node_type, node.start_line, node.name)

        return Chunk(
            chunk_id=chunk_id,
            file_path=file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            language=parsed.language,
            node_type=node.node_type,
            function_name=node.name,   # stored for ALL node types; node_type distinguishes classes
            class_name=class_name,
            raw_code=raw_code,
            docstring=docstring,
            has_docstring=has_docstring,
            imports=all_imports,
            token_count=tokens,
        )

    # ------------------------------------------------------------------ #
    # Docstring extraction
    # ------------------------------------------------------------------ #

    def _extract_docstring(self, node: ASTNode, parsed: ParsedFile) -> str | None:
        """Dispatch to the language-specific docstring extractor."""
        if parsed.language == "python":
            return self._extract_python_docstring(node, parsed)
        if parsed.language in ("javascript", "typescript", "tsx"):
            return self._extract_js_docstring(node, parsed)
        return None

    def _extract_python_docstring(self, node: ASTNode, parsed: ParsedFile) -> str | None:
        """Find a Python docstring by walking the live tree-sitter tree.

        Strategy: find the tree-sitter node that starts at ``node.start_byte``,
        unwrap any ``decorated_definition`` wrapper, navigate to the ``block``
        child, and return the text of the first ``string`` literal inside the
        first ``expression_statement``.
        """
        ts_node = _find_ts_node_at_byte(
            parsed.tree.root_node, node.start_byte
        )
        if ts_node is None:
            # Fallback: scan raw text (handles edge cases like injected code)
            return _py_docstring_from_text(node.text)

        # Unwrap decorated_definition to get the actual function/class node
        if ts_node.type == "decorated_definition":
            for child in ts_node.children:
                if child.type in (
                    "function_definition",
                    "async_function_definition",
                    "class_definition",
                ):
                    ts_node = child
                    break

        # Find the block child
        block = next(
            (c for c in ts_node.children if c.type == "block"), None
        )
        if block is None:
            return None

        # First named child of the block that is an expression_statement
        for stmt in block.children:
            if not stmt.is_named:
                continue
            if stmt.type == "expression_statement":
                # The first (and only) child of the expression_statement
                # should be the string literal
                for expr in stmt.children:
                    if expr.type == "string":
                        raw = parsed.source_bytes[
                            expr.start_byte:expr.end_byte
                        ].decode("utf-8", errors="replace")
                        return _clean_python_docstring(raw)
                break  # first expression_statement had no string → no docstring
            else:
                # First named statement is not an expression → no docstring
                break

        return None

    def _extract_js_docstring(self, node: ASTNode, parsed: ParsedFile) -> str | None:
        """Extract a JSDoc comment that immediately precedes *node* in source.

        Looks at the lines in the raw source immediately before
        ``node.start_line`` for a block comment ending in ``*/``.
        """
        lines = parsed.source_bytes.decode("utf-8", errors="replace").split("\n")
        before_idx = node.start_line - 2  # 0-indexed line just before the function

        if before_idx < 0:
            return None

        # The line immediately before the function should end with '*/'
        stripped = lines[before_idx].strip()
        if not (stripped.endswith("*/") or stripped == "*/"):
            return None

        # Walk backwards from before_idx to find '/**' or '/*'
        start_idx = before_idx
        while start_idx >= 0:
            if "/**" in lines[start_idx] or (
                "/*" in lines[start_idx] and "/**" not in lines[start_idx]
            ):
                break
            start_idx -= 1

        if start_idx < 0:
            return None

        comment = "\n".join(lines[start_idx : before_idx + 1])
        return _clean_jsdoc(comment)

    # ------------------------------------------------------------------ #
    # Import extraction
    # ------------------------------------------------------------------ #

    def _extract_module_imports(self, parsed: ParsedFile) -> list[str]:
        """Return all top-level import statements as raw strings."""
        root = parsed.tree.root_node
        imports: list[str] = []

        if parsed.language == "python":
            import_types = {"import_statement", "import_from_statement"}
        elif parsed.language in ("javascript", "typescript", "tsx"):
            import_types = {"import_statement"}
        else:
            import_types = set()

        for child in root.children:
            if child.type in import_types:
                text = parsed.source_bytes[
                    child.start_byte:child.end_byte
                ].decode("utf-8", errors="replace").strip()
                imports.append(text)

        return imports

    def _extract_local_imports(self, node: ASTNode, parsed: ParsedFile) -> list[str]:
        """Return import statements declared *inside* a function body.

        E.g. ``import urllib.request`` inside ``fetch_data``.
        """
        # We work from the raw text of the node to keep this fast
        local: list[str] = []
        if parsed.language != "python":
            return local

        # Simple line-based scan of the node text
        for line in node.text.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                local.append(stripped)

        return local


# ---------------------------------------------------------------------------
# tree-sitter helpers
# ---------------------------------------------------------------------------

_SEMANTIC_TYPES = frozenset(
    {
        "function_definition",
        "async_function_definition",
        "class_definition",
        "decorated_definition",
    }
)


def _find_ts_node_at_byte(root: "TSNode", target_byte: int) -> "TSNode | None":
    """DFS search: find the shallowest semantic node that starts at *target_byte*."""
    if root.start_byte == target_byte and root.type in _SEMANTIC_TYPES:
        return root
    for child in root.children:
        result = _find_ts_node_at_byte(child, target_byte)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# Docstring cleaning helpers
# ---------------------------------------------------------------------------


def _clean_python_docstring(raw: str) -> str | None:
    """Strip enclosing triple-quotes and normalise indentation."""
    for quote in ('"""', "'''"):
        if raw.startswith(quote) and raw.endswith(quote) and len(raw) >= len(quote) * 2:
            inner = raw[len(quote) : -len(quote)]
            return _dedent(inner).strip() or None
    # Single-quoted fallback
    for quote in ('"', "'"):
        if raw.startswith(quote) and raw.endswith(quote) and len(raw) >= 2:
            return raw[1:-1].strip() or None
    return raw.strip() or None


def _dedent(text: str) -> str:
    """Remove common leading whitespace from all non-empty lines."""
    lines = text.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return text
    indent = min(len(l) - len(l.lstrip()) for l in non_empty)
    return "\n".join(l[indent:] for l in lines)


def _clean_jsdoc(comment: str) -> str | None:
    """Strip ``/**``, ``*/``, and leading ``*`` from JSDoc comment lines."""
    lines = comment.strip().splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped in ("/**", "/*", "*/"):
            continue
        if stripped.startswith("* "):
            cleaned.append(stripped[2:])
        elif stripped.startswith("*"):
            cleaned.append(stripped[1:])
        else:
            cleaned.append(stripped)
    result = "\n".join(cleaned).strip()
    return result or None


# ---------------------------------------------------------------------------
# Fallback: regex-based Python docstring extraction from raw text
# (used when the byte-range lookup fails — edge case for injected snippets)
# ---------------------------------------------------------------------------

_TRIPLE_DOUBLE = re.compile(r'"""(.*?)"""', re.DOTALL)
_TRIPLE_SINGLE = re.compile(r"'''(.*?)'''", re.DOTALL)


def _py_docstring_from_text(text: str) -> str | None:
    """Extract the first triple-quoted string from *text* as a fallback."""
    # Skip the first line (def/class signature) to avoid matching decorators
    body = text[text.find("\n") + 1 :] if "\n" in text else ""
    for pattern in (_TRIPLE_DOUBLE, _TRIPLE_SINGLE):
        m = pattern.search(body)
        if m:
            return _dedent(m.group(1)).strip() or None
    return None

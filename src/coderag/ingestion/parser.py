"""
ingestion/parser.py — Language-aware AST parsing via tree-sitter.

Responsibilities:
  - Maintain one Parser instance per language (created lazily)
  - Parse a source file's bytes and return the tree-sitter Tree + language tag
  - Expose a thin ParsedFile dataclass that Phase 2 (chunking) consumes

tree-sitter API notes (v0.23+):
  - Each language grammar is a separate package: tree-sitter-python, etc.
  - ``Language(grammar_module.language())`` creates the Language object
  - ``Parser(language)`` creates a parser bound to that language
  - ``parser.parse(bytes)`` returns a ``Tree``; ``tree.root_node`` is the AST root
  - Nodes have ``.type``, ``.start_point``, ``.end_point``, ``.children``
    and ``.text`` (bytes) properties

Design choice: lazy import of grammar modules so that importing this file
does NOT pull in every grammar package — only the ones actually used.
This keeps startup time fast when only a subset of languages are needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tree_sitter import Language, Node, Tree

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parsed output types
# ---------------------------------------------------------------------------


@dataclass
class ParsedFile:
    """Result of parsing a single source file.

    Attributes:
        source_path:    Absolute path to the parsed file.
        relative_path:  Path relative to the repository root.
        language:       Canonical language name (e.g. ``"python"``).
        source_bytes:   Raw UTF-8 content of the file.
        tree:           tree-sitter parse tree.  Access via ``.root_node``.
        parse_errors:   True if the tree contains any ERROR nodes (the file
                        parsed but has syntax issues — we still store the
                        partial tree so chunking can extract what it can).
    """

    source_path: Path
    relative_path: Path
    language: str
    source_bytes: bytes
    tree: "Tree"
    parse_errors: bool = False
    line_count: int = 0


@dataclass
class ASTNode:
    """Flat, serialisable representation of a single tree-sitter node.

    Used to represent function / class / method nodes extracted from the tree
    before Phase 2 turns them into full :class:`~coderag.chunking.Chunk` objects.
    """

    node_type: str
    """tree-sitter node type (e.g. ``"function_definition"``)."""

    start_line: int
    """1-indexed start line."""

    end_line: int
    """1-indexed end line (inclusive)."""

    start_byte: int
    end_byte: int

    name: str | None = None
    """Identifier extracted from the node (function / class name)."""

    text: str = ""
    """Raw source text of the node (decoded UTF-8)."""

    children: list["ASTNode"] = field(default_factory=list)
    """Direct child ASTNodes of interest (e.g. methods inside a class)."""


# ---------------------------------------------------------------------------
# Grammar loader (lazy)
# ---------------------------------------------------------------------------


def _load_language(language_name: str) -> "Language":
    """Return a tree-sitter ``Language`` object for *language_name*.

    Raises:
        ImportError: If the required grammar package is not installed.
        ValueError:  If *language_name* is not supported.
    """
    from tree_sitter import Language  # type: ignore[import-untyped]

    lang = language_name.lower()

    if lang == "python":
        try:
            import tree_sitter_python as ts_lang  # type: ignore[import-untyped]
            return Language(ts_lang.language())
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-python is required for Python parsing. "
                "Install it with: pip install tree-sitter-python"
            ) from exc

    if lang in ("javascript", "js"):
        try:
            import tree_sitter_javascript as ts_lang  # type: ignore[import-untyped]
            return Language(ts_lang.language())
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-javascript is required for JavaScript parsing. "
                "Install it with: pip install tree-sitter-javascript"
            ) from exc

    if lang in ("typescript", "ts"):
        try:
            import tree_sitter_typescript as ts_lang  # type: ignore[import-untyped]
            return Language(ts_lang.language_typescript())
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-typescript is required for TypeScript parsing. "
                "Install it with: pip install tree-sitter-typescript"
            ) from exc

    if lang == "tsx":
        try:
            import tree_sitter_typescript as ts_lang  # type: ignore[import-untyped]
            return Language(ts_lang.language_tsx())
        except ImportError as exc:
            raise ImportError(
                "tree-sitter-typescript is required for TSX parsing."
            ) from exc

    raise ValueError(
        f"Language {language_name!r} is not supported by the parser. "
        "Supported: python, javascript, typescript, tsx."
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class ASTParser:
    """Parses source files into tree-sitter ASTs.

    One ``ASTParser`` instance should be shared across all files — it caches
    one ``Parser`` per language to avoid repeated grammar loading overhead.

    Usage::

        parser = ASTParser(supported_languages=["python", "javascript"])
        parsed = parser.parse_file(source_file)
        print(parsed.tree.root_node.type)          # "module"
        print(parsed.parse_errors)                 # False (if clean)
    """

    def __init__(self, supported_languages: list[str] | None = None) -> None:
        """
        Args:
            supported_languages: Languages to pre-load parsers for.
                                 If ``None``, parsers are created lazily on demand.
        """
        self._parsers: dict[str, "object"] = {}  # lang -> tree_sitter.Parser
        self._languages: dict[str, "Language"] = {}

        if supported_languages:
            for lang in supported_languages:
                try:
                    self._get_parser(lang)
                except (ImportError, ValueError) as exc:
                    logger.warning("Could not pre-load parser for %s: %s", lang, exc)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def parse_file(self, source_file: "object") -> ParsedFile | None:
        """Parse *source_file* (a :class:`~coderag.ingestion.walker.SourceFile`)
        and return a :class:`ParsedFile`, or ``None`` if the language is
        unsupported or the file cannot be read.
        """
        from coderag.ingestion.walker import SourceFile

        if not isinstance(source_file, SourceFile):
            raise TypeError(f"Expected SourceFile, got {type(source_file)}")

        language = source_file.language
        if language is None:
            logger.debug("Skipping %s (no language detected)", source_file.path)
            return None

        try:
            parser = self._get_parser(language)
        except (ImportError, ValueError) as exc:
            logger.debug("No parser for %s (%s): %s", language, source_file.path, exc)
            return None

        # Read file bytes
        try:
            source_bytes = source_file.path.read_bytes()
        except OSError as exc:
            logger.warning("Could not read %s: %s", source_file.path, exc)
            return None

        # Parse
        tree = parser.parse(source_bytes)  # type: ignore[union-attr]
        has_errors = self._tree_has_errors(tree.root_node)

        if has_errors:
            logger.debug("Syntax errors detected in %s (partial parse kept)", source_file.path)

        line_count = source_bytes.count(b"\n") + 1

        return ParsedFile(
            source_path=source_file.path,
            relative_path=source_file.relative_path,
            language=language,
            source_bytes=source_bytes,
            tree=tree,
            parse_errors=has_errors,
            line_count=line_count,
        )

    def parse_bytes(
        self,
        source_bytes: bytes,
        language: str,
        *,
        relative_path: Path | None = None,
    ) -> ParsedFile | None:
        """Parse raw bytes instead of a file — convenient for tests.

        Args:
            source_bytes:  Raw UTF-8 content.
            language:      Language name (``"python"``, ``"javascript"``, etc.)
            relative_path: Optional path label for the result (default: ``<bytes>``).

        Returns:
            A :class:`ParsedFile` or ``None`` if the language is unsupported.
        """
        try:
            parser = self._get_parser(language)
        except (ImportError, ValueError) as exc:
            logger.debug("No parser for %s: %s", language, exc)
            return None

        tree = parser.parse(source_bytes)  # type: ignore[union-attr]
        has_errors = self._tree_has_errors(tree.root_node)
        label = relative_path or Path("<bytes>")
        line_count = source_bytes.count(b"\n") + 1

        return ParsedFile(
            source_path=label,
            relative_path=label,
            language=language,
            source_bytes=source_bytes,
            tree=tree,
            parse_errors=has_errors,
            line_count=line_count,
        )

    def extract_named_nodes(self, parsed: ParsedFile) -> list[ASTNode]:
        """Extract all function / class / method nodes from *parsed*.

        Returns a flat list of :class:`ASTNode` objects.  Phase 2 (chunking)
        uses this list to build :class:`~coderag.chunking.Chunk` objects.
        """
        extractor = _get_node_extractor(parsed.language)
        return extractor(parsed.tree.root_node, parsed.source_bytes)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _get_parser(self, language: str) -> "object":
        """Return (cached) tree-sitter Parser for *language*."""
        lang = language.lower()
        if lang not in self._parsers:
            from tree_sitter import Parser  # type: ignore[import-untyped]

            ts_language = _load_language(lang)
            self._languages[lang] = ts_language
            self._parsers[lang] = Parser(ts_language)
        return self._parsers[lang]

    @staticmethod
    def _tree_has_errors(node: "Node") -> bool:
        """Recursively check whether any ERROR or MISSING node exists."""
        if node.type in ("ERROR", "MISSING"):
            return True
        return any(ASTParser._tree_has_errors(child) for child in node.children)


# ---------------------------------------------------------------------------
# Language-specific node extractors
# ---------------------------------------------------------------------------

# Maps language name → extractor function
# Each extractor takes (root_node, source_bytes) and returns list[ASTNode]
_EXTRACTORS: dict[str, "object"] = {}


def _get_node_extractor(language: str) -> "object":
    lang = language.lower()
    if lang not in _EXTRACTORS:
        raise ValueError(f"No node extractor registered for {language!r}")
    return _EXTRACTORS[lang]


def _register_extractor(language: str):
    """Decorator to register a node-extractor function for a language."""
    def decorator(fn):
        _EXTRACTORS[language] = fn
        return fn
    return decorator


def _node_text(node: "Node", source_bytes: bytes) -> str:
    """Decode a node's text from *source_bytes*."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _make_ast_node(node: "Node", source_bytes: bytes, name: str | None = None) -> ASTNode:
    """Build an :class:`ASTNode` from a tree-sitter node."""
    return ASTNode(
        node_type=node.type,
        start_line=node.start_point[0] + 1,   # tree-sitter is 0-indexed
        end_line=node.end_point[0] + 1,
        start_byte=node.start_byte,
        end_byte=node.end_byte,
        name=name,
        text=_node_text(node, source_bytes),
    )


# ------------------------------------------------------------------
# Python extractor
# ------------------------------------------------------------------

@_register_extractor("python")
def _extract_python(root: "Node", source_bytes: bytes) -> list[ASTNode]:
    """Extract function_definition and class_definition nodes from Python AST."""
    results: list[ASTNode] = []
    _walk_python(root, source_bytes, results)
    return results


def _walk_python(node: "Node", source_bytes: bytes, results: list[ASTNode]) -> None:
    """Depth-first walk; collect function / class / decorated-definition nodes."""

    if node.type == "decorated_definition":
        # A decorated function or class — the span must include the decorator lines.
        # Find the inner semantic node (function or class definition).
        inner = next(
            (
                c for c in node.children
                if c.type in (
                    "function_definition",
                    "async_function_definition",
                    "class_definition",
                )
            ),
            None,
        )
        if inner is not None:
            name = _get_python_name(inner, source_bytes)
            # Use decorated_definition's span so decorator lines are included.
            ast_node = ASTNode(
                node_type=inner.type,   # semantic type, not "decorated_definition"
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                name=name,
                text=_node_text(node, source_bytes),
            )
            results.append(ast_node)
            # Recurse into the inner node's body for nested definitions.
            for child in inner.children:
                _walk_python(child, source_bytes, ast_node.children)
        return

    if node.type in ("function_definition", "async_function_definition"):
        name = _get_python_name(node, source_bytes)
        ast_node = _make_ast_node(node, source_bytes, name=name)
        results.append(ast_node)
        for child in node.children:
            _walk_python(child, source_bytes, ast_node.children)
        return

    if node.type == "class_definition":
        name = _get_python_name(node, source_bytes)
        ast_node = _make_ast_node(node, source_bytes, name=name)
        results.append(ast_node)
        for child in node.children:
            _walk_python(child, source_bytes, ast_node.children)
        return

    # Otherwise recurse normally into all children.
    for child in node.children:
        _walk_python(child, source_bytes, results)


def _get_python_name(node: "Node", source_bytes: bytes) -> str | None:
    """Extract the identifier name from a function_definition / class_definition."""
    for child in node.children:
        if child.type == "identifier":
            return _node_text(child, source_bytes)
    return None


# ------------------------------------------------------------------
# JavaScript extractor
# ------------------------------------------------------------------

_JS_FUNCTION_TYPES = frozenset(
    {
        "function_declaration",
        "function_expression",
        "arrow_function",
        "method_definition",
        "generator_function_declaration",
    }
)


@_register_extractor("javascript")
def _extract_javascript(root: "Node", source_bytes: bytes) -> list[ASTNode]:
    results: list[ASTNode] = []
    _walk_js(root, source_bytes, results)
    return results


def _walk_js(node: "Node", source_bytes: bytes, results: list[ASTNode]) -> None:
    if node.type in _JS_FUNCTION_TYPES or node.type == "class_declaration":
        name = _get_js_name(node, source_bytes)
        ast_node = _make_ast_node(node, source_bytes, name=name)
        results.append(ast_node)
        for child in node.children:
            _walk_js(child, source_bytes, ast_node.children)
        return

    for child in node.children:
        _walk_js(child, source_bytes, results)


def _get_js_name(node: "Node", source_bytes: bytes) -> str | None:
    """Try to extract an identifier from JS function / class nodes."""
    for child in node.children:
        if child.type in ("identifier", "property_identifier"):
            return _node_text(child, source_bytes)
        # Handle: const foo = function() {} — name is in the parent lexical_declaration
    return None


# ------------------------------------------------------------------
# TypeScript / TSX extractor (same grammar extensions as JS)
# ------------------------------------------------------------------

@_register_extractor("typescript")
def _extract_typescript(root: "Node", source_bytes: bytes) -> list[ASTNode]:
    return _extract_javascript(root, source_bytes)


@_register_extractor("tsx")
def _extract_tsx(root: "Node", source_bytes: bytes) -> list[ASTNode]:
    return _extract_javascript(root, source_bytes)

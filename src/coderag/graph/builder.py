"""
graph/builder.py — Build a call graph from Phase 2 chunks.

Algorithm
---------
1. **Node registration** — every Chunk becomes a graph node keyed by
   ``"{file_path}::{qualified_name}"``.  A per-file name index maps bare
   function/method names → list of matching node IDs (for resolution).

2. **Call-edge extraction** — each chunk's ``raw_code`` is re-parsed with
   tree-sitter to find all function-call expressions inside it.  Parsing
   the chunk text directly (rather than threading byte offsets from the
   original file tree) keeps the API simple and the code self-contained.

3. **Resolution** — call names are looked up in the same-file name index.
   If a match is found, a ``"calls"`` edge is added.  If not (external
   library, stdlib, …), an ``"external::<name>"`` pseudo-node is added with
   ``is_resolved=False`` on the edge.

4. **Import edges** — each chunk's ``imports`` list is turned into
   ``"imports"`` edges.  These are module-level relationships used by
   Phase 5 to expand retrieval across file boundaries.

Design choice: NetworkX DiGraph as the in-memory store because it gives
us BFS/DFS, degree queries, and easy serialisation (``nx.node_link_data``)
out of the box.  It can be serialised to JSON for persistence in Phase 4.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import networkx as nx

from coderag.chunking.models import Chunk
from coderag.graph.models import GraphEdge, GraphNode
from coderag.ingestion.parser import ASTParser

if TYPE_CHECKING:
    from tree_sitter import Node as TSNode

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node ID helpers
# ---------------------------------------------------------------------------


def chunk_to_node_id(chunk: Chunk) -> str:
    """Return the unique graph node ID for *chunk*.

    Format: ``"{file_path}::{qualified_name}"``
    e.g.    ``"sample.py::Calculator.add"``
    """
    return f"{chunk.file_path}::{chunk.qualified_name}"


# ---------------------------------------------------------------------------
# Call-name extraction
# ---------------------------------------------------------------------------


def _extract_call_names(source_bytes: bytes, language: str) -> list[str]:
    """Parse *source_bytes* and return every called function/method name.

    Works on an *isolated* snippet (the chunk's raw_code), so no byte-offset
    threading from the full-file tree is needed.

    Returns deduplicated list of name strings (may include stdlib / external
    names — the caller is responsible for filtering).
    """
    parser = ASTParser()
    parsed = parser.parse_bytes(source_bytes, language)
    if parsed is None:
        return []

    root = parsed.tree.root_node
    names: list[str] = []

    if language == "python":
        _walk_calls_python(root, source_bytes, names)
    elif language in ("javascript", "typescript", "tsx"):
        _walk_calls_js(root, source_bytes, names)

    # Deduplicate while preserving first-occurrence order
    seen: set[str] = set()
    deduped: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped


# ── Python call walker ────────────────────────────────────────────────────────


def _walk_calls_python(node: "TSNode", source_bytes: bytes, out: list[str]) -> None:
    """DFS walk; collect callee names from Python ``call`` nodes."""
    if node.type == "call":
        func = node.child_by_field_name("function")
        if func is not None:
            name = _python_callee_name(func, source_bytes)
            if name:
                out.append(name)
    for child in node.children:
        _walk_calls_python(child, source_bytes, out)


def _python_callee_name(func_node: "TSNode", source_bytes: bytes) -> str | None:
    """Extract a canonical callee name from a Python call's ``function`` field.

    -  ``foo()``       → ``"foo"``  (identifier)
    -  ``self.bar()``  → ``"bar"``  (attribute leaf)
    -  ``obj.x.y()``   → ``"y"``   (deepest attribute leaf)
    """
    if func_node.type == "identifier":
        return _text(func_node, source_bytes)

    if func_node.type == "attribute":
        # attribute field holds the method name; object field holds the receiver
        attr = func_node.child_by_field_name("attribute")
        if attr:
            return _text(attr, source_bytes)

    return None


# ── JavaScript call walker ────────────────────────────────────────────────────

# tree-sitter JS node types that represent a function call
_JS_CALL_TYPES = frozenset({"call_expression", "new_expression"})


def _walk_calls_js(node: "TSNode", source_bytes: bytes, out: list[str]) -> None:
    """DFS walk; collect callee names from JS ``call_expression`` / ``new_expression``."""
    if node.type in _JS_CALL_TYPES:
        func = node.child_by_field_name("function") or node.child_by_field_name("constructor")
        if func is not None:
            name = _js_callee_name(func, source_bytes)
            if name:
                out.append(name)
    for child in node.children:
        _walk_calls_js(child, source_bytes, out)


def _js_callee_name(func_node: "TSNode", source_bytes: bytes) -> str | None:
    """Extract a canonical callee name from a JS call's function node.

    -  ``foo()``         → ``"foo"``
    -  ``obj.method()``  → ``"method"``
    -  ``new EventEmitter()`` → ``"EventEmitter"``
    """
    if func_node.type == "identifier":
        return _text(func_node, source_bytes)

    if func_node.type == "member_expression":
        prop = func_node.child_by_field_name("property")
        if prop:
            return _text(prop, source_bytes)

    return None


def _text(node: "TSNode", source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------


class GraphBuilder:
    """Builds a NetworkX ``DiGraph`` from a list of :class:`~coderag.chunking.Chunk` objects.

    Usage::

        builder = GraphBuilder()
        graph = builder.build(chunks)

        # Inspect nodes
        for node_id, attrs in graph.nodes(data=True):
            print(node_id, attrs["node"].qualified_name)

        # Inspect edges
        for u, v, data in graph.edges(data=True):
            print(f"{u} --{data['edge_type']}--> {v}")
    """

    def __init__(self, add_external_nodes: bool = False) -> None:
        """
        Args:
            add_external_nodes: If True, calls to unresolved (external / stdlib)
                                names add pseudo-nodes prefixed with ``"external::"``.
                                Useful for visualisation; disable for a cleaner graph.
        """
        self.add_external_nodes = add_external_nodes

    def build(self, chunks: list[Chunk]) -> nx.DiGraph:
        """Build and return the call graph for all *chunks*.

        Args:
            chunks: All chunks produced by Phase 2 for a repository.
                    Chunks from multiple files can be passed together.

        Returns:
            Directed graph where each node carries a
            :class:`~coderag.graph.models.GraphNode` in the ``"node"`` attribute,
            and each edge carries edge metadata (``edge_type``, ``call_name``, …).
        """
        G: nx.DiGraph = nx.DiGraph()

        # ── Step 1: Add all chunk nodes ──────────────────────────────────
        # Build a per-file lookup: {file_path: {name: [node_id, ...]}}
        # We keep a list because multiple chunks can share a leaf name
        # (e.g. both top-level ``add`` and ``Calculator.add``).
        file_name_index: dict[str, dict[str, list[str]]] = {}

        for chunk in chunks:
            node_id = chunk_to_node_id(chunk)
            graph_node = GraphNode(
                node_id=node_id,
                chunk_id=chunk.chunk_id,
                file_path=chunk.file_path,
                qualified_name=chunk.qualified_name,
                language=chunk.language,
                node_type=chunk.node_type,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )
            G.add_node(node_id, node=graph_node)

            # Index by qualified name AND by bare function/method name
            idx = file_name_index.setdefault(chunk.file_path, {})
            _index_name(idx, chunk.qualified_name, node_id)
            if chunk.function_name and chunk.function_name != chunk.qualified_name:
                _index_name(idx, chunk.function_name, node_id)

        logger.info(
            "Graph: added %d nodes from %d files",
            G.number_of_nodes(),
            len(file_name_index),
        )

        # ── Step 2: Add CALLS edges ──────────────────────────────────────
        edges_added = 0
        for chunk in chunks:
            caller_id = chunk_to_node_id(chunk)
            call_names = _extract_call_names(
                chunk.raw_code.encode("utf-8", errors="replace"),
                chunk.language,
            )

            file_idx = file_name_index.get(chunk.file_path, {})
            for call_name in call_names:
                resolved_ids = file_idx.get(call_name, [])

                if resolved_ids:
                    for callee_id in resolved_ids:
                        if callee_id == caller_id:
                            continue  # skip self-calls
                        if not G.has_edge(caller_id, callee_id):
                            G.add_edge(
                                caller_id,
                                callee_id,
                                edge_type="calls",
                                call_name=call_name,
                                is_resolved=True,
                            )
                            edges_added += 1
                else:
                    # Unresolved: external / stdlib call
                    if self.add_external_nodes:
                        ext_id = f"external::{call_name}"
                        if not G.has_node(ext_id):
                            G.add_node(
                                ext_id,
                                node=GraphNode(
                                    node_id=ext_id,
                                    chunk_id="",
                                    file_path="",
                                    qualified_name=call_name,
                                    language=chunk.language,
                                    node_type="external",
                                    start_line=0,
                                    end_line=0,
                                ),
                            )
                        if not G.has_edge(caller_id, ext_id):
                            G.add_edge(
                                caller_id,
                                ext_id,
                                edge_type="calls",
                                call_name=call_name,
                                is_resolved=False,
                            )
                            edges_added += 1

        # ── Step 3: Add IMPORTS edges ────────────────────────────────────
        # Cross-file relationships derived from import statements.
        # These are directional: the importing file → the imported module.
        import_edges = 0
        for chunk in chunks:
            caller_id = chunk_to_node_id(chunk)
            for imp_stmt in chunk.imports:
                # Extract the module name (first token after "import" or "from")
                module_name = _parse_import_module(imp_stmt, chunk.language)
                if not module_name:
                    continue
                # Find any chunk from a file whose path contains the module name
                target_ids = _resolve_import(module_name, file_name_index)
                for target_id in target_ids:
                    if target_id != caller_id and not G.has_edge(caller_id, target_id):
                        G.add_edge(
                            caller_id,
                            target_id,
                            edge_type="imports",
                            call_name=module_name,
                            is_resolved=True,
                        )
                        import_edges += 1

        logger.info(
            "Graph: added %d call edges + %d import edges",
            edges_added,
            import_edges,
        )
        return G

    # ------------------------------------------------------------------ #
    # Graph serialisation helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def to_node_link(graph: nx.DiGraph) -> dict:
        """Serialise *graph* to a JSON-compatible node-link dict (networkx format)."""
        # Convert GraphNode objects to plain dicts for JSON serialisation
        G_copy = nx.DiGraph()
        for node_id, attrs in graph.nodes(data=True):
            node_obj = attrs.get("node")
            G_copy.add_node(node_id, **(node_obj.to_dict() if node_obj else {}))
        for u, v, data in graph.edges(data=True):
            G_copy.add_edge(u, v, **data)
        return nx.node_link_data(G_copy)

    @staticmethod
    def from_node_link(data: dict) -> nx.DiGraph:
        """Restore a graph from a node-link dict produced by :meth:`to_node_link`."""
        return nx.node_link_graph(data)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _index_name(idx: dict[str, list[str]], name: str, node_id: str) -> None:
    """Append *node_id* to the list at *idx[name]*, creating the list if absent."""
    if name not in idx:
        idx[name] = []
    if node_id not in idx[name]:
        idx[name].append(node_id)


def _parse_import_module(import_stmt: str, language: str) -> str | None:
    """Extract the top-level module name from an import statement string.

    Examples:
        ``"import os"``                    → ``"os"``
        ``"from pathlib import Path"``     → ``"pathlib"``
        ``"import urllib.request"``        → ``"urllib"``
        ``"import { foo } from './bar'"``  → ``"bar"``
    """
    s = import_stmt.strip()
    if language == "python":
        if s.startswith("from "):
            parts = s.split()
            return parts[1].split(".")[0] if len(parts) > 1 else None
        if s.startswith("import "):
            parts = s.split()
            return parts[1].split(".")[0] if len(parts) > 1 else None
    elif language in ("javascript", "typescript", "tsx"):
        # e.g.  import { foo } from './utils'  or  import bar from 'bar'
        if "from" in s:
            module = s.split("from")[-1].strip().strip("'\"").strip(";")
            # Remove relative path prefix
            return module.lstrip("./").split("/")[0] if module else None
    return None


def _resolve_import(module_name: str, file_name_index: dict[str, dict[str, list[str]]]) -> list[str]:
    """Return node IDs of ALL chunks in files whose path contains *module_name*."""
    matching: list[str] = []
    for file_path, name_idx in file_name_index.items():
        # Simple check: file path contains the module name as a path component
        parts = file_path.replace("\\", "/").split("/")
        stem = parts[-1].split(".")[0]  # filename without extension
        if stem == module_name or file_path.startswith(module_name):
            matching.extend(nid for nids in name_idx.values() for nid in nids)
    return matching

"""
graph/analyzer.py — Query interface for the call graph.

Wraps a NetworkX DiGraph produced by :class:`~coderag.graph.builder.GraphBuilder`
and exposes typed, high-level query methods used by Phase 5 (retrieval).

Core queries
------------
- ``callers(chunk)``    → chunks that call *chunk*
- ``callees(chunk)``    → chunks that *chunk* calls
- ``neighbors(chunk, depth)`` → all chunks reachable within N hops
  (both directions; used for context expansion during retrieval)

The analyzer works with node IDs (strings) internally and accepts/returns
:class:`~coderag.chunking.Chunk` objects at its public boundary, performing
the lookup via the ``chunk_id`` stored in each graph node's attributes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import networkx as nx

from coderag.graph.builder import chunk_to_node_id
from coderag.graph.models import GraphEdge, GraphNode

if TYPE_CHECKING:
    from coderag.chunking.models import Chunk

logger = logging.getLogger(__name__)


class CallGraphAnalyzer:
    """High-level query interface for a call graph.

    Usage::

        analyzer = CallGraphAnalyzer(graph, chunk_map)

        callees = analyzer.callees(my_chunk)
        for c in callees:
            print(c.citation)

        # Expand 2 hops in both directions
        context = analyzer.neighbors(my_chunk, depth=2)
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        chunk_map: dict[str, "Chunk"],
    ) -> None:
        """
        Args:
            graph:      The DiGraph produced by :class:`~coderag.graph.builder.GraphBuilder`.
            chunk_map:  Mapping ``{chunk_id: Chunk}`` — used to convert node IDs
                        back to :class:`~coderag.chunking.Chunk` objects.
                        Build it as: ``{c.chunk_id: c for c in all_chunks}``.
        """
        self.graph = graph
        self._chunk_map = chunk_map
        # Build reverse lookup: node_id → chunk
        self._node_to_chunk: dict[str, "Chunk"] = {}
        for node_id, attrs in graph.nodes(data=True):
            node_obj: GraphNode | None = attrs.get("node")
            if node_obj and node_obj.chunk_id and node_obj.chunk_id in chunk_map:
                self._node_to_chunk[node_id] = chunk_map[node_obj.chunk_id]

    # ------------------------------------------------------------------ #
    # Core query methods
    # ------------------------------------------------------------------ #

    def callees(
        self,
        chunk: "Chunk",
        edge_types: tuple[str, ...] = ("calls",),
    ) -> list["Chunk"]:
        """Return chunks that *chunk* directly calls.

        Args:
            chunk:      The source chunk.
            edge_types: Edge types to follow (default: only ``"calls"``).

        Returns:
            List of :class:`~coderag.chunking.Chunk` objects representing
            the called functions/methods.  External (unresolved) nodes are
            excluded since they have no corresponding Chunk.
        """
        node_id = chunk_to_node_id(chunk)
        if not self.graph.has_node(node_id):
            logger.debug("Node %s not in graph", node_id)
            return []

        result: list["Chunk"] = []
        for _, target_id, data in self.graph.out_edges(node_id, data=True):
            if data.get("edge_type") not in edge_types:
                continue
            target_chunk = self._node_to_chunk.get(target_id)
            if target_chunk is not None:
                result.append(target_chunk)
        return result

    def callers(
        self,
        chunk: "Chunk",
        edge_types: tuple[str, ...] = ("calls",),
    ) -> list["Chunk"]:
        """Return chunks that directly call *chunk*.

        Args:
            chunk:      The target chunk.
            edge_types: Edge types to follow (default: only ``"calls"``).

        Returns:
            List of :class:`~coderag.chunking.Chunk` objects representing
            the calling functions/methods.
        """
        node_id = chunk_to_node_id(chunk)
        if not self.graph.has_node(node_id):
            return []

        result: list["Chunk"] = []
        for source_id, _, data in self.graph.in_edges(node_id, data=True):
            if data.get("edge_type") not in edge_types:
                continue
            source_chunk = self._node_to_chunk.get(source_id)
            if source_chunk is not None:
                result.append(source_chunk)
        return result

    def neighbors(
        self,
        chunk: "Chunk",
        depth: int = 1,
        edge_types: tuple[str, ...] = ("calls",),
    ) -> list["Chunk"]:
        """Return all chunks reachable from *chunk* within *depth* hops.

        Traversal is bidirectional: both ``chunk → callee`` and
        ``caller → chunk`` edges are followed.  The source *chunk* is
        excluded from the result.

        Args:
            chunk:      The seed chunk.
            depth:      Maximum number of hops to traverse.
            edge_types: Edge types to follow.

        Returns:
            Deduplicated list of reachable :class:`~coderag.chunking.Chunk`
            objects, sorted by (file_path, start_line).
        """
        node_id = chunk_to_node_id(chunk)
        if not self.graph.has_node(node_id):
            return []

        visited: set[str] = {node_id}
        frontier: set[str] = {node_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                # Follow outgoing edges (callees)
                for _, target, data in self.graph.out_edges(nid, data=True):
                    if data.get("edge_type") in edge_types and target not in visited:
                        next_frontier.add(target)
                # Follow incoming edges (callers)
                for source, _, data in self.graph.in_edges(nid, data=True):
                    if data.get("edge_type") in edge_types and source not in visited:
                        next_frontier.add(source)
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        # Convert to chunks, excluding the seed
        result: list["Chunk"] = []
        for nid in visited - {node_id}:
            c = self._node_to_chunk.get(nid)
            if c is not None:
                result.append(c)

        result.sort(key=lambda c: (c.file_path, c.start_line))
        return result

    # ------------------------------------------------------------------ #
    # Graph introspection helpers
    # ------------------------------------------------------------------ #

    def get_edges(self, chunk: "Chunk") -> list[GraphEdge]:
        """Return all graph edges touching *chunk* (both directions)."""
        node_id = chunk_to_node_id(chunk)
        edges: list[GraphEdge] = []

        for _, target, data in self.graph.out_edges(node_id, data=True):
            edges.append(
                GraphEdge(
                    source=node_id,
                    target=target,
                    edge_type=data.get("edge_type", "unknown"),
                    call_name=data.get("call_name", ""),
                    is_resolved=data.get("is_resolved", True),
                )
            )
        for source, _, data in self.graph.in_edges(node_id, data=True):
            edges.append(
                GraphEdge(
                    source=source,
                    target=node_id,
                    edge_type=data.get("edge_type", "unknown"),
                    call_name=data.get("call_name", ""),
                    is_resolved=data.get("is_resolved", True),
                )
            )
        return edges

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self, edge_type: str | None = None) -> int:
        if edge_type is None:
            return self.graph.number_of_edges()
        return sum(
            1 for _, _, d in self.graph.edges(data=True)
            if d.get("edge_type") == edge_type
        )

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    def print_summary(self, file_path: str | None = None) -> None:
        """Print a human-readable adjacency summary to stdout.

        Args:
            file_path: If provided, only show nodes from this file.
        """
        nodes = list(self.graph.nodes(data=True))
        if file_path:
            nodes = [
                (nid, attrs) for nid, attrs in nodes
                if attrs.get("node") and attrs["node"].file_path == file_path
            ]

        print(f"\n{'='*60}")
        print(f"  Call Graph Summary  ({len(nodes)} nodes shown)")
        print(f"{'='*60}")

        for node_id, attrs in sorted(nodes):
            node_obj: GraphNode | None = attrs.get("node")
            if node_obj and node_obj.node_type == "external":
                continue  # skip pseudo-nodes in summary

            out_edges = [
                (v, d) for _, v, d in self.graph.out_edges(node_id, data=True)
                if d.get("edge_type") == "calls"
            ]
            in_edges = [
                (u, d) for u, _, d in self.graph.in_edges(node_id, data=True)
                if d.get("edge_type") == "calls"
            ]

            callees_str = ", ".join(
                v.split("::")[-1] for v, _ in out_edges
            ) or "(none)"
            callers_str = ", ".join(
                u.split("::")[-1] for u, _ in in_edges
            ) or "(none)"

            qname = node_id.split("::")[-1] if "::" in node_id else node_id
            print(f"\n  [{node_obj.node_type if node_obj else '?'}] {qname}")
            print(f"    → calls:     {callees_str}")
            print(f"    ← called by: {callers_str}")

        print(f"\n  Edges: {self.edge_count('calls')} calls, "
              f"{self.edge_count('imports')} imports\n")

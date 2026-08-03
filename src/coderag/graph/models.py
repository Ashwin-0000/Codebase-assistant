"""
graph/models.py — Lightweight data classes for call graph nodes and edges.

These are separate from the NetworkX graph attributes to give callers a
typed, serialisable view of the graph structure without depending on
NetworkX internals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GraphNode:
    """An entry in the call graph — corresponds 1-to-1 with a Chunk.

    Attributes:
        node_id:       Unique string key: ``"{file_path}::{qualified_name}"``.
        chunk_id:      Back-reference to the :class:`~coderag.chunking.Chunk`
                       this node was derived from.
        file_path:     Repository-relative path (forward slashes).
        qualified_name: Human-readable identifier (e.g. ``"Calculator.add"``).
        language:      Canonical language name.
        node_type:     tree-sitter node type (``"function_definition"``, …).
        start_line:    1-indexed first line of the code unit.
        end_line:      1-indexed last line (inclusive).
    """

    node_id: str
    chunk_id: str
    file_path: str
    qualified_name: str
    language: str
    node_type: str
    start_line: int
    end_line: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class GraphEdge:
    """A directed relationship between two graph nodes.

    Attributes:
        source:     ``node_id`` of the calling / importing node.
        target:     ``node_id`` of the called / imported node.
        edge_type:  ``"calls"`` — direct function/method invocation, or
                    ``"imports"`` — module-level import relationship.
        call_name:  The exact name token used in the source (e.g. ``"greet"``
                    or ``"multiply"``).  Useful for debugging resolution.
        is_resolved: True when the target was found in the indexed codebase;
                    False for calls to external / unindexed code (edge still
                    added for completeness, target set to ``"external::<name>"``).
    """

    source: str
    target: str
    edge_type: str        # "calls" | "imports"
    call_name: str
    is_resolved: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

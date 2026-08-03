"""
Graph sub-package — call and import graph construction and querying.

Public API:
  GraphBuilder(add_external_nodes)  → builds nx.DiGraph from chunks
  CallGraphAnalyzer(graph, chunk_map) → callers / callees / neighbors
  chunk_to_node_id(chunk)           → the node ID string for a chunk
  GraphNode, GraphEdge              → typed data classes
"""

from coderag.graph.analyzer import CallGraphAnalyzer
from coderag.graph.builder import GraphBuilder, chunk_to_node_id
from coderag.graph.models import GraphEdge, GraphNode

__all__ = [
    "GraphBuilder",
    "CallGraphAnalyzer",
    "chunk_to_node_id",
    "GraphNode",
    "GraphEdge",
]

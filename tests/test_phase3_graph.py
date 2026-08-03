"""
Phase 3 tests — call graph construction and querying.

Tests verify:
  - All nodes from Phase 2 chunks appear in the graph
  - Known call edges exist:
      sample.py::main → sample.py::greet
      sample.py::main → sample.py::add
      sample.py::Calculator._compute_internal → sample.py::_clamp (nested)
      sample.js::processEvents → sample.js::multiply
  - callers() returns the correct inverse of callees()
  - neighbors() at depth 0, 1, 2 returns the right sets
  - Graph serialisation round-trips without data loss
  - CallGraphAnalyzer.print_summary() runs without crashing
  - External node behaviour (add_external_nodes flag)
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from coderag.chunking import Chunk, ChunkExtractor
from coderag.graph import CallGraphAnalyzer, GraphBuilder, chunk_to_node_id
from coderag.ingestion.parser import ASTParser

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"
SAMPLE_PY = FIXTURE_REPO / "sample.py"
SAMPLE_JS = FIXTURE_REPO / "sample.js"


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _parse_and_chunk(path: Path, language: str) -> list[Chunk]:
    parser = ASTParser()
    parsed = parser.parse_bytes(path.read_bytes(), language, relative_path=Path(path.name))
    assert parsed is not None
    extractor = ChunkExtractor()
    return extractor.extract(parsed)


@pytest.fixture(scope="module")
def py_chunks() -> list[Chunk]:
    return _parse_and_chunk(SAMPLE_PY, "python")


@pytest.fixture(scope="module")
def js_chunks() -> list[Chunk]:
    return _parse_and_chunk(SAMPLE_JS, "javascript")


@pytest.fixture(scope="module")
def py_graph(py_chunks: list[Chunk]) -> nx.DiGraph:
    builder = GraphBuilder(add_external_nodes=False)
    return builder.build(py_chunks)


@pytest.fixture(scope="module")
def py_analyzer(py_graph: nx.DiGraph, py_chunks: list[Chunk]) -> CallGraphAnalyzer:
    chunk_map = {c.chunk_id: c for c in py_chunks}
    return CallGraphAnalyzer(py_graph, chunk_map)


@pytest.fixture(scope="module")
def js_graph(js_chunks: list[Chunk]) -> nx.DiGraph:
    return GraphBuilder(add_external_nodes=False).build(js_chunks)


# ── Graph structure ───────────────────────────────────────────────────────────


class TestGraphStructure:
    def test_all_chunks_are_nodes(
        self, py_chunks: list[Chunk], py_graph: nx.DiGraph
    ) -> None:
        for chunk in py_chunks:
            node_id = chunk_to_node_id(chunk)
            assert py_graph.has_node(node_id), (
                f"Chunk {chunk.citation} missing from graph"
            )

    def test_graph_is_directed(self, py_graph: nx.DiGraph) -> None:
        assert isinstance(py_graph, nx.DiGraph)

    def test_node_has_metadata(self, py_graph: nx.DiGraph) -> None:
        for node_id, attrs in py_graph.nodes(data=True):
            assert "node" in attrs, f"Node {node_id} missing 'node' attribute"
            from coderag.graph.models import GraphNode
            assert isinstance(attrs["node"], GraphNode)

    def test_node_ids_contain_file_and_name(
        self, py_chunks: list[Chunk]
    ) -> None:
        for chunk in py_chunks:
            nid = chunk_to_node_id(chunk)
            assert "::" in nid
            file_part, name_part = nid.split("::", 1)
            assert file_part == chunk.file_path
            assert name_part == chunk.qualified_name

    def test_graph_has_call_edges(self, py_graph: nx.DiGraph) -> None:
        call_edges = [
            (u, v) for u, v, d in py_graph.edges(data=True)
            if d.get("edge_type") == "calls"
        ]
        assert len(call_edges) > 0, "Graph should have at least one call edge"

    def test_edge_has_edge_type_attribute(self, py_graph: nx.DiGraph) -> None:
        for u, v, data in py_graph.edges(data=True):
            assert "edge_type" in data, f"Edge {u}→{v} missing edge_type"
            assert data["edge_type"] in ("calls", "imports")

    def test_edge_has_call_name_attribute(self, py_graph: nx.DiGraph) -> None:
        for u, v, data in py_graph.edges(data=True):
            assert "call_name" in data, f"Edge {u}→{v} missing call_name"


# ── Known call relationships — Python ────────────────────────────────────────


class TestPythonCallEdges:
    def _node_id(self, chunks: list[Chunk], name: str) -> str:
        """Find the node_id for a chunk by function name."""
        chunk = next((c for c in chunks if c.function_name == name), None)
        assert chunk is not None, f"No chunk found with function_name={name!r}"
        return chunk_to_node_id(chunk)

    def test_main_calls_greet(
        self, py_chunks: list[Chunk], py_graph: nx.DiGraph
    ) -> None:
        main_id = self._node_id(py_chunks, "main")
        greet_id = self._node_id(py_chunks, "greet")
        assert py_graph.has_edge(main_id, greet_id), (
            "main() must have a 'calls' edge to greet()"
        )

    def test_main_calls_add(
        self, py_chunks: list[Chunk], py_graph: nx.DiGraph
    ) -> None:
        main_id = self._node_id(py_chunks, "main")
        # top-level add (class_name=None)
        add_chunk = next(
            (c for c in py_chunks if c.function_name == "add" and c.class_name is None),
            None,
        )
        assert add_chunk is not None, "Top-level add() chunk not found"
        add_id = chunk_to_node_id(add_chunk)
        assert py_graph.has_edge(main_id, add_id), (
            "main() must have a 'calls' edge to top-level add()"
        )

    def test_main_calls_calculator(
        self, py_chunks: list[Chunk], py_graph: nx.DiGraph
    ) -> None:
        main_id = self._node_id(py_chunks, "main")
        calc_chunk = next(
            (c for c in py_chunks if c.function_name == "Calculator"
             and c.node_type == "class_definition"),
            None,
        )
        assert calc_chunk is not None, "Calculator class chunk not found"
        calc_id = chunk_to_node_id(calc_chunk)
        assert py_graph.has_edge(main_id, calc_id), (
            "main() must have a 'calls' edge to Calculator (constructor call)"
        )

    def test_compute_internal_calls_clamp(
        self, py_chunks: list[Chunk], py_graph: nx.DiGraph
    ) -> None:
        """_compute_internal calls the nested _clamp function."""
        compute_chunk = next(
            (c for c in py_chunks if c.function_name == "_compute_internal"),
            None,
        )
        clamp_chunk = next(
            (c for c in py_chunks if c.function_name == "_clamp"),
            None,
        )
        assert compute_chunk is not None, "_compute_internal chunk not found"
        assert clamp_chunk is not None, "_clamp chunk not found"

        compute_id = chunk_to_node_id(compute_chunk)
        clamp_id = chunk_to_node_id(clamp_chunk)
        assert py_graph.has_edge(compute_id, clamp_id), (
            "_compute_internal must have a 'calls' edge to _clamp"
        )

    def test_call_edge_has_correct_call_name(
        self, py_chunks: list[Chunk], py_graph: nx.DiGraph
    ) -> None:
        main_id = self._node_id(py_chunks, "main")
        greet_id = self._node_id(py_chunks, "greet")
        data = py_graph.get_edge_data(main_id, greet_id)
        assert data is not None
        assert data["call_name"] == "greet"
        assert data["is_resolved"] is True


# ── Known call relationships — JavaScript ────────────────────────────────────


class TestJavaScriptCallEdges:
    def test_process_events_calls_multiply(
        self, js_chunks: list[Chunk], js_graph: nx.DiGraph
    ) -> None:
        pe_chunk = next(
            (c for c in js_chunks if c.function_name == "processEvents"), None
        )
        mul_chunk = next(
            (c for c in js_chunks if c.function_name == "multiply"), None
        )
        assert pe_chunk is not None, "processEvents chunk not found"
        assert mul_chunk is not None, "multiply chunk not found"

        pe_id = chunk_to_node_id(pe_chunk)
        mul_id = chunk_to_node_id(mul_chunk)
        assert js_graph.has_edge(pe_id, mul_id), (
            "processEvents must have a 'calls' edge to multiply"
        )

    def test_process_events_calls_eventemitter(
        self, js_chunks: list[Chunk], js_graph: nx.DiGraph
    ) -> None:
        pe_chunk = next(
            (c for c in js_chunks if c.function_name == "processEvents"), None
        )
        ee_chunk = next(
            (c for c in js_chunks if c.function_name == "EventEmitter"), None
        )
        assert pe_chunk is not None
        assert ee_chunk is not None
        assert js_graph.has_edge(chunk_to_node_id(pe_chunk), chunk_to_node_id(ee_chunk))


# ── CallGraphAnalyzer ─────────────────────────────────────────────────────────


class TestCallGraphAnalyzer:
    def _get_chunk(self, chunks: list[Chunk], name: str) -> Chunk:
        chunk = next((c for c in chunks if c.function_name == name), None)
        assert chunk is not None, f"Chunk {name!r} not found"
        return chunk

    def test_callees_of_main(
        self, py_chunks: list[Chunk], py_analyzer: CallGraphAnalyzer
    ) -> None:
        main = self._get_chunk(py_chunks, "main")
        callees = py_analyzer.callees(main)
        callee_names = {c.function_name for c in callees}
        assert "greet" in callee_names, f"greet not in callees of main: {callee_names}"
        assert "add" in callee_names, f"add not in callees of main: {callee_names}"

    def test_callers_of_greet(
        self, py_chunks: list[Chunk], py_analyzer: CallGraphAnalyzer
    ) -> None:
        greet = self._get_chunk(py_chunks, "greet")
        callers = py_analyzer.callers(greet)
        caller_names = {c.function_name for c in callers}
        assert "main" in caller_names, (
            f"main not listed as caller of greet: {caller_names}"
        )

    def test_callers_callees_are_inverse(
        self, py_chunks: list[Chunk], py_analyzer: CallGraphAnalyzer
    ) -> None:
        """For every A → B call edge, B.callers must include A."""
        main = self._get_chunk(py_chunks, "main")
        for callee in py_analyzer.callees(main):
            callers_of_callee = {c.chunk_id for c in py_analyzer.callers(callee)}
            assert main.chunk_id in callers_of_callee, (
                f"main not in callers of {callee.qualified_name}"
            )

    def test_neighbors_depth_0_empty(
        self, py_chunks: list[Chunk], py_analyzer: CallGraphAnalyzer
    ) -> None:
        main = self._get_chunk(py_chunks, "main")
        neighbors = py_analyzer.neighbors(main, depth=0)
        assert neighbors == []

    def test_neighbors_depth_1_includes_callees(
        self, py_chunks: list[Chunk], py_analyzer: CallGraphAnalyzer
    ) -> None:
        main = self._get_chunk(py_chunks, "main")
        neighbors = py_analyzer.neighbors(main, depth=1)
        neighbor_names = {c.function_name for c in neighbors}
        assert "greet" in neighbor_names

    def test_neighbors_depth_2_transitive(
        self, py_chunks: list[Chunk], py_analyzer: CallGraphAnalyzer
    ) -> None:
        """At depth 2, neighbors should reach at least greet and its callers."""
        greet = self._get_chunk(py_chunks, "greet")
        neighbors_d2 = py_analyzer.neighbors(greet, depth=2)
        # main calls greet, so main should be reachable at depth 1
        # main calls add too — add should be reachable at depth 2
        names = {c.function_name for c in neighbors_d2}
        assert "main" in names, "main must be reachable at depth 1 from greet"

    def test_unknown_chunk_returns_empty(
        self, py_chunks: list[Chunk], py_analyzer: CallGraphAnalyzer
    ) -> None:
        """Querying a chunk that isn't in the graph should return empty lists."""
        from coderag.chunking.models import make_chunk_id
        ghost = Chunk(
            chunk_id=make_chunk_id("ghost.py", "function_definition", 1, "ghost"),
            file_path="ghost.py",
            start_line=1, end_line=5,
            language="python",
            node_type="function_definition",
            function_name="ghost", class_name=None,
            raw_code="def ghost(): pass",
            docstring=None, has_docstring=False,
            imports=[], token_count=5,
        )
        assert py_analyzer.callees(ghost) == []
        assert py_analyzer.callers(ghost) == []
        assert py_analyzer.neighbors(ghost, depth=1) == []

    def test_get_edges_returns_graphedge_objects(
        self, py_chunks: list[Chunk], py_analyzer: CallGraphAnalyzer
    ) -> None:
        main = self._get_chunk(py_chunks, "main")
        edges = py_analyzer.get_edges(main)
        assert len(edges) > 0
        from coderag.graph.models import GraphEdge
        for e in edges:
            assert isinstance(e, GraphEdge)

    def test_node_count(
        self, py_chunks: list[Chunk], py_analyzer: CallGraphAnalyzer
    ) -> None:
        assert py_analyzer.node_count() == len(py_chunks)

    def test_edge_count(self, py_analyzer: CallGraphAnalyzer) -> None:
        assert py_analyzer.edge_count("calls") >= 3

    def test_print_summary_runs(
        self, py_chunks: list[Chunk], py_analyzer: CallGraphAnalyzer, capsys
    ) -> None:
        """print_summary() must not raise; output should mention 'call'."""
        py_analyzer.print_summary(file_path="sample.py")
        captured = capsys.readouterr()
        assert "calls" in captured.out.lower() or "call" in captured.out.lower()


# ── External nodes ────────────────────────────────────────────────────────────


class TestExternalNodes:
    def test_external_nodes_added_when_flag_on(self, py_chunks: list[Chunk]) -> None:
        builder = GraphBuilder(add_external_nodes=True)
        graph = builder.build(py_chunks)
        external_nodes = [
            n for n in graph.nodes()
            if n.startswith("external::")
        ]
        # main calls print() which is external
        assert len(external_nodes) > 0, (
            "With add_external_nodes=True, external calls (e.g. print) must appear"
        )

    def test_external_nodes_absent_when_flag_off(self, py_chunks: list[Chunk]) -> None:
        builder = GraphBuilder(add_external_nodes=False)
        graph = builder.build(py_chunks)
        external_nodes = [n for n in graph.nodes() if n.startswith("external::")]
        assert external_nodes == []


# ── Serialisation ─────────────────────────────────────────────────────────────


class TestGraphSerialisation:
    def test_to_node_link_is_json_serialisable(self, py_graph: nx.DiGraph) -> None:
        import json
        data = GraphBuilder.to_node_link(py_graph)
        # Must not raise
        serialised = json.dumps(data)
        assert "nodes" in serialised

    def test_round_trip_preserves_node_count(self, py_graph: nx.DiGraph) -> None:
        data = GraphBuilder.to_node_link(py_graph)
        restored = GraphBuilder.from_node_link(data)
        assert restored.number_of_nodes() == py_graph.number_of_nodes()

    def test_round_trip_preserves_edge_count(self, py_graph: nx.DiGraph) -> None:
        data = GraphBuilder.to_node_link(py_graph)
        restored = GraphBuilder.from_node_link(data)
        assert restored.number_of_edges() == py_graph.number_of_edges()


# ── Report: print graph for sample.py ────────────────────────────────────────


class TestPhase3Report:
    def test_print_graph_report(
        self, py_chunks: list[Chunk], py_analyzer: CallGraphAnalyzer, capsys
    ) -> None:
        """Phase 3 report: visualise the call graph for sample.py."""
        print("\n\n=== PHASE 3 REPORT: Call Graph for sample.py ===")
        py_analyzer.print_summary(file_path="sample.py")
        captured = capsys.readouterr()
        # Basic sanity: output contains at least one node name
        assert "main" in captured.out or "greet" in captured.out or "add" in captured.out

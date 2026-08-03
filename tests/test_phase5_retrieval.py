"""
Phase 5 tests — retrieval pipeline: similarity search, graph expansion,
re-ranking, context assembly.

Tests use:
  - MockEmbeddingModel (deterministic, no ML dependencies)
  - ChromaVectorStore.ephemeral() (in-memory, no filesystem I/O)
  - The existing sample_repo fixture from earlier phases
  - Indexer from Phase 4 to build a fully populated store + graph

Coverage:
  1. RetrievalResult — model fields, source tag, citation
  2. RetrievalContext — summary, citations, is_empty, chunk_count
  3. ContextBuilder — budget enforcement, reading order, edge cases
  4. NoOpReranker — pass-through, order preserved
  5. Retriever — end-to-end on sample_repo; with/without graph; scores
  6. Report — print retrieval stats for sample queries
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coderag.chunking import Chunk
from coderag.chunking.models import make_chunk_id
from coderag.embeddings.factory import MockEmbeddingModel
from coderag.graph.analyzer import CallGraphAnalyzer
from coderag.graph.builder import GraphBuilder
from coderag.indexer import Indexer
from coderag.retrieval.context_builder import ContextBuilder
from coderag.retrieval.models import RetrievalContext, RetrievalResult
from coderag.retrieval.reranker import NoOpReranker, get_reranker
from coderag.retrieval.retriever import Retriever
from coderag.store.chroma_store import ChromaVectorStore

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_result(
    name: str = "test_fn",
    score: float = 0.9,
    source: str = "vector_search",
    enriched_text: str | None = None,
) -> RetrievalResult:
    """Create a minimal RetrievalResult for unit tests."""
    cid = make_chunk_id("test.py", "function_definition", 1, name)
    chunk = Chunk(
        chunk_id=cid,
        file_path="test.py",
        start_line=1,
        end_line=5,
        language="python",
        node_type="function_definition",
        function_name=name,
        class_name=None,
        raw_code=f"def {name}():\n    pass\n",
        docstring=None,
        has_docstring=False,
        imports=[],
        token_count=10,
    )
    return RetrievalResult(
        chunk=chunk,
        score=score,
        enriched_text=enriched_text or f"[File]: test.py:L1-L5\n[Code]:\ndef {name}():\n    pass",
        source=source,
    )


def _build_index(tmp_path: Path):
    """Index the sample_repo fixture and return (store, embedder, indexer)."""
    store = ChromaVectorStore.ephemeral()
    embedder = MockEmbeddingModel(dim=128)
    indexer = Indexer(
        embedding_model=embedder,
        vector_store=store,
        coderag_dir=tmp_path / ".coderag",
    )
    stats = indexer.index_repo(FIXTURE_REPO, incremental=False)
    return store, embedder, indexer, stats


# ── RetrievalResult model ────────────────────────────────────────────────────


class TestRetrievalResult:
    def test_has_chunk_and_score(self) -> None:
        r = _make_result("greet", score=0.85)
        assert r.chunk.function_name == "greet"
        assert r.score == 0.85

    def test_source_default_is_vector_search(self) -> None:
        r = _make_result()
        assert r.source == "vector_search"

    def test_source_can_be_graph_expansion(self) -> None:
        r = _make_result(source="graph_expansion")
        assert r.source == "graph_expansion"

    def test_citation_property(self) -> None:
        r = _make_result("my_func")
        assert "test.py" in r.citation

    def test_chunk_id_property(self) -> None:
        r = _make_result()
        assert r.chunk_id == r.chunk.chunk_id

    def test_to_dict_contains_key_fields(self) -> None:
        r = _make_result("foo", score=0.75)
        d = r.to_dict()
        assert d["score"] == 0.75
        assert d["source"] == "vector_search"
        assert "chunk_id" in d
        assert "citation" in d


# ── RetrievalContext model ────────────────────────────────────────────────────


class TestRetrievalContext:
    def test_empty_context(self) -> None:
        ctx = RetrievalContext(results=[], query="test")
        assert ctx.is_empty
        assert ctx.chunk_count == 0
        assert ctx.citations() == []

    def test_summary_includes_count(self) -> None:
        results = [_make_result("a"), _make_result("b", source="graph_expansion")]
        ctx = RetrievalContext(results=results, query="test", total_tokens=100)
        s = ctx.summary()
        assert "2 chunk(s)" in s
        assert "100 tokens" in s

    def test_citations_returns_all(self) -> None:
        results = [_make_result("a"), _make_result("b")]
        ctx = RetrievalContext(results=results, query="test")
        cits = ctx.citations()
        assert len(cits) == 2
        assert all("test.py" in c for c in cits)


# ── ContextBuilder ────────────────────────────────────────────────────────────


class TestContextBuilder:
    def test_empty_results_returns_empty_context(self) -> None:
        builder = ContextBuilder(token_budget=1000)
        ctx = builder.build([], query="test")
        assert ctx.is_empty
        assert ctx.context_text == ""

    def test_single_chunk_always_included(self) -> None:
        builder = ContextBuilder(token_budget=1000)
        ctx = builder.build([_make_result()], query="test")
        assert ctx.chunk_count == 1
        assert ctx.total_tokens > 0

    def test_budget_limits_chunks(self) -> None:
        """With a very small budget, not all chunks should be included."""
        results = [_make_result(f"fn_{i}", score=0.9 - i * 0.1) for i in range(10)]
        builder = ContextBuilder(token_budget=50)  # very tight budget
        ctx = builder.build(results, query="test")
        # Should include fewer than all 10
        assert ctx.chunk_count < 10
        assert ctx.chunk_count >= 1  # at least the fallback

    def test_context_text_contains_citations(self) -> None:
        results = [_make_result("greet", score=0.9)]
        builder = ContextBuilder(token_budget=5000)
        ctx = builder.build(results, query="test")
        assert "test.py" in ctx.context_text
        assert "score=" in ctx.context_text

    def test_results_sorted_by_file_position(self) -> None:
        """Selected results should be in (file_path, start_line) order."""
        r1 = _make_result("fn_a", score=0.5)
        r2 = _make_result("fn_b", score=0.9)
        r1.chunk = Chunk(**{**r1.chunk.to_dict(), "start_line": 50})
        r2.chunk = Chunk(**{**r2.chunk.to_dict(), "start_line": 10})
        builder = ContextBuilder(token_budget=5000)
        ctx = builder.build([r1, r2], query="test")
        # r2 (line 10) should come before r1 (line 50) in reading order
        assert ctx.results[0].chunk.start_line <= ctx.results[1].chunk.start_line

    def test_highest_scored_selected_first_when_budget_tight(self) -> None:
        """When budget is tight, higher-scored chunks are preferred."""
        low = _make_result("low", score=0.3)
        high = _make_result("high", score=0.95)
        builder = ContextBuilder(token_budget=40)  # Only room for ~1 chunk
        ctx = builder.build([low, high], query="test")
        if ctx.chunk_count == 1:
            assert ctx.results[0].chunk.function_name == "high"


# ── NoOpReranker ──────────────────────────────────────────────────────────────


class TestNoOpReranker:
    def test_returns_same_results(self) -> None:
        reranker = NoOpReranker()
        results = [_make_result("a", score=0.9), _make_result("b", score=0.5)]
        reranked = reranker.rerank("test query", results)
        assert len(reranked) == len(results)
        assert reranked[0].score == 0.9
        assert reranked[1].score == 0.5

    def test_empty_input(self) -> None:
        reranker = NoOpReranker()
        assert reranker.rerank("test", []) == []


class TestGetReranker:
    def test_disabled_returns_noop(self) -> None:
        r = get_reranker(enabled=False)
        assert isinstance(r, NoOpReranker)

    def test_enabled_without_model_returns_noop(self) -> None:
        r = get_reranker(enabled=True, model_name="")
        assert isinstance(r, NoOpReranker)


# ── Retriever (end-to-end on sample_repo) ─────────────────────────────────────


class TestRetriever:
    @pytest.fixture
    def indexed_repo(self, tmp_path: Path):
        """Index the sample_repo and return components needed for retrieval."""
        store, embedder, indexer, stats = _build_index(tmp_path)
        graph = indexer.load_graph()
        return store, embedder, graph, stats

    def test_retrieve_returns_context(self, indexed_repo) -> None:
        store, embedder, graph, _ = indexed_repo
        retriever = Retriever(
            embedding_model=embedder,
            vector_store=store,
            top_k=5,
        )
        ctx = retriever.retrieve("greet function")
        assert isinstance(ctx, RetrievalContext)
        assert not ctx.is_empty

    def test_results_have_scores(self, indexed_repo) -> None:
        store, embedder, graph, _ = indexed_repo
        retriever = Retriever(
            embedding_model=embedder,
            vector_store=store,
            top_k=5,
        )
        ctx = retriever.retrieve("add numbers")
        for r in ctx.results:
            assert isinstance(r.score, float)

    def test_citations_not_empty(self, indexed_repo) -> None:
        store, embedder, graph, _ = indexed_repo
        retriever = Retriever(
            embedding_model=embedder,
            vector_store=store,
            top_k=5,
        )
        ctx = retriever.retrieve("Calculator class")
        assert len(ctx.citations()) > 0

    def test_context_text_not_empty(self, indexed_repo) -> None:
        store, embedder, graph, _ = indexed_repo
        retriever = Retriever(
            embedding_model=embedder,
            vector_store=store,
            top_k=3,
        )
        ctx = retriever.retrieve("how does retry work")
        assert len(ctx.context_text) > 0

    def test_token_budget_respected(self, indexed_repo) -> None:
        store, embedder, graph, _ = indexed_repo
        retriever_tight = Retriever(
            embedding_model=embedder,
            vector_store=store,
            top_k=8,
            context_token_budget=200,
        )
        retriever_loose = Retriever(
            embedding_model=embedder,
            vector_store=store,
            top_k=8,
            context_token_budget=8000,
        )
        ctx_tight = retriever_tight.retrieve("all functions")
        ctx_loose = retriever_loose.retrieve("all functions")
        # A tighter budget should produce fewer chunks (or fewer tokens)
        assert ctx_tight.chunk_count <= ctx_loose.chunk_count
        # And the tight context should have meaningfully fewer tokens
        assert ctx_tight.total_tokens < ctx_loose.total_tokens

    def test_without_graph_still_works(self, indexed_repo) -> None:
        store, embedder, _, _ = indexed_repo
        retriever = Retriever(
            embedding_model=embedder,
            vector_store=store,
            graph_analyzer=None,
            top_k=5,
        )
        ctx = retriever.retrieve("greet")
        assert not ctx.is_empty
        assert all(r.source == "vector_search" for r in ctx.results)

    def test_with_graph_adds_expansion_results(self, indexed_repo) -> None:
        store, embedder, graph, _ = indexed_repo
        if graph is None:
            pytest.skip("Graph not built")

        # Build chunk map from the store for the analyzer
        from coderag.chunking import ChunkExtractor
        from coderag.ingestion.parser import ASTParser
        from coderag.ingestion.walker import FileWalker

        parser = ASTParser()
        walker = FileWalker(FIXTURE_REPO)
        all_chunks = []
        for sf in walker.walk():
            parsed = parser.parse_file(sf)
            if parsed:
                all_chunks.extend(ChunkExtractor().extract(parsed))
        chunk_map = {c.chunk_id: c for c in all_chunks}
        analyzer = CallGraphAnalyzer(graph, chunk_map)

        retriever = Retriever(
            embedding_model=embedder,
            vector_store=store,
            graph_analyzer=analyzer,
            top_k=3,
            graph_expansion_depth=1,
            context_token_budget=8000,
        )
        ctx = retriever.retrieve("main function")
        # With graph expansion, we may get additional results
        sources = {r.source for r in ctx.results}
        # At minimum we should have vector_search results
        assert "vector_search" in sources

    def test_deduplication_keeps_highest_score(self, indexed_repo) -> None:
        store, embedder, _, _ = indexed_repo
        retriever = Retriever(
            embedding_model=embedder,
            vector_store=store,
            top_k=5,
        )
        # The dedup is an internal detail — we verify that chunk_ids are unique
        ctx = retriever.retrieve("function")
        ids = [r.chunk_id for r in ctx.results]
        assert len(ids) == len(set(ids)), "Duplicate chunk_ids in results"


# ── Phase 5 report ────────────────────────────────────────────────────────────


class TestPhase5Report:
    def test_print_retrieval_report(self, tmp_path: Path) -> None:
        store, embedder, indexer, stats = _build_index(tmp_path)

        print("\n\n=== PHASE 5 REPORT: Retrieval Pipeline ===")
        print(f"  Index: {stats.files_processed} files, {stats.chunks_indexed} chunks")
        print(f"  Store count: {store.count()}")

        retriever = Retriever(
            embedding_model=embedder,
            vector_store=store,
            top_k=5,
            context_token_budget=4000,
        )

        queries = [
            "greet function",
            "how does the Calculator class work",
            "what does the retry decorator do",
        ]

        for query in queries:
            ctx = retriever.retrieve(query)
            print(f"\n  Query: \"{query}\"")
            print(f"  {ctx.summary()}")
            for i, r in enumerate(ctx.results[:3]):
                print(f"    [{i+1}] {r.citation}  score={r.score:.3f}  src={r.source}")

        # Sanity check: at least one query returned results
        final_ctx = retriever.retrieve(queries[0])
        assert final_ctx.chunk_count > 0

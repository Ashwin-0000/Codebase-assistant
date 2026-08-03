"""
Phase 4 tests — embeddings, enrichment, vector store, and full index pipeline.

Tests use:
  - MockEmbeddingModel (deterministic, no ML dependencies)
  - ChromaVectorStore.ephemeral() (in-memory, no filesystem I/O)
  - The existing sample_repo fixture from Phase 1/2

Coverage:
  1. MockEmbeddingModel — consistent vectors, correct dimension
  2. ChunkEnricher — format, summary source priority, batch, LLM docgen path
  3. ChromaVectorStore — upsert, count, get_by_id, query, delete, clear
  4. Chunk serialisation round-trip through Chroma (all field types preserved)
  5. Incremental indexing — unchanged files skipped, changed files re-indexed
  6. Indexer — integration over sample_repo; stats sanity checks
  7. Report — print index stats for the sample_repo run
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from coderag.chunking import Chunk, ChunkExtractor
from coderag.chunking.models import make_chunk_id
from coderag.embeddings.factory import MockEmbeddingModel
from coderag.enricher import ChunkEnricher
from coderag.indexer import Indexer, IndexStats
from coderag.ingestion.parser import ASTParser
from coderag.store import SearchResult
from coderag.store.chroma_store import ChromaVectorStore

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"
SAMPLE_PY = FIXTURE_REPO / "sample.py"


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_chunk(
    function_name: str = "test_fn",
    *,
    chunk_id: str | None = None,
    raw_code: str = "def test_fn():\n    pass\n",
    docstring: str | None = None,
    class_name: str | None = None,
    imports: list[str] | None = None,
    is_split: bool = False,
    split_index: int | None = None,
    total_splits: int | None = None,
) -> Chunk:
    cid = chunk_id or make_chunk_id("test.py", "function_definition", 1, function_name)
    return Chunk(
        chunk_id=cid,
        file_path="test.py",
        start_line=1,
        end_line=raw_code.count("\n") + 1,
        language="python",
        node_type="function_definition",
        function_name=function_name,
        class_name=class_name,
        raw_code=raw_code,
        docstring=docstring,
        has_docstring=docstring is not None,
        imports=imports or ["import os"],
        token_count=len(raw_code) // 4,
        is_split_chunk=is_split,
        split_index=split_index,
        total_splits=total_splits,
    )


def _chunks_from_py() -> list[Chunk]:
    parser = ASTParser()
    parsed = parser.parse_bytes(
        SAMPLE_PY.read_bytes(), "python", relative_path=Path("sample.py")
    )
    assert parsed is not None
    return ChunkExtractor().extract(parsed)


# ── MockEmbeddingModel ────────────────────────────────────────────────────────


class TestMockEmbeddingModel:
    def test_embed_returns_correct_number(self) -> None:
        m = MockEmbeddingModel(dim=64)
        vecs = m.embed(["hello", "world"])
        assert len(vecs) == 2

    def test_embed_returns_correct_dimension(self) -> None:
        m = MockEmbeddingModel(dim=64)
        vecs = m.embed(["hello"])
        assert len(vecs[0]) == 64

    def test_same_text_same_vector(self) -> None:
        m = MockEmbeddingModel(dim=128)
        v1 = m.embed(["the quick brown fox"])
        v2 = m.embed(["the quick brown fox"])
        assert v1 == v2

    def test_different_text_different_vector(self) -> None:
        m = MockEmbeddingModel(dim=128)
        v1 = m.embed(["alpha"])
        v2 = m.embed(["beta"])
        assert v1 != v2

    def test_embed_one_matches_embed(self) -> None:
        m = MockEmbeddingModel(dim=64)
        assert m.embed_one("foo") == m.embed(["foo"])[0]

    def test_vectors_are_unit_length(self) -> None:
        import math
        m = MockEmbeddingModel(dim=64)
        for vec in m.embed(["apple", "banana", "cherry"]):
            norm = math.sqrt(sum(x * x for x in vec))
            assert abs(norm - 1.0) < 1e-6, f"Vector not unit-length: norm={norm}"

    def test_empty_batch_returns_empty(self) -> None:
        m = MockEmbeddingModel(dim=64)
        assert m.embed([]) == []

    def test_dimension_property(self) -> None:
        m = MockEmbeddingModel(dim=256)
        assert m.dimension == 256

    def test_model_name(self) -> None:
        m = MockEmbeddingModel(dim=64)
        assert "64" in m.model_name


# ── ChunkEnricher ─────────────────────────────────────────────────────────────


class TestChunkEnricher:
    def test_enriched_text_contains_file(self) -> None:
        chunk = _make_chunk()
        text = ChunkEnricher().enrich(chunk)
        assert "test.py" in text

    def test_enriched_text_contains_scope(self) -> None:
        chunk = _make_chunk()
        text = ChunkEnricher().enrich(chunk)
        assert "test_fn" in text

    def test_enriched_text_contains_code(self) -> None:
        chunk = _make_chunk(raw_code="def test_fn():\n    return 42\n")
        text = ChunkEnricher().enrich(chunk)
        assert "return 42" in text

    def test_docstring_used_as_summary(self) -> None:
        chunk = _make_chunk(docstring="Perform the main computation.")
        text = ChunkEnricher().enrich(chunk)
        assert "Perform the main computation" in text

    def test_no_docstring_no_llm_no_summary_line(self) -> None:
        """Without LLM, chunks lacking docstrings should still produce valid enriched text."""
        chunk = _make_chunk(docstring=None)
        text = ChunkEnricher().enrich(chunk)
        # Must still contain core fields
        assert "[File]:" in text
        assert "[Code]:" in text

    def test_llm_docgen_called_when_no_docstring(self) -> None:
        calls: list[Chunk] = []

        def fake_llm(c: Chunk) -> str:
            calls.append(c)
            return "Auto-generated summary."

        chunk = _make_chunk(docstring=None)
        text = ChunkEnricher(docgen_fn=fake_llm).enrich(chunk)
        assert len(calls) == 1
        assert "Auto-generated summary." in text

    def test_llm_not_called_when_docstring_present(self) -> None:
        calls: list[Chunk] = []

        def fake_llm(c: Chunk) -> str:
            calls.append(c)
            return "Should not be called."

        chunk = _make_chunk(docstring="Existing docstring.")
        ChunkEnricher(docgen_fn=fake_llm).enrich(chunk)
        assert calls == []

    def test_imports_included_when_present(self) -> None:
        chunk = _make_chunk(imports=["import os", "from pathlib import Path"])
        text = ChunkEnricher().enrich(chunk)
        assert "import os" in text

    def test_enrich_batch_returns_one_per_chunk(self) -> None:
        chunks = [_make_chunk(function_name=f"fn_{i}", chunk_id=make_chunk_id("t.py", "f", i, f"fn_{i}"))
                  for i in range(5)]
        enricher = ChunkEnricher()
        texts = enricher.enrich_batch(chunks)
        assert len(texts) == 5

    def test_class_method_shows_qualified_name(self) -> None:
        chunk = _make_chunk(function_name="add", class_name="Calculator")
        text = ChunkEnricher().enrich(chunk)
        assert "Calculator.add" in text

    def test_multiline_docstring_uses_first_line(self) -> None:
        chunk = _make_chunk(docstring="First line summary.\n\nMore detail here.")
        text = ChunkEnricher().enrich(chunk)
        assert "First line summary." in text


# ── ChromaVectorStore ─────────────────────────────────────────────────────────


class TestChromaVectorStore:
    @pytest.fixture
    def store(self) -> ChromaVectorStore:
        return ChromaVectorStore.ephemeral()

    @pytest.fixture
    def embedder(self) -> MockEmbeddingModel:
        return MockEmbeddingModel(dim=128)

    @pytest.fixture
    def sample_chunks(self) -> list[Chunk]:
        return [
            _make_chunk("alpha", chunk_id=make_chunk_id("f.py", "fn", 1, "alpha")),
            _make_chunk("beta",  chunk_id=make_chunk_id("f.py", "fn", 5, "beta"),
                        docstring="Beta does something."),
            _make_chunk("gamma", chunk_id=make_chunk_id("f.py", "fn", 10, "gamma"),
                        class_name="MyClass", imports=["import sys"]),
        ]

    def test_count_empty(self, store: ChromaVectorStore) -> None:
        assert store.count() == 0

    def test_upsert_increments_count(
        self,
        store: ChromaVectorStore,
        embedder: MockEmbeddingModel,
        sample_chunks: list[Chunk],
    ) -> None:
        texts = [f"text {i}" for i in range(len(sample_chunks))]
        embeddings = embedder.embed(texts)
        store.upsert(sample_chunks, embeddings, texts)
        assert store.count() == len(sample_chunks)

    def test_get_by_id_returns_chunk(
        self,
        store: ChromaVectorStore,
        embedder: MockEmbeddingModel,
        sample_chunks: list[Chunk],
    ) -> None:
        texts = ["t1", "t2", "t3"]
        store.upsert(sample_chunks, embedder.embed(texts), texts)

        chunk, doc = store.get_by_id(sample_chunks[0].chunk_id)
        assert chunk is not None
        assert chunk.chunk_id == sample_chunks[0].chunk_id
        assert doc == "t1"

    def test_get_by_id_missing_returns_none(
        self, store: ChromaVectorStore
    ) -> None:
        chunk, doc = store.get_by_id("nonexistent-id")
        assert chunk is None
        assert doc is None

    def test_delete_removes_chunks(
        self,
        store: ChromaVectorStore,
        embedder: MockEmbeddingModel,
        sample_chunks: list[Chunk],
    ) -> None:
        texts = ["t1", "t2", "t3"]
        store.upsert(sample_chunks, embedder.embed(texts), texts)
        store.delete([sample_chunks[0].chunk_id])
        assert store.count() == len(sample_chunks) - 1
        chunk, _ = store.get_by_id(sample_chunks[0].chunk_id)
        assert chunk is None

    def test_clear_removes_all(
        self,
        store: ChromaVectorStore,
        embedder: MockEmbeddingModel,
        sample_chunks: list[Chunk],
    ) -> None:
        texts = ["t1", "t2", "t3"]
        store.upsert(sample_chunks, embedder.embed(texts), texts)
        store.clear()
        assert store.count() == 0

    def test_query_returns_search_results(
        self,
        store: ChromaVectorStore,
        embedder: MockEmbeddingModel,
        sample_chunks: list[Chunk],
    ) -> None:
        texts = ["alpha code", "beta code", "gamma code"]
        store.upsert(sample_chunks, embedder.embed(texts), texts)
        query_vec = embedder.embed_one("alpha code")
        results = store.query(query_vec, top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_query_returns_best_match_first(
        self,
        store: ChromaVectorStore,
        embedder: MockEmbeddingModel,
        sample_chunks: list[Chunk],
    ) -> None:
        texts = ["alpha code", "totally different text", "another different text"]
        store.upsert(sample_chunks, embedder.embed(texts), texts)
        # Query with same text as first chunk — should be top result
        query_vec = embedder.embed_one("alpha code")
        results = store.query(query_vec, top_k=3)
        assert results[0].chunk.chunk_id == sample_chunks[0].chunk_id

    def test_query_scores_in_valid_range(
        self,
        store: ChromaVectorStore,
        embedder: MockEmbeddingModel,
        sample_chunks: list[Chunk],
    ) -> None:
        texts = ["t1", "t2", "t3"]
        store.upsert(sample_chunks, embedder.embed(texts), texts)
        results = store.query(embedder.embed_one("t1"), top_k=3)
        for r in results:
            assert -1.0 <= r.score <= 1.1, f"Score out of range: {r.score}"

    def test_upsert_is_idempotent(
        self,
        store: ChromaVectorStore,
        embedder: MockEmbeddingModel,
        sample_chunks: list[Chunk],
    ) -> None:
        """Upserting the same chunks twice should not duplicate them."""
        texts = ["t1", "t2", "t3"]
        store.upsert(sample_chunks, embedder.embed(texts), texts)
        store.upsert(sample_chunks, embedder.embed(texts), texts)
        assert store.count() == len(sample_chunks)


# ── Chunk serialisation round-trip ────────────────────────────────────────────


class TestChunkRoundTrip:
    """All Chunk fields must survive a write-read cycle through Chroma."""

    @pytest.fixture
    def store(self) -> ChromaVectorStore:
        return ChromaVectorStore.ephemeral()

    @pytest.fixture
    def embedder(self) -> MockEmbeddingModel:
        return MockEmbeddingModel(dim=64)

    def _roundtrip(
        self, store: ChromaVectorStore, embedder: MockEmbeddingModel, chunk: Chunk
    ) -> Chunk:
        text = f"enriched text for {chunk.chunk_id}"
        store.upsert([chunk], embedder.embed([text]), [text])
        restored, _ = store.get_by_id(chunk.chunk_id)
        assert restored is not None
        return restored

    def test_function_name_preserved(
        self, store: ChromaVectorStore, embedder: MockEmbeddingModel
    ) -> None:
        chunk = _make_chunk("my_func")
        r = self._roundtrip(store, embedder, chunk)
        assert r.function_name == "my_func"

    def test_class_name_preserved(
        self, store: ChromaVectorStore, embedder: MockEmbeddingModel
    ) -> None:
        chunk = _make_chunk("method", class_name="MyClass")
        r = self._roundtrip(store, embedder, chunk)
        assert r.class_name == "MyClass"

    def test_none_function_name_preserved(
        self, store: ChromaVectorStore, embedder: MockEmbeddingModel
    ) -> None:
        chunk = _make_chunk("Calculator")
        chunk = Chunk(**{**chunk.to_dict(), "function_name": None,
                        "node_type": "class_definition"})
        r = self._roundtrip(store, embedder, chunk)
        assert r.function_name is None

    def test_none_class_name_preserved(
        self, store: ChromaVectorStore, embedder: MockEmbeddingModel
    ) -> None:
        chunk = _make_chunk("standalone")
        r = self._roundtrip(store, embedder, chunk)
        assert r.class_name is None

    def test_docstring_preserved(
        self, store: ChromaVectorStore, embedder: MockEmbeddingModel
    ) -> None:
        chunk = _make_chunk(docstring="Compute something.")
        r = self._roundtrip(store, embedder, chunk)
        assert r.docstring == "Compute something."
        assert r.has_docstring is True

    def test_none_docstring_preserved(
        self, store: ChromaVectorStore, embedder: MockEmbeddingModel
    ) -> None:
        chunk = _make_chunk(docstring=None)
        r = self._roundtrip(store, embedder, chunk)
        assert r.docstring is None
        assert r.has_docstring is False

    def test_imports_list_preserved(
        self, store: ChromaVectorStore, embedder: MockEmbeddingModel
    ) -> None:
        chunk = _make_chunk(imports=["import os", "from pathlib import Path"])
        r = self._roundtrip(store, embedder, chunk)
        assert r.imports == ["import os", "from pathlib import Path"]

    def test_split_fields_preserved(
        self, store: ChromaVectorStore, embedder: MockEmbeddingModel
    ) -> None:
        chunk = _make_chunk(is_split=True, split_index=1, total_splits=3)
        r = self._roundtrip(store, embedder, chunk)
        assert r.is_split_chunk is True
        assert r.split_index == 1
        assert r.total_splits == 3

    def test_none_split_fields_preserved(
        self, store: ChromaVectorStore, embedder: MockEmbeddingModel
    ) -> None:
        chunk = _make_chunk(is_split=False, split_index=None, total_splits=None)
        r = self._roundtrip(store, embedder, chunk)
        assert r.is_split_chunk is False
        assert r.split_index is None
        assert r.total_splits is None

    def test_start_end_line_preserved(
        self, store: ChromaVectorStore, embedder: MockEmbeddingModel
    ) -> None:
        code = "def f():\n    pass\n    return 1\n"
        chunk = _make_chunk(raw_code=code)
        r = self._roundtrip(store, embedder, chunk)
        assert r.start_line == chunk.start_line
        assert r.end_line == chunk.end_line

    def test_token_count_preserved(
        self, store: ChromaVectorStore, embedder: MockEmbeddingModel
    ) -> None:
        chunk = _make_chunk()
        r = self._roundtrip(store, embedder, chunk)
        assert r.token_count == chunk.token_count


# ── Indexer integration ───────────────────────────────────────────────────────


class TestIndexer:
    @pytest.fixture
    def store(self) -> ChromaVectorStore:
        return ChromaVectorStore.ephemeral()

    @pytest.fixture
    def embedder(self) -> MockEmbeddingModel:
        return MockEmbeddingModel(dim=64)

    @pytest.fixture
    def indexer(
        self, store: ChromaVectorStore, embedder: MockEmbeddingModel, tmp_path: Path
    ) -> Indexer:
        return Indexer(
            embedding_model=embedder,
            vector_store=store,
            coderag_dir=tmp_path / ".coderag",
        )

    def test_index_returns_stats(
        self, indexer: Indexer, store: ChromaVectorStore
    ) -> None:
        stats = indexer.index_repo(FIXTURE_REPO)
        assert isinstance(stats, IndexStats)

    def test_index_processes_files(
        self, indexer: Indexer, store: ChromaVectorStore
    ) -> None:
        stats = indexer.index_repo(FIXTURE_REPO)
        assert stats.files_processed >= 2  # sample.py + sample.js at minimum

    def test_index_creates_chunks(
        self, indexer: Indexer, store: ChromaVectorStore
    ) -> None:
        stats = indexer.index_repo(FIXTURE_REPO)
        assert stats.chunks_indexed >= 8

    def test_store_has_chunks_after_index(
        self, indexer: Indexer, store: ChromaVectorStore
    ) -> None:
        indexer.index_repo(FIXTURE_REPO)
        assert store.count() >= 8

    def test_incremental_skips_unchanged_files(
        self, indexer: Indexer, store: ChromaVectorStore
    ) -> None:
        # First index
        stats1 = indexer.index_repo(FIXTURE_REPO, incremental=False)
        # Second index (incremental) — nothing changed
        stats2 = indexer.index_repo(FIXTURE_REPO, incremental=True)
        assert stats2.files_skipped >= stats1.files_processed
        assert stats2.files_processed == 0

    def test_non_incremental_reindexes_all(
        self, indexer: Indexer, store: ChromaVectorStore
    ) -> None:
        stats1 = indexer.index_repo(FIXTURE_REPO, incremental=False)
        stats2 = indexer.index_repo(FIXTURE_REPO, incremental=False)
        assert stats2.files_processed == stats1.files_processed

    def test_graph_persisted(
        self, indexer: Indexer, tmp_path: Path
    ) -> None:
        indexer.index_repo(FIXTURE_REPO)
        graph_path = tmp_path / ".coderag" / "graph.json"
        assert graph_path.exists()
        data = json.loads(graph_path.read_text())
        assert "nodes" in data

    def test_load_graph_after_index(
        self, indexer: Indexer
    ) -> None:
        indexer.index_repo(FIXTURE_REPO)
        graph = indexer.load_graph()
        assert graph is not None
        assert graph.number_of_nodes() >= 8

    def test_elapsed_seconds_positive(
        self, indexer: Indexer
    ) -> None:
        stats = indexer.index_repo(FIXTURE_REPO)
        assert stats.elapsed_seconds >= 0

    def test_errors_list_empty_on_success(
        self, indexer: Indexer
    ) -> None:
        stats = indexer.index_repo(FIXTURE_REPO)
        assert stats.errors == []


# ── Phase 4 report ────────────────────────────────────────────────────────────


class TestPhase4Report:
    def test_print_index_stats(self, tmp_path: Path) -> None:
        store = ChromaVectorStore.ephemeral()
        embedder = MockEmbeddingModel(dim=64)
        indexer = Indexer(
            embedding_model=embedder,
            vector_store=store,
            coderag_dir=tmp_path / ".coderag",
        )

        stats = indexer.index_repo(FIXTURE_REPO)

        print("\n\n=== PHASE 4 REPORT: Index Stats ===")
        print(stats.summary())
        print(f"  Files processed : {stats.files_processed}")
        print(f"  Files skipped   : {stats.files_skipped}")
        print(f"  Chunks indexed  : {stats.chunks_indexed}")
        print(f"  Store count     : {store.count()}")
        print(f"  Time            : {stats.elapsed_seconds:.2f}s")

        # Spot-check: query for a known function
        query_vec = embedder.embed_one("[Scope]: greet\n[Code]:\ndef greet(name: str):")
        results = store.query(query_vec, top_k=3)
        print("\n  Top-3 results for query about 'greet':")
        for i, r in enumerate(results):
            print(f"    [{i+1}] {r.chunk.citation}  score={r.score:.3f}")

        assert stats.chunks_indexed > 0
        assert len(results) > 0

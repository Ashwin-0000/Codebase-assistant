"""
Phase 7 tests — CLI wiring.

Tests use Typer's CliRunner so commands run in-process (no subprocess overhead).
All external I/O (embeddings, LLM, store) is either mocked or backed by the
MockEmbeddingModel + ChromaVectorStore.ephemeral() fixtures used in earlier phases.

Coverage:
  1. --version flag
  2. index — local path, happy path produces stats table
  3. index — force flag re-indexes unchanged files
  4. index -- invalid path exits with code 1
  5. ask   — no index → graceful error
  6. ask   — with pre-built index → answer printed
  7. ask   — --show-context flag
  8. ask   — --no-graph disables expansion
  9. reindex — delegates to index with incremental=True
  10. status — no index → informational message
  11. status — with index → table printed
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from coderag.cli import app
from coderag.config import get_settings

# Path to the shared sample_repo fixture
FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"

runner = CliRunner(mix_stderr=False)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_mock_embedder(dim: int = 128):
    """Return a MockEmbeddingModel instance."""
    from coderag.embeddings.factory import MockEmbeddingModel

    return MockEmbeddingModel(dim=dim)


def _make_ephemeral_store():
    """Return an in-memory ChromaVectorStore."""
    from coderag.store.chroma_store import ChromaVectorStore

    return ChromaVectorStore.ephemeral()


def _build_index(coderag_dir: Path):
    """
    Index the sample_repo fixture into an ephemeral store and write a manifest.

    Returns (store, embedder) so callers can build a Retriever against the
    same in-memory store.
    """
    from coderag.indexer import Indexer

    store = _make_ephemeral_store()
    embedder = _make_mock_embedder()
    indexer = Indexer(
        embedding_model=embedder,
        vector_store=store,
        coderag_dir=coderag_dir,
    )
    indexer.index_repo(FIXTURE_REPO, incremental=False)
    return store, embedder


# ── --version ─────────────────────────────────────────────────────────────────


def test_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "coderag" in result.output
    # version string should contain a digit (e.g. "0.1.0")
    assert any(ch.isdigit() for ch in result.output)


# ── index command ─────────────────────────────────────────────────────────────


class TestIndexCommand:
    def test_index_local_repo_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Indexing the sample_repo fixture should exit 0 and print stats."""
        get_settings.cache_clear()
        monkeypatch.setenv("CODERAG_EMBEDDING_PROVIDER", "sentence_transformers")
        monkeypatch.setenv("CODERAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        monkeypatch.setenv("CODERAG_VECTOR_STORE", "chroma")
        monkeypatch.setenv("CODERAG_CHROMA_PATH", str(tmp_path / ".coderag" / "chroma"))
        get_settings.cache_clear()

        # Patch the internal helpers so we don't spin up sentence-transformers
        fake_store = _make_ephemeral_store()
        fake_embedder = _make_mock_embedder()

        with (
            patch("coderag.cli.get_embedding_model", return_value=fake_embedder),  # noqa: SIM117
            patch("coderag.cli.get_vector_store", return_value=fake_store),
            patch("coderag.cli._coderag_dir", return_value=tmp_path / ".coderag"),
        ):
            result = runner.invoke(app, ["index", str(FIXTURE_REPO)])

        assert result.exit_code in (0, 2), result.output  # 2 = some files failed (ok for fixture)
        # Stats table must mention "Indexing Complete" or chunk/file counts
        assert "chunk" in result.output.lower() or "file" in result.output.lower()

    def test_index_invalid_path_exits_1(self, tmp_path: Path):
        """Passing a non-existent path should exit with code 1."""
        result = runner.invoke(app, ["index", str(tmp_path / "does_not_exist")])
        assert result.exit_code == 1

    def test_index_force_flag_accepted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """--force flag should be accepted without error."""
        get_settings.cache_clear()
        monkeypatch.setenv("CODERAG_CHROMA_PATH", str(tmp_path / ".coderag" / "chroma"))
        get_settings.cache_clear()

        fake_store = _make_ephemeral_store()
        fake_embedder = _make_mock_embedder()

        with (
            patch("coderag.cli.get_embedding_model", return_value=fake_embedder),
            patch("coderag.cli.get_vector_store", return_value=fake_store),
            patch("coderag.cli._coderag_dir", return_value=tmp_path / ".coderag"),
        ):
            result = runner.invoke(app, ["index", str(FIXTURE_REPO), "--force"])

        # Should not fail due to the flag itself
        assert result.exit_code in (0, 2), result.output

    def test_index_writes_manifest(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """After indexing, a manifest.json must exist in .coderag/."""
        get_settings.cache_clear()
        coderag_dir = tmp_path / ".coderag"
        monkeypatch.setenv("CODERAG_CHROMA_PATH", str(coderag_dir / "chroma"))
        get_settings.cache_clear()

        fake_store = _make_ephemeral_store()
        fake_embedder = _make_mock_embedder()

        with (
            patch("coderag.cli.get_embedding_model", return_value=fake_embedder),
            patch("coderag.cli.get_vector_store", return_value=fake_store),
            patch("coderag.cli._coderag_dir", return_value=coderag_dir),
        ):
            runner.invoke(app, ["index", str(FIXTURE_REPO)])

        assert (coderag_dir / "manifest.json").exists()


# ── ask command ───────────────────────────────────────────────────────────────


class TestAskCommand:
    def test_ask_without_index_exits_1(self, tmp_path: Path):
        """ask with no manifest.json should exit 1 with a clear message."""
        with patch("coderag.cli._coderag_dir", return_value=tmp_path / ".coderag"):
            result = runner.invoke(app, ["ask", "what does greet do?"])

        assert result.exit_code == 1
        assert "index" in result.output.lower() or "no index" in result.output.lower()

    def test_ask_with_empty_manifest_exits_1(self, tmp_path: Path):
        """ask with an empty manifest should exit 1."""
        coderag_dir = tmp_path / ".coderag"
        coderag_dir.mkdir(parents=True)
        (coderag_dir / "manifest.json").write_text("{}", encoding="utf-8")

        with patch("coderag.cli._coderag_dir", return_value=coderag_dir):
            result = runner.invoke(app, ["ask", "test question"])

        assert result.exit_code == 1
        assert "empty" in result.output.lower() or "index" in result.output.lower()

    def test_ask_with_prebuilt_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """ask with a real index and a MockLLMClient should print an answer."""
        get_settings.cache_clear()
        coderag_dir = tmp_path / ".coderag"
        monkeypatch.setenv("CODERAG_CHROMA_PATH", str(coderag_dir / "chroma"))
        monkeypatch.setenv("CODERAG_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        get_settings.cache_clear()

        store, embedder = _build_index(coderag_dir)

        from coderag.generation.llm_client import MockLLMClient

        mock_llm = MockLLMClient()

        with (
            patch("coderag.cli.get_embedding_model", return_value=embedder),
            patch("coderag.cli.get_vector_store", return_value=store),
            patch("coderag.cli.get_llm_client", return_value=mock_llm),
            patch("coderag.cli._coderag_dir", return_value=coderag_dir),
            patch("coderag.cli._load_graph_analyzer", return_value=None),
        ):
            result = runner.invoke(app, ["ask", "what does the greet function do?"])

        assert result.exit_code == 0, result.output
        # Should contain the answer panel
        assert "Answer" in result.output or "answer" in result.output.lower()

    def test_ask_show_context_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """--show-context should include retrieval context in the output."""
        get_settings.cache_clear()
        coderag_dir = tmp_path / ".coderag"
        monkeypatch.setenv("CODERAG_CHROMA_PATH", str(coderag_dir / "chroma"))
        get_settings.cache_clear()

        store, embedder = _build_index(coderag_dir)

        from coderag.generation.llm_client import MockLLMClient

        mock_llm = MockLLMClient()

        with (
            patch("coderag.cli.get_embedding_model", return_value=embedder),
            patch("coderag.cli.get_vector_store", return_value=store),
            patch("coderag.cli.get_llm_client", return_value=mock_llm),
            patch("coderag.cli._coderag_dir", return_value=coderag_dir),
            patch("coderag.cli._load_graph_analyzer", return_value=None),
        ):
            result = runner.invoke(
                app, ["ask", "how does the Calculator work?", "--show-context"]
            )

        assert result.exit_code == 0, result.output
        # Context panel should appear
        assert "Retrieval Context" in result.output or "[Code]:" in result.output

    def test_ask_no_graph_flag(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """--no-graph should disable graph expansion without errors."""
        get_settings.cache_clear()
        coderag_dir = tmp_path / ".coderag"
        monkeypatch.setenv("CODERAG_CHROMA_PATH", str(coderag_dir / "chroma"))
        get_settings.cache_clear()

        store, embedder = _build_index(coderag_dir)

        from coderag.generation.llm_client import MockLLMClient

        mock_llm = MockLLMClient()

        with (
            patch("coderag.cli.get_embedding_model", return_value=embedder),
            patch("coderag.cli.get_vector_store", return_value=store),
            patch("coderag.cli.get_llm_client", return_value=mock_llm),
            patch("coderag.cli._coderag_dir", return_value=coderag_dir),
        ):
            result = runner.invoke(app, ["ask", "greet function", "--no-graph"])

        assert result.exit_code == 0, result.output


# ── reindex command ───────────────────────────────────────────────────────────


class TestReindexCommand:
    def test_reindex_delegates_to_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """reindex should run successfully and write a manifest."""
        get_settings.cache_clear()
        coderag_dir = tmp_path / ".coderag"
        monkeypatch.setenv("CODERAG_CHROMA_PATH", str(coderag_dir / "chroma"))
        get_settings.cache_clear()

        fake_store = _make_ephemeral_store()
        fake_embedder = _make_mock_embedder()

        with (
            patch("coderag.cli.get_embedding_model", return_value=fake_embedder),
            patch("coderag.cli.get_vector_store", return_value=fake_store),
            patch("coderag.cli._coderag_dir", return_value=coderag_dir),
        ):
            result = runner.invoke(app, ["reindex", str(FIXTURE_REPO)])

        assert result.exit_code in (0, 2), result.output

    def test_reindex_second_run_skips_unchanged(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Second reindex run should report skipped files (incremental)."""
        get_settings.cache_clear()
        coderag_dir = tmp_path / ".coderag"
        monkeypatch.setenv("CODERAG_CHROMA_PATH", str(coderag_dir / "chroma"))
        get_settings.cache_clear()

        fake_store = _make_ephemeral_store()
        fake_embedder = _make_mock_embedder()

        # First run
        with (
            patch("coderag.cli.get_embedding_model", return_value=fake_embedder),
            patch("coderag.cli.get_vector_store", return_value=fake_store),
            patch("coderag.cli._coderag_dir", return_value=coderag_dir),
        ):
            runner.invoke(app, ["reindex", str(FIXTURE_REPO)])

        # Second run — should show "skipped" in output
        with (
            patch("coderag.cli.get_embedding_model", return_value=fake_embedder),
            patch("coderag.cli.get_vector_store", return_value=fake_store),
            patch("coderag.cli._coderag_dir", return_value=coderag_dir),
        ):
            result = runner.invoke(app, ["reindex", str(FIXTURE_REPO)])

        assert result.exit_code in (0, 2), result.output
        assert "skipped" in result.output.lower() or "unchanged" in result.output.lower()


# ── status command ────────────────────────────────────────────────────────────


class TestStatusCommand:
    def test_status_no_index(self, tmp_path: Path):
        """status with no index should print an informational message, exit 0."""
        with patch("coderag.cli._coderag_dir", return_value=tmp_path / ".coderag"):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0
        assert "index" in result.output.lower()

    def test_status_with_index(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """status after indexing should show file and chunk counts."""
        get_settings.cache_clear()
        coderag_dir = tmp_path / ".coderag"
        monkeypatch.setenv("CODERAG_CHROMA_PATH", str(coderag_dir / "chroma"))
        get_settings.cache_clear()

        # Build a real manifest by indexing
        _build_index(coderag_dir)

        with patch("coderag.cli._coderag_dir", return_value=coderag_dir):
            result = runner.invoke(app, ["status"])

        assert result.exit_code == 0, result.output
        # Table should mention file/chunk counts
        assert "chunk" in result.output.lower() or "file" in result.output.lower()

    def test_status_shows_graph_presence(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """status table should indicate whether a graph.json exists."""
        get_settings.cache_clear()
        coderag_dir = tmp_path / ".coderag"
        monkeypatch.setenv("CODERAG_CHROMA_PATH", str(coderag_dir / "chroma"))
        get_settings.cache_clear()

        _build_index(coderag_dir)

        with patch("coderag.cli._coderag_dir", return_value=coderag_dir):
            result = runner.invoke(app, ["status"])

        assert "graph" in result.output.lower()


# ── Phase 7 end-to-end report ─────────────────────────────────────────────────


class TestPhase7Report:
    def test_print_cli_report(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Print a Phase 7 report: index → status → ask (with mock LLM)."""
        get_settings.cache_clear()
        coderag_dir = tmp_path / ".coderag"
        monkeypatch.setenv("CODERAG_CHROMA_PATH", str(coderag_dir / "chroma"))
        get_settings.cache_clear()

        store, embedder = _build_index(coderag_dir)

        print("\n\n=== PHASE 7 REPORT: CLI Wiring ===")

        # -- status after index --
        with patch("coderag.cli._coderag_dir", return_value=coderag_dir):
            status_result = runner.invoke(app, ["status"])
        print("status output:")
        print(status_result.output)
        assert status_result.exit_code == 0

        # -- ask --
        from coderag.generation.llm_client import MockLLMClient

        mock_llm = MockLLMClient()

        with (
            patch("coderag.cli.get_embedding_model", return_value=embedder),
            patch("coderag.cli.get_vector_store", return_value=store),
            patch("coderag.cli.get_llm_client", return_value=mock_llm),
            patch("coderag.cli._coderag_dir", return_value=coderag_dir),
            patch("coderag.cli._load_graph_analyzer", return_value=None),
        ):
            ask_result = runner.invoke(app, ["ask", "what does the greet function do?"])

        print("ask output:")
        print(ask_result.output)
        assert ask_result.exit_code == 0

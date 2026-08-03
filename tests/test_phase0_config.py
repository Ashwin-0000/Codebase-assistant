"""
Phase 0 tests — configuration loading and logging setup.

These tests verify:
  1. Settings loads successfully with defaults.
  2. Environment-variable overrides are respected.
  3. Invalid values raise a clear validation error.
  4. configure_logging() doesn't crash and applies the correct level.
  5. The get_settings() lru_cache resets correctly between tests (via fixture).
"""

from __future__ import annotations

import logging

import pytest

from coderag.config import Settings, get_settings
from coderag.logging_config import configure_logging


# ------------------------------------------------------------------ #
# Settings — default values
# ------------------------------------------------------------------ #


class TestSettingsDefaults:
    """Verify that default values are what we documented in .env.example."""

    def test_embedding_defaults(self, isolated_settings: Settings) -> None:
        assert isolated_settings.embedding_provider == "sentence_transformers"
        assert isolated_settings.embedding_model == "all-MiniLM-L6-v2"

    def test_llm_defaults(self, isolated_settings: Settings) -> None:
        assert isolated_settings.llm_provider == "openai"
        assert isolated_settings.llm_model == "gpt-4o-mini"

    def test_vector_store_default(self, isolated_settings: Settings) -> None:
        assert isolated_settings.vector_store == "chroma"

    def test_chunk_defaults(self, isolated_settings: Settings) -> None:
        assert isolated_settings.chunk_max_tokens == 512
        assert isolated_settings.chunk_overlap_tokens == 64

    def test_retrieval_defaults(self, isolated_settings: Settings) -> None:
        assert isolated_settings.top_k == 8
        assert isolated_settings.graph_expansion_depth == 1
        assert isolated_settings.context_token_budget == 8000
        assert isolated_settings.reranking_enabled is False

    def test_languages_default(self, isolated_settings: Settings) -> None:
        assert "python" in isolated_settings.languages
        assert "javascript" in isolated_settings.languages
        assert "typescript" in isolated_settings.languages


# ------------------------------------------------------------------ #
# Settings — environment variable overrides
# ------------------------------------------------------------------ #


class TestSettingsOverrides:
    """Verify that env-var overrides are picked up correctly."""

    def test_embedding_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CODERAG_EMBEDDING_MODEL", "text-embedding-3-small")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.embedding_model == "text-embedding-3-small"
        get_settings.cache_clear()

    def test_top_k_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CODERAG_TOP_K", "16")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.top_k == 16
        get_settings.cache_clear()

    def test_languages_comma_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The languages field must accept a comma-separated string from env."""
        monkeypatch.setenv("CODERAG_LANGUAGES", "python , rust , go")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.languages == ["python", "rust", "go"]
        get_settings.cache_clear()

    def test_skip_docgen_flag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CODERAG_SKIP_DOCGEN", "true")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.skip_docgen is True
        get_settings.cache_clear()


# ------------------------------------------------------------------ #
# Settings — validation errors
# ------------------------------------------------------------------ #


class TestSettingsValidation:
    """Verify that bad values are caught at startup, not silently accepted."""

    def test_invalid_log_level(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CODERAG_LOG_LEVEL", "VERBOSE")
        get_settings.cache_clear()
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="log_level"):
            get_settings()
        get_settings.cache_clear()

    def test_invalid_embedding_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CODERAG_EMBEDDING_PROVIDER", "mythical_provider")
        get_settings.cache_clear()
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            get_settings()
        get_settings.cache_clear()

    def test_chunk_max_tokens_must_be_positive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CODERAG_CHUNK_MAX_TOKENS", "0")
        get_settings.cache_clear()
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            get_settings()
        get_settings.cache_clear()


# ------------------------------------------------------------------ #
# Settings — singleton caching
# ------------------------------------------------------------------ #


class TestSettingsCaching:
    """Verify that get_settings() returns the same object on repeated calls."""

    def test_singleton(self, isolated_settings: Settings) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2, "get_settings() must return the cached singleton"


# ------------------------------------------------------------------ #
# Logging
# ------------------------------------------------------------------ #


class TestLogging:
    """Basic smoke tests for configure_logging()."""

    def test_configure_logging_info(self) -> None:
        """configure_logging should set the root logger to INFO without crashing."""
        configure_logging(level="INFO")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_configure_logging_debug(self) -> None:
        configure_logging(level="DEBUG")
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_configure_logging_creates_file(self, tmp_path: pytest.TempPathFactory) -> None:
        """configure_logging should create the log file if a path is given."""
        log_file = str(tmp_path / "subdir" / "coderag.log")
        configure_logging(level="INFO", log_file=log_file)
        # Emit a test message so the file is flushed
        logging.getLogger("coderag.test").info("test message")
        import pathlib

        assert pathlib.Path(log_file).exists()

    def test_noisy_loggers_quietened(self) -> None:
        """Third-party loggers should be set to WARNING after configure_logging."""
        configure_logging(level="DEBUG")
        for name in ("httpx", "chromadb"):
            assert logging.getLogger(name).level == logging.WARNING

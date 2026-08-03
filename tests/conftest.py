"""
Shared pytest fixtures for the CodeRAG test suite.

Fixtures here are available to all test modules without explicit import
(pytest discovers conftest.py automatically).
"""

from __future__ import annotations

import pytest

from coderag.config import Settings, get_settings


@pytest.fixture(autouse=False)
def isolated_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Return a fresh Settings instance with a minimal test configuration.

    Clears the lru_cache between tests so settings don't bleed across.
    Sets only the variables needed so tests don't depend on a real .env file.
    """
    # Prevent loading any real .env file by pointing to a non-existent path
    monkeypatch.setenv("CODERAG_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("CODERAG_EMBEDDING_PROVIDER", "sentence_transformers")
    monkeypatch.setenv("CODERAG_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Clear the cached singleton so get_settings() re-reads env vars
    get_settings.cache_clear()
    settings = get_settings()
    yield settings
    # Clean up: clear cache again so subsequent tests start fresh
    get_settings.cache_clear()

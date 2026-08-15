"""
Configuration management for CodeRAG.

All settings are loaded from environment variables (with a .env file fallback).
Pydantic-Settings is used so every field is type-validated at startup and
a clear error is raised for any misconfiguration.

Design choice: Pydantic-Settings over a hand-rolled config loader because it
gives us free type coercion, validation, and IDE autocomplete on settings fields.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal, Union

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object.  Populated from environment / .env file."""

    model_config = SettingsConfigDict(
        env_prefix="CODERAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Don't crash on extra env vars that don't match our fields
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Embedding
    # ------------------------------------------------------------------ #
    embedding_provider: Literal["sentence_transformers", "openai", "voyage", "cohere"] = (
        "sentence_transformers"
    )
    embedding_model: str = "all-MiniLM-L6-v2"

    # ------------------------------------------------------------------ #
    # LLM / Generation
    # ------------------------------------------------------------------ #
    llm_provider: Literal["openai", "anthropic", "ollama", "local", "mock"] = "openai"
    llm_model: str = "gpt-4o-mini"
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    # ------------------------------------------------------------------ #
    # API keys  (no CODERAG_ prefix — standard env var names)
    # ------------------------------------------------------------------ #
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    voyage_api_key: str = Field(default="", alias="VOYAGE_API_KEY")
    cohere_api_key: str = Field(default="", alias="COHERE_API_KEY")

    # ------------------------------------------------------------------ #
    # Vector store
    # ------------------------------------------------------------------ #
    vector_store: Literal["chroma", "qdrant", "pgvector"] = "chroma"
    chroma_path: Path = Path(".coderag/chroma")

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #
    chunk_max_tokens: int = Field(default=512, gt=0)
    chunk_overlap_tokens: int = Field(default=64, ge=0)
    languages: Union[list[str], str] = ["python", "javascript", "typescript"]
    skip_docgen: bool = False

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    top_k: int = Field(default=8, gt=0)
    graph_expansion_depth: int = Field(default=1, ge=0)
    context_token_budget: int = Field(default=8000, gt=0)
    reranking_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    log_level: str = "INFO"
    log_file: str = ""

    @field_validator("languages", mode="before")
    @classmethod
    def parse_languages(cls, v: object) -> list[str]:
        """Accept a comma-separated string or a list."""
        if isinstance(v, str):
            return [lang.strip().lower() for lang in v.split(",") if lang.strip()]
        return v  # type: ignore[return-value]

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return (and cache) the global Settings instance.

    Using lru_cache means the .env file is parsed exactly once per process.
    Tests can call ``get_settings.cache_clear()`` to reset between runs.
    """
    return Settings()

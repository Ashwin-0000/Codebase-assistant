"""
web/models.py — Pydantic request/response models for the CodeRAG Web API.

All models use strict typing and field validation so that any mis-shaped
request is caught and returned as a clean 422 before it reaches the
pipeline code.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    """Body for ``POST /api/index``."""

    repo: str = Field(
        ...,
        description="Local path (absolute or relative) or a GitHub URL to index.",
        examples=["/home/user/myproject", "https://github.com/org/repo"],
    )
    force: bool = Field(
        False,
        description="Re-index every file even if its SHA-256 hash is unchanged.",
    )


class AskRequest(BaseModel):
    """Body for ``POST /api/ask``."""

    question: str = Field(..., min_length=1, max_length=2000, description="Natural-language question.")
    top_k: int = Field(8, ge=1, le=50, description="Number of chunks to retrieve.")
    provider: Optional[str] = Field(None, description="LLM provider override (openai/anthropic/ollama).")
    model: Optional[str] = Field(None, description="LLM model override.")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class AskResponse(BaseModel):
    """Response from ``POST /api/ask``."""

    answer: str
    citations: list[str]
    context_chunks: int
    provider: str
    model: str
    total_tokens: int
    latency_seconds: float


class StatusResponse(BaseModel):
    """Response from ``GET /api/status``."""

    indexed: bool
    files: int
    chunks: int
    last_indexed: Optional[str]
    graph_present: bool
    chroma_path: str
    embedding_provider: str
    llm_provider: str


class IndexJobResponse(BaseModel):
    """Immediate response from ``POST /api/index``."""

    job_id: str
    status: str
    message: str


class IndexProgressResponse(BaseModel):
    """Response from ``GET /api/index/progress/{job_id}``."""

    job_id: str
    status: str  # "started" | "running" | "done" | "error"
    files_processed: int = 0
    chunks_indexed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

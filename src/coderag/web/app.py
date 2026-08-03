"""
web/app.py — FastAPI application for the CodeRAG Web UI.

Endpoints
---------
GET  /                              Serve the single-page application
GET  /api/status                    Index statistics
POST /api/index                     Trigger background indexing
GET  /api/index/progress/{job_id}   Poll indexing job progress
POST /api/ask                       Retrieve + generate an answer
GET  /api/docs                      Interactive Swagger UI
GET  /api/redoc                     ReDoc documentation

Design
------
- The embedding model and vector store are loaded *lazily* on first use,
  protected by a threading.Lock so concurrent requests don't double-initialise.
- Indexing jobs run in a background thread (FastAPI BackgroundTasks) with a
  shared in-memory job-status dict.  Poll ``/api/index/progress/{job_id}``
  at ~1 s intervals from the frontend.
- The blocking pipeline operations (_sync_ask, _run_indexing) are either run
  directly as BackgroundTasks or via asyncio.run_in_executor so the uvicorn
  event loop stays unblocked.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from coderag.web.models import (
    AskRequest,
    AskResponse,
    IndexJobResponse,
    IndexProgressResponse,
    IndexRequest,
    StatusResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_pipeline: dict[str, Any] = {}
_pipeline_lock = threading.Lock()
_index_jobs: dict[str, dict] = {}

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    from coderag.config import get_settings
    from coderag.logging_config import configure_logging

    cfg = get_settings()
    configure_logging(level=cfg.log_level)
    _pipeline["settings"] = cfg
    logger.info(
        "CodeRAG Web UI starting  embedding=%s  llm=%s",
        cfg.embedding_provider,
        cfg.llm_provider,
    )
    yield
    _pipeline.clear()
    logger.info("CodeRAG Web UI stopped")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CodeRAG",
    description="Codebase RAG assistant — index a repo and ask natural-language questions about it.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (built into the installed package)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_coderag_dir() -> Path:
    return Path(".coderag")


def _get_manifest() -> dict:
    path = _get_coderag_dir() / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return {}


def _get_or_init_pipeline() -> dict:
    """Lazily initialise (and cache) the embedding model and vector store."""
    with _pipeline_lock:
        if "embedder" in _pipeline:
            return _pipeline

        from coderag.embeddings.factory import get_embedding_model
        from coderag.store.factory import get_vector_store

        cfg = _pipeline["settings"]
        logger.info(
            "Loading embedding model %s/%s …",
            cfg.embedding_provider,
            cfg.embedding_model,
        )
        _pipeline["embedder"] = get_embedding_model(
            provider=cfg.embedding_provider,
            model_name=cfg.embedding_model,
            api_key=cfg.openai_api_key or None,
        )
        _pipeline["store"] = get_vector_store(
            backend=cfg.vector_store,
            path=str(cfg.chroma_path),
        )
        logger.info("Pipeline initialised")
        return _pipeline


# ---------------------------------------------------------------------------
# Routes — SPA entry point
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def serve_spa():
    """Serve the single-page application."""
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html)
    return {
        "message": "CodeRAG API is running. "
        "Install the package with static files or access /api/docs for the REST API."
    }


# ---------------------------------------------------------------------------
# Routes — /api/status
# ---------------------------------------------------------------------------


@app.get("/api/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Return current index statistics."""
    coderag_dir = _get_coderag_dir()
    manifest = _get_manifest()

    total_chunks = sum(
        e.get("chunk_count", len(e.get("chunk_ids", [])))
        for e in manifest.values()
    )
    timestamps = [
        e.get("indexed_at", "") for e in manifest.values() if e.get("indexed_at")
    ]
    last_indexed = max(timestamps)[:19].replace("T", " ") if timestamps else None

    cfg = _pipeline.get("settings")
    embedding_provider = cfg.embedding_provider if cfg else "unknown"
    llm_provider = cfg.llm_provider if cfg else "unknown"

    return StatusResponse(
        indexed=bool(manifest),
        files=len(manifest),
        chunks=total_chunks,
        last_indexed=last_indexed,
        graph_present=(coderag_dir / "graph.json").exists(),
        chroma_path=str(coderag_dir / "chroma"),
        embedding_provider=embedding_provider,
        llm_provider=llm_provider,
    )


# ---------------------------------------------------------------------------
# Routes — /api/index
# ---------------------------------------------------------------------------


@app.post("/api/index", response_model=IndexJobResponse, status_code=202)
async def start_indexing(
    request: IndexRequest, background_tasks: BackgroundTasks
) -> IndexJobResponse:
    """Start indexing a repository in a background thread.

    Returns a ``job_id`` immediately; poll
    ``GET /api/index/progress/{job_id}`` to track progress.
    """
    job_id = uuid.uuid4().hex[:8]
    _index_jobs[job_id] = {
        "status": "started",
        "files_processed": 0,
        "chunks_indexed": 0,
        "files_skipped": 0,
        "files_failed": 0,
        "elapsed_seconds": 0.0,
        "error": None,
    }
    background_tasks.add_task(_run_indexing, job_id, request.repo, request.force)
    return IndexJobResponse(
        job_id=job_id,
        status="started",
        message="Indexing started in the background.",
    )


@app.get("/api/index/progress/{job_id}", response_model=IndexProgressResponse)
async def get_index_progress(job_id: str) -> IndexProgressResponse:
    """Poll the status of an indexing job."""
    if job_id not in _index_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return IndexProgressResponse(job_id=job_id, **_index_jobs[job_id])


# ---------------------------------------------------------------------------
# Routes — /api/ask
# ---------------------------------------------------------------------------


@app.post("/api/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """Retrieve relevant code chunks and generate a grounded answer.

    Returns 400 if the index is empty.
    """
    manifest = _get_manifest()
    if not manifest:
        raise HTTPException(
            status_code=400,
            detail="No index found. Index a repository first via POST /api/index.",
        )

    loop = asyncio.get_event_loop()
    try:
        result: AskResponse = await loop.run_in_executor(None, _sync_ask, request)
    except Exception as exc:
        logger.exception("Ask failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


# ---------------------------------------------------------------------------
# Background / thread-pool workers
# ---------------------------------------------------------------------------


def _run_indexing(job_id: str, repo: str, force: bool) -> None:
    """Synchronous indexing — runs in a FastAPI BackgroundTask thread."""
    from coderag.indexer import Indexer
    from coderag.ingestion.repo import resolve_repo

    t0 = time.perf_counter()
    job = _index_jobs[job_id]
    job["status"] = "running"

    try:
        repo_path = resolve_repo(repo)
        pl = _get_or_init_pipeline()
        indexer = Indexer(
            embedding_model=pl["embedder"],
            vector_store=pl["store"],
            coderag_dir=_get_coderag_dir(),
        )
        stats = indexer.index_repo(repo_path, incremental=not force)

        job.update(
            status="done",
            files_processed=stats.files_processed,
            chunks_indexed=stats.chunks_indexed,
            files_skipped=stats.files_skipped,
            files_failed=stats.files_failed,
            elapsed_seconds=round(time.perf_counter() - t0, 2),
        )
        logger.info(
            "Index job %s done: %d files, %d chunks",
            job_id,
            stats.files_processed,
            stats.chunks_indexed,
        )

    except Exception as exc:
        logger.exception("Index job %s failed", job_id)
        job.update(
            status="error",
            error=str(exc),
            elapsed_seconds=round(time.perf_counter() - t0, 2),
        )


def _sync_ask(request: AskRequest) -> AskResponse:
    """Synchronous ask pipeline — runs in asyncio's default thread pool."""
    from coderag.generation.generator import AnswerGenerator
    from coderag.generation.llm_client import MockLLMClient, get_llm_client
    from coderag.retrieval.retriever import Retriever

    t0 = time.perf_counter()
    cfg = _pipeline["settings"]
    pl = _get_or_init_pipeline()

    # Resolve LLM provider / model
    llm_provider = request.provider or cfg.llm_provider
    llm_model = request.model or cfg.llm_model

    api_key = ""
    if llm_provider == "openai":
        api_key = cfg.openai_api_key
    elif llm_provider == "anthropic":
        api_key = cfg.anthropic_api_key

    # Graceful fallback to mock when no API key is configured
    if not api_key and llm_provider in ("openai", "anthropic"):
        logger.warning(
            "No API key for %s — falling back to MockLLMClient", llm_provider
        )
        llm_client: Any = MockLLMClient()
        llm_provider = "mock"
        llm_model = "mock-model"
    else:
        llm_client = get_llm_client(
            provider=llm_provider,
            model=llm_model,
            api_key=api_key,
            ollama_base_url=cfg.ollama_base_url,
        )

    retriever = Retriever(
        embedding_model=pl["embedder"],
        vector_store=pl["store"],
        top_k=request.top_k,
        graph_expansion_depth=0,  # keep web API snappy; graph is optional
        context_token_budget=cfg.context_token_budget,
    )

    context = retriever.retrieve(request.question)
    generator = AnswerGenerator(llm_client=llm_client)
    answer = generator.generate(context)

    return AskResponse(
        answer=answer.answer,
        citations=answer.citations,
        context_chunks=answer.context_chunks,
        provider=answer.provider,
        model=answer.model,
        total_tokens=answer.usage.total_tokens,
        latency_seconds=round(time.perf_counter() - t0, 3),
    )

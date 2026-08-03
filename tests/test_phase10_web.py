"""
Phase 10 tests — Web UI (FastAPI / TestClient).

All external I/O is mocked so these tests run without:
  - A real embedding model
  - A real vector store
  - A real .coderag directory on disk
  - Any LLM API keys

Coverage:
  1. GET  /                              → 200 or valid JSON (static files may not exist)
  2. GET  /api/status (no index)         → 200, indexed=False
  3. GET  /api/status (with index)       → 200, correct file/chunk counts
  4. POST /api/index                     → 202, job_id returned
  5. GET  /api/index/progress/<job_id>   → 200, progress dict
  6. GET  /api/index/progress/<missing>  → 404
  7. POST /api/ask (no index)            → 400
  8. POST /api/ask (with index+pipeline) → 200, AskResponse schema
  9. POST /api/ask (invalid payload)     → 422
 10. GET  /api/docs                      → 200 (Swagger UI)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Helpers ───────────────────────────────────────────────────────────────


def _make_manifest(files: int = 3, chunks_per_file: int = 5) -> dict:
    """Return a fake manifest dict with ``files`` entries."""
    return {
        f"src/module{i}.py": {
            "chunk_count": chunks_per_file,
            "indexed_at": "2024-06-01T12:00:00Z",
            "chunk_ids": [f"cid-{i}-{j}" for j in range(chunks_per_file)],
        }
        for i in range(files)
    }


def _make_ask_response() -> dict:
    """Return a fake AskResponse payload (matching the model)."""
    return {
        "answer": "The `greet` function prints a personalised greeting.",
        "citations": ["src/module0.py:L1-L10", "src/module1.py:L5-L20"],
        "context_chunks": 2,
        "provider": "mock",
        "model": "mock-model",
        "total_tokens": 120,
        "latency_seconds": 0.042,
    }


# ── Fixture: TestClient with lifespan mocked ──────────────────────────────


@pytest.fixture
def client(monkeypatch, tmp_path):
    """
    Build a TestClient whose lifespan mock-initialises _pipeline["settings"]
    without loading a real embedding model or touching disk.
    """
    from coderag.config import Settings

    # Provide a dummy Settings object (no .env needed)
    dummy_settings = Settings(
        _env_file=None,         # type: ignore[call-arg]
        embedding_provider="sentence_transformers",
        embedding_model="all-MiniLM-L6-v2",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        chroma_path=tmp_path / ".coderag" / "chroma",
    )

    from coderag.web import app as web_app

    with patch.object(web_app, "_pipeline", {"settings": dummy_settings}):
        with patch.object(web_app, "_get_coderag_dir", return_value=tmp_path):
            with TestClient(web_app.app, raise_server_exceptions=True) as c:
                yield c


# ══════════════════════════════════════════════════════════════════════════
#  Tests — static / docs
# ══════════════════════════════════════════════════════════════════════════


class TestServedFiles:
    def test_root_returns_200(self, client):
        """GET / should return 200 regardless of whether static files exist."""
        res = client.get("/")
        assert res.status_code == 200

    def test_swagger_ui_available(self, client):
        """Swagger UI must always be served (no extras needed)."""
        res = client.get("/api/docs")
        assert res.status_code == 200
        assert "swagger" in res.text.lower()


# ══════════════════════════════════════════════════════════════════════════
#  Tests — /api/status
# ══════════════════════════════════════════════════════════════════════════


class TestStatusEndpoint:
    def test_status_no_index(self, client, tmp_path):
        """Without a manifest.json, indexed should be False."""
        from coderag.web import app as web_app

        with patch.object(web_app, "_get_manifest", return_value={}):
            res = client.get("/api/status")

        assert res.status_code == 200
        data = res.json()
        assert data["indexed"] is False
        assert data["files"] == 0
        assert data["chunks"] == 0
        assert data["last_indexed"] is None

    def test_status_with_index(self, client, tmp_path):
        """With a manifest, indexed=True and counts should aggregate correctly."""
        from coderag.web import app as web_app

        manifest = _make_manifest(files=4, chunks_per_file=6)
        with patch.object(web_app, "_get_manifest", return_value=manifest):
            res = client.get("/api/status")

        assert res.status_code == 200
        data = res.json()
        assert data["indexed"] is True
        assert data["files"] == 4
        assert data["chunks"] == 24   # 4 × 6
        assert data["last_indexed"] is not None
        assert data["embedding_provider"] == "sentence_transformers"
        assert data["llm_provider"] == "openai"

    def test_status_graph_present(self, client, tmp_path):
        """graph_present should be True when graph.json exists."""
        from coderag.web import app as web_app

        graph_file = tmp_path / "graph.json"
        graph_file.write_text("{}")

        with patch.object(web_app, "_get_manifest", return_value=_make_manifest(1)):
            with patch.object(web_app, "_get_coderag_dir", return_value=tmp_path):
                res = client.get("/api/status")

        assert res.json()["graph_present"] is True

    def test_status_schema_fields_present(self, client, tmp_path):
        """Response must include all StatusResponse fields."""
        from coderag.web import app as web_app

        expected_keys = {
            "indexed", "files", "chunks", "last_indexed",
            "graph_present", "chroma_path",
            "embedding_provider", "llm_provider",
        }
        with patch.object(web_app, "_get_manifest", return_value={}):
            res = client.get("/api/status")

        assert res.status_code == 200
        assert expected_keys.issubset(res.json().keys())


# ══════════════════════════════════════════════════════════════════════════
#  Tests — /api/index
# ══════════════════════════════════════════════════════════════════════════


class TestIndexEndpoint:
    def test_start_indexing_returns_job_id(self, client):
        """POST /api/index should return 202 with a job_id immediately."""
        from coderag.web import app as web_app

        # Background task will fail (no real repo), but the *response* is immediate
        with patch.object(web_app, "_run_indexing"):
            res = client.post("/api/index", json={"repo": "/fake/path"})

        assert res.status_code == 202
        data = res.json()
        assert "job_id" in data
        assert len(data["job_id"]) == 8
        assert data["status"] == "started"

    def test_force_flag_accepted(self, client):
        """The force flag must be accepted without error."""
        from coderag.web import app as web_app

        with patch.object(web_app, "_run_indexing"):
            res = client.post("/api/index", json={"repo": "/path", "force": True})

        assert res.status_code == 202

    def test_missing_repo_field_returns_422(self, client):
        """Request without required 'repo' field → 422 Unprocessable Entity."""
        res = client.post("/api/index", json={})
        assert res.status_code == 422


# ══════════════════════════════════════════════════════════════════════════
#  Tests — /api/index/progress
# ══════════════════════════════════════════════════════════════════════════


class TestIndexProgressEndpoint:
    def test_progress_not_found(self, client):
        """Unknown job_id → 404."""
        res = client.get("/api/index/progress/nonexistent")
        assert res.status_code == 404

    def test_progress_returns_job_state(self, client):
        """After starting a job, progress should be retrievable."""
        from coderag.web import app as web_app

        with patch.object(web_app, "_run_indexing"):
            start_res = client.post("/api/index", json={"repo": "/fake/path"})

        job_id = start_res.json()["job_id"]

        prog_res = client.get(f"/api/index/progress/{job_id}")
        assert prog_res.status_code == 200

        data = prog_res.json()
        assert data["job_id"] == job_id
        assert data["status"] in {"started", "running", "done", "error"}

    def test_progress_schema_fields(self, client):
        """Progress response must include all IndexProgressResponse fields."""
        from coderag.web import app as web_app

        expected_keys = {
            "job_id", "status", "files_processed", "chunks_indexed",
            "files_skipped", "files_failed", "elapsed_seconds", "error",
        }
        with patch.object(web_app, "_run_indexing"):
            start_res = client.post("/api/index", json={"repo": "/x"})
        job_id = start_res.json()["job_id"]

        prog_res = client.get(f"/api/index/progress/{job_id}")
        assert expected_keys.issubset(prog_res.json().keys())


# ══════════════════════════════════════════════════════════════════════════
#  Tests — /api/ask
# ══════════════════════════════════════════════════════════════════════════


class TestAskEndpoint:
    def test_ask_without_index_returns_400(self, client):
        """No index → 400 with helpful detail."""
        from coderag.web import app as web_app

        with patch.object(web_app, "_get_manifest", return_value={}):
            res = client.post("/api/ask", json={"question": "what does greet do?"})

        assert res.status_code == 400
        assert "index" in res.json()["detail"].lower()

    def test_ask_missing_question_returns_422(self, client):
        """Missing required 'question' field → 422."""
        res = client.post("/api/ask", json={"top_k": 5})
        assert res.status_code == 422

    def test_ask_empty_question_returns_422(self, client):
        """Empty string for 'question' → 422 (min_length=1)."""
        res = client.post("/api/ask", json={"question": ""})
        assert res.status_code == 422

    def test_ask_invalid_top_k_returns_422(self, client):
        """top_k outside [1, 50] → 422."""
        res = client.post("/api/ask", json={"question": "hi", "top_k": 0})
        assert res.status_code == 422

    def test_ask_returns_answer_response(self, client):
        """With a non-empty manifest and mocked pipeline, ask should return 200."""
        from coderag.web import app as web_app
        from coderag.web.models import AskResponse

        fake_answer = _make_ask_response()

        with patch.object(web_app, "_get_manifest", return_value=_make_manifest(2)):
            with patch.object(web_app, "_sync_ask", return_value=AskResponse(**fake_answer)):
                res = client.post("/api/ask", json={"question": "how does auth work?"})

        assert res.status_code == 200
        data = res.json()
        assert data["answer"] == fake_answer["answer"]
        assert data["citations"] == fake_answer["citations"]
        assert data["provider"] == "mock"
        assert data["total_tokens"] == 120

    def test_ask_response_schema(self, client):
        """Response must contain all AskResponse fields."""
        from coderag.web import app as web_app
        from coderag.web.models import AskResponse

        expected_keys = {
            "answer", "citations", "context_chunks",
            "provider", "model", "total_tokens", "latency_seconds",
        }

        with patch.object(web_app, "_get_manifest", return_value=_make_manifest(1)):
            with patch.object(web_app, "_sync_ask", return_value=AskResponse(**_make_ask_response())):
                res = client.post("/api/ask", json={"question": "explain parser"})

        assert expected_keys.issubset(res.json().keys())

    def test_ask_pipeline_error_returns_500(self, client):
        """If _sync_ask raises, the endpoint should return 500."""
        from coderag.web import app as web_app

        with patch.object(web_app, "_get_manifest", return_value=_make_manifest(1)):
            with patch.object(web_app, "_sync_ask", side_effect=RuntimeError("model exploded")):
                res = client.post("/api/ask", json={"question": "hello"})

        assert res.status_code == 500
        assert "model exploded" in res.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════
#  Tests — API request validation
# ══════════════════════════════════════════════════════════════════════════


class TestRequestValidation:
    @pytest.mark.parametrize("payload,expected_status", [
        ({"question": "hi"},                     200),   # minimal valid
        ({"question": "hi", "top_k": 1},         200),   # min top_k
        ({"question": "hi", "top_k": 50},        200),   # max top_k
        ({"question": "hi", "provider": "mock"}, 200),   # provider override
    ])
    def test_ask_valid_payloads(self, client, payload, expected_status):
        """Valid ask payloads should not be rejected at validation."""
        from coderag.web import app as web_app
        from coderag.web.models import AskResponse

        with patch.object(web_app, "_get_manifest", return_value=_make_manifest(1)):
            with patch.object(web_app, "_sync_ask", return_value=AskResponse(**_make_ask_response())):
                res = client.post("/api/ask", json=payload)

        assert res.status_code == expected_status

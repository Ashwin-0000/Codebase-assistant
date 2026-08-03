# Changelog

All notable changes to CodeRAG will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.0] — 2024-08

### Added

#### Phase 10 — Web UI
- `src/coderag/web/__init__.py` — package marker
- `src/coderag/web/models.py` — Pydantic request/response models (`IndexRequest`, `AskRequest`, `AskResponse`, `StatusResponse`, `IndexJobResponse`, `IndexProgressResponse`)
- `src/coderag/web/app.py` — FastAPI application:
  - `GET /` → SPA shell
  - `GET /api/status` → index statistics
  - `POST /api/index` → async background indexing with job tracking
  - `GET /api/index/progress/{job_id}` → polling endpoint
  - `POST /api/ask` → retrieve + generate (thread-pool, fallback to mock LLM)
  - `GET /api/docs` / `/api/redoc` → interactive API documentation
- `src/coderag/web/static/index.html` — semantic, accessible HTML shell (ARIA labels, single `<h1>`, `role="log"` for messages)
- `src/coderag/web/static/style.css` — premium dark theme: glassmorphism cards, CSS variable design tokens, animated skeleton loaders, typing indicator, citation chips, responsive mobile sidebar
- `src/coderag/web/static/app.js` — vanilla ES2022 SPA: status polling, indexing progress polling, ask flow with typing indicator, safe Markdown→HTML formatter (tokenises code blocks before HTML-escaping)
- `coderag serve` CLI command — starts uvicorn, auto-opens browser, friendly Rich panel output
- `tests/test_phase10_web.py` — 20+ parametrized tests covering all endpoints, schema validation, error paths (400/422/404/500)
- `pyproject.toml`: `web` optional extras (`fastapi`, `uvicorn[standard]`, `python-multipart`); `httpx` in dev deps; `package-data` for static files
- `docker-compose.yml`: `coderag-web` service with port mapping and health check

#### Phase 0 — Project scaffolding
- `pyproject.toml` with all dependencies (tree-sitter, sentence-transformers, ChromaDB, Typer, Rich)
- Pydantic-Settings `config.py` — typed, validated configuration from env vars / `.env`
- `logging_config.py` — Rich console handler + optional file handler

#### Phase 1 — Repo ingestion & parsing
- `ingestion/repo.py` — `resolve_repo()` clones GitHub URLs or validates local paths
- `ingestion/walker.py` — `.gitignore`-aware recursive file walker (`pathspec`)
- `ingestion/parser.py` — tree-sitter AST parser for Python, JavaScript, TypeScript; produces `ParsedFile` / `ASTNode` objects

#### Phase 2 — Semantic chunking
- `chunking/extractor.py` — converts `ParsedFile` → list of `Chunk` objects (function / class granularity)
- `chunking/splitter.py` — token-aware `TokenSplitter`; long functions split with configurable overlap
- `chunking/models.py` — `Chunk` dataclass with deterministic `chunk_id`

#### Phase 3 — Call graph construction
- `graph/builder.py` — builds a `networkx.DiGraph` of caller → callee edges from AST nodes
- `graph/analyzer.py` — `CallGraphAnalyzer.expand()` traverses N hops from seed chunks
- `graph/models.py` — `GraphEdge` dataclass

#### Phase 4 — Enrichment & embedding
- `enricher.py` — `ChunkEnricher` prepends imports + docstring to each chunk's text
- `embeddings/base.py` + `factory.py` — `EmbeddingModel` ABC; `MockEmbeddingModel` for tests; factory supporting `sentence_transformers` and `openai`
- `store/base.py` + `chroma_store.py` + `factory.py` — `VectorStore` ABC; ChromaDB implementation with `upsert`, `query`, `delete`, `count`
- `indexer.py` — `Indexer` orchestrates Phases 1–4; SHA-256 file hashing for incremental re-indexing; writes `manifest.json`

#### Phase 5 — Retrieval
- `retrieval/retriever.py` — `Retriever` performs vector similarity search, optional call-graph expansion, and token-budget trimming
- `retrieval/reranker.py` — optional cross-encoder re-ranker (`get_reranker` factory)
- `retrieval/models.py` — `RetrievalResult` / `RetrievalContext` dataclasses

#### Phase 6 — Answer generation
- `generation/prompt.py` — `PromptBuilder` assembles system + user messages; `SYSTEM_PROMPT` with citation rules
- `generation/llm_client.py` — `LLMClient` ABC; `OpenAIClient`, `AnthropicClient`, `MockLLMClient`; `get_llm_client` factory
- `generation/generator.py` — `AnswerGenerator` orchestrates prompt → LLM → `GeneratedAnswer`
- `generation/models.py` — `GeneratedAnswer` / `TokenUsage` dataclasses

#### Phase 7 — CLI
- `cli.py` — Typer CLI with `index`, `ask`, `reindex`, `status` commands; Rich progress spinners and tables

#### Phase 8 — Evaluation harness
- `tests/eval/run_eval.py` — end-to-end evaluation: index → retrieve → generate against golden QA pairs
- `tests/eval/eval_dataset.json` — 10 golden QA pairs for the sample repo fixture
- Metrics: retrieval recall and keyword recall; markdown report output

#### Phase 9 — Packaging & deployment
- Multi-stage `Dockerfile` with non-root user and persistent volume
- `docker-compose.yml` with `coderag` and `coderag-eval` services
- `.github/workflows/ci.yml` — lint + type-check + tests (3 OS × 2 Python versions) + eval + Docker build
- `.github/workflows/release.yml` — build → GitHub Release → PyPI (OIDC) → GHCR on tag push
- `MANIFEST.in`, `CONTRIBUTING.md`, `SECURITY.md`, `LICENSE`
- PyPI classifiers, keywords, and project URLs in `pyproject.toml`

---

[Unreleased]: https://github.com/your-org/coderag/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/coderag/releases/tag/v0.1.0

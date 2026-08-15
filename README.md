# CodeRAG — Codebase RAG Assistant

[![CI](https://github.com/your-org/coderag/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/coderag/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/coderag.svg)](https://pypi.org/project/coderag/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Index a GitHub repository and ask natural-language questions about it —
> with answers grounded in the actual source code, citing file paths and line numbers.

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Project scaffolding, config, logging | ✅ Done |
| 1 | Repo ingestion & language-aware parsing | ✅ Done |
| 2 | Semantic chunking (function / class granularity) | ✅ Done |
| 3 | Call graph construction | ✅ Done |
| 4 | Enrichment & embedding | ✅ Done |
| 5 | Retrieval pipeline | ✅ Done |
| 6 | Answer generation with citations | ✅ Done |
| 7 | CLI wiring | ✅ Done |
| 8 | Evaluation harness | ✅ Done |
| 9 | Packaging & deployment | ✅ Done |
| 10 | Web UI | ✅ Done |

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- (Optional) An OpenAI or Anthropic API key for LLM-based generation

### 2. Install

```bash
# Navigate to the repository directory
cd "path/to/rag pipeline github"

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Cmd):
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

# Install in editable mode with dev & web dependencies
pip install -e ".[dev,web]"
```

### 3. Configure

```bash
# Windows (PowerShell / Cmd)
copy .env.example .env

# macOS / Linux
cp .env.example .env

# Edit .env and fill in the settings you need.
# Defaults work offline without an API key (using sentence-transformers embeddings).
```

### 4. Run tests

```bash
pytest
```

### 5. Use the CLI

```bash
# Index a repository (Phase 7+)
coderag index /path/to/repo

# Or index from GitHub
coderag index https://github.com/owner/repo

# Ask a question (Phase 7+)
coderag ask "how does the authentication flow work?"

# Incrementally re-index after code changes
coderag reindex /path/to/repo
```

---

## Docker

```bash
# Pull the latest image
docker pull ghcr.io/your-org/coderag:latest

# Index a local repository
docker run --rm \
  -v "$(pwd):/workspace:ro" \
  -v coderag_data:/home/coderag/.coderag \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  ghcr.io/your-org/coderag:latest \
  index /workspace

# Ask a question
docker run --rm \
  -v coderag_data:/home/coderag/.coderag \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  ghcr.io/your-org/coderag:latest \
  ask "how does the authentication flow work?"
```

Or use Docker Compose (see [docker-compose.yml](docker-compose.yml)).

---

## Web UI

CodeRAG ships with a browser-based interface built on FastAPI + vanilla JS.

### Install web extras

```bash
pip install 'coderag[web]'
```

### Start the server

```bash
coderag serve                  # opens http://127.0.0.1:8000
coderag serve --port 9000      # custom port
coderag serve --no-open        # don't auto-open browser
```

Or run uvicorn directly (useful for production):

```bash
uvicorn coderag.web.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Launch the web UI container
docker compose up coderag-web
```

Then open [http://localhost:8000](http://localhost:8000).

### REST API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/status` | Index statistics |
| `POST` | `/api/index` | Start background indexing |
| `GET` | `/api/index/progress/{job_id}` | Poll indexing progress |
| `POST` | `/api/ask` | Ask a question |
| `GET` | `/api/docs` | Interactive Swagger UI |

---

## Configuration

All settings are controlled via environment variables or a `.env` file.
Copy `.env.example` to `.env` for a full annotated reference.

| Variable | Default | Description |
|----------|---------|-------------|
| `CODERAG_EMBEDDING_PROVIDER` | `sentence_transformers` | `sentence_transformers` / `openai` / `voyage` / `cohere` |
| `CODERAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model name within the provider |
| `CODERAG_LLM_PROVIDER` | `openai` | `openai` / `anthropic` / `ollama` |
| `CODERAG_LLM_MODEL` | `gpt-4o-mini` | Model name within the provider |
| `OPENAI_API_KEY` | — | Required when using OpenAI provider |
| `ANTHROPIC_API_KEY` | — | Required when using Anthropic provider |
| `CODERAG_VECTOR_STORE` | `chroma` | `chroma` / `qdrant` / `pgvector` |
| `CODERAG_CHROMA_PATH` | `.coderag/chroma` | Persistence directory for Chroma |
| `CODERAG_CHUNK_MAX_TOKENS` | `512` | Soft token limit per chunk |
| `CODERAG_TOP_K` | `8` | Number of similarity results to retrieve |
| `CODERAG_GRAPH_EXPANSION_DEPTH` | `1` | Call-graph expansion hops |
| `CODERAG_CONTEXT_TOKEN_BUDGET` | `8000` | Max tokens in the LLM context window |
| `CODERAG_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

---

## Project Structure

```
coderag/
├── src/
│   └── coderag/
│       ├── __init__.py          # Package version
│       ├── cli.py               # Typer CLI entry point
│       ├── config.py            # Pydantic-Settings configuration
│       ├── logging_config.py    # Logging setup (Rich + file handler)
│       ├── ingestion/           # Phase 1: file walking + AST parsing
│       ├── chunking/            # Phase 2: semantic chunking
│       ├── graph/               # Phase 3: call graph
│       ├── embeddings/          # Phase 4: embedding model interface
│       ├── store/               # Phase 4: vector store interface
│       ├── retrieval/           # Phase 5: similarity search + graph expansion
│       └── generation/          # Phase 6: prompt assembly + LLM generation
├── tests/
│   ├── conftest.py              # Shared fixtures
│   └── test_phase0_config.py   # Phase 0 tests
├── .env.example                 # Annotated config template
├── .gitignore
├── pyproject.toml               # Build config, dependencies, tool settings
└── README.md
```

---

## Architecture Overview

```
  ┌─────────────────────────────────────────────────┐
  │                   CLI (Typer)                   │
  └──────────────────────┬──────────────────────────┘
                         │
           ┌─────────────▼─────────────┐
           │       index command        │
           │  1. Walk + parse files     │  ← ingestion/
           │  2. Semantic chunking      │  ← chunking/
           │  3. Build call graph       │  ← graph/
           │  4. Enrich + embed         │  ← embeddings/
           │  5. Store in vector DB     │  ← store/
           └─────────────┬─────────────┘
                         │ (persisted index)
           ┌─────────────▼─────────────┐
           │        ask command         │
           │  5. Similarity search      │  ← retrieval/
           │  6. Graph expansion        │  ← graph/
           │  7. Re-rank (optional)     │  ← retrieval/
           │  8. Generate answer        │  ← generation/
           └───────────────────────────┘
```

---

## Development

```bash
# Run all tests with coverage
pytest --cov=coderag --cov-report=term-missing

# Lint
ruff check src/ tests/

# Type-check
mypy src/
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding conventions,
and pull request guidelines.

---

## Security

To report a security vulnerability, see [SECURITY.md](SECURITY.md).

---

## License

MIT — see [LICENSE](LICENSE).

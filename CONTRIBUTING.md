# Contributing to CodeRAG

Thank you for your interest in contributing! This document explains how to get
set up, the project conventions, and the review process.

---

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Project Structure](#project-structure)
- [Adding a New Language](#adding-a-new-language)
- [Adding a New Vector Store](#adding-a-new-vector-store)
- [Adding a New LLM Provider](#adding-a-new-llm-provider)
- [Pull Request Guidelines](#pull-request-guidelines)

---

## Development Setup

### Prerequisites

- Python 3.11 or 3.12
- [Git](https://git-scm.com/)
- (Optional) Docker, for container testing

### Install in Editable Mode

```bash
# Clone
git clone https://github.com/your-org/coderag.git
cd coderag

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# Install the package plus dev extras
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Fill in any API keys you need for manual testing.
# All keys are optional — tests use mocks by default.
```

---

## Running Tests

```bash
# All tests
pytest

# A single phase
pytest tests/test_phase2_chunking.py

# With coverage report
pytest --cov=coderag --cov-report=term-missing

# Evaluation harness (no real API needed — uses MockLLMClient)
python tests/eval/run_eval.py --provider mock
```

> **CI**: every pull request runs the full matrix (Ubuntu / Windows / macOS, Python 3.11 + 3.12) via GitHub Actions.

---

## Code Style

| Tool | Config |
|------|--------|
| Formatter | `ruff format` |
| Linter | `ruff check` (E, F, I, N, W, UP) |
| Type-checker | `mypy src/` (non-strict) |
| Line length | 100 characters |

Run everything before pushing:

```bash
ruff format src/ tests/
ruff check src/ tests/
mypy src/
```

The CI will fail if ruff finds any errors.

---

## Project Structure

```
src/coderag/
├── __init__.py          # Package version
├── cli.py               # Typer CLI (Phase 7)
├── config.py            # Pydantic-Settings (Phase 0)
├── logging_config.py    # Rich + file logging (Phase 0)
├── indexer.py           # Orchestrates Phases 1–4
├── enricher.py          # Docstring enrichment
├── ingestion/           # Phase 1: file walking + AST parsing
│   ├── repo.py          # Resolve local / GitHub repos
│   ├── walker.py        # Gitignore-aware file walker
│   └── parser.py        # tree-sitter AST parser
├── chunking/            # Phase 2: semantic chunking
│   ├── extractor.py     # ASTNode → Chunk
│   ├── splitter.py      # Token-aware chunk splitting
│   └── models.py        # Chunk dataclass
├── graph/               # Phase 3: call graph
│   ├── builder.py       # Build NetworkX call graph
│   ├── analyzer.py      # Graph expansion for retrieval
│   └── models.py        # Graph edge / node models
├── embeddings/          # Phase 4: embedding model interface
│   ├── base.py          # Abstract EmbeddingModel
│   └── factory.py       # get_embedding_model() factory
├── store/               # Phase 4: vector store interface
│   ├── base.py          # Abstract VectorStore
│   ├── chroma_store.py  # ChromaDB implementation
│   └── factory.py       # get_vector_store() factory
├── retrieval/           # Phase 5: similarity + graph retrieval
│   ├── retriever.py     # Retriever orchestrator
│   ├── reranker.py      # Optional cross-encoder re-ranking
│   └── models.py        # RetrievalResult / RetrievalContext
└── generation/          # Phase 6: prompt + LLM generation
    ├── generator.py     # AnswerGenerator orchestrator
    ├── llm_client.py    # LLMClient ABC + provider impls
    ├── prompt.py        # PromptBuilder
    └── models.py        # GeneratedAnswer / TokenUsage
```

---

## Adding a New Language

1. **Add a tree-sitter grammar** to `pyproject.toml` dependencies
   (e.g. `tree-sitter-rust>=0.23`).
2. **Extend `ASTParser.LANGUAGE_MAP`** in `ingestion/parser.py` to include the
   new language name and import its grammar module.
3. **Teach the walker** by adding the file extension → language mapping in
   `ingestion/walker.py` (`EXTENSION_MAP`).
4. **Add docstring extraction** if needed — implement a
   `_extract_<lang>_docstring` method in `ChunkExtractor` (`chunking/extractor.py`).
5. **Add fixture files** under `tests/fixtures/sample_repo/` (e.g. `sample.rs`).
6. **Write tests** in `tests/test_phase2_chunking.py`.

---

## Adding a New Vector Store

All stores implement `coderag.store.base.VectorStore` (abstract base class).

1. Create `src/coderag/store/<name>_store.py`.
2. Implement `upsert`, `query`, `delete`, `count`, and `clear`.
3. Register the new store in `src/coderag/store/factory.py` (`get_vector_store`).
4. Add a `Literal` entry in `config.py` (`vector_store` field).
5. Write integration tests in `tests/test_phase4_embedding.py`.

---

## Adding a New LLM Provider

All clients implement `coderag.generation.llm_client.LLMClient` (abstract base class).

1. Create a new class in `src/coderag/generation/llm_client.py`.
2. Implement `chat(messages, *, temperature, max_tokens) -> (str, TokenUsage)`.
3. Implement the `model_name` and `provider_name` properties.
4. Register the provider in `get_llm_client()`.
5. Add a `Literal` entry in `config.py` (`llm_provider` field).
6. Write tests in `tests/test_phase6_generation.py`.

---

## Pull Request Guidelines

1. **Open an issue first** for non-trivial changes so we can agree on the
   approach before you invest time coding.
2. **Keep PRs focused** — one logical change per PR.
3. **Write tests** — new code should come with unit tests. Bug fixes should
   add a regression test.
4. **Update the README** if you change user-visible behaviour (new config keys,
   new commands, etc.).
5. **Pass CI** — the PR merge button is blocked until lint, type-check, and
   all tests are green.
6. **Squash before merge** — we use a linear, squashed commit history on `main`.

---

## Licence

By contributing, you agree that your contributions will be released under the
[MIT Licence](LICENSE).

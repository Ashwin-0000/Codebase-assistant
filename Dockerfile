# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Use a slim Python image for the build stage to keep the final image small.
FROM python:3.11-slim AS builder

# Install build tools needed by some native extensions (e.g. tree-sitter).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency definitions first so Docker can cache the pip install layer.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Create a wheel so the install in the final stage is fast and clean.
RUN pip install --upgrade pip \
 && pip install build \
 && python -m build --wheel --outdir /wheels .


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Labels (OCI image spec)
LABEL org.opencontainers.image.title="CodeRAG"
LABEL org.opencontainers.image.description="Codebase RAG assistant — index a repo and ask natural-language questions about it."
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/your-org/coderag"

# Runtime system deps (git needed for GitPython repo cloning).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security.
RUN useradd --create-home --shell /bin/bash coderag
WORKDIR /home/coderag/app

# Install the wheel built in the previous stage.
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl \
 && rm -rf /wheels

# Switch to non-root user.
USER coderag

# Persist the Chroma vector store and index data as a volume.
VOLUME ["/home/coderag/.coderag"]

# Default environment — can all be overridden via docker run -e.
ENV CODERAG_CHROMA_PATH=/home/coderag/.coderag/chroma \
    CODERAG_EMBEDDING_PROVIDER=sentence_transformers \
    CODERAG_EMBEDDING_MODEL=all-MiniLM-L6-v2 \
    CODERAG_LLM_PROVIDER=openai \
    CODERAG_LLM_MODEL=gpt-4o-mini \
    CODERAG_LOG_LEVEL=INFO

ENTRYPOINT ["coderag"]
CMD ["--help"]

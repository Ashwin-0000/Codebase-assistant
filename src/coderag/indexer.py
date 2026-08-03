"""
indexer.py — Main indexing pipeline: walk → parse → chunk → graph → enrich → embed → store.

The ``Indexer`` class orchestrates all prior phases (1-4) into a single
``index_repo()`` call.  It also handles:

- **Incremental re-indexing**: tracks SHA-256 hashes of every indexed file
  in a ``manifest.json`` sidecar; on ``reindex()``, only files whose hash
  has changed are re-processed.

- **Graph persistence**: the NetworkX call graph is serialised to
  ``graph.json`` (via ``nx.node_link_data``) next to the manifest, so that
  Phase 5 can reconstruct the graph without re-parsing.

- **Progress reporting**: yields :class:`IndexStats` with per-run numbers.

Directory layout created under ``<store_path>/../``::

    .coderag/
      chroma/              ← ChromaDB collection files
      manifest.json        ← {file_path: {sha256, chunk_ids, indexed_at}}
      graph.json           ← serialised NetworkX DiGraph
      docgen_cache.json    ← LLM-generated summaries (optional)
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

from coderag.chunking.extractor import ChunkExtractor
from coderag.chunking.models import Chunk
from coderag.embeddings.base import EmbeddingModel
from coderag.enricher import ChunkEnricher
from coderag.graph.builder import GraphBuilder
from coderag.ingestion.parser import ASTParser
from coderag.ingestion.walker import FileWalker
from coderag.store.base import VectorStore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stats / manifest data classes
# ---------------------------------------------------------------------------


@dataclass
class IndexStats:
    """Summary statistics from an indexing run."""

    repo_path: str
    files_processed: int = 0
    files_skipped: int = 0         # unchanged files (incremental run)
    files_failed: int = 0
    chunks_indexed: int = 0
    chunks_deleted: int = 0        # removed for changed files
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Indexed {self.files_processed} file(s) "
            f"({self.chunks_indexed} chunks) "
            f"in {self.elapsed_seconds:.1f}s  "
            f"[skipped={self.files_skipped} failed={self.files_failed}]"
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


class Indexer:
    """Orchestrates the full 4-phase indexing pipeline.

    Usage::

        from coderag.embeddings import MockEmbeddingModel
        from coderag.store import ChromaVectorStore

        store = ChromaVectorStore.ephemeral()
        embedder = MockEmbeddingModel(dim=384)
        indexer = Indexer(embedding_model=embedder, vector_store=store)
        stats = indexer.index_repo(Path("/path/to/repo"))
        print(stats.summary())
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        *,
        enricher: ChunkEnricher | None = None,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        coderag_dir: Path | None = None,
        embed_batch_size: int = 32,
    ) -> None:
        """
        Args:
            embedding_model:  The model used to generate vectors.
            vector_store:     The store where chunks and embeddings are persisted.
            enricher:         Pre-embedding text builder; a no-LLM default is
                              used if ``None``.
            max_tokens:       Token budget for the chunker.
            overlap_tokens:   Overlap between split chunks.
            coderag_dir:      Directory for manifest/graph/cache files.
                              Defaults to ``.coderag/`` in the CWD.
            embed_batch_size: Number of texts sent to the embedder per call.
        """
        self.embedder = embedding_model
        self.store = vector_store
        self.enricher = enricher or ChunkEnricher()
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.coderag_dir = coderag_dir or Path(".coderag")
        self.embed_batch_size = embed_batch_size

        self._parser = ASTParser()
        self._chunk_extractor = ChunkExtractor(
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        self._graph_builder = GraphBuilder(add_external_nodes=False)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def index_repo(
        self,
        repo_path: Path,
        *,
        incremental: bool = True,
    ) -> IndexStats:
        """Index (or re-index) a local repository.

        Args:
            repo_path:    Root of the local repository to index.
            incremental:  If True (default), skip files whose SHA-256 hash
                          matches the stored manifest (unchanged since last run).

        Returns:
            :class:`IndexStats` with numbers for this run.
        """
        start = time.perf_counter()
        stats = IndexStats(repo_path=str(repo_path))

        manifest = self._load_manifest() if incremental else {}
        walker = FileWalker(repo_path)
        all_source_files = list(walker.walk())

        all_chunks: list[Chunk] = []

        for source_file in all_source_files:
            try:
                file_hash = _sha256(source_file.path)

                rel = source_file.relative_path.as_posix()
                if incremental and manifest.get(rel, {}).get("sha256") == file_hash:
                    stats.files_skipped += 1
                    # Still need existing chunks for graph building
                    for chunk_id in manifest[rel].get("chunk_ids", []):
                        chunk, _ = self.store.get_by_id(chunk_id)
                        if chunk is not None:
                            all_chunks.append(chunk)
                    continue

                # Parse + chunk
                parsed = self._parser.parse_file(source_file)
                if parsed is None:
                    stats.files_skipped += 1
                    continue

                new_chunks = self._chunk_extractor.extract(parsed)
                if not new_chunks:
                    stats.files_skipped += 1
                    continue

                # Remove old chunks for this file (incremental cleanup)
                old_chunk_ids = manifest.get(rel, {}).get("chunk_ids", [])
                if old_chunk_ids:
                    self.store.delete(old_chunk_ids)
                    stats.chunks_deleted += len(old_chunk_ids)

                # Enrich + embed + store
                enriched_texts = self.enricher.enrich_batch(new_chunks)
                embeddings = self._embed_in_batches(enriched_texts)
                self.store.upsert(new_chunks, embeddings, enriched_texts)

                # Update manifest
                manifest[rel] = {
                    "sha256": file_hash,
                    "chunk_ids": [c.chunk_id for c in new_chunks],
                    "chunk_count": len(new_chunks),
                    "indexed_at": _now_iso(),
                }

                all_chunks.extend(new_chunks)
                stats.files_processed += 1
                stats.chunks_indexed += len(new_chunks)

                logger.debug(
                    "Indexed %s → %d chunks", rel, len(new_chunks)
                )

            except Exception as exc:
                logger.exception("Failed to index %s: %s", source_file.relative_path, exc)
                stats.files_failed += 1
                stats.errors.append(f"{source_file.relative_path}: {exc}")

        # Build and persist the call graph
        if all_chunks:
            graph = self._graph_builder.build(all_chunks)
            self._save_graph(graph)
            logger.info(
                "Call graph: %d nodes, %d edges",
                graph.number_of_nodes(),
                graph.number_of_edges(),
            )

        self._save_manifest(manifest)
        stats.elapsed_seconds = time.perf_counter() - start

        logger.info(stats.summary())
        return stats

    def load_graph(self) -> nx.DiGraph | None:
        """Load the persisted call graph from disk.  Returns ``None`` if not found."""
        path = self._graph_path()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text("utf-8"))
            return GraphBuilder.from_node_link(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not load graph: %s", e)
            return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _embed_in_batches(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* in batches of ``embed_batch_size``."""
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i : i + self.embed_batch_size]
            all_vecs.extend(self.embedder.embed(batch))
        return all_vecs

    # ── Manifest (incremental indexing) ─────────────────────────────────

    def _manifest_path(self) -> Path:
        return self.coderag_dir / "manifest.json"

    def _graph_path(self) -> Path:
        return self.coderag_dir / "graph.json"

    def _load_manifest(self) -> dict:
        path = self._manifest_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_manifest(self, manifest: dict) -> None:
        path = self._manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _save_graph(self, graph: nx.DiGraph) -> None:
        path = self._graph_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = GraphBuilder.to_node_link(graph)
        path.write_text(
            json.dumps(data, ensure_ascii=False),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    """Return the hex-encoded SHA-256 hash of *path*'s contents."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

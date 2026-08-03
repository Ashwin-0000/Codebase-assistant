"""
store/chroma_store.py — ChromaDB-backed vector store.

ChromaDB persists a vector index + metadata to a local directory.  Each
chunk is stored as a ChromaDB document with:
  - ``id``:        chunk_id (deterministic, stable across re-index runs)
  - ``embedding``: pre-computed float vector
  - ``document``:  the enriched text that was embedded (for reference)
  - ``metadata``:  all Chunk fields, serialised to Chroma-compatible types

Chroma metadata constraints (as of v0.6):
  - Values must be ``str | int | float | bool`` — no ``None``, no ``list``.
  - We convert:
      ``None``   → ``""`` (empty string) for str fields
      ``None``   → ``-1`` (sentinel int) for int fields
      ``list``   → JSON string
      ``bool``   → native Python bool (Chroma handles this fine)
"""

from __future__ import annotations

import json
import logging

import chromadb  # type: ignore

from coderag.chunking.models import Chunk
from coderag.store.base import SearchResult, VectorStore

logger = logging.getLogger(__name__)

# Sentinel value for "this field was None" when stored as an int
_NONE_INT = -999


class ChromaVectorStore(VectorStore):
    """Vector store backed by a local ChromaDB collection.

    Usage (persistent)::

        store = ChromaVectorStore("/path/to/db")
        store.upsert(chunks, embeddings, enriched_texts)
        results = store.query(query_vec, top_k=5)

    Usage (ephemeral — for tests)::

        store = ChromaVectorStore.ephemeral()
    """

    def __init__(
        self,
        path: str,
        collection_name: str = "coderag",
        *,
        hnsw_space: str = "cosine",
    ) -> None:
        """
        Args:
            path:            Directory path for ChromaDB persistence.
            collection_name: Name of the ChromaDB collection.
            hnsw_space:      Distance metric (``"cosine"`` or ``"l2"``).
        """
        self._client = chromadb.PersistentClient(path=path)
        self._collection_name = collection_name
        self._hnsw_space = hnsw_space
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": hnsw_space},
        )
        logger.info(
            "ChromaVectorStore: collection=%r path=%r count=%d",
            collection_name,
            path,
            self._collection.count(),
        )

    @classmethod
    def ephemeral(cls, collection_name: str = "coderag") -> "ChromaVectorStore":
        """Create an in-memory store (no filesystem I/O — ideal for tests)."""
        instance: ChromaVectorStore = object.__new__(cls)
        instance._client = chromadb.EphemeralClient()
        instance._collection_name = collection_name
        instance._hnsw_space = "cosine"
        instance._collection = instance._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return instance

    # ------------------------------------------------------------------ #
    # VectorStore interface
    # ------------------------------------------------------------------ #

    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        enriched_texts: list[str],
    ) -> None:
        """Add or overwrite chunks in the collection."""
        if not chunks:
            return

        ids = [c.chunk_id for c in chunks]
        metadatas = [_chunk_to_meta(c) for c in chunks]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=enriched_texts,
            metadatas=metadatas,
        )
        logger.debug("Upserted %d chunks", len(chunks))

    def query(
        self,
        embedding: list[float],
        top_k: int = 8,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """Return top-k most similar chunks, sorted by descending cosine similarity."""
        n = min(top_k, self.count())
        if n == 0:
            return []

        kwargs: dict = dict(
            query_embeddings=[embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        results: list[SearchResult] = []
        for chunk_id, doc, meta, dist in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            chunk = _meta_to_chunk(chunk_id, meta)
            # For cosine space: distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
            score = 1.0 - dist
            results.append(SearchResult(chunk=chunk, score=score, enriched_text=doc))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def get_by_id(self, chunk_id: str) -> tuple[Chunk | None, str | None]:
        """Retrieve a single chunk by its ID.

        Returns:
            ``(Chunk, enriched_text)`` if found, ``(None, None)`` otherwise.
        """
        raw = self._collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )
        if not raw["ids"]:
            return None, None

        chunk = _meta_to_chunk(raw["ids"][0], raw["metadatas"][0])
        doc = raw["documents"][0]
        return chunk, doc

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by ID."""
        if chunk_ids:
            self._collection.delete(ids=chunk_ids)
            logger.debug("Deleted %d chunks", len(chunk_ids))

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        """Delete all documents from the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self._hnsw_space},
        )
        logger.warning("ChromaVectorStore: collection cleared")

    def get_file_chunk_ids(self, file_path: str) -> list[str]:
        """Return all chunk IDs for a given file_path (for incremental re-index)."""
        raw = self._collection.get(
            where={"file_path": file_path},
            include=[],
        )
        return raw["ids"]


# ---------------------------------------------------------------------------
# Chroma serialisation helpers
# ---------------------------------------------------------------------------


def _chunk_to_meta(chunk: Chunk) -> dict:
    """Convert a Chunk to a flat Chroma-compatible metadata dict."""
    return {
        # ── str fields ────────────────────────────────────────────────
        "file_path":      chunk.file_path,
        "language":       chunk.language,
        "node_type":      chunk.node_type,
        "function_name":  chunk.function_name or "",
        "class_name":     chunk.class_name or "",
        "raw_code":       chunk.raw_code,
        "docstring":      chunk.docstring or "",
        # ── int fields ────────────────────────────────────────────────
        "start_line":     chunk.start_line,
        "end_line":       chunk.end_line,
        "token_count":    chunk.token_count,
        "split_index":    chunk.split_index if chunk.split_index is not None else _NONE_INT,
        "total_splits":   chunk.total_splits if chunk.total_splits is not None else _NONE_INT,
        # ── bool fields (stored as Python bool — Chroma handles them) -
        "has_docstring":  chunk.has_docstring,
        "is_split_chunk": chunk.is_split_chunk,
        # ── serialised list ───────────────────────────────────────────
        "imports":        json.dumps(chunk.imports),
    }


def _meta_to_chunk(chunk_id: str, meta: dict) -> Chunk:
    """Reconstruct a Chunk from Chroma metadata."""
    split_index_raw = int(meta.get("split_index", _NONE_INT))
    total_splits_raw = int(meta.get("total_splits", _NONE_INT))

    return Chunk(
        chunk_id=chunk_id,
        file_path=meta["file_path"],
        language=meta["language"],
        node_type=meta["node_type"],
        function_name=meta["function_name"] or None,
        class_name=meta["class_name"] or None,
        raw_code=meta["raw_code"],
        docstring=meta["docstring"] or None,
        start_line=int(meta["start_line"]),
        end_line=int(meta["end_line"]),
        token_count=int(meta["token_count"]),
        has_docstring=bool(meta.get("has_docstring", False)),
        is_split_chunk=bool(meta.get("is_split_chunk", False)),
        split_index=split_index_raw if split_index_raw != _NONE_INT else None,
        total_splits=total_splits_raw if total_splits_raw != _NONE_INT else None,
        imports=json.loads(meta.get("imports", "[]")),
    )

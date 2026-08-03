"""
store/factory.py — Build the configured VectorStore at runtime.
"""

from __future__ import annotations

import logging

from coderag.store.base import VectorStore

logger = logging.getLogger(__name__)


def get_vector_store(
    backend: str = "chroma",
    path: str = ".coderag/chroma",
    collection_name: str = "coderag",
    **kwargs,
) -> VectorStore:
    """Construct and return the requested vector store.

    Args:
        backend:         One of ``"chroma"``.
        path:            Persistence directory (used by ``chroma``).
        collection_name: ChromaDB collection name.
        **kwargs:        Extra arguments forwarded to the concrete constructor.

    Returns:
        A ready-to-use :class:`~coderag.store.base.VectorStore`.
    """
    b = backend.lower()

    if b == "chroma":
        from coderag.store.chroma_store import ChromaVectorStore

        logger.info("Vector store: chroma path=%r collection=%r", path, collection_name)
        return ChromaVectorStore(path=path, collection_name=collection_name, **kwargs)

    raise ValueError(
        f"Unknown vector store backend {backend!r}. Valid choices: 'chroma'."
    )

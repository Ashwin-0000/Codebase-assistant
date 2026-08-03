"""
embeddings/sentence_transformer.py — Local, zero-API-key embedding model.

Uses ``sentence-transformers`` (backed by PyTorch) to run embeddings entirely
on the local machine.  This is the default provider — no API keys needed,
works offline, and ``all-MiniLM-L6-v2`` is small enough to run on CPU in
reasonable time (≈ 30 ms/chunk on a modern laptop).

Model is loaded lazily on first call so that importing this module doesn't
incur the PyTorch startup cost if the provider is never used.
"""

from __future__ import annotations

import logging

from coderag.embeddings.base import EmbeddingModel

logger = logging.getLogger(__name__)

_BATCH_SIZE = 64  # process at most N texts per encode() call to limit memory


class SentenceTransformerEmbedder(EmbeddingModel):
    """Embedding model backed by ``sentence-transformers``.

    Usage::

        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        vectors = embedder.embed(["hello world", "foo bar"])
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Args:
            model_name: Any model available on the Hugging Face Hub or cached
                        locally (e.g. ``"all-MiniLM-L6-v2"``,
                        ``"BAAI/bge-small-en-v1.5"``).
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        logger.info("Loading SentenceTransformer model: %s", model_name)
        self._st = SentenceTransformer(model_name)
        self._dim: int = self._st.get_sentence_embedding_dimension()
        self._name = model_name
        logger.info("Model loaded (dim=%d)", self._dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* in batches; returns a list of float lists."""
        if not texts:
            return []

        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            vecs = self._st.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            all_vecs.extend(vec.tolist() for vec in vecs)

        return all_vecs

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._name

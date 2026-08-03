"""
Embeddings sub-package.

Public API:
  EmbeddingModel          — abstract base class
  get_embedding_model()   — factory (reads provider from args or settings)
  MockEmbeddingModel      — deterministic test double
"""

from coderag.embeddings.base import EmbeddingModel
from coderag.embeddings.factory import MockEmbeddingModel, get_embedding_model

__all__ = ["EmbeddingModel", "get_embedding_model", "MockEmbeddingModel"]

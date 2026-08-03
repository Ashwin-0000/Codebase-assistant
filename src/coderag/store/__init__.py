"""
Store sub-package — vector database abstraction.

Public API:
  VectorStore          — abstract base class
  SearchResult         — result from query()
  ChromaVectorStore    — ChromaDB implementation
  get_vector_store()   — factory function
"""

from coderag.store.base import SearchResult, VectorStore
from coderag.store.chroma_store import ChromaVectorStore
from coderag.store.factory import get_vector_store

__all__ = ["VectorStore", "SearchResult", "ChromaVectorStore", "get_vector_store"]

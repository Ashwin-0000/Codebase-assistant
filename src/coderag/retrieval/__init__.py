"""Retrieval sub-package (Phase 5 — similarity search + graph expansion).

Public API
----------
- :class:`Retriever` — main retrieval pipeline orchestrator.
- :class:`RetrievalResult` — a single scored chunk from retrieval.
- :class:`RetrievalContext` — fully assembled, token-budgeted context.
- :class:`ContextBuilder` — context assembly utility.
- :class:`Reranker`, :class:`NoOpReranker` — re-ranking interface.
"""

from coderag.retrieval.context_builder import ContextBuilder
from coderag.retrieval.models import RetrievalContext, RetrievalResult
from coderag.retrieval.reranker import NoOpReranker, Reranker, get_reranker
from coderag.retrieval.retriever import Retriever

__all__ = [
    "ContextBuilder",
    "NoOpReranker",
    "Reranker",
    "RetrievalContext",
    "RetrievalResult",
    "Retriever",
    "get_reranker",
]

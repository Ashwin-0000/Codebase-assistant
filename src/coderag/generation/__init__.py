"""Generation sub-package (Phase 6 — prompt assembly + LLM answer generation).

Public API
----------
- :class:`AnswerGenerator` — main generation pipeline orchestrator.
- :class:`GeneratedAnswer` — the final answer with citations.
- :class:`TokenUsage` — token usage statistics from the LLM call.
- :class:`PromptBuilder` — prompt assembly from retrieval context.
- :class:`LLMClient` — abstract LLM client interface.
- :func:`get_llm_client` — factory for LLM clients.
"""

from coderag.generation.generator import AnswerGenerator
from coderag.generation.llm_client import (
    LLMClient,
    MockLLMClient,
    get_llm_client,
)
from coderag.generation.models import GeneratedAnswer, TokenUsage
from coderag.generation.prompt import PromptBuilder

__all__ = [
    "AnswerGenerator",
    "GeneratedAnswer",
    "LLMClient",
    "MockLLMClient",
    "PromptBuilder",
    "TokenUsage",
    "get_llm_client",
]

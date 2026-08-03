"""
generation/models.py — Data classes for LLM-generated answers.

Provides :class:`GeneratedAnswer`, the final output of the answer generation
pipeline.  It bundles the answer text, source citations, model metadata,
and token usage so callers (CLI, Web UI) have everything they need.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenUsage:
    """Token counts for a single LLM call.

    Attributes:
        prompt_tokens:     Tokens consumed by the prompt (system + user).
        completion_tokens: Tokens generated in the response.
        total_tokens:      Sum of prompt + completion tokens.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class GeneratedAnswer:
    """The final answer produced by the generation pipeline.

    Attributes:
        answer:     The natural-language answer with inline citations.
        citations:  Deduplicated list of ``file:line`` references from
                    the retrieval context that the answer is grounded in.
        query:      The original user question.
        model:      The LLM model name used (e.g. ``gpt-4o-mini``).
        provider:   The LLM provider (``openai`` / ``anthropic`` / ``ollama``).
        usage:      Token usage statistics (when available from the API).
        context_chunks: Number of code chunks provided to the LLM.
    """

    answer: str
    citations: list[str] = field(default_factory=list)
    query: str = ""
    model: str = ""
    provider: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    context_chunks: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.answer.strip()

    def formatted(self) -> str:
        """Return the answer with a citation footer."""
        lines = [self.answer]
        if self.citations:
            lines.append("")
            lines.append("Sources:")
            for i, cite in enumerate(self.citations, 1):
                lines.append(f"  [{i}] {cite}")
        return "\n".join(lines)

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"Answer: {len(self.answer)} chars, "
            f"{len(self.citations)} citations, "
            f"model={self.provider}/{self.model}, "
            f"tokens={self.usage.total_tokens}"
        )

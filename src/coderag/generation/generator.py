"""
generation/generator.py — Answer generation orchestrator.

The :class:`AnswerGenerator` ties together prompt assembly, LLM invocation,
and citation extraction into a single ``generate(context)`` call:

1. **Build prompt** from the retrieval context (system + user messages).
2. **Call LLM** via the configured :class:`~coderag.generation.llm_client.LLMClient`.
3. **Extract citations** from the retrieval context and attach them.
4. Return a :class:`~coderag.generation.models.GeneratedAnswer`.

The generator is stateless — all configuration is injected at construction.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from coderag.generation.llm_client import LLMClient
from coderag.generation.models import GeneratedAnswer
from coderag.generation.prompt import PromptBuilder

if TYPE_CHECKING:
    from coderag.retrieval.models import RetrievalContext

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Orchestrates the full answer-generation pipeline.

    Usage::

        from coderag.generation import AnswerGenerator
        from coderag.generation.llm_client import get_llm_client

        client = get_llm_client("openai", model="gpt-4o-mini", api_key="sk-...")
        generator = AnswerGenerator(llm_client=client)
        answer = generator.generate(retrieval_context)
        print(answer.formatted())
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        prompt_builder: PromptBuilder | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        """
        Args:
            llm_client:     The LLM client to use for generation.
            prompt_builder: Override the default prompt builder.
            temperature:    Sampling temperature for generation.
            max_tokens:     Maximum tokens in the LLM response.
        """
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, context: "RetrievalContext") -> GeneratedAnswer:
        """Generate an answer grounded in the retrieval context.

        Args:
            context: A :class:`~coderag.retrieval.models.RetrievalContext`
                     from Phase 5 retrieval.

        Returns:
            A :class:`~coderag.generation.models.GeneratedAnswer` with
            the answer text, citations, and model metadata.
        """
        if context.is_empty:
            logger.warning("Empty retrieval context — returning fallback answer")
            return GeneratedAnswer(
                answer="I don't have enough context to answer this question. "
                       "No relevant code chunks were found in the index.",
                citations=[],
                query=context.query,
                model=self.llm_client.model_name,
                provider=self.llm_client.provider_name,
                context_chunks=0,
            )

        # 1. Build the prompt
        messages = self.prompt_builder.build(context)

        # 2. Call the LLM
        try:
            answer_text, usage = self.llm_client.chat(
                messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            logger.exception("LLM call failed: %s", exc)
            return GeneratedAnswer(
                answer=f"Error generating answer: {exc}",
                citations=context.citations(),
                query=context.query,
                model=self.llm_client.model_name,
                provider=self.llm_client.provider_name,
                context_chunks=context.chunk_count,
            )

        # 3. Collect citations from the retrieval context
        citations = _deduplicate_citations(context.citations())

        # 4. Assemble the answer
        answer = GeneratedAnswer(
            answer=answer_text.strip(),
            citations=citations,
            query=context.query,
            model=self.llm_client.model_name,
            provider=self.llm_client.provider_name,
            usage=usage,
            context_chunks=context.chunk_count,
        )

        logger.info(answer.summary())
        return answer


def _deduplicate_citations(citations: list[str]) -> list[str]:
    """Remove duplicate citations while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result

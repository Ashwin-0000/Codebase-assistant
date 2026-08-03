"""
generation/prompt.py — Prompt assembly for the answer-generation LLM call.

Builds the system and user messages from a
:class:`~coderag.retrieval.models.RetrievalContext`.

Design decisions
----------------
- **System prompt**: instructs the model to answer based solely on the
  provided code context, cite file paths and line numbers, and say
  "I don't know" when the context is insufficient.
- **User message**: embeds the full ``context_text`` from Phase 5 plus
  the user's question.
- Messages are returned as ``list[dict]`` compatible with both the OpenAI
  and Anthropic chat-completion APIs.
"""

from __future__ import annotations

import logging

from coderag.retrieval.models import RetrievalContext

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are CodeRAG, an AI assistant that answers questions about a codebase.

RULES:
1. Answer ONLY based on the code context provided below. Do not guess or
   use outside knowledge about the codebase.
2. When referencing code, ALWAYS cite the file path and line numbers
   (e.g., `src/auth.py:L12-L45`).
3. If the context does not contain enough information to answer the
   question, say "I don't have enough context to answer this question"
   and explain what additional code would be needed.
4. Be concise but thorough. Explain the *why*, not just the *what*.
5. When describing control flow, list the functions/methods in call order.
6. Format code references with backticks: `function_name()`, `ClassName`.
"""

USER_TEMPLATE = """\
## Code Context

{context}

---

## Question

{query}
"""


class PromptBuilder:
    """Assemble chat messages from a retrieval context.

    Usage::

        builder = PromptBuilder()
        messages = builder.build(retrieval_context)
        # → [{"role": "system", ...}, {"role": "user", ...}]
    """

    def __init__(
        self,
        system_prompt: str | None = None,
    ) -> None:
        """
        Args:
            system_prompt: Override the default system prompt.  Pass ``None``
                           to use the built-in CodeRAG prompt.
        """
        self.system_prompt = system_prompt or SYSTEM_PROMPT

    def build(self, context: RetrievalContext) -> list[dict[str, str]]:
        """Build the message list for the LLM.

        Args:
            context: A :class:`~coderag.retrieval.models.RetrievalContext`
                     from Phase 5 retrieval.

        Returns:
            List of message dicts with ``role`` and ``content`` keys.
        """
        user_content = USER_TEMPLATE.format(
            context=context.context_text or "(no code context available)",
            query=context.query,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        logger.debug(
            "Prompt built: system=%d chars, user=%d chars",
            len(self.system_prompt),
            len(user_content),
        )
        return messages

    def build_user_content(self, context: RetrievalContext) -> str:
        """Return just the user message content (used by Anthropic API)."""
        return USER_TEMPLATE.format(
            context=context.context_text or "(no code context available)",
            query=context.query,
        )

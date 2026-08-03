"""
generation/llm_client.py — Thin wrappers around LLM provider APIs.

Each client implements a minimal ``chat(messages) → (text, usage)`` contract.
The clients are intentionally thin — no retry logic, no streaming, no caching
— so that higher-level concerns live in the :class:`AnswerGenerator`.

Supported providers
-------------------
- **OpenAI** (``gpt-4o``, ``gpt-4o-mini``, etc.)
- **Anthropic** (``claude-3.5-sonnet``, etc.)
- **Ollama** (local models via the OpenAI-compatible API)

The factory function :func:`get_llm_client` reads provider/model/key from
the arguments (or falls back to :class:`~coderag.config.Settings`).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from coderag.generation.models import TokenUsage

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM chat-completion clients."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> tuple[str, TokenUsage]:
        """Send a chat-completion request.

        Args:
            messages:    List of ``{"role": ..., "content": ...}`` dicts.
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens:  Maximum tokens in the response.

        Returns:
            ``(answer_text, token_usage)`` tuple.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier being used."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """The provider name (``openai`` / ``anthropic`` / ``ollama``)."""


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class OpenAIClient(LLMClient):
    """Chat-completion client for OpenAI (and OpenAI-compatible APIs).

    Also used for Ollama by setting ``base_url`` to the Ollama endpoint.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str = "",
        *,
        base_url: str | None = None,
        provider: str = "openai",
    ) -> None:
        from openai import OpenAI

        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)
        self._model = model
        self._provider = provider
        logger.info("OpenAI client: model=%s provider=%s", model, provider)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> tuple[str, TokenUsage]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text = response.choices[0].message.content or ""
        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            )

        logger.debug("OpenAI response: %d chars, %s", len(text), usage)
        return text, usage

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return self._provider


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


class AnthropicClient(LLMClient):
    """Chat-completion client for Anthropic (Claude models)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str = "",
    ) -> None:
        from anthropic import Anthropic

        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key

        self._client = Anthropic(**kwargs)
        self._model = model
        logger.info("Anthropic client: model=%s", model)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> tuple[str, TokenUsage]:
        # Anthropic API separates system from user/assistant messages
        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        kwargs: dict = dict(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=chat_messages,
        )
        if system_msg:
            kwargs["system"] = system_msg

        response = self._client.messages.create(**kwargs)

        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
        )

        logger.debug("Anthropic response: %d chars, %s", len(text), usage)
        return text, usage

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "anthropic"


# ---------------------------------------------------------------------------
# Mock (for testing)
# ---------------------------------------------------------------------------


class MockLLMClient(LLMClient):
    """Deterministic mock client for unit tests.

    Returns a canned answer that references the first citation from the
    context, so tests can verify citation pass-through.
    """

    def __init__(self, model: str = "mock-model") -> None:
        self._model = model
        self.call_count = 0
        self.last_messages: list[dict[str, str]] = []

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> tuple[str, TokenUsage]:
        self.call_count += 1
        self.last_messages = messages

        # Extract the query from the user message
        user_msg = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )

        answer = (
            f"Based on the provided code context, here is the answer to your question.\n\n"
            f"The code handles this functionality as described in the context above.\n\n"
            f"This is a mock response (call #{self.call_count})."
        )

        usage = TokenUsage(
            prompt_tokens=len(user_msg) // 4,
            completion_tokens=len(answer) // 4,
            total_tokens=(len(user_msg) + len(answer)) // 4,
        )

        return answer, usage

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return "mock"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_llm_client(
    provider: str = "openai",
    model: str | None = None,
    api_key: str = "",
    *,
    ollama_base_url: str = "http://localhost:11434",
) -> LLMClient:
    """Construct and return the requested LLM client.

    Args:
        provider:        One of ``"openai"``, ``"anthropic"``, ``"ollama"``.
        model:           Model name.  Defaults depend on provider.
        api_key:         API key (required for cloud providers).
        ollama_base_url: Base URL for the Ollama server.

    Returns:
        A ready-to-use :class:`LLMClient`.

    Raises:
        ValueError: If *provider* is unknown.
    """
    p = provider.lower()

    if p == "openai":
        return OpenAIClient(
            model=model or "gpt-4o-mini",
            api_key=api_key,
        )

    if p == "anthropic":
        return AnthropicClient(
            model=model or "claude-sonnet-4-20250514",
            api_key=api_key,
        )

    if p == "ollama":
        # Ollama exposes an OpenAI-compatible API
        return OpenAIClient(
            model=model or "llama3.1",
            api_key="ollama",  # Ollama ignores the key but OpenAI client requires one
            base_url=f"{ollama_base_url.rstrip('/')}/v1",
            provider="ollama",
        )

    raise ValueError(
        f"Unknown LLM provider {provider!r}. Valid choices: 'openai', 'anthropic', 'ollama'."
    )

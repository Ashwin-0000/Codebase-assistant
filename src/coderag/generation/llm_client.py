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
    """Question-aware local synthesizer for offline use (zero API cost).

    Analyzes the user's question and retrieved code context to produce a
    focused, question-relevant answer locally — without calling any LLM API.

    Uses keyword matching, relevance scoring, and question-type detection
    to generate answers that actually address what the user asked, rather
    than dumping a generic formatted template.
    """

    # Common stop-words to ignore when extracting question keywords
    _STOP_WORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "shall",
        "should", "may", "might", "must", "can", "could", "about", "above",
        "after", "again", "all", "also", "and", "any", "because", "before",
        "between", "both", "but", "by", "came", "come", "each", "for",
        "from", "get", "got", "had", "has", "have", "he", "her", "here",
        "him", "his", "how", "i", "if", "in", "into", "it", "its", "just",
        "let", "like", "make", "many", "me", "more", "most", "much", "my",
        "no", "not", "now", "of", "on", "one", "only", "or", "other", "our",
        "out", "over", "own", "said", "same", "she", "so", "some", "still",
        "such", "take", "than", "that", "the", "their", "them", "then",
        "there", "these", "they", "this", "those", "through", "to", "too",
        "under", "up", "us", "very", "want", "was", "way", "we", "well",
        "were", "what", "when", "where", "which", "while", "who", "whom",
        "why", "with", "would", "you", "your", "tell", "explain", "show",
        "describe", "give", "me", "please", "code", "codebase", "repository",
        "repo", "project", "work", "works", "used", "using", "use",
    })

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

        user_msg = next(
            (m["content"] for m in messages if m["role"] == "user"), ""
        )

        question = ""
        raw_ctx = ""
        if "## Question" in user_msg:
            question = user_msg.split("## Question", 1)[1].strip()
        if "## Code Context" in user_msg and "## Question" in user_msg:
            try:
                after_header = user_msg.split("## Code Context", 1)[1]
                raw_ctx = after_header.split("## Question", 1)[0].strip()
                if raw_ctx.endswith("---"):
                    raw_ctx = raw_ctx[:-3].strip()
            except Exception:
                raw_ctx = ""

        if not raw_ctx or raw_ctx == "(no code context available)":
            answer = (
                "I don't have enough context to answer this question. "
                "No relevant code chunks were found in the index."
            )
        else:
            answer = self._synthesize_local_answer(question, raw_ctx)

        usage = TokenUsage(
            prompt_tokens=len(user_msg) // 4,
            completion_tokens=len(answer) // 4,
            total_tokens=(len(user_msg) + len(answer)) // 4,
        )

        return answer, usage

    # ------------------------------------------------------------------ #
    # Question-aware answer synthesis
    # ------------------------------------------------------------------ #

    def _synthesize_local_answer(self, question: str, raw_ctx: str) -> str:
        """Synthesize a question-aware answer from retrieved code chunks."""
        import re

        chunks = self._parse_chunks(raw_ctx)
        if not chunks:
            return f"Based on the retrieved context for your question *\"{question}\"*:\n\n{raw_ctx}"

        # Extract keywords from the question
        keywords = self._extract_keywords(question)

        # Score each chunk for relevance to the question
        scored_chunks = self._score_chunks(chunks, keywords, question)

        # Detect what kind of question this is
        q_type = self._detect_question_type(question)

        # Build a focused answer based on question type and relevant chunks
        return self._build_answer(question, scored_chunks, q_type, keywords)

    def _parse_chunks(self, raw_ctx: str) -> list[dict]:
        """Parse raw context text into structured chunk dicts."""
        chunks = []
        raw_blocks = raw_ctx.split("--- [")
        for block in raw_blocks:
            if not block.strip():
                continue
            lines = block.splitlines()
            header = lines[0] if lines else ""
            file_info = ""
            scope = ""
            lang = "python"
            code_lines: list[str] = []
            summary_lines: list[str] = []
            in_code = False
            in_summary = False

            for line in lines:
                if line.startswith("[File]:"):
                    file_info = line.replace("[File]:", "").strip()
                    in_code = False
                    in_summary = False
                elif line.startswith("[Scope]:"):
                    scope = line.replace("[Scope]:", "").strip()
                    in_code = False
                    in_summary = False
                elif line.startswith("[Language]:"):
                    lang = line.replace("[Language]:", "").strip()
                    in_code = False
                    in_summary = False
                elif line.startswith("[Summary]:"):
                    in_summary = True
                    in_code = False
                    rest = line.replace("[Summary]:", "").strip()
                    if rest:
                        summary_lines.append(rest)
                elif line.startswith("[Code]:"):
                    in_code = True
                    in_summary = False
                elif in_code:
                    code_lines.append(line)
                elif in_summary:
                    summary_lines.append(line)

            code = "\n".join(code_lines).strip()
            summary = " ".join(summary_lines).strip()
            if file_info or scope or code:
                chunks.append({
                    "header": header,
                    "file": file_info,
                    "scope": scope,
                    "lang": lang,
                    "code": code,
                    "summary": summary,
                    "relevance": 0.0,
                })
        return chunks

    def _extract_keywords(self, question: str) -> list[str]:
        """Extract meaningful keywords from the question."""
        import re
        # Normalize and tokenize
        words = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question.lower())
        # Filter stop words, keep meaningful terms
        keywords = [w for w in words if w not in self._STOP_WORDS and len(w) > 1]
        # Also extract quoted terms and backtick terms as exact phrases
        exact = re.findall(r'[`"\']([^`"\']+)[`"\']', question)
        keywords.extend(e.lower() for e in exact)
        # Deduplicate while preserving order
        seen: set[str] = set()
        result: list[str] = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                result.append(k)
        return result

    def _score_chunks(
        self,
        chunks: list[dict],
        keywords: list[str],
        question: str,
    ) -> list[dict]:
        """Score and sort chunks by relevance to the question keywords."""
        q_lower = question.lower()
        for chunk in chunks:
            score = 0.0
            searchable = " ".join([
                chunk.get("scope", ""),
                chunk.get("file", ""),
                chunk.get("code", ""),
                chunk.get("summary", ""),
                chunk.get("header", ""),
            ]).lower()

            for kw in keywords:
                # Exact match in scope/function name is highly valuable
                if kw in chunk.get("scope", "").lower():
                    score += 5.0
                # Match in file path
                if kw in chunk.get("file", "").lower():
                    score += 3.0
                # Match in code
                if kw in chunk.get("code", "").lower():
                    score += 2.0
                # Match in summary
                if kw in chunk.get("summary", "").lower():
                    score += 2.5

            # Bonus: if the scope/function name appears directly in the question
            scope = chunk.get("scope", "")
            if scope and scope != "<anonymous>":
                scope_lower = scope.lower()
                # Check various forms: exact, snake_case parts, camelCase
                if scope_lower in q_lower:
                    score += 10.0
                else:
                    # Check individual parts of snake_case names
                    parts = scope_lower.replace("-", "_").split("_")
                    matching_parts = sum(1 for p in parts if p and p in q_lower)
                    if matching_parts > 0:
                        score += matching_parts * 2.0

            chunk["relevance"] = score

        # Sort by relevance descending, then preserve original order as tiebreaker
        chunks.sort(key=lambda c: c["relevance"], reverse=True)
        return chunks

    def _detect_question_type(self, question: str) -> str:
        """Detect the question type to tailor the response format."""
        q = question.lower().strip().rstrip("?").strip()

        if any(q.startswith(w) for w in ("how does", "how do", "how is", "how are", "how to")):
            return "how"
        if any(q.startswith(w) for w in ("what does", "what do", "what is", "what are", "what's")):
            return "what"
        if any(q.startswith(w) for w in ("why does", "why do", "why is", "why are")):
            return "why"
        if any(q.startswith(w) for w in ("where does", "where do", "where is", "where are")):
            return "where"
        if any(q.startswith(w) for w in ("list", "enumerate", "show all", "find all")):
            return "list"
        if any(w in q for w in ("difference between", "compare", "vs", "versus")):
            return "compare"
        if any(w in q for w in ("purpose", "role", "responsible", "does this do", "does it do")):
            return "purpose"
        if any(w in q for w in ("architecture", "structure", "overview", "organize", "layout")):
            return "overview"
        if any(w in q for w in ("error", "bug", "issue", "fix", "wrong", "fail", "broken")):
            return "debug"
        return "general"

    def _build_answer(
        self,
        question: str,
        chunks: list[dict],
        q_type: str,
        keywords: list[str],
    ) -> str:
        """Build a focused answer tailored to the question type."""
        # Separate high-relevance chunks from low-relevance ones
        relevant = [c for c in chunks if c["relevance"] > 0]
        if not relevant:
            # Fall back to all chunks if none scored above 0
            relevant = chunks

        # Cap at a reasonable number for display
        primary = relevant[:4]
        secondary = relevant[4:8]

        output_parts: list[str] = []

        # Build the answer header based on question type
        output_parts.append(self._build_header(question, primary, q_type, keywords))

        # Add the primary relevant code sections with explanations
        output_parts.append(self._build_code_sections(primary, q_type, keywords))

        # Add secondary mentions if available
        if secondary:
            output_parts.append(self._build_related_section(secondary))

        # Add execution flow for "how" questions
        if q_type in ("how", "general", "overview"):
            flow = self._build_flow_section(primary)
            if flow:
                output_parts.append(flow)

        output_parts.append(
            "\n---\n*Running in **Offline Local Synthesizer** mode — "
            "for richer answers, configure an LLM provider (OpenAI / Anthropic / Ollama) "
            "in your `.env` file.*"
        )
        return "\n".join(output_parts)

    def _build_header(
        self,
        question: str,
        chunks: list[dict],
        q_type: str,
        keywords: list[str],
    ) -> str:
        """Build a question-aware header/overview."""
        # Collect unique scopes and files from relevant chunks
        scopes = [
            c["scope"] for c in chunks
            if c.get("scope") and c["scope"] != "<anonymous>"
        ]
        files = list(dict.fromkeys(
            c["file"].split(":")[0] for c in chunks if c.get("file")
        ))

        if q_type == "what":
            if scopes:
                intro = (
                    f"Based on the indexed code, **{scopes[0]}** "
                    f"{'(in `' + files[0] + '`)' if files else ''} "
                    f"is the most relevant component to your question."
                )
            elif files:
                intro = (
                    f"The relevant code is found in "
                    f"`{'`, `'.join(files[:3])}`."
                )
            else:
                intro = "Here is what the code reveals about your question."

        elif q_type == "how":
            if scopes:
                intro = (
                    f"The implementation involves "
                    f"**{'**, **'.join(scopes[:3])}** "
                    f"{'across `' + '`, `'.join(files[:3]) + '`' if files else ''}. "
                    f"Here's how it works:"
                )
            else:
                intro = "Here's how this is implemented based on the retrieved code:"

        elif q_type == "where":
            if files:
                intro = (
                    f"This is located in "
                    f"`{'`, `'.join(files[:4])}`"
                    f"{', specifically in ' + ', '.join(f'`{s}`' for s in scopes[:3]) if scopes else ''}."
                )
            else:
                intro = "Here is where the relevant code is located:"

        elif q_type == "why":
            intro = (
                "Based on the code structure and context, "
                "here is the reasoning I can infer:"
            )

        elif q_type == "list":
            intro = f"Here are the relevant items found ({len(chunks)} match{'es' if len(chunks) != 1 else ''}):"

        elif q_type == "overview":
            if files:
                intro = (
                    f"The codebase is organized across "
                    f"`{'`, `'.join(files[:5])}`. "
                    f"Here's a breakdown of the key components:"
                )
            else:
                intro = "Here's an overview based on the retrieved code:"

        elif q_type == "purpose":
            if scopes:
                intro = (
                    f"The purpose of **{scopes[0]}** "
                    f"{'(defined in `' + files[0] + '`)' if files else ''} "
                    f"can be understood from its implementation:"
                )
            else:
                intro = "Here's what the code is designed to do:"

        elif q_type == "debug":
            intro = (
                "Based on the retrieved code, here are the relevant "
                "components that may be related to the issue:"
            )

        else:
            # "general" and other types
            if keywords and scopes:
                matching = [s for s in scopes if any(k in s.lower() for k in keywords)]
                if matching:
                    intro = (
                        f"Regarding your question about "
                        f"**{'**, **'.join(matching[:2])}** — "
                        f"here's what the code shows:"
                    )
                else:
                    intro = (
                        f"The most relevant code to your question involves "
                        f"**{'**, **'.join(scopes[:3])}**. "
                        f"Here's a breakdown:"
                    )
            elif files:
                intro = (
                    f"Based on the code found in "
                    f"`{'`, `'.join(files[:3])}`, here's what's relevant:"
                )
            else:
                intro = "Here's what the retrieved code reveals:"

        return f"### Answer\n\n{intro}\n"

    def _build_code_sections(
        self,
        chunks: list[dict],
        q_type: str,
        keywords: list[str],
    ) -> str:
        """Build code sections with question-aware annotations."""
        sections: list[str] = []
        sections.append("### Key Code\n")

        for i, c in enumerate(chunks, 1):
            scope = c.get("scope", "")
            title = scope if scope and scope != "<anonymous>" else f"Code section #{i}"
            loc = f" — `{c['file']}`" if c.get("file") else ""

            # Build an explanation line based on what we know
            explanation = self._explain_chunk(c, keywords, q_type)

            section = f"**{i}. {title}**{loc}\n"
            if explanation:
                section += f"\n{explanation}\n"
            if c.get("code"):
                section += f"\n```{c.get('lang', 'python')}\n{c['code']}\n```\n"

            sections.append(section)

        return "\n".join(sections)

    def _explain_chunk(self, chunk: dict, keywords: list[str], q_type: str) -> str:
        """Generate a brief explanation of a chunk's relevance to the question."""
        scope = chunk.get("scope", "")
        code = chunk.get("code", "")
        summary = chunk.get("summary", "")
        file_info = chunk.get("file", "")

        # If there's a summary from the enricher, use it
        if summary:
            return summary

        # Otherwise infer from the code structure
        explanations: list[str] = []

        if not code:
            return ""

        # Detect what the code defines
        if "class " in code:
            class_names = [
                line.split("class ")[1].split("(")[0].split(":")[0].strip()
                for line in code.splitlines()
                if line.strip().startswith("class ")
            ]
            if class_names:
                explanations.append(f"Defines class{'es' if len(class_names) > 1 else ''} `{'`, `'.join(class_names)}`.")

        if "def " in code:
            func_names = [
                line.split("def ")[1].split("(")[0].strip()
                for line in code.splitlines()
                if line.strip().startswith("def ") or line.strip().startswith("async def ")
            ]
            if func_names and not explanations:
                explanations.append(f"Defines `{'`, `'.join(func_names[:3])}`.")

        # Check for keyword matches and note them
        matching_kws = [k for k in keywords if k in code.lower()]
        if matching_kws and len(matching_kws) <= 5:
            explanations.append(
                f"Contains references to: {', '.join(f'`{k}`' for k in matching_kws[:4])}."
            )

        return " ".join(explanations)

    def _build_related_section(self, chunks: list[dict]) -> str:
        """Build a brief 'related code' section for lower-relevance chunks."""
        items: list[str] = []
        for c in chunks:
            scope = c.get("scope", "")
            file_info = c.get("file", "")
            name = f"`{scope}`" if scope and scope != "<anonymous>" else "anonymous block"
            loc = f" in `{file_info}`" if file_info else ""
            items.append(f"- {name}{loc}")

        return "### Also Related\n\n" + "\n".join(items) + "\n"

    def _build_flow_section(self, chunks: list[dict]) -> str:
        """Build an execution/dependency flow from the chunks."""
        steps = []
        for c in chunks:
            scope = c.get("scope", "")
            if scope and scope != "<anonymous>":
                loc = f" (`{c['file']}`)" if c.get("file") else ""
                steps.append(f"- `{scope}`{loc}")

        if len(steps) < 2:
            return ""

        return "### Component Flow\n\n" + "\n".join(steps) + "\n"

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

    if p in ("local", "mock"):
        return MockLLMClient(model=model or "local-synthesizer")

    if p == "ollama":
        # Ollama exposes an OpenAI-compatible API
        return OpenAIClient(
            model=model or "llama3.1",
            api_key="ollama",  # Ollama ignores the key but OpenAI client requires one
            base_url=f"{ollama_base_url.rstrip('/')}/v1",
            provider="ollama",
        )

    raise ValueError(
        f"Unknown LLM provider {provider!r}. Valid choices: 'openai', 'anthropic', 'ollama', 'local'."
    )

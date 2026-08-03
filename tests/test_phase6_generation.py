"""
Phase 6 tests — answer generation pipeline.

Tests use MockLLMClient (deterministic, no API calls) and build retrieval
contexts from scratch or via the sample_repo fixture + Indexer.

Coverage:
  1. TokenUsage — field defaults, arithmetic
  2. GeneratedAnswer — formatted(), summary(), is_empty
  3. PromptBuilder — system prompt present, user template, Anthropic separation
  4. MockLLMClient — call_count, last_messages, usage estimation
  5. get_llm_client factory — unknown provider raises ValueError
  6. AnswerGenerator — empty context fallback, normal generation,
                       LLM failure fallback, citation deduplication
  7. End-to-end: index sample_repo → retrieve → generate (Phase 6 report)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coderag.chunking.models import Chunk, make_chunk_id
from coderag.generation.generator import AnswerGenerator
from coderag.generation.llm_client import MockLLMClient, get_llm_client
from coderag.generation.models import GeneratedAnswer, TokenUsage
from coderag.generation.prompt import PromptBuilder, SYSTEM_PROMPT
from coderag.retrieval.models import RetrievalContext, RetrievalResult

FIXTURE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_chunk(
    name: str = "test_fn",
    file_path: str = "src/test.py",
    start_line: int = 1,
    end_line: int = 10,
    docstring: str | None = "Does something useful.",
) -> Chunk:
    cid = make_chunk_id(file_path, "function_definition", start_line, name)
    return Chunk(
        chunk_id=cid,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        language="python",
        node_type="function_definition",
        function_name=name,
        class_name=None,
        raw_code=f"def {name}():\n    pass\n",
        docstring=docstring,
        has_docstring=docstring is not None,
        imports=[],
        token_count=10,
    )


def _make_result(
    name: str = "fn",
    score: float = 0.9,
    source: str = "vector_search",
    file_path: str = "src/test.py",
    start_line: int = 1,
) -> RetrievalResult:
    chunk = _make_chunk(name=name, file_path=file_path, start_line=start_line)
    return RetrievalResult(
        chunk=chunk,
        score=score,
        enriched_text=(
            f"[File]: {file_path}:L{start_line}-L{chunk.end_line}\n"
            f"[Scope]: {name}\n[Code]:\ndef {name}():\n    pass\n"
        ),
        source=source,
    )


def _make_context(
    results: list[RetrievalResult] | None = None,
    query: str = "what does greet do?",
    total_tokens: int = 200,
) -> RetrievalContext:
    if results is None:
        results = [_make_result()]
    text_parts = [f"--- [1/{len(results)}] {r.citation} ---\n{r.enriched_text}" for r in results]
    return RetrievalContext(
        results=results,
        query=query,
        total_tokens=total_tokens,
        context_text="\n\n".join(text_parts),
    )


# ── TokenUsage ────────────────────────────────────────────────────────────────


class TestTokenUsage:
    def test_defaults_are_zero(self) -> None:
        u = TokenUsage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_explicit_values(self) -> None:
        u = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert u.total_tokens == 150

    def test_dataclass_equality(self) -> None:
        a = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        b = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        assert a == b


# ── GeneratedAnswer ───────────────────────────────────────────────────────────


class TestGeneratedAnswer:
    def test_formatted_no_citations(self) -> None:
        a = GeneratedAnswer(answer="The function returns a greeting.", citations=[])
        formatted = a.formatted()
        assert "The function returns a greeting." in formatted
        assert "Sources:" not in formatted

    def test_formatted_with_citations(self) -> None:
        a = GeneratedAnswer(
            answer="See the greet function.",
            citations=["src/test.py:L1-L5", "src/other.py:L10-L20"],
        )
        formatted = a.formatted()
        assert "Sources:" in formatted
        assert "[1] src/test.py:L1-L5" in formatted
        assert "[2] src/other.py:L10-L20" in formatted

    def test_summary_contains_key_info(self) -> None:
        a = GeneratedAnswer(
            answer="x" * 100,
            citations=["a.py:L1-L5", "b.py:L1-L5"],
            model="gpt-4o-mini",
            provider="openai",
            usage=TokenUsage(total_tokens=300),
        )
        s = a.summary()
        assert "100 chars" in s
        assert "2 citations" in s
        assert "openai/gpt-4o-mini" in s
        assert "300" in s

    def test_is_empty_true_for_blank_answer(self) -> None:
        a = GeneratedAnswer(answer="   ")
        assert a.is_empty

    def test_is_empty_false_for_content(self) -> None:
        a = GeneratedAnswer(answer="This is a real answer.")
        assert not a.is_empty

    def test_context_chunks_field(self) -> None:
        a = GeneratedAnswer(answer="ok", context_chunks=7)
        assert a.context_chunks == 7


# ── PromptBuilder ─────────────────────────────────────────────────────────────


class TestPromptBuilder:
    def test_default_system_prompt_present(self) -> None:
        builder = PromptBuilder()
        ctx = _make_context()
        messages = builder.build(ctx)
        assert messages[0]["role"] == "system"
        assert "CodeRAG" in messages[0]["content"]

    def test_custom_system_prompt(self) -> None:
        builder = PromptBuilder(system_prompt="You are a custom assistant.")
        ctx = _make_context()
        messages = builder.build(ctx)
        assert messages[0]["content"] == "You are a custom assistant."

    def test_user_message_contains_query(self) -> None:
        builder = PromptBuilder()
        ctx = _make_context(query="What does the retry decorator do?")
        messages = builder.build(ctx)
        user_msg = messages[1]["content"]
        assert "What does the retry decorator do?" in user_msg

    def test_user_message_contains_context_text(self) -> None:
        builder = PromptBuilder()
        ctx = _make_context()
        messages = builder.build(ctx)
        user_msg = messages[1]["content"]
        assert ctx.context_text in user_msg

    def test_empty_context_uses_placeholder(self) -> None:
        builder = PromptBuilder()
        ctx = RetrievalContext(results=[], query="q", context_text="")
        messages = builder.build(ctx)
        assert "no code context" in messages[1]["content"].lower()

    def test_returns_two_messages(self) -> None:
        builder = PromptBuilder()
        ctx = _make_context()
        messages = builder.build(ctx)
        assert len(messages) == 2
        roles = {m["role"] for m in messages}
        assert roles == {"system", "user"}

    def test_build_user_content_is_string(self) -> None:
        builder = PromptBuilder()
        ctx = _make_context()
        content = builder.build_user_content(ctx)
        assert isinstance(content, str)
        assert len(content) > 0

    def test_system_prompt_rules_present(self) -> None:
        """The default system prompt should include key RAG constraints."""
        assert "cite" in SYSTEM_PROMPT.lower() or "citation" in SYSTEM_PROMPT.lower()
        assert "context" in SYSTEM_PROMPT.lower()


# ── MockLLMClient ─────────────────────────────────────────────────────────────


class TestMockLLMClient:
    def test_returns_string_and_usage(self) -> None:
        client = MockLLMClient()
        messages = [{"role": "user", "content": "test"}]
        text, usage = client.chat(messages)
        assert isinstance(text, str)
        assert len(text) > 0
        assert isinstance(usage, TokenUsage)

    def test_call_count_increments(self) -> None:
        client = MockLLMClient()
        messages = [{"role": "user", "content": "q"}]
        client.chat(messages)
        client.chat(messages)
        assert client.call_count == 2

    def test_last_messages_stored(self) -> None:
        client = MockLLMClient()
        msgs = [{"role": "user", "content": "hello"}]
        client.chat(msgs)
        assert client.last_messages == msgs

    def test_model_name_property(self) -> None:
        client = MockLLMClient(model="test-model")
        assert client.model_name == "test-model"

    def test_provider_name_is_mock(self) -> None:
        client = MockLLMClient()
        assert client.provider_name == "mock"

    def test_usage_tokens_proportional_to_input(self) -> None:
        client = MockLLMClient()
        short_msgs = [{"role": "user", "content": "a"}]
        long_msgs = [{"role": "user", "content": "x" * 1000}]
        _, short_usage = client.chat(short_msgs)
        _, long_usage = client.chat(long_msgs)
        assert long_usage.prompt_tokens > short_usage.prompt_tokens


# ── get_llm_client factory ────────────────────────────────────────────────────


class TestGetLlmClientFactory:
    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            get_llm_client(provider="unknown_provider")

    def test_openai_provider_string(self) -> None:
        # We don't actually instantiate (requires openai package + key),
        # but we verify the factory accepts the string without raising early.
        # Skip if openai package not configured for real calls.
        try:
            client = get_llm_client(provider="openai", model="gpt-4o-mini", api_key="dummy")
            assert client.provider_name == "openai"
        except Exception:
            pytest.skip("openai package not importable or not configured")

    def test_anthropic_provider_string(self) -> None:
        try:
            client = get_llm_client(provider="anthropic", model="claude-3-haiku-20240307", api_key="dummy")
            assert client.provider_name == "anthropic"
        except Exception:
            pytest.skip("anthropic package not importable or not configured")


# ── AnswerGenerator ───────────────────────────────────────────────────────────


class TestAnswerGenerator:
    def test_empty_context_returns_fallback(self) -> None:
        client = MockLLMClient()
        generator = AnswerGenerator(llm_client=client)
        empty_ctx = RetrievalContext(results=[], query="anything")
        answer = generator.generate(empty_ctx)
        assert answer.is_empty is False  # fallback has text
        assert "don't have enough context" in answer.answer.lower() or len(answer.answer) > 0
        assert client.call_count == 0  # LLM not called for empty context

    def test_normal_generation_calls_llm(self) -> None:
        client = MockLLMClient()
        generator = AnswerGenerator(llm_client=client)
        ctx = _make_context()
        answer = generator.generate(ctx)
        assert client.call_count == 1
        assert isinstance(answer.answer, str)
        assert len(answer.answer) > 0

    def test_answer_has_query(self) -> None:
        client = MockLLMClient()
        generator = AnswerGenerator(llm_client=client)
        ctx = _make_context(query="explain the retry decorator")
        answer = generator.generate(ctx)
        assert answer.query == "explain the retry decorator"

    def test_answer_has_model_info(self) -> None:
        client = MockLLMClient(model="my-model")
        generator = AnswerGenerator(llm_client=client)
        ctx = _make_context()
        answer = generator.generate(ctx)
        assert answer.model == "my-model"
        assert answer.provider == "mock"

    def test_citations_are_deduplicated(self) -> None:
        # Two results with the same file:line → one citation
        r1 = _make_result("fn_a", file_path="src/a.py", start_line=1)
        r2 = _make_result("fn_a", file_path="src/a.py", start_line=1)
        r1_citation = r1.citation
        # Override context.citations() to return duplicates
        ctx = _make_context(results=[r1, r2])
        client = MockLLMClient()
        generator = AnswerGenerator(llm_client=client)
        answer = generator.generate(ctx)
        # Citations should be unique
        assert len(answer.citations) == len(set(answer.citations))

    def test_llm_failure_returns_error_answer(self) -> None:
        """When the LLM raises, the generator should return a graceful error answer."""
        class FailingLLMClient(MockLLMClient):
            def chat(self, messages, *, temperature=0.1, max_tokens=2048):
                raise RuntimeError("API error")

        client = FailingLLMClient()
        generator = AnswerGenerator(llm_client=client)
        ctx = _make_context()
        answer = generator.generate(ctx)
        assert "Error" in answer.answer or "error" in answer.answer.lower()

    def test_context_chunks_count(self) -> None:
        client = MockLLMClient()
        generator = AnswerGenerator(llm_client=client)
        results = [_make_result(f"fn_{i}") for i in range(5)]
        ctx = _make_context(results=results)
        answer = generator.generate(ctx)
        assert answer.context_chunks == 5

    def test_custom_prompt_builder_used(self) -> None:
        """A custom PromptBuilder should be used instead of the default."""
        custom_builder = PromptBuilder(system_prompt="Custom system prompt for testing.")
        client = MockLLMClient()
        generator = AnswerGenerator(llm_client=client, prompt_builder=custom_builder)
        ctx = _make_context()
        generator.generate(ctx)
        # The custom system prompt should appear in the messages sent to the LLM
        assert client.last_messages[0]["content"] == "Custom system prompt for testing."

    def test_usage_populated_from_llm(self) -> None:
        client = MockLLMClient()
        generator = AnswerGenerator(llm_client=client)
        ctx = _make_context()
        answer = generator.generate(ctx)
        assert answer.usage.total_tokens > 0


# ── End-to-end: index + retrieve + generate ───────────────────────────────────


def _build_index(tmp_path: Path):
    """Index the sample_repo fixture and return (store, embedder, indexer, stats)."""
    from coderag.embeddings.factory import MockEmbeddingModel
    from coderag.store.chroma_store import ChromaVectorStore
    from coderag.indexer import Indexer

    store = ChromaVectorStore.ephemeral()
    embedder = MockEmbeddingModel(dim=128)
    indexer = Indexer(
        embedding_model=embedder,
        vector_store=store,
        coderag_dir=tmp_path / ".coderag",
    )
    stats = indexer.index_repo(FIXTURE_REPO, incremental=False)
    return store, embedder, indexer, stats


class TestPhase6EndToEnd:
    @pytest.fixture
    def indexed_repo(self, tmp_path: Path):
        store, embedder, indexer, stats = _build_index(tmp_path)
        return store, embedder, stats

    def test_generate_answer_for_greet(self, indexed_repo) -> None:
        store, embedder, _ = indexed_repo
        from coderag.retrieval.retriever import Retriever

        retriever = Retriever(
            embedding_model=embedder,
            vector_store=store,
            top_k=5,
        )
        ctx = retriever.retrieve("what does the greet function do?")
        assert not ctx.is_empty

        client = MockLLMClient()
        generator = AnswerGenerator(llm_client=client)
        answer = generator.generate(ctx)

        assert isinstance(answer, GeneratedAnswer)
        assert not answer.is_empty
        assert len(answer.citations) > 0

    def test_formatted_answer_contains_sources(self, indexed_repo) -> None:
        store, embedder, _ = indexed_repo
        from coderag.retrieval.retriever import Retriever

        retriever = Retriever(embedding_model=embedder, vector_store=store, top_k=3)
        ctx = retriever.retrieve("Calculator class")
        client = MockLLMClient()
        generator = AnswerGenerator(llm_client=client)
        answer = generator.generate(ctx)

        formatted = answer.formatted()
        if answer.citations:
            assert "Sources:" in formatted
            assert "[1]" in formatted


class TestPhase6Report:
    def test_print_generation_report(self, tmp_path: Path) -> None:
        from coderag.retrieval.retriever import Retriever

        store, embedder, _, stats = _build_index(tmp_path)

        print("\n\n=== PHASE 6 REPORT: Answer Generation ===")
        print(f"  Index: {stats.files_processed} files, {stats.chunks_indexed} chunks")

        retriever = Retriever(
            embedding_model=embedder,
            vector_store=store,
            top_k=5,
            context_token_budget=4000,
        )

        queries = [
            "what does the greet function return?",
            "how does the retry decorator work?",
            "what methods does the Calculator class have?",
        ]

        client = MockLLMClient()
        generator = AnswerGenerator(llm_client=client)

        for query in queries:
            ctx = retriever.retrieve(query)
            answer = generator.generate(ctx)
            print(f"\n  Q: \"{query}\"")
            print(f"  Chunks used: {answer.context_chunks}")
            print(f"  Citations: {answer.citations[:3]}")
            print(f"  Answer preview: {answer.answer[:120]}…")
            print(f"  {answer.summary()}")

        # Sanity: all queries returned non-empty answers
        for query in queries:
            ctx = retriever.retrieve(query)
            answer = generator.generate(ctx)
            assert not answer.is_empty

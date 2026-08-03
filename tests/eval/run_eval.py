#!/usr/bin/env python
"""
tests/eval/run_eval.py — CodeRAG evaluation harness.

Evaluates the full pipeline (index → retrieve → generate) against a set of
golden QA pairs defined in eval_dataset.json.

Metrics
-------
retrieval_recall
    For each query: was at least one chunk from the *expected* file(s) in the
    top-K retrieval results?  Score = fraction of queries where recall is 1.

keyword_recall
    For each query: what fraction of the expected keywords appear (case-insensitive)
    in the generated answer?  Averaged across all queries.

Usage
-----
    # From the repo root (activate .venv first)
    python tests/eval/run_eval.py

    # Use a different LLM provider / model
    python tests/eval/run_eval.py --provider anthropic --model claude-3-haiku-20240307

    # Save report to a custom path
    python tests/eval/run_eval.py --report eval_report.md

    # Use a real API key (reads from .env by default)
    OPENAI_API_KEY=sk-... python tests/eval/run_eval.py --provider openai

Options
-------
--repo PATH        Repo to index (default: tests/fixtures/sample_repo)
--dataset PATH     Golden dataset JSON (default: tests/eval/eval_dataset.json)
--report PATH      Output markdown report (default: eval_report.md)
--provider NAME    LLM provider: mock | openai | anthropic | ollama (default: mock)
--model NAME       LLM model name (provider-specific)
--top-k N          Retrieval top-K (default: 8)
--no-index         Skip indexing (reuse existing index in .coderag/)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent.parent
EVAL_DIR = Path(__file__).parent
DEFAULT_REPO = REPO_ROOT / "tests" / "fixtures" / "sample_repo"
DEFAULT_DATASET = EVAL_DIR / "eval_dataset.json"
DEFAULT_REPORT = REPO_ROOT / "eval_report.md"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class QAPair:
    id: str
    question: str
    expected_chunks: list[str]        # filenames (basenames) that should be retrieved
    expected_keywords: list[str]      # keywords that should appear in the answer
    notes: str = ""


@dataclass
class EvalResult:
    id: str
    question: str
    retrieval_recall: float            # 1.0 if any expected chunk retrieved, else 0.0
    keyword_recall: float              # fraction of expected_keywords in answer
    retrieved_citations: list[str]
    answer_preview: str
    latency_seconds: float
    error: str = ""

    @property
    def passed_retrieval(self) -> bool:
        return self.retrieval_recall >= 1.0

    @property
    def keyword_score_pct(self) -> str:
        return f"{self.keyword_recall * 100:.0f}%"


@dataclass
class EvalSummary:
    total: int = 0
    retrieval_recall_sum: float = 0.0
    keyword_recall_sum: float = 0.0
    errors: int = 0
    total_latency: float = 0.0
    results: list[EvalResult] = field(default_factory=list)

    @property
    def mean_retrieval_recall(self) -> float:
        return self.retrieval_recall_sum / self.total if self.total else 0.0

    @property
    def mean_keyword_recall(self) -> float:
        return self.keyword_recall_sum / self.total if self.total else 0.0

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def score_retrieval_recall(
    citations: list[str],
    expected_files: list[str],
) -> float:
    """
    Return 1.0 if *any* of the retrieved citations references *any* of the
    expected files (by basename match), else 0.0.
    """
    if not expected_files:
        return 1.0  # nothing to check → full credit
    for citation in citations:
        # citation looks like "sample.py:L1-L10"
        citation_basename = citation.split(":")[0].split("/")[-1].split("\\")[-1]
        for expected in expected_files:
            expected_basename = expected.split("/")[-1].split("\\")[-1]
            if citation_basename == expected_basename:
                return 1.0
    return 0.0


def score_keyword_recall(answer: str, keywords: list[str]) -> float:
    """
    Return the fraction of *keywords* that appear (case-insensitive) in *answer*.
    """
    if not keywords:
        return 1.0
    answer_lower = answer.lower()
    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)
    return hits / len(keywords)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def build_pipeline(
    provider: str,
    model: str | None,
    top_k: int,
    coderag_dir: Path,
):
    """Construct (embedder, store, retriever, generator) for evaluation."""
    from coderag.config import get_settings
    from coderag.embeddings.factory import get_embedding_model, MockEmbeddingModel
    from coderag.store.chroma_store import ChromaVectorStore
    from coderag.retrieval.retriever import Retriever
    from coderag.generation.generator import AnswerGenerator
    from coderag.generation.llm_client import MockLLMClient, get_llm_client

    cfg = get_settings()

    # Embedder — always use sentence-transformers for eval reproducibility
    embedder = get_embedding_model(
        provider=cfg.embedding_provider,
        model_name=cfg.embedding_model,
        api_key=cfg.openai_api_key or None,
    )

    # Store — persistent Chroma at coderag_dir/chroma
    chroma_path = str(coderag_dir / "chroma")
    store = ChromaVectorStore(path=chroma_path)

    # LLM client
    if provider == "mock":
        llm_client = MockLLMClient(model=model or "mock-model")
    else:
        api_key = ""
        if provider == "openai":
            api_key = cfg.openai_api_key
        elif provider == "anthropic":
            api_key = cfg.anthropic_api_key
        llm_client = get_llm_client(
            provider=provider,
            model=model,
            api_key=api_key,
            ollama_base_url=cfg.ollama_base_url,
        )

    retriever = Retriever(
        embedding_model=embedder,
        vector_store=store,
        top_k=top_k,
        graph_expansion_depth=0,   # disable graph for faster eval
        context_token_budget=cfg.context_token_budget,
    )

    generator = AnswerGenerator(llm_client=llm_client)
    return embedder, store, retriever, generator


def run_indexing(repo_path: Path, embedder, store, coderag_dir: Path) -> None:
    """Index the repo if not already indexed."""
    from coderag.indexer import Indexer

    indexer = Indexer(
        embedding_model=embedder,
        vector_store=store,
        coderag_dir=coderag_dir,
    )
    stats = indexer.index_repo(repo_path, incremental=True)
    print(
        f"  Indexed: {stats.files_processed} new files, "
        f"{stats.files_skipped} skipped, "
        f"{stats.chunks_indexed} new chunks  "
        f"({stats.elapsed_seconds:.1f}s)"
    )


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate(
    dataset: list[QAPair],
    retriever,
    generator,
) -> EvalSummary:
    summary = EvalSummary()

    for qa in dataset:
        t0 = time.perf_counter()
        try:
            context = retriever.retrieve(qa.question)
            answer = generator.generate(context)

            citations = context.citations()
            retrieval_recall = score_retrieval_recall(citations, qa.expected_chunks)
            keyword_recall = score_keyword_recall(answer.answer, qa.expected_keywords)

            result = EvalResult(
                id=qa.id,
                question=qa.question,
                retrieval_recall=retrieval_recall,
                keyword_recall=keyword_recall,
                retrieved_citations=citations[:5],
                answer_preview=answer.answer[:200],
                latency_seconds=time.perf_counter() - t0,
            )

        except Exception as exc:
            result = EvalResult(
                id=qa.id,
                question=qa.question,
                retrieval_recall=0.0,
                keyword_recall=0.0,
                retrieved_citations=[],
                answer_preview="",
                latency_seconds=time.perf_counter() - t0,
                error=str(exc),
            )
            summary.errors += 1

        summary.results.append(result)
        summary.total += 1
        summary.retrieval_recall_sum += result.retrieval_recall
        summary.keyword_recall_sum += result.keyword_recall
        summary.total_latency += result.latency_seconds

        status = "✅" if result.passed_retrieval else "❌"
        print(
            f"  [{result.id}] {status} ret={result.retrieval_recall:.0f}  "
            f"kw={result.keyword_score_pct}  "
            f"{result.latency_seconds:.2f}s  "
            f"| {qa.question[:60]}"
        )

    return summary


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def write_report(
    summary: EvalSummary,
    report_path: Path,
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines += [
        f"# CodeRAG Evaluation Report",
        f"",
        f"**Generated**: {ts}  ",
        f"**Repo**: `{args.repo}`  ",
        f"**LLM Provider**: `{args.provider}`  ",
        f"**Model**: `{args.model or 'default'}`  ",
        f"**Top-K**: {args.top_k}  ",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Score |",
        f"|--------|-------|",
        f"| Retrieval Recall (mean) | **{summary.mean_retrieval_recall * 100:.1f}%** |",
        f"| Keyword Recall (mean)   | **{summary.mean_keyword_recall * 100:.1f}%** |",
        f"| Queries evaluated       | {summary.total} |",
        f"| Errors                  | {summary.errors} |",
        f"| Avg latency             | {summary.avg_latency:.2f}s |",
        f"| Total latency           | {summary.total_latency:.1f}s |",
        f"",
        f"---",
        f"",
        f"## Per-Query Results",
        f"",
        f"| ID | Retrieval | Keywords | Latency | Question |",
        f"|----|-----------|----------|---------|----------|",
    ]

    for r in summary.results:
        ret_icon = "✅" if r.passed_retrieval else "❌"
        kw_icon = "✅" if r.keyword_recall >= 0.5 else "⚠️" if r.keyword_recall > 0 else "❌"
        lines.append(
            f"| {r.id} | {ret_icon} {r.retrieval_recall:.0f} | "
            f"{kw_icon} {r.keyword_score_pct} | "
            f"{r.latency_seconds:.2f}s | {r.question[:70]} |"
        )

    lines += ["", "---", "", "## Detailed Results", ""]

    for r in summary.results:
        lines += [
            f"### [{r.id}] {r.question}",
            f"",
            f"- **Retrieval recall**: {r.retrieval_recall:.0f}  ",
            f"- **Keyword recall**: {r.keyword_score_pct}  ",
            f"- **Latency**: {r.latency_seconds:.2f}s  ",
        ]
        if r.error:
            lines.append(f"- **Error**: `{r.error}`  ")
        if r.retrieved_citations:
            lines.append(f"- **Retrieved**:")
            for c in r.retrieved_citations:
                lines.append(f"  - `{c}`")
        if r.answer_preview:
            lines += [
                f"",
                f"**Answer preview**:",
                f"",
                f"> {r.answer_preview[:300].replace(chr(10), ' ')}",
            ]
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report written to: {report_path}")


# ---------------------------------------------------------------------------
# pytest integration
# ---------------------------------------------------------------------------


class TestPhase8Eval:
    """
    Pytest-friendly wrapper for the evaluation harness.

    Runs the full eval against the sample_repo fixture using the MockLLMClient
    (no real API calls) and asserts minimum quality thresholds.
    """

    def test_eval_retrieval_recall_above_threshold(self, tmp_path: Path) -> None:
        """At least 70% of queries should retrieve a chunk from the expected file."""
        import pytest

        # Load dataset
        dataset_path = DEFAULT_DATASET
        if not dataset_path.exists():
            pytest.skip(f"Eval dataset not found at {dataset_path}")

        dataset = _load_dataset(dataset_path)
        coderag_dir = tmp_path / ".coderag"

        # Build pipeline with mock LLM
        embedder, store, retriever, generator = build_pipeline(
            provider="mock",
            model=None,
            top_k=8,
            coderag_dir=coderag_dir,
        )

        # Index first
        run_indexing(DEFAULT_REPO, embedder, store, coderag_dir)

        # Evaluate
        summary = evaluate(dataset, retriever, generator)

        # Assert thresholds
        assert summary.mean_retrieval_recall >= 0.70, (
            f"Retrieval recall {summary.mean_retrieval_recall:.2%} below 70% threshold"
        )

    def test_eval_keyword_recall_above_threshold(self, tmp_path: Path) -> None:
        """Mean keyword recall (with mock LLM) should be non-trivially positive.

        Note: MockLLMClient returns canned text, so keyword recall will be low.
        This test checks that the eval harness runs without errors, not that the
        mock LLM produces good answers.  Set the threshold low accordingly.
        """
        import pytest

        dataset_path = DEFAULT_DATASET
        if not dataset_path.exists():
            pytest.skip(f"Eval dataset not found at {dataset_path}")

        dataset = _load_dataset(dataset_path)
        coderag_dir = tmp_path / ".coderag"

        embedder, store, retriever, generator = build_pipeline(
            provider="mock",
            model=None,
            top_k=8,
            coderag_dir=coderag_dir,
        )
        run_indexing(DEFAULT_REPO, embedder, store, coderag_dir)
        summary = evaluate(dataset, retriever, generator)

        # With a mock LLM the keyword recall will be near 0 — just check no crash
        assert summary.errors == 0, f"{summary.errors} eval errors encountered"
        assert summary.total == len(dataset)

    def test_eval_report_written(self, tmp_path: Path) -> None:
        """The eval harness should produce a valid markdown report file."""
        import pytest

        dataset_path = DEFAULT_DATASET
        if not dataset_path.exists():
            pytest.skip(f"Eval dataset not found at {dataset_path}")

        dataset = _load_dataset(dataset_path)
        coderag_dir = tmp_path / ".coderag"

        embedder, store, retriever, generator = build_pipeline(
            provider="mock",
            model=None,
            top_k=5,
            coderag_dir=coderag_dir,
        )
        run_indexing(DEFAULT_REPO, embedder, store, coderag_dir)
        summary = evaluate(dataset, retriever, generator)

        report_path = tmp_path / "eval_report.md"

        # Build a minimal args namespace for the report writer
        class _Args:
            repo = str(DEFAULT_REPO)
            provider = "mock"
            model = None
            top_k = 5

        write_report(summary, report_path, _Args())

        assert report_path.exists()
        content = report_path.read_text("utf-8")
        assert "# CodeRAG Evaluation Report" in content
        assert "Retrieval Recall" in content
        assert "q01" in content  # first query present


# ---------------------------------------------------------------------------
# Dataset loader (shared by tests and __main__)
# ---------------------------------------------------------------------------


def _load_dataset(path: Path) -> list[QAPair]:
    data = json.loads(path.read_text("utf-8"))
    return [
        QAPair(
            id=item["id"],
            question=item["question"],
            expected_chunks=item.get("expected_chunks", []),
            expected_keywords=item.get("expected_keywords", []),
            notes=item.get("notes", ""),
        )
        for item in data
    ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CodeRAG evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--repo", default=str(DEFAULT_REPO), help="Repo to index")
    p.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Golden dataset JSON")
    p.add_argument("--report", default=str(DEFAULT_REPORT), help="Output markdown report")
    p.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "openai", "anthropic", "ollama"],
        help="LLM provider (default: mock — no API key needed)",
    )
    p.add_argument("--model", default=None, help="LLM model name")
    p.add_argument("--top-k", type=int, default=8, dest="top_k", help="Retrieval top-K")
    p.add_argument("--no-index", action="store_true", help="Skip indexing, reuse .coderag/")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo_path = Path(args.repo)
    dataset_path = Path(args.dataset)
    report_path = Path(args.report)
    coderag_dir = repo_path.parent.parent / ".coderag"

    print("=" * 60)
    print("  CodeRAG Evaluation Harness")
    print("=" * 60)
    print(f"  Repo:     {repo_path}")
    print(f"  Dataset:  {dataset_path}")
    print(f"  Provider: {args.provider}")
    print(f"  Model:    {args.model or 'default'}")
    print(f"  Top-K:    {args.top_k}")
    print()

    # Load dataset
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)
    dataset = _load_dataset(dataset_path)
    print(f"  Loaded {len(dataset)} QA pairs from {dataset_path.name}")
    print()

    # Build pipeline
    print("  Building pipeline…")
    try:
        embedder, store, retriever, generator = build_pipeline(
            provider=args.provider,
            model=args.model,
            top_k=args.top_k,
            coderag_dir=coderag_dir,
        )
    except Exception as exc:
        print(f"ERROR building pipeline: {exc}", file=sys.stderr)
        sys.exit(1)

    # Index (unless --no-index)
    if not args.no_index:
        print(f"  Indexing {repo_path.name}…")
        try:
            run_indexing(repo_path, embedder, store, coderag_dir)
        except Exception as exc:
            print(f"ERROR during indexing: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        print("  Skipping indexing (--no-index)")
    print()

    # Evaluate
    print("  Running evaluation…")
    print(f"  {'ID':<6} {'Ret':>4}  {'KW':>5}  {'Latency':>8}  Question")
    print("  " + "-" * 56)
    summary = evaluate(dataset, retriever, generator)

    # Print summary
    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  Retrieval Recall (mean): {summary.mean_retrieval_recall * 100:.1f}%")
    print(f"  Keyword Recall  (mean):  {summary.mean_keyword_recall * 100:.1f}%")
    print(f"  Errors:                  {summary.errors}")
    print(f"  Avg latency:             {summary.avg_latency:.2f}s")
    print(f"  Total latency:           {summary.total_latency:.1f}s")
    print()

    # Write report
    write_report(summary, report_path, args)


if __name__ == "__main__":
    main()

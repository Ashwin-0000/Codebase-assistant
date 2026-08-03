"""
CodeRAG CLI — fully wired entry point.

Commands
--------
  index    — Walk, parse, chunk, graph-build, enrich, embed, and store a repo.
  ask      — Query the index with a natural-language question.
  reindex  — Incremental update: re-processes only changed files.
  status   — Show index statistics without re-indexing.

All heavy lifting is delegated to the underlying pipeline modules (Phases 1–6).
The CLI's job is to read settings, construct the right objects, call the
pipeline, and present results clearly with Rich formatting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

from coderag import __version__
from coderag.config import get_settings
from coderag.logging_config import configure_logging

# Module-level factory imports — kept here (not inside functions) so that the
# test suite can patch them via ``patch("coderag.cli.get_embedding_model", ...)``.
from coderag.embeddings.factory import get_embedding_model  # noqa: E402
from coderag.store.factory import get_vector_store           # noqa: E402
from coderag.generation.llm_client import get_llm_client    # noqa: E402
from coderag.retrieval.reranker import get_reranker          # noqa: E402

app = typer.Typer(
    name="coderag",
    help="Codebase RAG assistant — index a repo and ask questions about it.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()
err_console = Console(stderr=True)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def _setup() -> None:
    """Bootstrap logging from the current settings."""
    cfg = get_settings()
    configure_logging(level=cfg.log_level, log_file=cfg.log_file)


def _coderag_dir() -> Path:
    """Return the .coderag working directory (from settings)."""
    return Path(".coderag")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="Show version and exit."),
) -> None:
    """CodeRAG — answer natural-language questions about a codebase."""
    _setup()
    if version:
        console.print(f"[bold cyan]coderag[/] version [bold]{__version__}[/]")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@app.command()
def index(
    repo: str = typer.Argument(..., help="Local path or GitHub URL of the repository to index."),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Re-index every file even if its hash is unchanged (disables incremental).",
    ),
    embed_provider: Optional[str] = typer.Option(
        None, "--embed-provider",
        help="Override embedding provider (sentence_transformers / openai).",
    ),
    embed_model: Optional[str] = typer.Option(
        None, "--embed-model",
        help="Override embedding model name.",
    ),
) -> None:
    """[bold]Index[/] a repository — walk, parse, chunk, embed, and store.

    On subsequent runs only files whose SHA-256 hash has changed are
    re-processed (pass [cyan]--force[/] to override).
    """
    _setup()
    cfg = get_settings()

    # --- Resolve the repo path (local or GitHub URL) ---
    from coderag.ingestion.repo import resolve_repo

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Resolving repository…", total=None)
            repo_path = resolve_repo(repo)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        err_console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=1)

    console.print(f"[dim]Repository:[/] {repo_path}")

    # --- Build pipeline components from settings ---
    provider = embed_provider or cfg.embedding_provider
    model = embed_model or cfg.embedding_model
    chroma_path = str(cfg.chroma_path)
    coderag_dir = _coderag_dir()

    try:
        from coderag.indexer import Indexer

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(f"Loading embedding model [cyan]{provider}/{model}[/]…", total=None)
            embedder = get_embedding_model(
                provider=provider,
                model_name=model,
                api_key=cfg.openai_api_key or None,
            )

        store = get_vector_store(
            backend=cfg.vector_store,
            path=chroma_path,
        )

        indexer = Indexer(
            embedding_model=embedder,
            vector_store=store,
            max_tokens=cfg.chunk_max_tokens,
            overlap_tokens=cfg.chunk_overlap_tokens,
            coderag_dir=coderag_dir,
        )

    except Exception as exc:
        err_console.print(f"[bold red]Setup error:[/] {exc}")
        logger.exception("Indexer setup failed")
        raise typer.Exit(code=1)

    # --- Run the indexing pipeline ---
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Indexing [cyan]{repo_path.name}[/] ({'full' if force else 'incremental'})…",
            total=None,
        )
        try:
            stats = indexer.index_repo(repo_path, incremental=not force)
            progress.update(task, completed=True)
        except Exception as exc:
            progress.stop()
            err_console.print(f"[bold red]Indexing error:[/] {exc}")
            logger.exception("index_repo failed")
            raise typer.Exit(code=1)

    # --- Print results table ---
    console.print()
    _print_index_stats(stats)

    if stats.errors:
        console.print(
            f"\n[yellow]⚠  {len(stats.errors)} file(s) had errors:[/]"
        )
        for err in stats.errors[:10]:
            console.print(f"   • {err}")
        if len(stats.errors) > 10:
            console.print(f"   … and {len(stats.errors) - 10} more (see logs)")

    if stats.files_failed > 0:
        raise typer.Exit(code=2)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question about the codebase."),
    top_k: int = typer.Option(0, "--top-k", "-k", help="Override top-K retrieval (0 = use config)."),
    no_graph: bool = typer.Option(False, "--no-graph", help="Disable call-graph expansion."),
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p",
        help="Override LLM provider (openai / anthropic / ollama).",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m",
        help="Override LLM model name.",
    ),
    show_context: bool = typer.Option(
        False, "--show-context",
        help="Print the raw retrieval context before the answer.",
    ),
) -> None:
    """[bold]Ask[/] a natural-language question about the indexed codebase.

    Runs the full retrieval → generation pipeline and prints the answer
    with source citations.
    """
    _setup()
    cfg = get_settings()
    coderag_dir = _coderag_dir()

    # --- Check that an index exists ---
    manifest_path = coderag_dir / "manifest.json"
    if not manifest_path.exists():
        err_console.print(
            "[bold red]No index found.[/]  "
            "Run [cyan]coderag index <repo>[/] first."
        )
        raise typer.Exit(code=1)

    # --- Load manifest to check it's non-empty ---
    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        manifest = {}

    if not manifest:
        err_console.print(
            "[bold red]Index is empty.[/]  "
            "Run [cyan]coderag index <repo>[/] to populate it."
        )
        raise typer.Exit(code=1)

    # --- Build components ---
    llm_provider = provider or cfg.llm_provider
    llm_model = model or cfg.llm_model
    k = top_k or cfg.top_k

    try:
        from coderag.generation.generator import AnswerGenerator
        from coderag.retrieval.retriever import Retriever

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task("Loading models…", total=None)

            embedder = get_embedding_model(
                provider=cfg.embedding_provider,
                model_name=cfg.embedding_model,
                api_key=cfg.openai_api_key or None,
            )
            store = get_vector_store(
                backend=cfg.vector_store,
                path=str(cfg.chroma_path),
            )
            llm_client = get_llm_client(
                provider=llm_provider,
                model=llm_model,
                api_key=_pick_api_key(llm_provider, cfg),
                ollama_base_url=cfg.ollama_base_url,
            )

        # Load the call graph (optional)
        graph_analyzer = None
        if not no_graph:
            graph_analyzer = _load_graph_analyzer(coderag_dir, store)

        reranker = get_reranker(
            enabled=cfg.reranking_enabled,
            model_name=cfg.reranker_model,
        )

        retriever = Retriever(
            embedding_model=embedder,
            vector_store=store,
            graph_analyzer=graph_analyzer,
            reranker=reranker,
            top_k=k,
            graph_expansion_depth=cfg.graph_expansion_depth if not no_graph else 0,
            context_token_budget=cfg.context_token_budget,
        )

        generator = AnswerGenerator(
            llm_client=llm_client,
        )

    except Exception as exc:
        err_console.print(f"[bold red]Setup error:[/] {exc}")
        logger.exception("ask setup failed")
        raise typer.Exit(code=1)

    # --- Retrieve ---
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Searching index…", total=None)
        try:
            context = retriever.retrieve(question)
        except Exception as exc:
            err_console.print(f"[bold red]Retrieval error:[/] {exc}")
            logger.exception("retrieve failed")
            raise typer.Exit(code=1)

    if context.is_empty:
        console.print(
            Panel(
                "[yellow]No relevant code found in the index.[/]\n"
                "Try rephrasing your question or re-indexing with more files.",
                title="[bold yellow]No Results[/]",
                border_style="yellow",
            )
        )
        raise typer.Exit(code=0)

    if show_context:
        console.print(Panel(
            context.context_text,
            title=f"[dim]Retrieval Context[/] ({context.chunk_count} chunk(s), {context.total_tokens} tokens)",
            border_style="dim",
        ))

    # --- Generate ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(
            f"Generating answer with [cyan]{llm_provider}/{llm_model}[/]…",
            total=None,
        )
        try:
            answer = generator.generate(context)
        except Exception as exc:
            err_console.print(f"[bold red]Generation error:[/] {exc}")
            logger.exception("generate failed")
            raise typer.Exit(code=1)

    # --- Print answer ---
    _print_answer(answer, context)


@app.command()
def reindex(
    repo: str = typer.Argument(..., help="Local path or GitHub URL of the repository."),
    embed_provider: Optional[str] = typer.Option(
        None, "--embed-provider", help="Override embedding provider.",
    ),
    embed_model: Optional[str] = typer.Option(
        None, "--embed-model", help="Override embedding model name.",
    ),
) -> None:
    """[bold]Incrementally re-index[/] a repository — only changed files are re-processed.

    Uses SHA-256 file hashing to detect changes.  Pass the same repo path
    used during the original [cyan]coderag index[/] run.
    """
    # Reindex is identical to index with incremental=True (which is the default).
    # We expose it as a separate sub-command for discoverability and clarity.
    ctx = typer.get_current_context()
    ctx.invoke(
        index,
        repo=repo,
        force=False,
        embed_provider=embed_provider,
        embed_model=embed_model,
    )


@app.command()
def status() -> None:
    """[bold]Show index statistics[/] — chunk count, file count, last indexed."""
    _setup()
    coderag_dir = _coderag_dir()
    manifest_path = coderag_dir / "manifest.json"
    graph_path = coderag_dir / "graph.json"

    if not manifest_path.exists():
        console.print(
            "[yellow]No index found.[/]  Run [cyan]coderag index <repo>[/] to create one."
        )
        raise typer.Exit(code=0)

    try:
        manifest = json.loads(manifest_path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        err_console.print(f"[bold red]Could not read manifest:[/] {exc}")
        raise typer.Exit(code=1)

    total_files = len(manifest)
    total_chunks = sum(
        entry.get("chunk_count", len(entry.get("chunk_ids", [])))
        for entry in manifest.values()
    )

    # Newest indexed_at timestamp
    timestamps = [
        e.get("indexed_at", "") for e in manifest.values() if e.get("indexed_at")
    ]
    last_indexed = max(timestamps) if timestamps else "unknown"

    table = Table(
        title="[bold]CodeRAG Index Status[/]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")

    table.add_row("Indexed files", str(total_files))
    table.add_row("Total chunks", str(total_chunks))
    table.add_row("Last indexed", last_indexed[:19].replace("T", " ") if last_indexed != "unknown" else "unknown")
    table.add_row("Manifest path", str(manifest_path))
    table.add_row("Graph present", "✓ yes" if graph_path.exists() else "✗ no")
    table.add_row("Chroma path", str(coderag_dir / "chroma"))

    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _print_index_stats(stats: object) -> None:
    """Render a Rich table of IndexStats."""
    table = Table(
        title="[bold]Indexing Complete[/]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Files indexed", str(stats.files_processed))        # type: ignore[attr-defined]
    table.add_row("Files skipped (unchanged)", str(stats.files_skipped))  # type: ignore[attr-defined]
    table.add_row("Files failed", str(stats.files_failed))             # type: ignore[attr-defined]
    table.add_row("Chunks indexed", str(stats.chunks_indexed))         # type: ignore[attr-defined]
    table.add_row("Chunks deleted (stale)", str(stats.chunks_deleted)) # type: ignore[attr-defined]
    table.add_row("Elapsed", f"{stats.elapsed_seconds:.1f}s")          # type: ignore[attr-defined]

    console.print(table)


def _print_answer(answer: object, context: object) -> None:
    """Render a Rich panel for the generated answer + citations."""
    from coderag.generation.models import GeneratedAnswer
    from coderag.retrieval.models import RetrievalContext

    assert isinstance(answer, GeneratedAnswer)
    assert isinstance(context, RetrievalContext)

    # Answer panel
    console.print(
        Panel(
            answer.answer,
            title="[bold green]Answer[/]",
            border_style="green",
            padding=(1, 2),
        )
    )

    # Citations
    if answer.citations:
        console.print()
        cit_table = Table(
            title="[bold]Sources[/]",
            box=box.SIMPLE,
            show_header=True,
            header_style="bold dim",
        )
        cit_table.add_column("#", style="dim", width=4)
        cit_table.add_column("File : Lines", style="cyan")
        for i, cite in enumerate(answer.citations, 1):
            cit_table.add_row(str(i), cite)
        console.print(cit_table)

    # Footer
    console.print(
        f"\n[dim]  {context.chunk_count} chunk(s) · "
        f"{context.total_tokens} context tokens · "
        f"model: {answer.provider}/{answer.model} · "
        f"tokens used: {answer.usage.total_tokens}[/]"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pick_api_key(provider: str, cfg: object) -> str:
    """Return the right API key for *provider* from settings."""
    p = provider.lower()
    if p == "openai":
        return cfg.openai_api_key  # type: ignore[attr-defined]
    if p == "anthropic":
        return cfg.anthropic_api_key  # type: ignore[attr-defined]
    return ""  # Ollama / local


def _load_graph_analyzer(coderag_dir: Path, store: object):
    """
    Load the persisted call graph and build a CallGraphAnalyzer.

    Returns None if the graph file is missing or the store is empty.
    """
    graph_path = coderag_dir / "graph.json"
    if not graph_path.exists():
        logger.debug("No graph.json found — skipping graph expansion")
        return None

    try:
        from coderag.graph.builder import GraphBuilder
        from coderag.graph.analyzer import CallGraphAnalyzer
        from coderag.store.base import VectorStore

        data = json.loads(graph_path.read_text("utf-8"))
        graph = GraphBuilder.from_node_link(data)

        if graph.number_of_nodes() == 0:
            return None

        # Build the chunk_map needed by CallGraphAnalyzer from the store
        assert isinstance(store, VectorStore)
        chunk_map = _build_chunk_map_from_store(store)

        if not chunk_map:
            return None

        analyzer = CallGraphAnalyzer(graph, chunk_map)
        logger.info(
            "Call graph loaded: %d nodes, %d edges",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return analyzer

    except Exception as exc:
        logger.warning("Could not load call graph: %s", exc)
        return None


def _build_chunk_map_from_store(store: object) -> dict:
    """
    Build a {chunk_id: Chunk} mapping by querying all chunks from the store.

    We use a broad zero-vector query to pull chunks out, which works because
    the store's ``query()`` method returns whichever chunks are closest to the
    query vector — using a zero vector gives a representative cross-section.
    This is a best-effort approach; graph expansion will only work for chunks
    that appear in these results.
    """
    from coderag.store.base import VectorStore

    assert isinstance(store, VectorStore)

    try:
        count = store.count()
        if count == 0:
            return {}

        # Pull up to 2000 chunks using a zero vector to get a broad sample
        dim_probe: list[float] = []
        sample = store.query(embedding=dim_probe or [0.0] * 384, top_k=min(count, 2000))
        return {sr.chunk.chunk_id: sr.chunk for sr in sample}
    except Exception as exc:
        logger.warning("Could not build chunk map: %s", exc)
        return {}




@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-H", help="Host to bind."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on."),
    reload: bool = typer.Option(False, "--reload", help="Enable hot-reload (dev mode)."),
    no_open: bool = typer.Option(False, "--no-open", help="Do not open the browser automatically."),
) -> None:
    """[bold]Serve[/] the CodeRAG Web UI.

    Starts a local uvicorn server and (by default) opens the browser.
    Requires: [cyan]pip install 'coderag\\[web\\]'[/]
    """
    _setup()

    try:
        import uvicorn  # noqa: F401 — just checking it's installed
    except ImportError:
        err_console.print(
            "[bold red]Error:[/] The web extras are not installed.\n"
            "Run: [cyan]pip install 'coderag\\[web\\]'[/]"
        )
        raise typer.Exit(code=1)

    url = f"http://{host}:{port}"
    console.print(
        Panel(
            f"[bold cyan]CodeRAG Web UI[/]\n\n"
            f"  [dim]URL:[/]  [link={url}]{url}[/link]\n"
            f"  [dim]API:[/]  [link={url}/api/docs]{url}/api/docs[/link]\n\n"
            f"[dim]Press Ctrl+C to stop[/]",
            border_style="bright_blue",
        )
    )

    if not no_open:
        import threading
        import webbrowser

        def _open_browser() -> None:
            import time
            time.sleep(1.5)          # let the server start first
            webbrowser.open(url)

        threading.Thread(target=_open_browser, daemon=True).start()

    import uvicorn as _uvicorn

    _uvicorn.run(
        "coderag.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    app()

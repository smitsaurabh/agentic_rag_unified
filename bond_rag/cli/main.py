"""
Bond RAG — Click CLI.

Commands
────────
  ingest    Ingest PDF(s) into the vector store
  query     One-shot question with sources
  chat      Interactive multi-turn session
  list      Show indexed ISINs and files
  status    Store / registry statistics
  remove    Remove a file's chunks from the store
  models    List locally available Ollama models

Examples
────────
  bond-rag ingest data/bond_2024.pdf
  bond-rag ingest data/ --all
  bond-rag query "What is the coupon rate for XS1234567890?"
  bond-rag query "Redemption terms" --isin XS1234567890 --sources
  bond-rag chat --isin XS1234567890
  bond-rag list
  bond-rag status
  bond-rag remove bond_2024.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def _get_pipeline():
    """Lazy import — keeps CLI startup fast."""
    from bond_rag.rag.pipeline import BondRAGPipeline
    return BondRAGPipeline()


# ── Root group ─────────────────────────────────────────────────────────────────

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="bond-rag")
def cli() -> None:
    """Bond RAG — ISIN-aware RAG for financial bond PDFs."""


# ── ingest ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--all", "ingest_all", is_flag=True,
              help="Ingest every PDF in PATH (requires PATH to be a directory).")
@click.option("--force", is_flag=True, help="Re-ingest even if already loaded.")
def ingest(path: str, ingest_all: bool, force: bool) -> None:
    """Ingest a PDF file or directory of PDFs."""
    rag = _get_pipeline()
    p   = Path(path)

    if p.is_dir():
        if not ingest_all:
            console.print("[yellow]PATH is a directory. Pass --all to ingest all PDFs.[/yellow]")
            sys.exit(1)
        results = rag.ingest_directory(p, force=force)
    else:
        results = [rag.ingest(p, force=force)]

    t = Table(title="Ingestion Results", show_lines=True)
    t.add_column("File",         style="cyan")
    t.add_column("Pages",        justify="right")
    t.add_column("Chunks",       justify="right", style="green")
    t.add_column("ISINs",        style="magenta")
    t.add_column("Status")

    for r in results:
        _status_str = {
            "loaded":         "[green]✓ loaded[/green]",
            "already_loaded": "[yellow]already loaded[/yellow]",
            "skipped":        "[dim]skipped[/dim]",
        }.get(r.status, r.status)

        t.add_row(
            r.filename,
            str(r.total_pages),
            str(r.chunks_added),
            ", ".join(r.isins) or "—",
            _status_str,
        )

    console.print(t)
    rag.close()


# ── query ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("question", nargs=-1, required=True)
@click.option("--top-k",  default=6,      help="Chunks to retrieve.")
@click.option("--isin",   multiple=True,  help="Restrict to ISIN(s).")
@click.option("--sources", is_flag=True,  help="Print retrieved source chunks.")
@click.option("--stream",  is_flag=True,  help="Stream answer token by token.")
def query(
    question: tuple,
    top_k: int,
    isin: tuple,
    sources: bool,
    stream: bool,
) -> None:
    """Ask a one-shot question against the indexed documents."""
    q   = " ".join(question)
    rag = _get_pipeline()
    console.rule("[bold]Question[/bold]")
    console.print(f"[bold cyan]{q}[/bold cyan]\n")

    isins: Optional[list[str]] = list(isin) if isin else None

    if stream:
        console.print("[bold green]Answer[/bold green]")
        for token in rag.stream_query(q, top_k=top_k, force_isins=isins):
            print(token, end="", flush=True)
        print()
    else:
        result = rag.query(q, top_k=top_k, force_isins=isins)
        console.print(Panel(result.answer, title="[bold green]Answer[/bold green]", expand=False))
        if sources:
            _print_sources(result.sources)

    rag.close()


# ── chat ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--top-k", default=6)
@click.option("--isin", multiple=True, help="Restrict session to these ISIN(s).")
def chat(top_k: int, isin: tuple) -> None:
    """Interactive multi-turn chat session."""
    rag   = _get_pipeline()
    isins: Optional[list[str]] = list(isin) if isin else None

    console.rule("[bold]Bond RAG — Chat[/bold]")
    if isins:
        console.print(f"[yellow]ISIN filter: {isins}[/yellow]")
    console.print("Type your question and press Enter. Type [bold]exit[/bold] to quit.\n")

    history: list[tuple[str, str]] = []

    while True:
        try:
            q = console.input("[bold cyan]You:[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye.[/dim]")
            break

        enriched = _enrich_with_history(q, history)
        result   = rag.query(enriched, top_k=top_k, force_isins=isins)
        history.append((q, result.answer))

        console.print()
        console.print(Panel(result.answer, title="[bold green]Assistant[/bold green]", expand=False))
        _print_citations(result.sources)
        console.print()

    rag.close()


# ── list ───────────────────────────────────────────────────────────────────────

@cli.command("list")
def list_cmd() -> None:
    """Show all indexed ISINs and ingested files."""
    rag = _get_pipeline()

    isins = rag.list_isins()
    if isins:
        t = Table(title="Indexed ISINs", show_lines=True)
        t.add_column("#", justify="right", style="dim")
        t.add_column("ISIN", style="magenta bold")
        for i, isin in enumerate(isins, 1):
            t.add_row(str(i), isin)
        console.print(t)
    else:
        console.print("[yellow]No ISINs indexed yet.[/yellow]")

    console.print()

    files = rag.list_files()
    if files:
        tf = Table(title="Ingested Files", show_lines=True)
        tf.add_column("Filename",    style="cyan")
        tf.add_column("Pages",       justify="right")
        tf.add_column("Chunks",      justify="right")
        tf.add_column("ISINs",       style="magenta")
        tf.add_column("Ingested At", style="dim")
        for f in files:
            tf.add_row(
                f.get("filename", "?"),
                str(f.get("total_pages", "?")),
                str(f.get("chunks", "?")),
                ", ".join(f.get("isins", [])) or "—",
                (f.get("ingested_at") or "")[:19],
            )
        console.print(tf)
    else:
        console.print("[yellow]No files ingested yet.[/yellow]")

    rag.close()


# ── status ─────────────────────────────────────────────────────────────────────

@cli.command()
def status() -> None:
    """Show vector store and registry statistics."""
    rag   = _get_pipeline()
    stats = rag.stats()
    console.print(Panel(
        "\n".join([
            f"[bold]Files indexed:[/bold]      {stats.get('total_files', 0)}",
            f"[bold]Total pages:[/bold]        {stats.get('total_pages', 0)}",
            f"[bold]Registry chunks:[/bold]    {stats.get('total_chunks', 0)}",
            f"[bold]Vector store chunks:[/bold]{stats.get('vector_store_chunks', 0)}",
            f"[bold]Distinct ISINs:[/bold]     {stats.get('total_isins', 0)}",
        ]),
        title="[bold]System Status[/bold]",
        expand=False,
    ))
    rag.close()


# ── remove ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("filename")
@click.confirmation_option(prompt="Remove this file's chunks from the store?")
def remove(filename: str) -> None:
    """Remove all chunks for FILENAME from the store."""
    rag = _get_pipeline()
    n   = rag.remove(filename)
    console.print(f"[green]Removed {n} chunks for '{filename}'.[/green]")
    rag.close()


# ── models ─────────────────────────────────────────────────────────────────────

@cli.command()
def models() -> None:
    """List locally available Ollama models."""
    rag   = _get_pipeline()
    names = rag.llm.list_local_models()
    t = Table(title="Available Ollama Models")
    t.add_column("Model", style="cyan")
    for name in names:
        t.add_row(name)
    console.print(t)
    rag.close()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_sources(chunks) -> None:  # type: ignore[no-untyped-def]
    if not chunks:
        return
    console.rule("[dim]Sources[/dim]")
    for i, c in enumerate(chunks, 1):
        console.print(
            f"[dim][{i}] {c.citation()}  score={c.final_score:.3f}[/dim]\n"
            f"[dim]{c.text[:300]}…[/dim]\n"
        )


def _print_citations(chunks) -> None:  # type: ignore[no-untyped-def]
    if not chunks:
        return
    cites = " | ".join(c.citation() for c in chunks[:3])
    console.print(f"[dim]Sources: {cites}[/dim]")


def _enrich_with_history(
    query: str,
    history: list[tuple[str, str]],
    window: int = 2,
) -> str:
    if not history:
        return query
    parts = [
        f"Previous Q: {q}\nPrevious A: {a[:300]}"
        for q, a in history[-window:]
    ]
    parts.append(f"Current Q: {query}")
    return "\n\n".join(parts)

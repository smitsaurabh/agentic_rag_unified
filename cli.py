#!/usr/bin/env python3
"""
Agentic RAG — Command Line Interface
─────────────────────────────────────
Usage:
    python cli.py --help
    python cli.py query "Has BharatSteel defaulted?"
    python cli.py stream "What are Arjun Power covenants?" --domain instruments
    python cli.py interactive
    python cli.py schema
    python cli.py db setup --seed
    python cli.py health

The CLI can operate in two modes:
  1. Direct mode  — imports and calls agents/tools directly (no server needed).
  2. HTTP mode    — calls the running FastAPI server (use --host to point at it).

By default it runs in direct mode (faster, no server required).
Pass --http to force HTTP mode.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

# ─── App & console ───────────────────────────────────────────────────────────

app = typer.Typer(
    name="rag",
    help="Agentic RAG — Financial Intelligence CLI",
    add_completion=False,
    rich_markup_mode="rich",
)
db_app  = typer.Typer(help="Database management commands")
pdf_app = typer.Typer(help="Bond PDF pipeline commands")
app.add_typer(db_app,  name="db")
app.add_typer(pdf_app, name="pdf")

console = Console()
err_console = Console(stderr=True, style="bold red")

# ─── Shared state ─────────────────────────────────────────────────────────────

class _State:
    host: str = "http://localhost:8000"
    http_mode: bool = False
    no_color: bool = False
    verbose: bool = False

state = _State()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_env():
    """Load .env if present (for direct mode)."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)


def _direct_query(
    query: str,
    company: Optional[str],
    domain: Optional[str],
    provider: Optional[str],
    force_web: bool,
    force_sql: bool,
    force_pdf: bool = False,
) -> dict:
    """Run the query pipeline directly (no HTTP)."""
    _load_env()
    from agents.orchestrator import run_query
    state_out = run_query(
        query=query,
        company=company,
        domain=domain,
        llm_provider=provider,
        force_web=force_web,
        force_sql=force_sql,
        force_pdf=force_pdf,
    )
    web_sources = []
    if state_out.get("web_result") and state_out["web_result"].get("results"):
        web_sources = [r.url for r in state_out["web_result"]["results"]]
    sources_used = []
    if state_out.get("sql_has_data"):
        sources_used.append("sql")
    if state_out.get("pdf_has_data"):
        sources_used.append("pdf")
    if state_out.get("web_has_data"):
        sources_used.append("web")
    return {
        "answer": state_out.get("final_answer", ""),
        "domain": state_out.get("domain", ""),
        "intent": state_out.get("intent", ""),
        "entities": state_out.get("entities", []),
        "sources_used": sources_used,
        "sql_executed": state_out.get("sql_result", {}).get("sql"),
        "sql_row_count": state_out.get("sql_result", {}).get("row_count"),
        "web_sources": web_sources,
        "pdf_sources": state_out.get("pdf_result", {}).get("sources", []),
        "pdf_isins": state_out.get("pdf_result", {}).get("isins_found", []),
        "confidence": state_out.get("confidence", "Medium"),
        "processing_steps": state_out.get("processing_steps", []),
    }


def _http_query(payload: dict) -> dict:
    """Call the FastAPI /query endpoint."""
    import httpx
    try:
        r = httpx.post(f"{state.host}/query", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        err_console.print(f"HTTP request failed: {exc}")
        if state.verbose:
            import traceback; traceback.print_exc()
        raise typer.Exit(1)


def _render_response(
    result: dict,
    show_sql: bool = False,
    show_steps: bool = False,
    output: str = "pretty",
):
    """Render the API/direct response based on requested output format."""

    if output == "json":
        console.print_json(json.dumps(result, indent=2, default=str))
        return

    if output == "markdown":
        console.print(result.get("answer", ""))
        return

    # ── pretty (default) ────────────────────────────────────────────────────
    # Header panel
    domain = result.get("domain", "—")
    conf = result.get("confidence", "—")
    entities = ", ".join(result.get("entities", [])) or "—"
    sources = " + ".join(s.upper() for s in result.get("sources_used", [])) or "none"
    conf_color = {"High": "green", "Medium": "yellow", "Low": "red"}.get(conf, "white")

    meta = (
        f"[bold]Domain:[/bold] {domain}   "
        f"[bold]Sources:[/bold] {sources}   "
        f"[bold]Confidence:[/bold] [{conf_color}]{conf}[/{conf_color}]   "
        f"[bold]Entities:[/bold] {entities}"
    )
    console.print(Panel(meta, title="[bold cyan]Agentic RAG[/bold cyan]", border_style="cyan"))

    # Processing steps
    if show_steps:
        console.print("\n[bold]Processing steps:[/bold]")
        for step in result.get("processing_steps", []):
            console.print(f"  {step}")
        console.print()

    # SQL
    if show_sql and result.get("sql_executed"):
        sql_rows = result.get("sql_row_count", 0)
        console.print(
            Panel(
                result["sql_executed"],
                title=f"[bold yellow]Generated SQL[/bold yellow] ({sql_rows} rows)",
                border_style="yellow",
            )
        )

    # Answer
    console.print(Markdown(result.get("answer", "_No answer generated._")))

    # Web sources
    web_sources = result.get("web_sources", [])
    if web_sources:
        console.print("\n[dim]Web sources:[/dim]")
        for i, url in enumerate(web_sources, 1):
            console.print(f"  [dim][{i}] {url}[/dim]")

    # PDF sources + ISINs
    pdf_sources = result.get("pdf_sources", [])
    pdf_isins   = result.get("pdf_isins", [])
    if pdf_sources:
        console.print("\n[dim]Bond document sources:[/dim]")
        for src in pdf_sources:
            console.print(f"  [dim]📄 {src}[/dim]")
    if pdf_isins:
        console.print(f"  [dim]ISINs: {', '.join(pdf_isins)}[/dim]")


# ─── Global options callback ──────────────────────────────────────────────────

@app.callback()
def main(
    host: str = typer.Option("http://localhost:8000", "--host", help="API server URL (HTTP mode only)"),
    http: bool = typer.Option(False, "--http", help="Force HTTP mode (call running server)"),
    no_color: bool = typer.Option(False, "--no-color", help="Disable colour output"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full stack traces"),
):
    state.host = host
    state.http_mode = http
    state.no_color = no_color
    state.verbose = verbose
    if no_color:
        console._highlight = False


# ─── Command: query ───────────────────────────────────────────────────────────

@app.command()
def query(
    question: str = typer.Argument(..., help="Natural language question"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d",
        help="Domain hint: promoters | company | financials | instruments | industry"),
    company: Optional[str] = typer.Option(None, "--company", "-c",
        help="Company name hint for entity extraction"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p",
        help="LLM provider override: anthropic | openai"),
    force_web: bool = typer.Option(False, "--force-web", help="Always run web search"),
    force_sql: bool = typer.Option(False, "--force-sql", help="Always run SQL lookup"),
    force_pdf: bool = typer.Option(False, "--force-pdf", help="Always search bond PDFs"),
    show_sql: bool = typer.Option(False, "--show-sql", help="Print the generated SQL"),
    show_steps: bool = typer.Option(False, "--show-steps", "-s", help="Print agent steps"),
    output: str = typer.Option("pretty", "--output", "-o",
        help="Output format: pretty | json | markdown"),
):
    """
    Ask a financial intelligence question.

    Examples:

      rag query "Has BharatSteel ever defaulted?"

      rag query "What are the covenants on Arjun Power NCD?" --domain instruments --show-sql

      rag query "How volatile are earnings?" --company BharatSteel --show-steps
    """
    payload = {
        "query": question,
        "company": company,
        "domain": domain,
        "llm_provider": provider,
        "force_web": force_web,
        "force_sql": force_sql,
        "force_pdf": force_pdf,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        progress.add_task("Thinking...", total=None)
        try:
            if state.http_mode:
                result = _http_query(payload)
            else:
                result = _direct_query(
                    question, company, domain, provider, force_web, force_sql, force_pdf
                )
        except Exception as exc:
            err_console.print(f"Error: {exc}")
            if state.verbose:
                import traceback; traceback.print_exc()
            raise typer.Exit(1)

    _render_response(result, show_sql=show_sql, show_steps=show_steps, output=output)


# ─── Command: stream ─────────────────────────────────────────────────────────

@app.command()
def stream(
    question: str = typer.Argument(..., help="Natural language question"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d"),
    company: Optional[str] = typer.Option(None, "--company", "-c"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    force_web: bool = typer.Option(False, "--force-web"),
    force_sql: bool = typer.Option(False, "--force-sql"),
):
    """
    Stream the answer progressively (SSE).

    Requires the API server to be running. Automatically switches to HTTP mode.

    Example:

      rag stream "How volatile are BharatSteel earnings?" --domain financials
    """
    import httpx

    payload = {
        "query": question,
        "company": company,
        "domain": domain,
        "llm_provider": provider,
        "force_web": force_web,
        "force_sql": force_sql,
    }

    console.print(Panel(
        f"[bold]{question}[/bold]",
        title="[cyan]Streaming query[/cyan]",
        border_style="cyan",
    ))

    answer_parts: list[str] = []
    steps_printed = False

    try:
        with httpx.Client(timeout=120) as client:
            with client.stream("POST", f"{state.host}/query/stream", json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")
                    content = event.get("content", "")

                    if etype == "step":
                        console.print(f"  [dim]{content}[/dim]")
                        steps_printed = True
                    elif etype == "answer":
                        if steps_printed:
                            console.print()
                            steps_printed = False
                        console.print(content, end="")
                        answer_parts.append(content)
                    elif etype == "meta":
                        console.print("\n")
                        conf = content.get("confidence", "—")
                        conf_color = {"High": "green", "Medium": "yellow", "Low": "red"}.get(conf, "white")
                        console.print(
                            f"[dim]Domain:[/dim] {content.get('domain')}   "
                            f"[dim]Confidence:[/dim] [{conf_color}]{conf}[/{conf_color}]"
                        )
                        if content.get("web_sources"):
                            console.print("\n[dim]Web sources:[/dim]")
                            for i, url in enumerate(content["web_sources"], 1):
                                console.print(f"  [dim][{i}] {url}[/dim]")
                    elif etype == "error":
                        err_console.print(f"\nError: {content}")

    except Exception as exc:
        err_console.print(f"\nStreaming failed: {exc}")
        console.print("\n[yellow]Tip: Make sure the server is running: uvicorn main:app --reload[/yellow]")
        if state.verbose:
            import traceback; traceback.print_exc()
        raise typer.Exit(1)


# ─── Command: interactive ─────────────────────────────────────────────────────

@app.command()
def interactive(
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Default domain"),
    company: Optional[str] = typer.Option(None, "--company", "-c", help="Default company"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="LLM provider"),
):
    """
    Start an interactive REPL session.

    In-session commands:

      /domain instruments      Set the active domain
      /company Arjun Power     Set the company hint
      /provider openai         Switch LLM provider
      /clear                   Clear session context
      /help                    Show commands
      exit / quit              Leave
    """
    _load_env()

    console.print(Panel(
        "[bold cyan]Agentic RAG — Interactive Mode[/bold cyan]\n"
        "Type your question and press Enter.\n"
        "Use [bold]/help[/bold] for in-session commands. Type [bold]exit[/bold] to quit.",
        border_style="cyan",
    ))

    current_domain = domain
    current_company = company
    current_provider = provider

    def _show_help():
        t = Table(show_header=False, box=box.SIMPLE)
        t.add_column(style="bold yellow")
        t.add_column()
        t.add_row("/domain <name>",   "Set domain: promoters | company | financials | instruments | industry")
        t.add_row("/company <name>",  "Set company hint")
        t.add_row("/provider <name>", "Switch LLM: anthropic | openai")
        t.add_row("/clear",           "Reset domain/company/provider to defaults")
        t.add_row("/help",            "Show this help")
        t.add_row("exit / quit",      "Exit the session")
        console.print(t)

    while True:
        try:
            prompt_parts = []
            if current_domain:
                prompt_parts.append(f"[{current_domain}]")
            if current_company:
                prompt_parts.append(f"({current_company})")
            prefix = " ".join(prompt_parts) + " " if prompt_parts else ""

            user_input = console.input(f"[bold green]{prefix}> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if user_input.startswith("/"):
            parts = user_input[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "help":
                _show_help()
            elif cmd == "domain":
                current_domain = arg or None
                console.print(f"[dim]Domain set to: {current_domain or 'auto'}[/dim]")
            elif cmd == "company":
                current_company = arg or None
                console.print(f"[dim]Company set to: {current_company or 'none'}[/dim]")
            elif cmd == "provider":
                current_provider = arg or None
                console.print(f"[dim]Provider set to: {current_provider or 'default'}[/dim]")
            elif cmd == "clear":
                current_domain = domain
                current_company = company
                current_provider = provider
                console.print("[dim]Context cleared.[/dim]")
            else:
                console.print(f"[yellow]Unknown command /{cmd} — type /help[/yellow]")
            continue

        # Run the query
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console,
        ) as progress:
            progress.add_task("Thinking...", total=None)
            try:
                if state.http_mode:
                    result = _http_query({
                        "query": user_input,
                        "company": current_company,
                        "domain": current_domain,
                        "llm_provider": current_provider,
                        "force_web": False,
                        "force_sql": False,
                    })
                else:
                    result = _direct_query(
                        user_input, current_company, current_domain,
                        current_provider, False, False
                    )
            except Exception as exc:
                err_console.print(f"Error: {exc}")
                if state.verbose:
                    import traceback; traceback.print_exc()
                continue

        _render_response(result)
        console.print()


# ─── Command: schema ─────────────────────────────────────────────────────────

@app.command()
def schema(
    table: Optional[str] = typer.Option(None, "--table", "-t", help="Filter to a specific table"),
):
    """
    Inspect the database schema.

    Examples:

      rag schema

      rag schema --table instrument_covenant
    """
    _load_env()
    try:
        from tools.mysql_tool import get_db_schema
        raw = get_db_schema(force_refresh=True)
    except Exception as exc:
        err_console.print(f"Could not connect to database: {exc}")
        console.print("[yellow]Ensure MySQL is running and MYSQL_* env vars are set.[/yellow]")
        raise typer.Exit(1)

    lines = raw.splitlines()
    if table:
        lines = [l for l in lines if l.upper().startswith(f"TABLE {table.upper()}")]
        if not lines:
            err_console.print(f"Table '{table}' not found in schema.")
            raise typer.Exit(1)

    for line in lines:
        # Highlight TABLE keyword
        styled = line.replace("TABLE ", "[bold cyan]TABLE [/bold cyan]", 1)
        console.print(styled)


# ─── Command: domains ────────────────────────────────────────────────────────

@app.command()
def domains():
    """List supported query domains and their coverage."""
    t = Table(title="Supported Domains", box=box.ROUNDED, border_style="cyan")
    t.add_column("ID",          style="bold yellow", no_wrap=True)
    t.add_column("Label",       style="bold")
    t.add_column("Description")

    rows = [
        ("promoters",   "Company Promoters",   "Defaults, regulatory actions, group structure, directors, PE, ratings"),
        ("company",     "Company Generic",      "Governance, ESG, debt restructuring, capital market access"),
        ("financials",  "Company Financials",   "Earnings volatility, hedging, lender mix, DSCR/ICR trends"),
        ("instruments", "Debt Instruments",     "Covenants, repayment, prepayment, options, charge creation"),
        ("industry",    "Industry Analysis",    "Cyclicality, regulation, disruption, default history"),
    ]
    for r in rows:
        t.add_row(*r)
    console.print(t)


# ─── Command: health ─────────────────────────────────────────────────────────

@app.command()
def health(
    check: Optional[str] = typer.Option(None, "--check",
        help="Component to check: db | llm | web"),
):
    """
    Check system health: database, LLM API, and web search.

    Examples:

      rag health

      rag health --check db
    """
    _load_env()

    results: dict[str, tuple[bool, str]] = {}

    def _check_db():
        try:
            from tools.mysql_tool import get_db_schema
            s = get_db_schema(force_refresh=True)
            n = s.count("TABLE ")
            return True, f"{n} tables found"
        except Exception as exc:
            return False, str(exc)

    def _check_llm():
        try:
            from llm.provider import get_router_llm
            from langchain_core.messages import HumanMessage
            llm = get_router_llm()
            llm.invoke([HumanMessage(content="ping")])
            from config import settings
            return True, f"Provider: {settings.llm_provider}, Model: {settings.anthropic_model if settings.llm_provider == 'anthropic' else settings.openai_model}"
        except Exception as exc:
            return False, str(exc)

    def _check_web():
        try:
            from tools.web_search_tool import WebSearchTool
            tool = WebSearchTool()
            if not tool._available:
                return False, "TAVILY_API_KEY not configured"
            result = tool.search("test", n_results=1)
            if result["success"]:
                return True, f"{len(result['results'])} result(s) returned"
            return False, result.get("error", "unknown")
        except Exception as exc:
            return False, str(exc)

    checks = {"db": _check_db, "llm": _check_llm, "web": _check_web}
    to_run = {check: checks[check]} if check and check in checks else checks

    t = Table(title="System Health", box=box.ROUNDED)
    t.add_column("Component", style="bold")
    t.add_column("Status")
    t.add_column("Detail")

    all_ok = True
    for name, fn in to_run.items():
        with Progress(SpinnerColumn(), TextColumn(f"Checking {name}..."), transient=True, console=console) as p:
            p.add_task("", total=None)
            ok, detail = fn()
        status = "[green]✓ OK[/green]" if ok else "[red]✗ FAIL[/red]"
        if not ok:
            all_ok = False
        t.add_row(name.upper(), status, detail)

    console.print(t)
    if not all_ok:
        raise typer.Exit(1)


# ─── Command group: db ───────────────────────────────────────────────────────

@db_app.command("setup")
def db_setup(
    seed: bool = typer.Option(False, "--seed", help="Also load sample seed data"),
    reset: bool = typer.Option(False, "--reset", help="Drop and recreate all tables (DESTRUCTIVE)"),
):
    """
    Create the database schema (and optionally load sample data).

    Examples:

      rag db setup

      rag db setup --seed

      rag db setup --seed --reset
    """
    _load_env()
    import sqlalchemy as sa
    from sqlalchemy import text
    from config import settings

    schema_dir = os.path.join(os.path.dirname(__file__), "schema")
    ddl_path  = os.path.join(schema_dir, "ddl.sql")
    seed_path = os.path.join(schema_dir, "seed.sql")

    if not os.path.exists(ddl_path):
        err_console.print(f"DDL file not found: {ddl_path}")
        raise typer.Exit(1)

    if reset:
        confirm = typer.confirm(
            "[bold red]This will DROP all tables. Are you sure?[/bold red]",
            default=False,
        )
        if not confirm:
            console.print("Aborted.")
            raise typer.Exit(0)

    engine = sa.create_engine(settings.mysql_sync_url)

    def _exec_sql_file(path: str, label: str):
        with open(path) as f:
            content = f.read()
        # Split on semicolons, skip empty/comment-only blocks
        statements = [s.strip() for s in content.split(";") if s.strip() and not s.strip().startswith("--")]
        with engine.connect() as conn:
            if reset and label == "DDL":
                # Drop all user tables in reverse order
                insp = sa.inspect(engine)
                tables = insp.get_table_names()
                conn.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
                for tbl in tables:
                    conn.execute(text(f"DROP TABLE IF EXISTS `{tbl}`"))
                conn.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
                conn.commit()
                console.print(f"[yellow]Dropped {len(tables)} existing tables.[/yellow]")
            for stmt in statements:
                try:
                    conn.execute(text(stmt))
                except Exception as exc:
                    if state.verbose:
                        console.print(f"[dim]Skipped: {exc}[/dim]")
            conn.commit()
        console.print(f"[green]✓[/green] {label} applied ({len(statements)} statements)")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as p:
        p.add_task("Setting up schema...", total=None)
        _exec_sql_file(ddl_path, "DDL")

    if seed:
        if not os.path.exists(seed_path):
            err_console.print(f"Seed file not found: {seed_path}")
            raise typer.Exit(1)
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as p:
            p.add_task("Loading seed data...", total=None)
            _exec_sql_file(seed_path, "Seed data")

    console.print("[bold green]Database setup complete.[/bold green]")


@db_app.command("stats")
def db_stats():
    """Show row counts for all tables."""
    _load_env()
    import sqlalchemy as sa
    from sqlalchemy import text, inspect as sa_inspect
    from config import settings

    engine = sa.create_engine(settings.mysql_sync_url)
    try:
        insp = sa_inspect(engine)
        tables = insp.get_table_names()
    except Exception as exc:
        err_console.print(f"Cannot connect: {exc}")
        raise typer.Exit(1)

    t = Table(title="Database Statistics", box=box.ROUNDED)
    t.add_column("Table",   style="bold")
    t.add_column("Rows",    justify="right", style="cyan")

    total = 0
    with engine.connect() as conn:
        for tbl in sorted(tables):
            try:
                row = conn.execute(text(f"SELECT COUNT(*) FROM `{tbl}`")).fetchone()
                n = row[0]
            except Exception:
                n = "—"
            t.add_row(tbl, str(n))
            if isinstance(n, int):
                total += n

    console.print(t)
    console.print(f"[dim]Total rows across all tables: {total}[/dim]")


# ─── Command group: pdf ───────────────────────────────────────────────────────

@pdf_app.command("ingest")
def pdf_ingest(
    path: str = typer.Argument(..., help="Path to a PDF file or directory of PDFs"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirectories"),
):
    """
    Ingest one or more bond PDF documents into the vector store.

    Examples:

      rag pdf ingest ~/Downloads/bond_prospectus.pdf

      rag pdf ingest ~/Downloads/bonds/ --recursive
    """
    _load_env()
    from tools.pdf_tool import PDFTool
    tool = PDFTool()

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True, console=console) as p:
        p.add_task(f"Ingesting {path}...", total=None)
        try:
            result = tool.ingest(path=path, recursive=recursive)
        except Exception as exc:
            err_console.print(f"Ingestion failed: {exc}")
            raise typer.Exit(1)

    # Display result
    if isinstance(result, dict):
        ingested   = result.get("ingested", result.get("files_ingested", 0))
        skipped    = result.get("skipped", result.get("files_skipped", 0))
        chunks     = result.get("chunks_added", result.get("chunk_count", "?"))
        console.print(f"[green]✓ Ingested[/green] {ingested} file(s), skipped {skipped}, {chunks} chunks added")
        isins = result.get("isins_found", [])
        if isins:
            console.print(f"  ISINs detected: [cyan]{', '.join(isins)}[/cyan]")
    else:
        console.print(f"[green]✓[/green] {result}")


@pdf_app.command("list")
def pdf_list():
    """List all ingested bond PDF documents."""
    _load_env()
    from tools.pdf_tool import PDFTool

    try:
        files = PDFTool().list_files()
    except Exception as exc:
        err_console.print(f"Error: {exc}")
        raise typer.Exit(1)

    if not files:
        console.print("[yellow]No PDFs ingested yet. Run: rag pdf ingest <path>[/yellow]")
        return

    t = Table(title=f"Ingested Bond Documents ({len(files)})", box=box.ROUNDED, border_style="cyan")
    t.add_column("Filename",  style="bold")
    t.add_column("ISINs",     style="cyan")
    t.add_column("Chunks",    justify="right")
    t.add_column("Status")

    for f in files:
        name   = f.get("filename", f.get("source", "—"))
        isins  = ", ".join(f.get("isins", [])) or "—"
        chunks = str(f.get("chunk_count", f.get("chunks", "—")))
        status = f.get("status", "ingested")
        t.add_row(name, isins, chunks, status)

    console.print(t)


@pdf_app.command("isins")
def pdf_isins():
    """List all ISINs found across ingested bond documents."""
    _load_env()
    from tools.pdf_tool import PDFTool

    try:
        isins = PDFTool().list_isins()
    except Exception as exc:
        err_console.print(f"Error: {exc}")
        raise typer.Exit(1)

    if not isins:
        console.print("[yellow]No ISINs found. Ingest bond PDFs first: rag pdf ingest <path>[/yellow]")
        return

    console.print(f"[bold]ISINs across all ingested documents ({len(isins)}):[/bold]")
    for isin in sorted(isins):
        console.print(f"  [cyan]{isin}[/cyan]")


@pdf_app.command("stats")
def pdf_stats_cmd():
    """Show bond RAG pipeline statistics."""
    _load_env()
    from tools.pdf_tool import PDFTool

    try:
        s = PDFTool().stats()
    except Exception as exc:
        err_console.print(f"Error: {exc}")
        raise typer.Exit(1)

    t = Table(title="Bond RAG Pipeline Stats", box=box.ROUNDED)
    t.add_column("Metric", style="bold")
    t.add_column("Value",  justify="right", style="cyan")

    for k, v in s.items():
        if k != "error":
            t.add_row(str(k).replace("_", " ").title(), str(v))

    console.print(t)


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()

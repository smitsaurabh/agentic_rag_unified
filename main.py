"""
Agentic RAG — FastAPI Application
──────────────────────────────────

Endpoints:
  GET  /chat              — browser chat UI
  POST /query             — JSON Q&A
  POST /query/stream      — SSE streaming Q&A
  GET  /health            — liveness check
  GET  /schema            — DB schema (debug)
  GET  /domains           — supported query domains
  POST /pdf/ingest        — ingest a PDF file or directory
  GET  /pdf/files         — list ingested PDF files
  GET  /pdf/isins         — list all known ISINs
  GET  /pdf/stats         — bond RAG pipeline stats

Run:
  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

import structlog
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from agents.orchestrator import run_query
from config import settings
from models.schemas import QueryRequest, QueryResponse

# ─── Logging ──────────────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()
logging.basicConfig(level=logging.INFO)


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    log.info("startup", llm_provider=settings.llm_provider, db=settings.mysql_database)

    # Warm up the embedding model + reranker in a background thread
    # so the first query doesn't pay the cold-start cost (~1-2s).
    import asyncio, concurrent.futures
    def _warmup():
        try:
            from tools.pdf_tool import PDFTool
            tool = PDFTool()
            # Trigger lazy initialisation of pipeline, embedder and reranker
            tool.retrieve(query="warmup", top_k=1)
            log.info("warmup_complete", model="BAAI/bge-large-en-v1.5+reranker")
        except Exception as exc:
            log.warning("warmup_skipped", reason=str(exc))

    loop = asyncio.get_event_loop()
    loop.run_in_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1), _warmup)

    yield
    log.info("shutdown")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG — Financial Intelligence API",
    description=(
        "Multi-agent RAG system for financial credit analysis. "
        "Queries MySQL (structured data), bond PDFs (prospectuses), and the web, "
        "then synthesises a domain-aware answer."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Middleware ────────────────────────────────────────────────────────────────

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.3f}s"
    return response


# ─── Chat UI ──────────────────────────────────────────────────────────────────

@app.get("/chat", include_in_schema=False)
async def chat_ui():
    chat_file = os.path.join(os.path.dirname(__file__), "chat.html")
    if not os.path.exists(chat_file):
        raise HTTPException(status_code=404, detail="chat.html not found")
    return FileResponse(chat_file, media_type="text/html")


# ─── Health / meta ────────────────────────────────────────────────────────────

@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "llm_provider": settings.llm_provider, "version": "2.0.0"}


@app.get("/domains")
async def list_domains():
    return {
        "domains": [
            {"id": "promoters",   "label": "Company Promoters",  "description": "Promoter background, defaults, regulatory issues, group structure"},
            {"id": "company",     "label": "Company Generic",     "description": "Governance, ESG, debt restructuring, capital market access"},
            {"id": "financials",  "label": "Company Financials",  "description": "Earnings volatility, hedging, liability mix, financing diversification"},
            {"id": "instruments", "label": "Debt Instruments",    "description": "Covenants, repayment, prepayment, options, charge creation"},
            {"id": "industry",    "label": "Industry Analysis",   "description": "Cyclicality, regulation, disruption risk, strategic importance"},
        ]
    }


@app.get("/schema")
async def db_schema():
    try:
        from tools.mysql_tool import get_db_schema
        return {"schema": get_db_schema()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"DB unavailable: {exc}")


# ─── Q&A endpoints ────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    log.info("query_received", query=request.query[:120], domain=request.domain)

    try:
        state = run_query(
            query=request.query,
            company=request.company,
            domain=request.domain,
            llm_provider=request.llm_provider,
            force_web=request.force_web,
            force_sql=request.force_sql,
            force_pdf=request.force_pdf,
        )
    except Exception as exc:
        log.error("query_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc))

    web_sources: list[str] = []
    if state.get("web_result") and state["web_result"].get("results"):
        web_sources = [r.url for r in state["web_result"]["results"]]

    pdf_sources: list[str] = state.get("pdf_result", {}).get("sources", [])
    pdf_isins: list[str] = state.get("pdf_result", {}).get("isins_found", [])

    sources_used = []
    if state.get("sql_has_data"):
        sources_used.append("sql")
    if state.get("pdf_has_data"):
        sources_used.append("pdf")
    if state.get("web_has_data"):
        sources_used.append("web")

    return QueryResponse(
        answer=state.get("final_answer", "No answer generated."),
        domain=state.get("domain", "general"),
        intent=state.get("intent", request.query),
        entities=state.get("entities", []),
        sources_used=sources_used,
        sql_executed=state.get("sql_result", {}).get("sql"),
        sql_row_count=state.get("sql_result", {}).get("row_count"),
        web_sources=web_sources,
        pdf_sources=pdf_sources,
        pdf_isins=pdf_isins,
        confidence=state.get("confidence", "Medium"),
        processing_steps=state.get("processing_steps", []),
    )


@app.post("/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    import asyncio
    loop = asyncio.get_event_loop()

    async def event_generator():
        try:
            # Run the synchronous graph in a thread pool so the event loop
            # is not blocked during the full agent execution.
            state = await loop.run_in_executor(
                None,
                lambda: run_query(
                    query=request.query,
                    company=request.company,
                    domain=request.domain,
                    llm_provider=request.llm_provider,
                    force_web=request.force_web,
                    force_sql=request.force_sql,
                    force_pdf=request.force_pdf,
                ),
            )

            for step in state.get("processing_steps", []):
                yield f"data: {json.dumps({'type': 'step', 'content': step})}\n\n"

            answer = state.get("final_answer", "")
            chunk_size = 80
            for i in range(0, len(answer), chunk_size):
                yield f"data: {json.dumps({'type': 'answer', 'content': answer[i:i+chunk_size]})}\n\n"

            web_sources = []
            if state.get("web_result") and state["web_result"].get("results"):
                web_sources = [r.url for r in state["web_result"]["results"]]

            meta = {
                "domain": state.get("domain"),
                "entities": state.get("entities", []),
                "confidence": state.get("confidence", "Medium"),
                "sql_row_count": state.get("sql_result", {}).get("row_count"),
                "web_sources": web_sources,
                "pdf_sources": state.get("pdf_result", {}).get("sources", []),
                "pdf_isins": state.get("pdf_result", {}).get("isins_found", []),
            }
            yield f"data: {json.dumps({'type': 'meta', 'content': meta})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── PDF / Bond RAG endpoints ─────────────────────────────────────────────────

class IngestPathRequest(BaseModel):
    path: str
    recursive: bool = False


@app.post("/pdf/ingest")
async def pdf_ingest_path(request: IngestPathRequest):
    """Ingest PDFs from a server-side file path or directory."""
    try:
        from tools.pdf_tool import PDFTool
        tool = PDFTool()
        result = tool.ingest(path=request.path, recursive=request.recursive)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/pdf/upload")
async def pdf_upload(file: UploadFile = File(...)):
    """Upload a PDF file and ingest it into the bond RAG pipeline."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    upload_dir = Path(settings.bond_rag_data_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    try:
        content = await file.read()
        dest.write_bytes(content)

        from tools.pdf_tool import PDFTool
        tool = PDFTool()
        result = tool.ingest(path=dest)
        return {"filename": file.filename, "size_bytes": len(content), **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/pdf/files")
async def pdf_list_files():
    """List all ingested bond PDF documents."""
    try:
        from tools.pdf_tool import PDFTool
        return {"files": PDFTool().list_files()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/pdf/isins")
async def pdf_list_isins():
    """List all ISINs found across ingested bond documents."""
    try:
        from tools.pdf_tool import PDFTool
        return {"isins": PDFTool().list_isins()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/pdf/stats")
async def pdf_stats():
    """Return bond RAG pipeline statistics."""
    try:
        from tools.pdf_tool import PDFTool
        return PDFTool().stats()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

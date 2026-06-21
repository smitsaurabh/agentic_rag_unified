# Agentic RAG — Financial Intelligence System

A multi-agent RAG system for financial credit analysis. Combines three data sources — a MySQL structured database, bond PDF prospectuses, and real-time web search — orchestrated by a LangGraph state machine and served via a FastAPI backend.

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│                LangGraph Orchestrator           │
│                                                 │
│  [Router] ──classify domain & sources──►        │
│       │    (ISIN fast-path: skips LLM)          │
│       │                                         │
│  ┌────▼──────────────────────────────────────┐  │
│  │         Parallel Agents Node              │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐  │  │
│  │  │SQL Agent │ │PDF Agent │ │Web Agent │  │  │
│  │  │(MySQL)   │ │(ChromaDB)│ │(Tavily)  │  │  │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘  │  │
│  └───────┴────────────┴────────────┴────────┘  │
│                       │                         │
│              [Synthesiser LLM]                  │
│          (Anthropic Claude / OpenAI)            │
│                       │                         │
│        [In-memory TTL Result Cache]             │
└─────────────────────────────────────────────────┘
    │
    ▼
Structured Answer
```

**Routing logic (parallel execution):**
- For **ISIN queries**: fast-path bypasses the LLM router (regex detection) → directly runs SQL + PDF + Web agents in parallel
- For **other queries**: LLM classifies domain + sources → chosen agents run concurrently via `ThreadPoolExecutor`
- **Synthesiser** combines whichever sources returned data (all three results merged simultaneously)
- **Result cache**: identical queries served from memory within a 5-minute TTL (skips all agents)

---

## Quick Start

### 1. Install dependencies

```bash
cd agentic-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes `rag` a global CLI command
```

> **Bond PDF dependencies** require extra system packages:
> ```bash
> # macOS
> brew install tesseract
>
> # Ubuntu / Debian
> apt install tesseract-ocr
> ```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY (or OPENAI_API_KEY), MYSQL_* vars, TAVILY_API_KEY
```

### 3. Set up MySQL

```bash
rag db setup --seed          # create schema + load sample companies
```

### 4. (Optional) Ingest bond PDFs

```bash
rag pdf ingest ~/Downloads/bond_prospectus.pdf
rag pdf ingest ~/Downloads/bonds/ --recursive   # scan a directory
```

### 5. Start the server

```bash
uvicorn main:app --reload --port 8080
```

> **Note:** Port 8000 may already be in use by another service. Use `lsof -i :8000` to check,
> or just run on 8080 as shown above.

### 6. Open chat UI

[http://localhost:8080/chat](http://localhost:8080/chat)

---

## Project Structure

```
agentic-rag/
├── main.py                   # FastAPI application
├── cli.py                    # Typer CLI (rag command)
├── config.py                 # Pydantic settings (reads .env)
├── chat.html                 # Dark-themed browser chat UI
├── setup.py                  # pip install -e . entry point
├── requirements.txt
├── .env.example
│
├── agents/
│   └── orchestrator.py       # LangGraph state machine (parallel agents + cache)
│
├── cache/
│   └── query_cache.py        # In-memory TTL result cache (5-min default)
│
├── llm/
│   └── provider.py           # Anthropic / OpenAI abstraction
│
├── prompts/
│   └── domain_prompts.py     # All system prompts + synthesis templates
│
├── tools/
│   ├── mysql_tool.py         # NL → SQL → execute
│   ├── web_search_tool.py    # Tavily web search
│   └── pdf_tool.py           # bond_rag PDF retrieval wrapper
│
├── models/
│   └── schemas.py            # Pydantic models + AgentState TypedDict
│
├── bond_rag/                 # Embedded bond PDF RAG pipeline
│   ├── core/config.py        # BOND_RAG__ settings
│   ├── ingestion/            # PDF parsing, OCR, ISIN-aware chunking
│   ├── retrieval/            # ChromaDB + BM25 + cross-encoder reranker
│   └── rag/pipeline.py       # BondRAGPipeline (ingest / retrieve)
│
└── schema/
    ├── ddl.sql               # MySQL schema (16 tables)
    └── seed.sql              # Sample companies + instruments
```

---

## CLI Reference

```bash
# Query (direct mode — no server needed)
rag query "Has BharatSteel ever defaulted?"
rag query "What are the covenants on Arjun Power NCD?" --domain instruments --show-sql
rag query "What does ISIN INE123A01234 prospectus say about call options?" --force-pdf

# Stream (requires running server)
rag stream "How volatile are BharatSteel earnings?" --domain financials

# Interactive REPL
rag interactive
# In-session: /domain instruments | /company Arjun | /provider openai | /clear | exit

# Bond PDF commands
rag pdf ingest ~/Downloads/prospectus.pdf
rag pdf ingest ~/Downloads/bonds/ --recursive
rag pdf list          # show ingested files + ISINs
rag pdf isins         # list all known ISINs
rag pdf stats         # pipeline statistics

# Database
rag db setup --seed   # create schema + load sample data
rag db setup --seed --reset   # drop + recreate + seed
rag db stats          # row counts per table

# Utilities
rag schema            # print DB schema
rag domains           # list supported query domains
rag health            # check DB + LLM + web connectivity
rag --help
```

---

## API Reference

### Core

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/chat` | Browser chat UI |
| `POST` | `/query` | JSON Q&A response |
| `POST` | `/query/stream` | SSE streaming response |
| `GET`  | `/health` | Liveness check |
| `GET`  | `/api/health` | Liveness check (alias) |
| `GET`  | `/domains` | Supported domains |
| `GET`  | `/schema` | DB schema (debug) |

### Bond PDF Pipeline

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/pdf/ingest` | Ingest from server-side path |
| `POST` | `/pdf/upload` | Upload PDF file via multipart |
| `GET`  | `/pdf/files` | List ingested documents |
| `GET`  | `/pdf/isins` | List all known ISINs |
| `GET`  | `/pdf/stats` | Pipeline statistics |

### Query Request Body

```json
{
  "query": "What are the financial covenants?",
  "company": "Arjun Power",
  "domain": "instruments",
  "llm_provider": "anthropic",
  "force_sql": false,
  "force_pdf": false,
  "force_web": false
}
```

### SSE Event Types

```
data: {"type": "step",   "content": "⚡ Router (fast-path): ISIN detected — SQL+PDF+Web"}
data: {"type": "step",   "content": "⚡ Parallel Agents: running SQL+PDF+WEB concurrently"}
data: {"type": "step",   "content": "✅ Parallel Agents: all done in 3.2s"}
data: {"type": "answer", "content": "## Executive Summary..."}
data: {"type": "meta",   "content": {"domain": "...", "confidence": "High", "pdf_isins": [...], ...}}
data: [DONE]
```

---

## Supported Query Domains

| Domain | Covers |
|--------|--------|
| `promoters` | Group structure, defaults, regulatory actions, directors, PE, ratings |
| `company` | Governance, ESG, debt restructuring, capital market access |
| `financials` | DSCR, ICR, hedging, lender mix, earnings volatility |
| `instruments` | Covenants, repayment schedule, prepayment, options, charges |
| `industry` | Cyclicality, regulation, disruption risk, default history |
| `general` | Catch-all (auto-detected) |

The router also recognises **ISIN patterns** (e.g. `INE976I07CV5`) via regex and immediately routes to SQL + PDF + Web **without calling the LLM router** (fast-path). Both ISINs in the query text and `force_pdf=true` trigger bond document retrieval.

---

## MySQL Schema

16 tables across 5 domains:

```
industry            promoter_group       promoter_default
regulatory_action   director             company
company_financials  financing_mix        lender_diversification
instrument          instrument_covenant  repayment_schedule
prepayment_clause   instrument_option    charge_creation
credit_rating_history
```

---

## Bond PDF Pipeline (bond_rag)

The embedded `bond_rag` package handles PDF ingestion and retrieval independently of the main LLM.

**Ingestion pipeline:**
1. PyMuPDF parses each page; Tesseract OCR handles scanned pages
2. ISIN regex two-pass annotation marks which ISIN each chunk belongs to
3. ISIN-aware chunker splits at 800 chars / 25% overlap
4. BAAI/bge-large-en-v1.5 embeds chunks → ChromaDB persistent store
5. SQLite registry tracks ingested files + checksums (deduplication)

**Retrieval (3-stage hybrid):**
1. Dense retrieval: ChromaDB cosine similarity (top-N candidates)
2. BM25 sparse retrieval over the same candidate set
3. Cross-encoder reranking (`ms-marco-MiniLM-L-12-v2`) → top-K final chunks (2× candidate multiplier for speed)

**Important:** Ollama is not required. The bond_rag LLM component is bypassed; synthesis is always performed by the main Anthropic Claude / OpenAI provider.

### Bond RAG Configuration

All `BOND_RAG__*` env vars (see `.env.example`):

```env
BOND_RAG__DB_DIR=db/bond_rag
BOND_RAG__DATA_DIR=data/pdfs
BOND_RAG__EMBED__DEVICE=cpu          # or cuda / mps
BOND_RAG__RETRIEVER__TOP_K=6
BOND_RAG__OCR__ENABLED=true
BOND_RAG__OCR__LANGUAGE=eng
```

---

## Environment Variables

```env
# LLM
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-opus-4-8

OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=secret
MYSQL_DATABASE=financial_rag

# Web search
TAVILY_API_KEY=tvly-...

# Agent knobs
MAX_SQL_ROWS=50
WEB_SEARCH_RESULTS=5
SYNTHESIS_TEMPERATURE=0.2

# Result cache
QUERY_CACHE_TTL=300          # seconds (0 to disable)
QUERY_CACHE_MAXSIZE=100      # max entries

# Bond RAG
BOND_RAG__DB_DIR=db/bond_rag
BOND_RAG__DATA_DIR=data/pdfs
BOND_RAG__RETRIEVER__TOP_K=6
BOND_RAG__EMBED__DEVICE=cpu
```

---

## Dependencies

| Stack | Version |
|-------|---------|
| Python | ≥ 3.11 |
| FastAPI | 0.111 |
| LangChain | 1.3.x |
| LangGraph | 1.2.x |
| LangChain Anthropic | 1.4.x |
| LangChain OpenAI | 1.3.x |
| PyMuPDF (fitz) | ≥ 1.24 |
| sentence-transformers | ≥ 3.0 |
| ChromaDB | ≥ 0.5 |
| rank-bm25 | ≥ 0.2 |
| codecarbon | ≥ 2.3 |
| Tesseract | system binary |
| Tavily | 0.3.x |
| Typer | 0.12.x |

---

## Performance Characteristics

After optimisations the typical end-to-end latency is:

| Query type | Before | After |
|-----------|--------|-------|
| ISIN query (SQL+PDF+Web) | ~10–16s | ~4–6s |
| General query (SQL+Web) | ~6–10s | ~3–5s |
| Repeat query (cached) | ~10–16s | < 10ms |
| First query after restart | +1–2s cold start | eliminated (warmup) |

**Key optimisations applied:**
- Agents run in parallel (`ThreadPoolExecutor`) instead of sequentially
- ISIN queries bypass the LLM router (deterministic regex fast-path, saves ~0.8s)
- SQL generation uses `gpt-4o-mini` / `claude-haiku` (3–5× faster than full model)
- Cross-encoder reranks 2× top-K candidates (was 3×)
- Embedding model + reranker are pre-warmed at server startup
- In-memory TTL cache (5-min default) for repeated identical queries

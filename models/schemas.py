"""
Pydantic models for API I/O and LangGraph state.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ─── API schemas ──────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="Natural language question from the user")
    company: Optional[str] = Field(None, description="Optional company name hint")
    domain: Optional[Literal[
        "promoters", "company", "financials", "instruments", "industry", "general"
    ]] = Field(None, description="Optional domain hint — auto-detected if not provided")
    llm_provider: Optional[Literal["anthropic", "openai"]] = Field(
        None, description="Override the default LLM provider for this request"
    )
    force_web: bool = Field(False, description="Always run web search even if DB has data")
    force_sql: bool = Field(False, description="Always attempt SQL lookup")
    force_pdf: bool = Field(False, description="Always attempt PDF retrieval")


class SourceInfo(BaseModel):
    type: Literal["database", "web", "pdf"]
    detail: str  # SQL query, URL list, or PDF filename


class QueryResponse(BaseModel):
    answer: str
    domain: str
    intent: str
    entities: list[str]
    sources_used: list[Literal["sql", "web", "pdf"]]
    sql_executed: Optional[str] = None
    sql_row_count: Optional[int] = None
    web_sources: list[str] = Field(default_factory=list)
    pdf_sources: list[str] = Field(default_factory=list)
    pdf_isins: list[str] = Field(default_factory=list)
    confidence: Literal["High", "Medium", "Low"] = "Medium"
    processing_steps: list[str] = Field(default_factory=list)


# ─── LangGraph state ──────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str
    company_hint: str
    domain_override: Optional[str]
    llm_provider_override: Optional[str]
    force_web: bool
    force_sql: bool
    force_pdf: bool

    # ── Router outputs ─────────────────────────────────────────────────────────
    domain: str
    data_sources: list[str]          # ["sql"], ["web"], or ["sql", "web"]
    entities: list[str]
    intent: str

    # ── SQL agent outputs ──────────────────────────────────────────────────────
    sql_result: dict[str, Any]       # from MySQLTool.run()
    sql_attempted: bool
    sql_has_data: bool

    # ── Web agent outputs ──────────────────────────────────────────────────────
    web_result: dict[str, Any]       # from WebSearchTool.search()
    web_attempted: bool
    web_has_data: bool

    # ── PDF agent outputs ──────────────────────────────────────────────────────
    pdf_result: dict[str, Any]       # from PDFTool.retrieve()
    pdf_attempted: bool
    pdf_has_data: bool

    # ── Synthesiser outputs ────────────────────────────────────────────────────
    final_answer: str
    processing_steps: list[str]
    confidence: str

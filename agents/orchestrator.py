"""
LangGraph Orchestration Graph  (LangGraph 1.x compatible)
──────────────────────────────────────────────────────────

Flow:
  router → sql_agent → pdf_agent → web_agent → synthesise
                  ↘ pdf_agent ↗ web_agent ↗
                           ↘ web_agent ↗
                  ↘ web_agent ↗

Routing:
  route_after_router → "sql_first"  if sql in sources
                     → "pdf_first"  if pdf in sources but not sql
                     → "web_only"   otherwise
  route_after_sql    → "need_pdf"   if pdf in sources
                     → "need_web"   if web in sources OR sql empty
                     → "skip_rest"  if sql has data and no other sources
  route_after_pdf    → "need_web"   if web in sources OR both sql & pdf empty
                     → "skip_web"   otherwise
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import time
from typing import Any

from langgraph.graph import StateGraph, END

from config import settings
from llm.provider import get_router_llm, get_synthesis_llm
from models.schemas import AgentState
from prompts.domain_prompts import (
    QUERY_CLASSIFIER_PROMPT,
    WEB_SYNTHESIZER_PROMPT,
    COMBINED_SYNTHESIZER_PROMPT,
    SQL_ONLY_SYNTHESIZER_PROMPT,
    PDF_ONLY_SYNTHESIZER_PROMPT,
    FULL_COMBINED_SYNTHESIZER_PROMPT,
    SQL_PDF_SYNTHESIZER_PROMPT,
    PDF_WEB_SYNTHESIZER_PROMPT,
    get_system_prompt,
)
from cache.query_cache import get_cached, set_cached
from tools.mysql_tool import MySQLTool
from tools.web_search_tool import WebSearchTool
from tools.pdf_tool import PDFTool

logger = logging.getLogger(__name__)

# Deterministic ISIN detector — 2-letter country code + 9 alphanumeric + 1 digit
_ISIN_RE = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b")


def _extract_isins(text: str) -> list[str]:
    """Return unique ISINs found in text (case-insensitive)."""
    return list(dict.fromkeys(_ISIN_RE.findall(text.upper())))


# ─── Singleton tools ──────────────────────────────────────────────────────────

_mysql_tool: MySQLTool | None = None
_web_tool: WebSearchTool | None = None
_pdf_tool: PDFTool | None = None


def _get_mysql_tool() -> MySQLTool:
    global _mysql_tool
    if _mysql_tool is None:
        _mysql_tool = MySQLTool()
    return _mysql_tool


def _get_web_tool() -> WebSearchTool:
    global _web_tool
    if _web_tool is None:
        _web_tool = WebSearchTool()
    return _web_tool


def _get_pdf_tool() -> PDFTool:
    global _pdf_tool
    if _pdf_tool is None:
        _pdf_tool = PDFTool()
    return _pdf_tool


# ─── Node: router ─────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    steps = list(state.get("processing_steps", []))

    # ── Fast path: ISIN detected → skip LLM router entirely ──────────────────
    query_isins = _extract_isins(state["query"])
    if query_isins and not state.get("force_web") and not state.get("domain_override"):
        sources = {"sql", "pdf", "web"}
        if state.get("force_sql"): sources.add("sql")
        if state.get("force_pdf"): sources.add("pdf")
        steps.append(
            f"⚡ Router (fast-path): ISIN {query_isins} detected "
            f"— SQL+PDF+Web, domain=instruments"
        )
        logger.info("Router fast-path: ISIN detected, skipping LLM — %s", query_isins)
        return {
            **state,
            "domain":         state.get("domain_override") or "instruments",
            "data_sources":   sorted(sources),
            "entities":       query_isins,
            "intent":         state["query"],
            "sql_attempted":  False,
            "sql_has_data":   False,
            "pdf_attempted":  False,
            "pdf_has_data":   False,
            "web_attempted":  False,
            "web_has_data":   False,
            "processing_steps": steps,
        }

    # ── Slow path: LLM classification ─────────────────────────────────────────
    steps.append("🔍 Router: classifying query domain and data sources")
    llm   = get_router_llm()
    chain = QUERY_CLASSIFIER_PROMPT | llm

    try:
        response = chain.invoke({"query": state["query"]})
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(
                line for line in raw.splitlines() if not line.startswith("```")
            ).strip()
        parsed = json.loads(raw)
    except Exception as exc:
        logger.warning("Router classification failed: %s", exc)
        parsed = {
            "domain":       state.get("domain_override") or "general",
            "data_sources": ["sql", "web"],
            "entities":     [state.get("company_hint", "")] if state.get("company_hint") else [],
            "intent":       state["query"],
        }

    if state.get("domain_override"):
        parsed["domain"] = state["domain_override"]

    sources: set[str] = set(parsed.get("data_sources", ["sql", "web"]))

    # ISIN override: ensure PDF is always included for ISIN mentions
    if query_isins:
        sources.update({"pdf", "sql", "web"})

    if state.get("force_web"):  sources.add("web")
    if state.get("force_sql"):  sources.add("sql")
    if state.get("force_pdf"):  sources.add("pdf")

    steps.append(
        f"✅ Router: domain={parsed['domain']}, "
        f"sources={sorted(sources)}, "
        f"entities={parsed.get('entities', [])}"
    )

    return {
        **state,
        "domain":         parsed.get("domain", "general"),
        "data_sources":   sorted(sources),
        "entities":       parsed.get("entities", []),
        "intent":         parsed.get("intent", state["query"]),
        "sql_attempted":  False,
        "sql_has_data":   False,
        "pdf_attempted":  False,
        "pdf_has_data":   False,
        "web_attempted":  False,
        "web_has_data":   False,
        "processing_steps": steps,
    }


# ─── Node: sql_agent ──────────────────────────────────────────────────────────

def sql_agent_node(state: AgentState) -> AgentState:
    steps = list(state.get("processing_steps", []))
    steps.append("🗄️  SQL Agent: generating and executing query")

    result = _get_mysql_tool().run(
        query=state["query"],
        entities=state.get("entities", []),
        domain=state.get("domain", "general"),
    )

    if result["success"] and result["has_data"]:
        steps.append(f"✅ SQL Agent: {result['row_count']} rows returned")
    elif result.get("error"):
        steps.append(f"⚠️  SQL Agent: {result['error']}")
    else:
        steps.append("⚠️  SQL Agent: no rows — will try other sources")

    return {
        **state,
        "sql_result": result,
        "sql_attempted": True,
        "sql_has_data": result.get("has_data", False),
        "processing_steps": steps,
    }


# ─── Node: pdf_agent ──────────────────────────────────────────────────────────

def pdf_agent_node(state: AgentState) -> AgentState:
    steps = list(state.get("processing_steps", []))
    steps.append("📄 PDF Agent: searching bond documents")

    result = _get_pdf_tool().retrieve(
        query=state["query"],
        top_k=settings.bond_rag_top_k,
    )

    n = len(result.get("chunks", []))
    if result["success"] and result["has_data"]:
        isins = result.get("isins_found", [])
        isin_str = f", ISINs: {isins}" if isins else ""
        steps.append(f"✅ PDF Agent: {n} chunks retrieved{isin_str}")
    elif result.get("error"):
        steps.append(f"⚠️  PDF Agent: {result['error']}")
    else:
        steps.append("⚠️  PDF Agent: no matching chunks in bond documents")

    return {
        **state,
        "pdf_result": result,
        "pdf_attempted": True,
        "pdf_has_data": result.get("has_data", False),
        "processing_steps": steps,
    }


# ─── Node: web_agent ──────────────────────────────────────────────────────────

def web_agent_node(state: AgentState) -> AgentState:
    steps = list(state.get("processing_steps", []))
    steps.append("🌐 Web Agent: searching the internet")

    tool = _get_web_tool()
    enriched = tool.build_financial_query(
        user_query=state["query"],
        entities=state.get("entities", []),
        domain=state.get("domain", "general"),
    )
    result = tool.search(query=enriched)

    n = len(result.get("results", []))
    if result["success"] and n:
        steps.append(f"✅ Web Agent: {n} results fetched")
    else:
        steps.append(f"⚠️  Web Agent: {result.get('error', 'no results')}")

    return {
        **state,
        "web_result": result,
        "web_attempted": True,
        "web_has_data": result.get("success", False) and n > 0,
        "processing_steps": steps,
    }


# ─── Node: synthesiser ────────────────────────────────────────────────────────

def synthesise_node(state: AgentState) -> AgentState:
    steps = list(state.get("processing_steps", []))
    has_sql = state.get("sql_has_data", False)
    has_pdf = state.get("pdf_has_data", False)
    has_web = state.get("web_has_data", False)

    llm = get_synthesis_llm()
    system_prompt = get_system_prompt(state.get("domain", "general"))

    if has_sql and has_pdf and has_web:
        steps.append("🧠 Synthesiser: combining DB + PDF + Web results")
        chain = FULL_COMBINED_SYNTHESIZER_PROMPT | llm
        response = chain.invoke({
            "system_prompt": system_prompt,
            "query": state["query"],
            "sql_results": state["sql_result"]["markdown_table"],
            "pdf_results": state["pdf_result"]["formatted"],
            "web_results": state["web_result"]["formatted"],
        })
        confidence = "High"

    elif has_sql and has_pdf:
        steps.append("🧠 Synthesiser: combining DB + PDF results")
        chain = SQL_PDF_SYNTHESIZER_PROMPT | llm
        response = chain.invoke({
            "system_prompt": system_prompt,
            "query": state["query"],
            "sql_results": state["sql_result"]["markdown_table"],
            "pdf_results": state["pdf_result"]["formatted"],
        })
        confidence = "High"

    elif has_pdf and has_web:
        steps.append("🧠 Synthesiser: combining PDF + Web results")
        chain = PDF_WEB_SYNTHESIZER_PROMPT | llm
        response = chain.invoke({
            "system_prompt": system_prompt,
            "query": state["query"],
            "pdf_results": state["pdf_result"]["formatted"],
            "web_results": state["web_result"]["formatted"],
        })
        confidence = "Medium"

    elif has_sql and has_web:
        steps.append("🧠 Synthesiser: combining DB + Web results")
        chain = COMBINED_SYNTHESIZER_PROMPT | llm
        response = chain.invoke({
            "system_prompt": system_prompt,
            "query": state["query"],
            "sql_results": state["sql_result"]["markdown_table"],
            "web_results": state["web_result"]["formatted"],
        })
        confidence = "High"

    elif has_sql:
        steps.append("🧠 Synthesiser: analysing DB results only")
        chain = SQL_ONLY_SYNTHESIZER_PROMPT | llm
        response = chain.invoke({
            "system_prompt": system_prompt,
            "query": state["query"],
            "sql_results": state["sql_result"]["markdown_table"],
        })
        confidence = "Medium"

    elif has_pdf:
        steps.append("🧠 Synthesiser: analysing PDF results only")
        chain = PDF_ONLY_SYNTHESIZER_PROMPT | llm
        response = chain.invoke({
            "system_prompt": system_prompt,
            "query": state["query"],
            "pdf_results": state["pdf_result"]["formatted"],
        })
        confidence = "Medium"

    elif has_web:
        steps.append("🧠 Synthesiser: analysing Web results only")
        chain = WEB_SYNTHESIZER_PROMPT | llm
        response = chain.invoke({
            "system_prompt": system_prompt,
            "query": state["query"],
            "web_results": state["web_result"]["formatted"],
        })
        confidence = "Medium"

    else:
        steps.append("⚠️  Synthesiser: no data found in DB, PDF, or Web")
        return {
            **state,
            "final_answer": (
                "## No Data Found\n\n"
                "Neither the database, bond documents, nor web search returned relevant data.\n\n"
                "**Suggestions:**\n"
                "- Verify the company name, ISIN, or instrument identifier\n"
                "- Ingest bond PDFs first: `rag pdf ingest <path>`\n"
                "- Check that `TAVILY_API_KEY` is set for web search\n"
                "- Ensure the MySQL database is seeded: `rag db setup --seed`"
            ),
            "confidence": "Low",
            "processing_steps": steps,
        }

    steps.append("✅ Synthesiser: answer generated")
    return {
        **state,
        "final_answer": response.content,
        "confidence": confidence,
        "processing_steps": steps,
    }


# ─── Node: parallel agents ────────────────────────────────────────────────────

def parallel_agents_node(state: AgentState) -> AgentState:
    """
    Run SQL, PDF, and Web agents concurrently using a thread pool.

    Each agent only writes its own fields (sql_result / pdf_result / web_result)
    so merging their outputs is safe with no conflicts.
    """
    sources  = state.get("data_sources", [])
    steps    = list(state.get("processing_steps", []))
    active   = [s for s in ("sql", "pdf", "web") if s in sources]

    steps.append(f"⚡ Parallel Agents: running {'+'.join(active).upper()} concurrently")
    base_state = {**state, "processing_steps": steps}

    agent_fns = {
        "sql": sql_agent_node,
        "pdf": pdf_agent_node,
        "web": web_agent_node,
    }

    t0 = time.perf_counter()
    results: dict[str, AgentState] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(active)) as pool:
        futures = {
            name: pool.submit(agent_fns[name], base_state)
            for name in active
        }
        for name, fut in futures.items():
            try:
                results[name] = fut.result(timeout=45)
            except Exception as exc:
                logger.error("Agent %s failed in parallel run: %s", name, exc)
                results[name] = base_state   # safe fallback

    elapsed = time.perf_counter() - t0

    # ── Merge agent results ────────────────────────────────────────────────────
    merged       = {**base_state}
    merged_steps = list(steps)

    for name, agent_state in results.items():
        # Collect the step(s) the agent appended (anything beyond base_state steps)
        for step in agent_state.get("processing_steps", []):
            if step not in merged_steps:
                merged_steps.append(step)

        if name == "sql":
            merged["sql_result"]    = agent_state.get("sql_result", {})
            merged["sql_attempted"] = agent_state.get("sql_attempted", False)
            merged["sql_has_data"]  = agent_state.get("sql_has_data", False)
        elif name == "pdf":
            merged["pdf_result"]    = agent_state.get("pdf_result", {})
            merged["pdf_attempted"] = agent_state.get("pdf_attempted", False)
            merged["pdf_has_data"]  = agent_state.get("pdf_has_data", False)
        elif name == "web":
            merged["web_result"]    = agent_state.get("web_result", {})
            merged["web_attempted"] = agent_state.get("web_attempted", False)
            merged["web_has_data"]  = agent_state.get("web_has_data", False)

    merged_steps.append(
        f"✅ Parallel Agents: all done in {elapsed:.1f}s "
        f"(sql={merged.get('sql_has_data')}, "
        f"pdf={merged.get('pdf_has_data')}, "
        f"web={merged.get('web_has_data')})"
    )
    merged["processing_steps"] = merged_steps
    return merged


# ─── Graph construction ───────────────────────────────────────────────────────

def build_graph() -> Any:
    builder = StateGraph(AgentState)

    builder.add_node("router",          router_node)
    builder.add_node("parallel_agents", parallel_agents_node)
    builder.add_node("synthesise",      synthesise_node)

    builder.set_entry_point("router")
    builder.add_edge("router",          "parallel_agents")
    builder.add_edge("parallel_agents", "synthesise")
    builder.add_edge("synthesise",      END)

    return builder.compile()


# ─── Public API ───────────────────────────────────────────────────────────────

_graph: Any = None


def _get_graph() -> Any:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query(
    query: str,
    company: str | None = None,
    domain: str | None = None,
    llm_provider: str | None = None,
    force_web: bool = False,
    force_sql: bool = False,
    force_pdf: bool = False,
) -> AgentState:
    # ── Cache check (skip on forced overrides) ────────────────────────────────
    use_cache = not (force_web or force_sql or force_pdf)
    if use_cache:
        cached = get_cached(query, domain=domain, company=company)
        if cached is not None:
            return cached

    initial: AgentState = {
        "query":                query,
        "company_hint":         company or "",
        "domain_override":      domain,
        "llm_provider_override": llm_provider,
        "force_web":            force_web,
        "force_sql":            force_sql,
        "force_pdf":            force_pdf,
        "processing_steps":     [],
    }
    result = _get_graph().invoke(initial)

    # ── Cache store ───────────────────────────────────────────────────────────
    if use_cache:
        set_cached(
            query, result,
            domain=domain,
            company=company,
            confidence=result.get("confidence", "Low"),
        )

    return result

"""
Web Search Tool
───────────────
Uses Tavily for web search (best-in-class for LLM RAG use cases).
Formats results into a clean structure for downstream synthesis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings


@dataclass
class WebResult:
    title: str
    url: str
    content: str
    score: float = 0.0
    published_date: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
            "published_date": self.published_date,
        }


class WebSearchTool:
    """
    Wraps Tavily search API.

    Falls back gracefully if Tavily key is not configured:
    returns an explanatory message so the agent can still synthesise
    from DB-only results.
    """

    def __init__(self) -> None:
        self._client = None
        self._available = bool(settings.tavily_api_key)
        if self._available:
            try:
                from tavily import TavilyClient
                self._client = TavilyClient(api_key=settings.tavily_api_key)
            except ImportError:
                self._available = False

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=5))
    def search(
        self,
        query: str,
        n_results: int | None = None,
        search_depth: str = "advanced",
    ) -> dict[str, Any]:
        """
        Search the web for the query.

        Returns:
            {
              "success": bool,
              "results": list[WebResult],
              "formatted": str,          # markdown-formatted for LLM consumption
              "error": str | None,
            }
        """
        if not self._available:
            return {
                "success": False,
                "results": [],
                "formatted": "_Web search unavailable: TAVILY_API_KEY not configured._",
                "error": "Tavily not configured",
            }

        k = n_results or settings.web_search_results
        try:
            raw = self._client.search(
                query=query,
                search_depth=search_depth,
                max_results=k,
                include_answer=False,
                include_raw_content=False,
            )
            results = [
                WebResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    score=r.get("score", 0.0),
                    published_date=r.get("published_date", ""),
                )
                for r in raw.get("results", [])
            ]
            formatted = self._format_results(results)
            return {
                "success": True,
                "results": results,
                "formatted": formatted,
                "error": None,
            }
        except Exception as exc:
            return {
                "success": False,
                "results": [],
                "formatted": f"_Web search failed: {exc}_",
                "error": str(exc),
            }

    @staticmethod
    def _format_results(results: list[WebResult]) -> str:
        if not results:
            return "_No web results found._"
        parts = []
        for i, r in enumerate(results, 1):
            date_str = f" ({r.published_date})" if r.published_date else ""
            parts.append(
                f"**[Source {i}]** [{r.title}]({r.url}){date_str}\n"
                f"Relevance score: {r.score:.2f}\n\n"
                f"{r.content.strip()}"
            )
        return "\n\n---\n\n".join(parts)

    def build_financial_query(self, user_query: str, entities: list[str], domain: str) -> str:
        """
        Enrich the raw user query with domain-specific financial keywords
        to maximise web search relevance.
        """
        domain_keywords = {
            "promoters": "promoter default regulatory penalty credit rating group company",
            "company": "corporate governance ESG debt restructuring rating",
            "financials": "financial results EBITDA earnings revenue debt equity ratio",
            "instruments": "bond NCD debenture covenants rating prepayment charge",
            "industry": "industry outlook credit rating default disruption regulation",
        }
        entity_str = " ".join(entities) if entities else ""
        kw = domain_keywords.get(domain, "financial credit analysis")
        return f"{entity_str} {user_query} {kw}".strip()

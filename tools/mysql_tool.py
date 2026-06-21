"""
MySQL Tool
──────────
Responsibilities:
  1. Maintain a connection pool to MySQL.
  2. Introspect the schema and cache it.
  3. Accept a natural-language query → generate SQL via LLM → execute → return results.
  4. Perform a semantic relevance check so the agent knows whether the DB
     actually returned useful data or just an empty/irrelevant result set.
"""

from __future__ import annotations

import json
import textwrap
from typing import Any

import sqlalchemy as sa
from sqlalchemy import text, inspect
from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings
from llm.provider import get_sql_llm
from prompts import SQL_GENERATOR_PROMPT


# ─── Engine singleton ─────────────────────────────────────────────────────────

_engine: sa.Engine | None = None


def _get_engine() -> sa.Engine:
    global _engine
    if _engine is None:
        _engine = sa.create_engine(
            settings.mysql_sync_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            connect_args={"connect_timeout": 10},
        )
    return _engine


# ─── Schema introspection ─────────────────────────────────────────────────────

_schema_cache: str | None = None


def get_db_schema(force_refresh: bool = False) -> str:
    """
    Return a text description of all tables + columns in the database.
    Cached after first call unless force_refresh=True.
    """
    global _schema_cache
    if _schema_cache and not force_refresh:
        return _schema_cache

    engine = _get_engine()
    insp = inspect(engine)
    lines: list[str] = []

    for table_name in insp.get_table_names():
        cols = insp.get_columns(table_name)
        fks = insp.get_foreign_keys(table_name)
        pk = insp.get_pk_constraint(table_name)

        col_defs = ", ".join(
            f"{c['name']} {str(c['type'])}"
            + (" PK" if c['name'] in (pk.get("constrained_columns") or []) else "")
            for c in cols
        )
        fk_defs = "; ".join(
            f"{fk['constrained_columns']} → {fk['referred_table']}.{fk['referred_columns']}"
            for fk in fks
        )
        lines.append(f"TABLE {table_name} ({col_defs})" + (f"  FK: {fk_defs}" if fk_defs else ""))

    _schema_cache = "\n".join(lines)
    return _schema_cache


# ─── SQL execution ────────────────────────────────────────────────────────────

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
def execute_sql(sql: str) -> list[dict[str, Any]]:
    """Run a SELECT query and return rows as list-of-dicts."""
    engine = _get_engine()
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        cols = list(result.keys())
        rows = [dict(zip(cols, row)) for row in result.fetchmany(settings.max_sql_rows)]
    return rows


def rows_to_markdown(rows: list[dict[str, Any]]) -> str:
    """Convert query results to a markdown table string."""
    if not rows:
        return "_No results found._"
    headers = list(rows[0].keys())
    header_row = " | ".join(headers)
    separator = " | ".join(["---"] * len(headers))
    data_rows = [
        " | ".join(str(row.get(h, "")) for h in headers)
        for row in rows
    ]
    return "\n".join(["| " + header_row + " |",
                      "| " + separator + " |"] +
                     ["| " + r + " |" for r in data_rows])


# ─── Main tool class ──────────────────────────────────────────────────────────

class MySQLTool:
    """
    Agentic MySQL tool.

    Call .run(query, entities, domain) to:
      1. Generate SQL from natural language.
      2. Execute the SQL.
      3. Return structured output with metadata.
    """

    def __init__(self) -> None:
        self._llm = get_sql_llm()
        self._chain = SQL_GENERATOR_PROMPT | self._llm

    def generate_sql(self, query: str, entities: list[str], domain: str) -> str:
        """Ask the LLM to translate the NL query into a SQL statement."""
        schema = get_db_schema()
        response = self._chain.invoke({
            "schema": schema,
            "query": query,
            "entities": ", ".join(entities) if entities else "none specified",
            "domain": domain,
            "max_rows": settings.max_sql_rows,
        })
        sql = response.content.strip()
        # Strip markdown fences if the model wraps in ```sql ... ```
        if sql.startswith("```"):
            sql = "\n".join(
                line for line in sql.splitlines()
                if not line.startswith("```")
            ).strip()
        return sql

    def run(
        self,
        query: str,
        entities: list[str],
        domain: str,
    ) -> dict[str, Any]:
        """
        Full pipeline: NL → SQL → execute → return result dict.

        Returns:
            {
              "success": bool,
              "sql": str,
              "rows": list[dict],
              "markdown_table": str,
              "row_count": int,
              "has_data": bool,
              "error": str | None,
            }
        """
        result: dict[str, Any] = {
            "success": False,
            "sql": "",
            "rows": [],
            "markdown_table": "",
            "row_count": 0,
            "has_data": False,
            "error": None,
        }

        try:
            sql = self.generate_sql(query, entities, domain)
            result["sql"] = sql

            if sql == "NO_SQL":
                result["error"] = "Schema does not contain data relevant to this query."
                return result

            rows = execute_sql(sql)
            md = rows_to_markdown(rows)

            result.update({
                "success": True,
                "rows": rows,
                "markdown_table": md,
                "row_count": len(rows),
                "has_data": len(rows) > 0,
            })

        except Exception as exc:
            result["error"] = str(exc)

        return result

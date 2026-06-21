"""
PDF Tool — wraps bond_rag's BondRAGPipeline.

This tool retrieves relevant chunks from ingested bond PDF documents
and formats them for the unified Anthropic/OpenAI synthesiser.
Intentionally decoupled from Ollama — synthesis is always done by the
main LLM provider (Anthropic Claude / OpenAI GPT-4o).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─── ISIN extraction helper ────────────────────────────────────────────────────

ISIN_RE = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b")


def extract_isins(text: str) -> list[str]:
    """Return unique ISINs found in *text*, in order of appearance."""
    return list(dict.fromkeys(ISIN_RE.findall(text)))


# ─── Lazy pipeline singleton ──────────────────────────────────────────────────

_pipeline: Any = None


def _get_pipeline() -> Any:
    global _pipeline
    if _pipeline is None:
        try:
            from bond_rag.rag.pipeline import BondRAGPipeline
            _pipeline = BondRAGPipeline()
            logger.info("BondRAGPipeline initialised")
        except ImportError as exc:
            msg = str(exc)
            if "bond_rag" in msg:
                logger.error("bond_rag package not found: %s", exc)
            else:
                logger.error(
                    "bond_rag dependency missing: %s — "
                    "run: pip install -r requirements.txt",
                    exc,
                )
            raise
    return _pipeline


# ─── PDFTool ──────────────────────────────────────────────────────────────────

class PDFTool:
    """
    Retrieve relevant chunks from bond PDF documents.

    Returns a dict compatible with AgentState['pdf_result']:
      {
        "success": bool,
        "has_data": bool,
        "chunks": list[dict],    # raw chunk dicts from BondRetriever
        "formatted": str,        # markdown-formatted text for synthesiser
        "isins_found": list[str],
        "sources": list[str],    # unique source filenames
        "error": str | None,
      }
    """

    def retrieve(
        self,
        query: str,
        top_k: int = 6,
        force_isins: list[str] | None = None,
        force_filename: str | None = None,
    ) -> dict[str, Any]:
        # Auto-extract ISINs from query if not forced
        if force_isins is None:
            force_isins = extract_isins(query) or None

        try:
            pipeline = _get_pipeline()
            chunks = pipeline.retrieve(
                query=query,
                top_k=top_k,
                force_isins=force_isins,
                force_filename=force_filename,
            )
        except Exception as exc:
            logger.warning("PDF retrieval failed: %s", exc)
            return {
                "success": False,
                "has_data": False,
                "chunks": [],
                "formatted": "",
                "isins_found": [],
                "sources": [],
                "error": str(exc),
            }

        if not chunks:
            return {
                "success": True,
                "has_data": False,
                "chunks": [],
                "formatted": "",
                "isins_found": force_isins or [],
                "sources": [],
                "error": None,
            }

        # Format chunks for the synthesiser prompt
        formatted_parts: list[str] = []
        seen_sources: set[str] = set()
        all_isins: list[str] = list(force_isins or [])

        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            isin = meta.get("active_isin") or meta.get("isin", "")
            source = meta.get("source", meta.get("filename", "unknown"))
            text = chunk.get("text", chunk.get("page_content", ""))

            seen_sources.add(source)
            if isin and isin not in all_isins:
                all_isins.append(isin)

            header = f"[Chunk {i}"
            if isin:
                header += f" | ISIN: {isin}"
            if source:
                header += f" | Source: {source}"
            header += "]"

            formatted_parts.append(f"{header}\n{text.strip()}")

        formatted = "\n\n---\n\n".join(formatted_parts)

        return {
            "success": True,
            "has_data": True,
            "chunks": chunks,
            "formatted": formatted,
            "isins_found": all_isins,
            "sources": sorted(seen_sources),
            "error": None,
        }

    # ── Pipeline delegation helpers ────────────────────────────────────────────

    def ingest(self, path: str | Path, recursive: bool = False) -> dict[str, Any]:
        """Ingest PDF(s) from a file or directory path."""
        pipeline = _get_pipeline()
        p = Path(path)

        if p.is_dir():
            # ingest_directory scans for *.pdf; recurse via glob pattern
            glob = "**/*.pdf" if recursive else "*.pdf"
            results = pipeline.ingest_directory(directory=p, glob=glob)
            ingested = sum(1 for r in results if not r.already_loaded)
            skipped  = sum(1 for r in results if r.already_loaded)
            chunks   = sum(r.chunks_added for r in results)
            isins    = sorted({isin for r in results for isin in r.isins})
            return {
                "files_ingested": ingested,
                "files_skipped":  skipped,
                "chunks_added":   chunks,
                "isins_found":    isins,
            }
        else:
            # Single file
            result = pipeline.ingest(pdf_path=p)
            return {
                "files_ingested": 0 if result.already_loaded else 1,
                "files_skipped":  1 if result.already_loaded else 0,
                "chunks_added":   result.chunks_added,
                "isins_found":    result.isins,
            }

    def list_files(self) -> list[dict[str, Any]]:
        """Return metadata for all ingested PDFs."""
        try:
            return _get_pipeline().list_files()
        except Exception:
            return []

    def list_isins(self) -> list[str]:
        """Return all ISINs across ingested documents."""
        try:
            return _get_pipeline().list_isins()
        except Exception:
            return []

    def stats(self) -> dict[str, Any]:
        """Return pipeline statistics (chunk count, doc count, etc.)."""
        try:
            return _get_pipeline().stats()
        except Exception as exc:
            return {"error": str(exc)}

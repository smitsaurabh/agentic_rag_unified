"""
Hybrid BM25 + dense retriever with cross-encoder reranking — production version.

Improvements over v1
─────────────────────
• Uses Pydantic RetrievedChunk.from_chroma_hit() factory for safe construction.
• ISIN/filename filter falls back gracefully if no results are found.
• Reranker is lazy-loaded and shared.
• Structured logging for each stage with timing.
• Full type annotations.
"""

from __future__ import annotations

import re
import time
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from bond_rag.core.config import RetrieverSettings, get_settings
from bond_rag.core.exceptions import NoResultsError, RetrievalError
from bond_rag.core.logging import get_logger
from bond_rag.core.models import RetrievedChunk
from bond_rag.retrieval.embedder import Embedder
from bond_rag.retrieval.vector_store import VectorStore

logger = get_logger(__name__)

_ISIN_RE     = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b")
_FILENAME_RE = re.compile(
    r"\b(prospectus|issuance|offering|supplement|indenture|circular|term[-\s]?sheet)\b",
    re.IGNORECASE,
)


class BondRetriever:
    """
    Three-stage retriever:
      1. ISIN/filename metadata pre-filter
      2. Hybrid dense (cosine) + BM25 fusion
      3. Cross-encoder reranking
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder:     Embedder,
        config: Optional[RetrieverSettings] = None,
    ) -> None:
        self.store    = vector_store
        self.embedder = embedder
        self.cfg      = config or get_settings().retriever
        self._reranker: Optional[CrossEncoder] = None

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            logger.info("Loading reranker model", model=self.cfg.reranker_model)
            try:
                self._reranker = CrossEncoder(self.cfg.reranker_model)
            except Exception as exc:
                raise RetrievalError("Failed to load reranker model") from exc
        return self._reranker

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        force_isins: Optional[list[str]] = None,
        force_filename: Optional[str] = None,
        raise_if_empty: bool = False,
    ) -> list[RetrievedChunk]:
        """
        Full three-stage retrieval.

        Parameters
        ----------
        query          : Natural language question
        top_k          : Override cfg.top_k
        force_isins    : Restrict to these ISINs (skips auto-detection)
        force_filename : Restrict to this filename
        raise_if_empty : Raise NoResultsError instead of returning []

        Returns
        -------
        Ranked RetrievedChunk list, best first.
        """
        k   = top_k or self.cfg.top_k
        t0  = time.perf_counter()

        # ── Stage 1: build metadata filter ───────────────────────────────────
        detected_isins, detected_file = _extract_query_context(query)
        isins    = force_isins    or (detected_isins    if self.cfg.use_isin_filter     else [])
        filename = force_filename or (detected_file     if self.cfg.use_filename_filter else None)
        where    = _build_where_filter(isins, filename)

        logger.debug("Retrieval started", query=query[:80], isins=isins, filename=filename)

        # ── Stage 2a: dense retrieval ─────────────────────────────────────────
        query_emb = self.embedder.embed_query(query)
        raw_hits  = self.store.query(query_emb, where=where)

        # Fallback: retry without filter if pre-filter returned nothing
        if not raw_hits and where:
            logger.info("Pre-filter returned 0 results — retrying without filter")
            raw_hits = self.store.query(query_emb)

        if not raw_hits:
            if raise_if_empty:
                raise NoResultsError(query)
            logger.warning("No chunks retrieved", query=query[:80])
            return []

        # ── Stage 2b: BM25 on candidate set ──────────────────────────────────
        texts       = [h["text"] for h in raw_hits]
        bm25_scores = _bm25_score(query, texts)

        # ── Stage 2c: fusion ──────────────────────────────────────────────────
        dense_arr = np.array([h["score"] for h in raw_hits])
        bm25_arr  = np.array(bm25_scores)
        fused     = (
            self.cfg.dense_weight * _minmax(dense_arr)
            + self.cfg.bm25_weight * _minmax(bm25_arr)
        )

        # Keep top 2×k candidates for reranking (was 3×k — reduced for speed)
        n_rerank = min(k * 2, len(raw_hits))
        top_idx  = np.argsort(fused)[::-1][:n_rerank]

        # ── Stage 3: cross-encoder rerank ─────────────────────────────────────
        rerank_pairs  = [(query, texts[i]) for i in top_idx]
        rerank_scores = self.reranker.predict(rerank_pairs).tolist()

        results: list[RetrievedChunk] = []
        for rank, orig_idx in enumerate(top_idx):
            hit = raw_hits[int(orig_idx)]
            results.append(
                RetrievedChunk.from_chroma_hit(
                    hit          = hit,
                    dense_score  = float(dense_arr[orig_idx]),
                    bm25_score   = float(bm25_arr[orig_idx]),
                    rerank_score = float(rerank_scores[rank]),
                )
            )

        results.sort(key=lambda x: x.final_score, reverse=True)
        final = results[:k]

        elapsed = time.perf_counter() - t0
        logger.info(
            "Retrieval complete",
            query=query[:80],
            candidates=len(raw_hits),
            returned=len(final),
            top_score=f"{final[0].final_score:.3f}" if final else "N/A",
            elapsed_s=f"{elapsed:.2f}",
        )
        return final


# ── Module-level helpers ───────────────────────────────────────────────────────

def _extract_query_context(query: str) -> tuple[list[str], Optional[str]]:
    isins = list(dict.fromkeys(_ISIN_RE.findall(query)))
    m     = _FILENAME_RE.search(query)
    return isins, m.group(0).lower() if m else None


def _build_where_filter(
    isins: list[str],
    filename: Optional[str],
) -> Optional[dict]:
    conditions = []

    if len(isins) == 1:
        conditions.append({"isin": {"$eq": isins[0]}})
    elif len(isins) > 1:
        conditions.append({"$or": [{"isin": {"$eq": i}} for i in isins]})

    if filename:
        conditions.append({"filename": {"$contains": filename}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _bm25_score(query: str, corpus: list[str]) -> list[float]:
    tokenised = [doc.lower().split() for doc in corpus]
    bm25      = BM25Okapi(tokenised)
    return bm25.get_scores(query.lower().split()).tolist()


def _minmax(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return np.zeros_like(arr) if mx == mn else (arr - mn) / (mx - mn)

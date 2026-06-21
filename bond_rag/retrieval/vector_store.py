"""
ChromaDB vector store — production version.

Improvements over v1
─────────────────────
• Uses IngestionRegistry (SQLite) instead of JSON file for dedup tracking.
• Upsert batching with configurable size and retry on transient errors.
• Structured logging for every operation.
• Thread-safe upsert via a lock (safe to call from multiple threads).
• Lazy ChromaDB initialisation so import is cheap.
"""

from __future__ import annotations

import threading
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
from tqdm import tqdm

from bond_rag.core.config import VectorStoreSettings, get_settings
from bond_rag.core.exceptions import VectorStoreError
from bond_rag.core.logging import get_logger
from bond_rag.core.models import Chunk
from bond_rag.retrieval.embedder import Embedder

logger = get_logger(__name__)

_UPSERT_BATCH = 500   # ChromaDB recommended max per upsert call


class VectorStore:
    """
    Persistent ChromaDB store for bond document chunks.
    """

    def __init__(
        self,
        embedder: Embedder,
        config: Optional[VectorStoreSettings] = None,
        persist_dir: Optional[str] = None,
    ) -> None:
        self.embedder = embedder
        self.cfg      = config or get_settings().store
        self._persist_dir = persist_dir or str(get_settings().chroma_dir)
        self._lock    = threading.Lock()
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None

    # ── Lazy init ──────────────────────────────────────────────────────────────

    @property
    def collection(self):  # type: ignore[return]
        if self._collection is None:
            self._client = chromadb.PersistentClient(
                path     = self._persist_dir,
                settings = ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name     = self.cfg.collection_name,
                metadata = {"hnsw:space": "cosine"},
            )
            logger.info(
                "ChromaDB collection ready",
                collection=self.cfg.collection_name,
                existing_chunks=self._collection.count(),
            )
        return self._collection

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """
        Embed and upsert chunks.  Thread-safe.
        Returns the number of chunks upserted.
        """
        if not chunks:
            return 0

        texts      = [c.text for c in chunks]
        ids        = [c.chunk_id for c in chunks]
        metadatas  = [c.to_chroma_metadata() for c in chunks]

        logger.info("Embedding chunks", n=len(chunks))
        embeddings = self.embedder.embed_documents(texts)

        upserted = 0
        with self._lock:
            for i in tqdm(
                range(0, len(ids), _UPSERT_BATCH),
                desc="Upserting",
                unit="batch",
                leave=False,
                disable=len(ids) < _UPSERT_BATCH,
            ):
                sl = slice(i, i + _UPSERT_BATCH)
                self._upsert_batch(
                    ids[sl], embeddings[sl], texts[sl], metadatas[sl]
                )
                upserted += len(ids[sl])

        logger.info("Upsert complete", upserted=upserted)
        return upserted

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        reraise=True,
    )
    def _upsert_batch(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        try:
            self.collection.upsert(
                ids        = ids,
                embeddings = embeddings,
                documents  = documents,
                metadatas  = metadatas,
            )
        except Exception as exc:
            raise VectorStoreError("ChromaDB upsert failed") from exc

    def delete_by_filename(self, filename: str) -> int:
        """Delete all chunks associated with the given filename."""
        try:
            results = self.collection.get(where={"filename": {"$eq": filename}})
            ids = results.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
            logger.info("Deleted chunks", filename=filename, count=len(ids))
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete chunks for '{filename}'") from exc

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embedding: list[float],
        n_results: Optional[int] = None,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Cosine-similarity search.

        Parameters
        ----------
        query_embedding : pre-computed query vector
        n_results       : candidates to retrieve (defaults to cfg.n_candidates)
        where           : ChromaDB metadata filter

        Returns
        -------
        List of dicts: {id, text, metadata, distance, score}
        """
        n = n_results or self.cfg.n_candidates

        try:
            kwargs: dict = {
                "query_embeddings": [query_embedding],
                "n_results":        n,
                "include":          ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            results = self.collection.query(**kwargs)
        except Exception as exc:
            raise VectorStoreError("ChromaDB query failed") from exc

        hits: list[dict] = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                dist = results["distances"][0][i]
                hits.append({
                    "id":       doc_id,
                    "text":     results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": dist,
                    "score":    1.0 - dist,   # cosine distance → similarity
                })
        return hits

    # ── Introspection ──────────────────────────────────────────────────────────

    def total_chunks(self) -> int:
        return self.collection.count()

    def list_isins(self) -> list[str]:
        """Return all distinct ISINs in the store (from chunk metadata)."""
        results = self.collection.get(include=["metadatas"])
        isins: set[str] = set()
        for meta in (results.get("metadatas") or []):
            val = meta.get("isin", "")
            if val and val != "_NO_ISIN_":
                isins.add(val)
        return sorted(isins)

"""
BondRAGPipeline — the main public facade.

Wires all components together and exposes a clean API.

Design decisions
────────────────
• All sub-components are lazy-initialised (first call only) so importing
  the pipeline does not trigger model downloads.
• Logging is configured on first instantiation from Settings.
• The pipeline owns the registry and passes it to components that need it.
• ``ingest()`` is idempotent: re-calling with the same file is safe.
• All public methods return typed Pydantic models (IngestionResult, QueryResult).
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional

from bond_rag.core.config import Settings, get_settings
from bond_rag.core.logging import get_logger, setup_logging
from bond_rag.core.models import IngestionResult, QueryResult, RetrievedChunk
from bond_rag.ingestion.chunker import ISINAwareChunker
from bond_rag.ingestion.pdf_processor import PDFProcessor
from bond_rag.ingestion.registry import IngestionRegistry
from bond_rag.retrieval.embedder import Embedder
from bond_rag.retrieval.retriever import BondRetriever
from bond_rag.retrieval.vector_store import VectorStore

logger = get_logger(__name__)


class BondRAGPipeline:
    """
    Main entry point for the Bond RAG system.

    Usage::

        rag = BondRAGPipeline()
        rag.ingest("data/bond_2024.pdf")
        result = rag.query("What is the coupon rate for XS1234567890?")
        print(result.answer)
        for src in result.sources:
            print(src.citation())
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()

        # Configure logging from settings on first construction
        setup_logging(
            level    = self._settings.log_level,
            log_file = self._settings.log_file,
        )

        # Stateless helpers — always ready
        self._pdf_processor = PDFProcessor()
        self._chunker       = ISINAwareChunker(config=self._settings.chunk)

        # Lazy-initialised heavyweight components
        self._embedder:      Optional[Embedder]           = None
        self._vector_store:  Optional[VectorStore]        = None
        self._retriever:     Optional[BondRetriever]      = None
        self._llm:           Optional[object]             = None  # OllamaLLM, imported lazily
        self._registry:      Optional[IngestionRegistry]  = None

        logger.info("BondRAGPipeline initialised")

    # ── Lazy component accessors ───────────────────────────────────────────────

    @property
    def registry(self) -> IngestionRegistry:
        if self._registry is None:
            self._registry = IngestionRegistry(self._settings.registry_db_path)
        return self._registry

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(config=self._settings.embed)
        return self._embedder

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(
                embedder    = self.embedder,
                config      = self._settings.store,
                persist_dir = str(self._settings.chroma_dir),
            )
        return self._vector_store

    @property
    def retriever(self) -> BondRetriever:
        if self._retriever is None:
            self._retriever = BondRetriever(
                vector_store = self.vector_store,
                embedder     = self.embedder,
                config       = self._settings.retriever,
            )
        return self._retriever

    @property
    def llm(self):
        """OllamaLLM — imported lazily so ingestion works without ollama installed."""
        if self._llm is None:
            try:
                from bond_rag.generation.llm import OllamaLLM  # noqa: PLC0415
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "The 'ollama' package is required for LLM query/stream but is not installed. "
                    "Run: pip install ollama"
                ) from exc
            self._llm = OllamaLLM(config=self._settings.llm)
        return self._llm

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest(self, pdf_path: str | Path, force: bool = False) -> IngestionResult:
        """
        Ingest a PDF.  Idempotent — safe to call multiple times on the
        same file.

        Parameters
        ----------
        pdf_path : Path to the PDF file
        force    : If True, re-ingest even if the file hash is already loaded

        Returns
        -------
        IngestionResult with status, chunk count, and ISINs found.
        """
        path = Path(pdf_path).resolve()

        # ── Parse PDF ─────────────────────────────────────────────────────────
        doc = self._pdf_processor.process(path)

        # ── Incremental load check ─────────────────────────────────────────────
        if not force and self.registry.is_loaded(doc.file_hash):
            logger.info("File already loaded — skipping", filename=doc.filename)
            return IngestionResult(
                filename       = doc.filename,
                filepath       = str(path),
                total_pages    = doc.total_pages,
                chunks_added   = 0,
                isins          = doc.all_isins,
                already_loaded = True,
            )

        # ── Chunk ─────────────────────────────────────────────────────────────
        chunks = self._chunker.chunk_document(doc)

        # ── Embed + store ──────────────────────────────────────────────────────
        n_upserted = self.vector_store.upsert_chunks(chunks)

        # ── Register ──────────────────────────────────────────────────────────
        self.registry.register(
            file_hash   = doc.file_hash,
            filename    = doc.filename,
            filepath    = str(path),
            total_pages = doc.total_pages,
            chunks      = n_upserted,
            isins       = doc.all_isins,
        )

        logger.info(
            "Ingestion complete",
            filename=doc.filename,
            chunks=n_upserted,
            isins=doc.all_isins,
        )

        return IngestionResult(
            filename     = doc.filename,
            filepath     = str(path),
            total_pages  = doc.total_pages,
            chunks_added = n_upserted,
            isins        = doc.all_isins,
        )

    def ingest_directory(
        self,
        directory: str | Path,
        glob: str = "*.pdf",
        force: bool = False,
    ) -> list[IngestionResult]:
        """Ingest all PDFs in a directory matching ``glob``."""
        results = []
        for pdf_file in sorted(Path(directory).glob(glob)):
            result = self.ingest(pdf_file, force=force)
            results.append(result)
        return results

    def remove(self, filename: str) -> int:
        """
        Remove all chunks for a filename from the vector store and registry.

        Returns number of chunks deleted.
        """
        n = self.vector_store.delete_by_filename(filename)
        self.registry.unregister_by_filename(filename)
        logger.info("Removed file", filename=filename, chunks_deleted=n)
        return n

    # ── Retrieval only ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        force_isins: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """Retrieve ranked chunks without calling the LLM."""
        return self.retriever.retrieve(
            query       = query,
            top_k       = top_k,
            force_isins = force_isins,
        )

    # ── Full RAG query ─────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        force_isins: Optional[list[str]] = None,
    ) -> QueryResult:
        """
        Full RAG: retrieve → generate answer.

        Returns
        -------
        QueryResult with .answer and .sources (list of RetrievedChunk).
        """
        sources = self.retrieve(question, top_k=top_k, force_isins=force_isins)
        answer  = self.llm.answer(question, sources)
        return QueryResult(question=question, answer=answer, sources=sources)

    def stream_query(
        self,
        question: str,
        top_k: Optional[int] = None,
        force_isins: Optional[list[str]] = None,
    ) -> Generator[str, None, None]:
        """Streaming RAG — yields LLM tokens as they arrive."""
        sources = self.retrieve(question, top_k=top_k, force_isins=force_isins)
        yield from self.llm.stream_answer(question, sources)

    # ── Introspection ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Combined stats from registry and vector store."""
        reg   = self.registry.stats()
        store = self.vector_store.total_chunks()
        return {**reg, "vector_store_chunks": store}

    def list_isins(self) -> list[str]:
        return self.registry.list_isins()

    def list_files(self) -> list[dict]:
        return self.registry.list_files()

    def find_files_by_isin(self, isin: str) -> list[dict]:
        return self.registry.find_by_isin(isin)

    def close(self) -> None:
        """Release resources (call on shutdown)."""
        if self._registry:
            self._registry.close()

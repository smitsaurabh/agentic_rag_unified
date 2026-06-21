"""
Exception hierarchy for the Bond RAG system.

All custom exceptions inherit from BondRAGError so callers can catch
the entire family with a single ``except BondRAGError``.

Hierarchy
─────────
BondRAGError
├── ConfigurationError          — bad settings / missing env vars
├── IngestionError              — failures during PDF ingestion
│   ├── PDFReadError            — can't open or parse the PDF
│   ├── ISINExtractionError     — ISIN regex produced unexpected results
│   └── RegistryError           — SQLite registry read/write failure
├── EmbeddingError              — model loading or encoding failure
├── VectorStoreError            — ChromaDB operation failure
│   └── CollectionNotFoundError — collection doesn't exist yet
├── RetrievalError              — query / reranking failure
│   └── NoResultsError          — query returned zero chunks
└── LLMError                    — Ollama communication failure
    ├── OllamaConnectionError   — server not reachable
    ├── OllamaModelNotFoundError — model not pulled locally
    └── LLMTimeoutError         — request exceeded timeout
"""

from __future__ import annotations


# ── Base ───────────────────────────────────────────────────────────────────────

class BondRAGError(Exception):
    """Base class for all Bond RAG exceptions."""

    def __init__(self, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (caused by: {self.cause})" if self.cause else base


# ── Configuration ──────────────────────────────────────────────────────────────

class ConfigurationError(BondRAGError):
    """Raised when the application is misconfigured."""


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestionError(BondRAGError):
    """Base class for ingestion failures."""


class PDFReadError(IngestionError):
    """Cannot open, read, or parse a PDF file."""

    def __init__(self, path: str, *, cause: BaseException | None = None) -> None:
        super().__init__(f"Failed to read PDF: {path}", cause=cause)
        self.path = path


class ISINExtractionError(IngestionError):
    """Unexpected error during ISIN extraction."""


class RegistryError(IngestionError):
    """SQLite registry read/write failure."""


# ── Embedding ─────────────────────────────────────────────────────────────────

class EmbeddingError(BondRAGError):
    """Model loading or encoding failure."""


# ── Vector store ──────────────────────────────────────────────────────────────

class VectorStoreError(BondRAGError):
    """ChromaDB operation failure."""


class CollectionNotFoundError(VectorStoreError):
    """The requested collection does not exist."""

    def __init__(self, collection_name: str) -> None:
        super().__init__(f"Collection not found: '{collection_name}'")
        self.collection_name = collection_name


# ── Retrieval ─────────────────────────────────────────────────────────────────

class RetrievalError(BondRAGError):
    """Query or reranking failure."""


class NoResultsError(RetrievalError):
    """The query returned zero chunks."""

    def __init__(self, query: str) -> None:
        super().__init__(f"No results found for query: '{query[:80]}'")
        self.query = query


# ── LLM ───────────────────────────────────────────────────────────────────────

class LLMError(BondRAGError):
    """Base class for LLM / Ollama failures."""


class OllamaConnectionError(LLMError):
    """Ollama server is not reachable."""

    def __init__(self, host: str, *, cause: BaseException | None = None) -> None:
        super().__init__(
            f"Cannot connect to Ollama at {host}. "
            "Ensure Ollama is running: `ollama serve`",
            cause=cause,
        )
        self.host = host


class OllamaModelNotFoundError(LLMError):
    """The requested model has not been pulled locally."""

    def __init__(self, model: str) -> None:
        super().__init__(
            f"Model '{model}' not found locally. "
            f"Pull it with: `ollama pull {model}`"
        )
        self.model = model


class LLMTimeoutError(LLMError):
    """LLM request exceeded the configured timeout."""

    def __init__(self, model: str, timeout: float) -> None:
        super().__init__(
            f"Model '{model}' did not respond within {timeout}s"
        )
        self.model = model
        self.timeout = timeout

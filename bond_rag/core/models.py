"""
Pydantic v2 domain models.

All data that flows through the pipeline is validated here.
Pydantic catches bad inputs early (wrong types, missing fields, constraint
violations) and generates accurate schema documentation for free.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


# ── Common constraints ─────────────────────────────────────────────────────────

NonEmptyStr   = Annotated[str,  Field(min_length=1)]
PositiveInt   = Annotated[int,  Field(gt=0)]
NormalFloat   = Annotated[float, Field(ge=0.0, le=1.0)]

ISIN_SENTINEL = "_NO_ISIN_"


# ─── Ingestion models ──────────────────────────────────────────────────────────

class PageRecord(BaseModel):
    """
    Text and metadata extracted from a single PDF page.
    Immutable after creation (frozen=True).
    """
    model_config = ConfigDict(frozen=True)

    filename:     NonEmptyStr
    filepath:     NonEmptyStr
    page_num:     PositiveInt          # 1-based
    text:         str                  # may be empty (scanned image page)
    isins:        tuple[str, ...]  = Field(default=())
    active_isin:  Optional[str]    = None   # governing ISIN for this page

    @field_validator("isins", mode="before")
    @classmethod
    def deduplicate_isins(cls, v: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        result = []
        for isin in v:
            if isin not in seen:
                seen.add(isin)
                result.append(isin)
        return tuple(result)


class DocumentRecord(BaseModel):
    """
    All pages extracted from one PDF, plus file-level metadata.
    """
    model_config = ConfigDict(frozen=True)

    filename:    NonEmptyStr
    filepath:    NonEmptyStr
    file_hash:   NonEmptyStr           # SHA-256 for deduplication
    total_pages: PositiveInt
    pages:       tuple[PageRecord, ...] = Field(default=())

    @computed_field  # type: ignore[misc]
    @property
    def all_isins(self) -> list[str]:
        """Return deduplicated ISINs in document order."""
        seen: set[str] = set()
        result = []
        for page in self.pages:
            for isin in page.isins:
                if isin not in seen:
                    seen.add(isin)
                    result.append(isin)
        return result

    @computed_field  # type: ignore[misc]
    @property
    def non_empty_pages(self) -> int:
        return sum(1 for p in self.pages if p.text.strip())


# ─── Chunking models ──────────────────────────────────────────────────────────

class Chunk(BaseModel):
    """
    A single text chunk ready for embedding and ChromaDB storage.
    """
    model_config = ConfigDict(frozen=True)

    chunk_id:     NonEmptyStr      # Deterministic MD5 for safe upsert
    isin:         str = ISIN_SENTINEL
    filename:     NonEmptyStr
    filepath:     NonEmptyStr
    page_start:   PositiveInt
    page_end:     PositiveInt
    chunk_index:  int = Field(ge=0)
    total_chunks: PositiveInt
    text:         NonEmptyStr
    all_isins_in_section: tuple[str, ...] = Field(default=())

    @field_validator("page_end")
    @classmethod
    def end_gte_start(cls, v: int, info) -> int:  # type: ignore[override]
        start = info.data.get("page_start", 1)
        if v < start:
            raise ValueError(f"page_end ({v}) must be >= page_start ({start})")
        return v

    def to_chroma_metadata(self) -> dict[str, str | int | float]:
        """
        Flat dict suitable for ChromaDB metadata storage.
        ChromaDB does not support nested objects or lists, so list fields
        are serialised as comma-separated strings.
        """
        return {
            "isin":         self.isin,
            "filename":     self.filename,
            "filepath":     self.filepath,
            "page_start":   self.page_start,
            "page_end":     self.page_end,
            "chunk_index":  self.chunk_index,
            "total_chunks": self.total_chunks,
            "all_isins":    ",".join(self.all_isins_in_section),
        }

    @staticmethod
    def make_id(filename: str, isin: str, idx: int, text: str) -> str:
        """Stable deterministic chunk ID."""
        key = f"{filename}|{isin}|{idx}|{text[:64]}"
        return hashlib.md5(key.encode()).hexdigest()


# ─── Retrieval models ─────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """
    A chunk returned by the retriever, annotated with scoring details.
    """
    model_config = ConfigDict(frozen=True)

    chunk_id:     NonEmptyStr
    text:         NonEmptyStr
    isin:         str
    filename:     NonEmptyStr
    page_start:   PositiveInt
    page_end:     PositiveInt
    chunk_index:  int = Field(ge=0)
    dense_score:  float
    bm25_score:   float
    rerank_score: float
    final_score:  float
    metadata:     dict[str, str | int | float] = Field(default_factory=dict)

    def citation(self) -> str:
        return (
            f"[{self.filename} | ISIN: {self.isin} | "
            f"pp. {self.page_start}–{self.page_end}]"
        )

    @staticmethod
    def from_chroma_hit(
        hit: dict,
        dense_score: float,
        bm25_score: float,
        rerank_score: float,
    ) -> "RetrievedChunk":
        meta = hit["metadata"]
        return RetrievedChunk(
            chunk_id     = hit["id"],
            text         = hit["text"],
            isin         = meta.get("isin", ISIN_SENTINEL),
            filename     = meta.get("filename", "unknown"),
            page_start   = int(meta.get("page_start", 1)),
            page_end     = int(meta.get("page_end", 1)),
            chunk_index  = int(meta.get("chunk_index", 0)),
            dense_score  = dense_score,
            bm25_score   = bm25_score,
            rerank_score = rerank_score,
            final_score  = rerank_score,
            metadata     = {k: v for k, v in meta.items()},
        )


# ─── Pipeline result models ───────────────────────────────────────────────────

class IngestionResult(BaseModel):
    """Returned by BondRAGPipeline.ingest()."""
    filename:       NonEmptyStr
    filepath:       NonEmptyStr
    total_pages:    int = Field(ge=0)
    chunks_added:   int = Field(ge=0)
    isins:          list[str] = Field(default_factory=list)
    already_loaded: bool = False
    skipped:        bool = False

    @computed_field  # type: ignore[misc]
    @property
    def status(self) -> str:
        if self.skipped:
            return "skipped"
        if self.already_loaded:
            return "already_loaded"
        return "loaded"


class QueryResult(BaseModel):
    """Returned by BondRAGPipeline.query() when return_sources=True."""
    question: NonEmptyStr
    answer:   str
    sources:  list[RetrievedChunk] = Field(default_factory=list)

    @computed_field  # type: ignore[misc]
    @property
    def source_citations(self) -> list[str]:
        return [s.citation() for s in self.sources]

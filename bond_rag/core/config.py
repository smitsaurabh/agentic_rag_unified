"""
Centralised, type-safe configuration using Pydantic Settings v2.

Every setting can be overridden via:
  1. A `.env` file in the project root.
  2. Environment variables prefixed with `BOND_RAG__` (double-underscore
     separates nested sections, e.g. `BOND_RAG__LLM__MODEL=mistral:7b`).

Usage::

    from bond_rag.core.config import get_settings

    settings = get_settings()          # cached singleton
    print(settings.llm.model)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─── Nested config sections ────────────────────────────────────────────────────

class ChunkSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOND_RAG__CHUNK__", extra="ignore")

    chunk_size: int = Field(default=800, gt=0, description="Characters per chunk window")
    overlap_ratio: float = Field(default=0.25, ge=0.0, lt=1.0, description="Fraction of chunk to overlap")
    min_chunk_size: int = Field(default=100, gt=0, description="Minimum chunk characters to keep")

    @property
    def step_size(self) -> int:
        return int(self.chunk_size * (1 - self.overlap_ratio))


class EmbedSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOND_RAG__EMBED__", extra="ignore")

    model_name: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="HuggingFace sentence-transformers model name",
    )
    query_prefix: str = Field(
        default="Represent this sentence for searching relevant passages: ",
        description="BGE query encoding prefix",
    )
    batch_size: int = Field(default=32, gt=0, description="Embedding batch size")
    normalize_embeddings: bool = Field(default=True)
    device: Optional[str] = Field(
        default=None,
        description="PyTorch device ('cpu', 'cuda', 'mps'). None = auto-detect",
    )


class VectorStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOND_RAG__STORE__", extra="ignore")

    collection_name: str = Field(default="bond_rag")
    n_candidates: int = Field(default=20, gt=0, description="Candidates before reranking")


class RetrieverSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOND_RAG__RETRIEVER__", extra="ignore")

    top_k: int = Field(default=6, gt=0, description="Final chunks sent to LLM")
    dense_weight: float = Field(default=0.65, ge=0.0, le=1.0)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    use_isin_filter: bool = Field(default=True)
    use_filename_filter: bool = Field(default=False)

    @property
    def bm25_weight(self) -> float:
        return 1.0 - self.dense_weight


class OCRSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOND_RAG__OCR__", extra="ignore")

    enabled: bool = Field(
        default=True,
        description="Enable OCR for scanned/image-only pages",
    )
    min_text_length: int = Field(
        default=50,
        ge=0,
        description=(
            "Pages whose extracted text is shorter than this (characters) are "
            "treated as scanned and sent to OCR"
        ),
    )
    dpi: int = Field(
        default=300,
        ge=72,
        le=600,
        description="Resolution for rendering pages before OCR (higher = more accurate, slower)",
    )
    language: str = Field(
        default="eng",
        description=(
            "Tesseract language code(s). Use '+' for multiple, e.g. 'eng+fra'. "
            "Run 'tesseract --list-langs' to see what is installed."
        ),
    )
    tesseract_cmd: Optional[str] = Field(
        default=None,
        description=(
            "Full path to the tesseract binary if it is not on PATH. "
            "e.g. '/usr/local/bin/tesseract' or 'C:/Program Files/Tesseract-OCR/tesseract.exe'"
        ),
    )
    preprocess: bool = Field(
        default=True,
        description=(
            "Apply image pre-processing before OCR (grayscale, deskew, denoise). "
            "Improves accuracy on low-quality scans. Requires opencv-python-headless."
        ),
    )


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BOND_RAG__LLM__", extra="ignore")

    model: str = Field(default="llama3.1:8b")
    ollama_host: str = Field(default="http://localhost:11434")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    request_timeout: float = Field(default=120.0, gt=0, description="Seconds before request timeout")

    system_prompt: str = Field(
        default=(
            "You are a precise financial analyst assistant specialising in bond markets. "
            "Answer questions strictly based on the provided context excerpts from bond "
            "prospectuses and issuance documents. "
            "If the context does not contain enough information to answer confidently, "
            "say 'Not found in the provided documents.' rather than guessing. "
            "Always cite the ISIN and source document name when referencing bond-specific data."
        )
    )

    @field_validator("ollama_host")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


# ─── Root settings ─────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    Root settings object.  All nested sections are composed here.

    Nested values can be overridden via environment variables using
    double-underscore notation, e.g.::

        BOND_RAG__LLM__MODEL=mistral:7b
        BOND_RAG__CHUNK__CHUNK_SIZE=1200
    """

    model_config = SettingsConfigDict(
        env_prefix="BOND_RAG__",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    db_dir: Path = Field(
        default=Path("db"),
        description="Directory for ChromaDB and SQLite registry",
    )
    data_dir: Path = Field(
        default=Path("data"),
        description="Default directory to scan for PDFs",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_file: Optional[Path] = Field(default=None, description="Optional log file path")

    # ── Nested sections ───────────────────────────────────────────────────────
    chunk:     ChunkSettings       = Field(default_factory=ChunkSettings)
    embed:     EmbedSettings       = Field(default_factory=EmbedSettings)
    store:     VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    retriever: RetrieverSettings   = Field(default_factory=RetrieverSettings)
    llm:       LLMSettings         = Field(default_factory=LLMSettings)
    ocr:       OCRSettings         = Field(default_factory=OCRSettings)

    @field_validator("db_dir", "data_dir", mode="after")
    @classmethod
    def ensure_dirs_exist(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def chroma_dir(self) -> Path:
        return self.db_dir / "chroma"

    @property
    def registry_db_path(self) -> Path:
        return self.db_dir / "registry.db"


# ─── Cached singleton ──────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    In tests, call ``get_settings.cache_clear()`` before overriding settings
    to ensure the cache is refreshed.
    """
    return Settings()

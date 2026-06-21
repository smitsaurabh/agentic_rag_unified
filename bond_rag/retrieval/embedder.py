"""
Embedder — production wrapper around sentence-transformers.

Improvements over v1
─────────────────────
• Auto device detection (CUDA → MPS → CPU).
• Tenacity retry with exponential backoff for transient GPU/OOM errors.
• Lazy model load — first call only.
• Structured logging for load time and batch throughput.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from bond_rag.core.config import EmbedSettings, get_settings
from bond_rag.core.exceptions import EmbeddingError
from bond_rag.core.logging import get_logger

logger = get_logger(__name__)


def _auto_device() -> str:
    """Return the best available PyTorch device string."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class Embedder:
    """
    Sentence-transformers embedder with auto-device detection and retry.
    """

    def __init__(self, config: Optional[EmbedSettings] = None) -> None:
        self.cfg    = config or get_settings().embed
        self._model: Optional[SentenceTransformer] = None
        self._device = self.cfg.device or _auto_device()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            t0 = time.perf_counter()
            logger.info("Loading embedding model", model=self.cfg.model_name, device=self._device)
            try:
                self._model = SentenceTransformer(
                    self.cfg.model_name,
                    device=self._device,
                )
            except Exception as exc:
                raise EmbeddingError(
                    f"Failed to load model '{self.cfg.model_name}'"
                ) from exc
            elapsed = time.perf_counter() - t0
            logger.info("Embedding model loaded", elapsed_s=f"{elapsed:.1f}")
        return self._model

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode document texts for indexing (no query prefix)."""
        if not texts:
            return []
        return self._encode(texts, show_progress_bar=len(texts) > 50)

    def embed_query(self, query: str) -> list[float]:
        """Encode a single query string (with BGE query prefix)."""
        prefixed = self.cfg.query_prefix + query
        return self._encode([prefixed])[0]

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Batch-encode multiple query strings."""
        prefixed = [self.cfg.query_prefix + q for q in queries]
        return self._encode(prefixed)

    # ── Internal ───────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type((RuntimeError, MemoryError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _encode(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        try:
            t0 = time.perf_counter()
            embeddings = self.model.encode(
                texts,
                batch_size           = self.cfg.batch_size,
                normalize_embeddings = self.cfg.normalize_embeddings,
                show_progress_bar    = show_progress_bar,
                convert_to_numpy     = True,
            )
            elapsed = time.perf_counter() - t0
            throughput = len(texts) / elapsed if elapsed > 0 else 0
            logger.debug(
                "Encoded batch",
                n=len(texts),
                elapsed_s=f"{elapsed:.2f}",
                docs_per_sec=f"{throughput:.0f}",
            )
            return embeddings.tolist()  # type: ignore[union-attr]
        except (RuntimeError, MemoryError):
            raise
        except Exception as exc:
            raise EmbeddingError("Encoding failed") from exc

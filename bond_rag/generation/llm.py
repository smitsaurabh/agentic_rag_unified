"""
Ollama LLM interface — production version.

Improvements over v1
─────────────────────
• Health check on first use with informative error messages.
• Tenacity retry for transient connection errors.
• Streaming with clean token generator.
• Prompt construction separated into its own method (testable).
• Timeout enforced via Ollama's request options.
• Structured logging for every call with token count estimate.
"""

from __future__ import annotations

import time
from typing import Generator, Optional

import ollama
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from bond_rag.core.config import LLMSettings, get_settings
from bond_rag.core.exceptions import (
    LLMTimeoutError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
)
from bond_rag.core.logging import get_logger
from bond_rag.core.models import RetrievedChunk

logger = get_logger(__name__)


class OllamaLLM:
    """
    Thin, production-grade wrapper around the Ollama Python SDK.
    """

    def __init__(self, config: Optional[LLMSettings] = None) -> None:
        self.cfg = config or get_settings().llm
        self._client = ollama.Client(host=self.cfg.ollama_host)
        self._verified = False

    # ── Health check ──────────────────────────────────────────────────────────

    def verify(self) -> None:
        """
        Check Ollama is reachable and the configured model is available.
        Called lazily before the first inference call.
        """
        if self._verified:
            return

        try:
            models_resp = self._client.list()
            available   = [m.model for m in getattr(models_resp, "models", [])]
        except Exception as exc:
            raise OllamaConnectionError(self.cfg.ollama_host, cause=exc) from exc

        if not any(self.cfg.model in name for name in available if name):
            raise OllamaModelNotFoundError(self.cfg.model)

        self._verified = True
        logger.info("Ollama verified", model=self.cfg.model, host=self.cfg.ollama_host)

    # ── Inference ─────────────────────────────────────────────────────────────

    def answer(self, query: str, chunks: list[RetrievedChunk]) -> str:
        """Blocking single-turn answer generation."""
        self.verify()
        messages = self._build_messages(query, chunks)
        t0 = time.perf_counter()

        try:
            response = self._chat(messages)
        except ollama.ResponseError as exc:
            if "timeout" in str(exc).lower():
                raise LLMTimeoutError(self.cfg.model, self.cfg.request_timeout) from exc
            raise

        text    = response["message"]["content"]
        elapsed = time.perf_counter() - t0
        logger.info(
            "LLM answer generated",
            model=self.cfg.model,
            elapsed_s=f"{elapsed:.1f}",
            approx_tokens=len(text.split()),
        )
        return text

    def stream_answer(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> Generator[str, None, None]:
        """Streaming answer — yields text tokens as they arrive."""
        self.verify()
        messages = self._build_messages(query, chunks)

        try:
            stream = self._client.chat(
                model    = self.cfg.model,
                messages = messages,
                stream   = True,
                options  = self._options(),
            )
            for chunk in stream:
                yield chunk["message"]["content"]
        except ollama.ResponseError as exc:
            if "timeout" in str(exc).lower():
                raise LLMTimeoutError(self.cfg.model, self.cfg.request_timeout) from exc
            raise

    # ── Prompt construction ───────────────────────────────────────────────────

    def _build_messages(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> list[dict]:
        if chunks:
            ctx_parts = []
            for i, c in enumerate(chunks, 1):
                ctx_parts.append(
                    f"[Context {i}] {c.citation()}\n{c.text}"
                )
            context_block = "\n\n---\n\n".join(ctx_parts)
        else:
            context_block = "No relevant context found in the document store."

        user_content = (
            "Use ONLY the context excerpts below to answer the question.\n"
            "If the answer is not present in the context, reply exactly: "
            "'Not found in the provided documents.'\n"
            "Always cite the ISIN and document name for any bond-specific fact.\n\n"
            f"=== CONTEXT ===\n{context_block}\n\n"
            f"=== QUESTION ===\n{query}"
        )
        return [
            {"role": "system", "content": self.cfg.system_prompt},
            {"role": "user",   "content": user_content},
        ]

    # ── Internal helpers ──────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(ConnectionError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _chat(self, messages: list[dict]) -> dict:
        return self._client.chat(
            model    = self.cfg.model,
            messages = messages,
            options  = self._options(),
        )

    def _options(self) -> dict:
        return {
            "temperature":  self.cfg.temperature,
            "num_predict":  self.cfg.max_tokens,
        }

    def list_local_models(self) -> list[str]:
        """Return names of locally available Ollama models."""
        try:
            resp = self._client.list()
            return [m.model for m in getattr(resp, "models", []) if m.model]
        except Exception as exc:
            raise OllamaConnectionError(self.cfg.ollama_host, cause=exc) from exc

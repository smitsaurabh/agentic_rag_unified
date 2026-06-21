"""
ISIN-Aware Chunker — production version.

Improvements over v1
────────────────────
• Uses frozen Pydantic models (no in-place mutation).
• Chunk boundaries are word-aligned to avoid splitting mid-word.
• Chunk.make_id() is called via the model's static method.
• Full structured logging.
• Defensive handling of single-page ISIN sections that are shorter
  than chunk_size (emits one chunk without windowing).
"""

from __future__ import annotations

from bond_rag.core.config import ChunkSettings, get_settings
from bond_rag.core.logging import get_logger
from bond_rag.core.models import ISIN_SENTINEL, Chunk, DocumentRecord, PageRecord

logger = get_logger(__name__)


class ISINAwareChunker:
    """
    Splits a ``DocumentRecord`` into ISIN-scoped overlapping text chunks
    with 25 % overlap (configurable via ``ChunkSettings.overlap_ratio``).
    """

    def __init__(self, config: ChunkSettings | None = None) -> None:
        self.cfg = config or get_settings().chunk

    # ── Public API ─────────────────────────────────────────────────────────────

    def chunk_document(self, doc: DocumentRecord) -> list[Chunk]:
        """
        Return all chunks for the document, ordered by ISIN section
        then chunk index.
        """
        sections = self._group_by_isin(list(doc.pages))
        all_chunks: list[Chunk] = []

        for isin_key, pages in sections.items():
            section_chunks = self._chunk_section(
                isin     = isin_key,
                pages    = pages,
                filename = doc.filename,
                filepath = doc.filepath,
            )
            all_chunks.extend(section_chunks)

        logger.info(
            "Chunking complete",
            filename=doc.filename,
            total_chunks=len(all_chunks),
            sections=len(sections),
        )
        return all_chunks

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _group_by_isin(
        pages: list[PageRecord],
    ) -> dict[str, list[PageRecord]]:
        """
        Group pages by active_isin, preserving document order.
        Pages without an active ISIN are grouped under ISIN_SENTINEL.
        """
        groups: dict[str, list[PageRecord]] = {}
        for page in pages:
            key = page.active_isin or ISIN_SENTINEL
            groups.setdefault(key, []).append(page)
        return groups

    def _chunk_section(
        self,
        isin: str,
        pages: list[PageRecord],
        filename: str,
        filepath: str,
    ) -> list[Chunk]:
        """
        Concatenate page texts for this ISIN section and apply a
        word-aligned sliding window with configured overlap.
        """
        # Concatenate pages; track character offsets per page
        parts: list[str] = []
        offsets: list[int] = []
        cursor = 0
        for page in pages:
            offsets.append(cursor)
            parts.append(page.text)
            cursor += len(page.text) + 1  # +1 for joining newline

        full_text  = "\n".join(parts)
        chunk_size = self.cfg.chunk_size
        step_size  = self.cfg.step_size
        min_size   = self.cfg.min_chunk_size

        all_isins: list[str] = []
        seen: set[str] = set()
        for p in pages:
            for isin_str in p.isins:
                if isin_str not in seen:
                    seen.add(isin_str)
                    all_isins.append(isin_str)

        # ── Word-aligned sliding windows ──────────────────────────────────────
        windows: list[tuple[int, int]] = []
        start = 0
        text_len = len(full_text)

        while start < text_len:
            end = min(start + chunk_size, text_len)
            # Align end to the next word boundary to avoid mid-word cuts
            if end < text_len:
                space_idx = full_text.find(" ", end)
                if space_idx != -1 and space_idx - end < 80:
                    end = space_idx

            windows.append((start, end))
            if end >= text_len:
                break
            start += step_size

        # Filter tiny tail chunks
        windows = [(s, e) for s, e in windows if (e - s) >= min_size]
        total   = len(windows)

        chunks: list[Chunk] = []
        for idx, (s, e) in enumerate(windows):
            text_slice = full_text[s:e].strip()
            if not text_slice:
                continue

            page_start, page_end = _pages_for_range(s, e, offsets, pages)

            chunks.append(
                Chunk(
                    chunk_id             = Chunk.make_id(filename, isin, idx, text_slice),
                    isin                 = isin,
                    filename             = filename,
                    filepath             = filepath,
                    page_start           = page_start,
                    page_end             = page_end,
                    chunk_index          = idx,
                    total_chunks         = total,
                    text                 = text_slice,
                    all_isins_in_section = tuple(all_isins),
                )
            )

        logger.debug(
            "Section chunked",
            isin=isin,
            pages=len(pages),
            chunks=len(chunks),
        )
        return chunks


# ── Module-level helpers ───────────────────────────────────────────────────────

def _pages_for_range(
    start_char: int,
    end_char: int,
    offsets: list[int],
    pages: list[PageRecord],
) -> tuple[int, int]:
    """Map a character range in concatenated text to (page_start, page_end)."""
    page_start = pages[0].page_num
    page_end   = pages[-1].page_num

    for i, offset in enumerate(offsets):
        next_offset = offsets[i + 1] if i + 1 < len(offsets) else float("inf")
        if offset <= start_char < next_offset:
            page_start = pages[i].page_num
        if offset <= end_char <= next_offset:
            page_end = pages[i].page_num
            break

    return page_start, page_end

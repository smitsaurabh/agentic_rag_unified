"""
Production PDF Processor — with scanned-page OCR support.

Page text extraction strategy
──────────────────────────────
Every page is first processed by PyMuPDF's native text extractor (fast,
lossless for digital PDFs).  If the extracted text is shorter than the
``ocr.min_text_length`` threshold (default 50 chars) the page is classified
as scanned / image-only, and the OCRHandler takes over:

  1. PyMuPDF renders the page to a high-resolution pixmap (300 DPI).
  2. Optional pre-processing: greyscale → deskew → adaptive threshold → denoise.
  3. Tesseract OCR extracts text from the processed image.
  4. The OCR'd text replaces the empty/short digital extraction.

Mixed PDFs (some digital, some scanned pages) are handled transparently —
each page is evaluated individually.

The ``ocr_stats`` field on ``DocumentRecord`` reports how many pages were
OCR'd so operators can monitor quality.

OCR is disabled by setting ``BOND_RAG__OCR__ENABLED=false`` in ``.env``.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Generator, Optional

import fitz  # PyMuPDF

from bond_rag.core.config import OCRSettings, get_settings
from bond_rag.core.exceptions import ISINExtractionError, PDFReadError
from bond_rag.core.logging import get_logger
from bond_rag.core.models import DocumentRecord, PageRecord

logger = get_logger(__name__)

# ISO 6166: 2 uppercase country letters + 9 alphanumeric chars + 1 check digit
_ISIN_RE = re.compile(r"\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b")

# Patterns that indicate a NEW bond section is starting
_SECTION_HEADER_RE = re.compile(
    r"(ISIN|International Securities Identification Number|Bond\s+ISIN"
    r"|Series\s+ISIN|Notes?\s+ISIN)\s*[:\-–]?\s*([A-Z]{2}[A-Z0-9]{9}[0-9])",
    re.IGNORECASE,
)


class PDFProcessor:
    """
    Reads a PDF and returns a ``DocumentRecord`` with per-page text
    and ISIN annotations.  Automatically detects and OCRs scanned pages.
    """

    def __init__(
        self,
        verbose: bool = True,
        ocr_config: Optional[OCRSettings] = None,
    ) -> None:
        self.verbose    = verbose
        self._ocr_cfg   = ocr_config or get_settings().ocr
        self._ocr_handler = None   # lazy — only created if OCR is actually needed

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, pdf_path: str | Path) -> DocumentRecord:
        """
        Full pipeline: open → stream pages (with OCR fallback for scanned pages)
        → annotate ISINs → return record.

        Raises
        ------
        PDFReadError if the file cannot be opened or pages cannot be read.
        """
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise PDFReadError(str(path))

        file_hash = _sha256(path)
        filename  = path.name
        size_mb   = path.stat().st_size / 1_048_576

        logger.info(
            "Processing PDF",
            filename=filename,
            size_mb=f"{size_mb:.1f}",
            ocr_enabled=self._ocr_cfg.enabled,
        )

        try:
            pages, n_ocr = self._extract_pages(path, filename)
        except PDFReadError:
            raise
        except Exception as exc:
            raise PDFReadError(str(path), cause=exc) from exc

        self._annotate_isins(pages)

        doc = DocumentRecord(
            filename    = filename,
            filepath    = str(path),
            file_hash   = file_hash,
            total_pages = len(pages),
            pages       = tuple(pages),
        )

        logger.info(
            "PDF processed",
            filename      = filename,
            total_pages   = doc.total_pages,
            ocr_pages     = n_ocr,
            digital_pages = doc.total_pages - n_ocr,
            isins         = doc.all_isins,
            non_empty_pages = doc.non_empty_pages,
        )
        return doc

    # ── Page extraction ────────────────────────────────────────────────────────

    def _extract_pages(
        self, path: Path, filename: str
    ) -> tuple[list[PageRecord], int]:
        """
        Open the PDF and extract text from every page.

        For each page:
          1. Try PyMuPDF native text extraction.
          2. If text is below the scanned-page threshold → fall back to OCR.

        Returns
        -------
        (pages, n_ocr_pages)
        """
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            raise PDFReadError(str(path), cause=exc) from exc

        pages:  list[PageRecord] = []
        n_ocr:  int = 0

        with doc:
            total = len(doc)
            logger.debug("Opened PDF", filename=filename, total_pages=total)

            for page_idx in range(total):
                page_num = page_idx + 1
                try:
                    fitz_page = doc[page_idx]
                    text, used_ocr = self._extract_page_text(
                        fitz_page, filename, page_num
                    )
                    if used_ocr:
                        n_ocr += 1
                except Exception as exc:
                    logger.warning(
                        "Failed to extract page — skipping",
                        filename=filename,
                        page=page_num,
                        error=str(exc),
                    )
                    text = ""

                pages.append(PageRecord(
                    filename = filename,
                    filepath = str(path),
                    page_num = page_num,
                    text     = text,
                    isins    = (),
                ))

        return pages, n_ocr

    def _extract_page_text(
        self,
        fitz_page: fitz.Page,
        filename: str,
        page_num: int,
    ) -> tuple[str, bool]:
        """
        Extract text from one page.

        Returns
        -------
        (text, used_ocr)  — used_ocr is True if OCR was needed.
        """
        # ── Step 1: try native digital text extraction ────────────────────────
        raw = fitz_page.get_text(
            "text",
            flags = fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP,
        )
        text = _clean_text(raw)

        # ── Step 2: decide if OCR is needed ───────────────────────────────────
        if not self._ocr_cfg.enabled:
            return text, False

        if not self._is_scanned_page(text):
            return text, False

        # ── Step 3: OCR fallback ──────────────────────────────────────────────
        logger.info(
            "Scanned page detected — running OCR",
            filename=filename,
            page=page_num,
            digital_chars=len(text),
        )

        ocr_text = self._ocr(fitz_page)

        if ocr_text:
            logger.debug(
                "OCR succeeded",
                filename=filename,
                page=page_num,
                ocr_chars=len(ocr_text),
            )
            return _clean_text(ocr_text), True

        # OCR returned nothing (blank page, pure figure, etc.)
        logger.warning(
            "OCR returned no text",
            filename=filename,
            page=page_num,
        )
        return text, True

    def _is_scanned_page(self, text: str) -> bool:
        """Return True when the page looks like a scanned image."""
        return len(text.strip()) < self._ocr_cfg.min_text_length

    def _ocr(self, fitz_page: fitz.Page) -> str:
        """Lazy-init the OCR handler and run it."""
        if self._ocr_handler is None:
            from bond_rag.ingestion.ocr import OCRHandler
            self._ocr_handler = OCRHandler(config=self._ocr_cfg)
        return self._ocr_handler.extract(fitz_page)

    # ── ISIN annotation ────────────────────────────────────────────────────────

    def _annotate_isins(self, pages: list[PageRecord]) -> None:
        """
        Two-pass ISIN annotation (mutates the list in-place by replacing items,
        since PageRecord is frozen).

        Pass 1 — Forward scan: find all ISINs on each page; propagate
                 active_isin forward as each new ISIN is seen.
        Pass 2 — Backward fill: propagate the first-seen ISIN backward to
                 cover preamble pages (table of contents, title pages) that
                 appear before the first ISIN occurrence.
        """
        try:
            # ── Pass 1: forward ──────────────────────────────────────────────
            forward: list[PageRecord] = []
            active: str | None = None

            for page in pages:
                found = _extract_isins(page.text)

                # Prefer ISINs from explicit header patterns (more reliable)
                header_isins = _extract_header_isins(page.text)
                if header_isins:
                    active = header_isins[0]
                    found  = list(dict.fromkeys(header_isins + found))
                elif found:
                    active = found[0]

                forward.append(
                    page.model_copy(update={"isins": tuple(found), "active_isin": active})
                )

            # ── Pass 2: backward fill for preamble pages ─────────────────────
            # Find the first page with an ISIN
            first_isin: str | None = None
            for p in forward:
                if p.active_isin:
                    first_isin = p.active_isin
                    break

            annotated: list[PageRecord] = []
            for page in forward:
                if page.active_isin is None and first_isin:
                    page = page.model_copy(update={"active_isin": first_isin})
                annotated.append(page)

            pages[:] = annotated

        except Exception as exc:
            raise ISINExtractionError(
                "Unexpected error during ISIN annotation"
            ) from exc


# ── Module-level helpers ───────────────────────────────────────────────────────

def _extract_isins(text: str) -> list[str]:
    """Extract deduplicated ISINs from a text block."""
    found = _ISIN_RE.findall(text)
    seen: set[str] = set()
    result = []
    for isin in found:
        if isin not in seen:
            seen.add(isin)
            result.append(isin)
    return result


def _extract_header_isins(text: str) -> list[str]:
    """Extract ISINs that appear in explicit header patterns like 'ISIN: XS123…'."""
    found = [m.group(2) for m in _SECTION_HEADER_RE.finditer(text)]
    return list(dict.fromkeys(found))


def _clean_text(text: str) -> str:
    """Normalise whitespace and strip control characters."""
    # Collapse 3+ newlines → paragraph boundary
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove non-printable control chars (keep \n, \t)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text.strip()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65_536), b""):
            h.update(block)
    return h.hexdigest()

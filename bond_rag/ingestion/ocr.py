"""
OCR handler for scanned / image-only PDF pages.

Strategy
────────
1.  PyMuPDF renders the page to a high-resolution pixmap (default 300 DPI).
2.  The pixmap is converted to a PIL Image (no temp file on disk).
3.  Optional pre-processing pipeline improves accuracy on low-quality scans:
      a. Convert to greyscale
      b. Deskew  (correct rotation up to ±10°)
      c. Denoise (median blur to remove scanner noise)
4.  Tesseract performs OCR via pytesseract.
5.  The extracted text is returned; the caller decides what to do with it.

Dependencies
────────────
Required:
    pip install pytesseract Pillow
    # macOS:   brew install tesseract
    # Ubuntu:  sudo apt install tesseract-ocr
    # Windows: https://github.com/UB-Mannheim/tesseract/wiki

Optional (for pre-processing):
    pip install opencv-python-headless

Language packs (example — install as needed):
    # Ubuntu: sudo apt install tesseract-ocr-deu  (German)
    # macOS:  brew install tesseract-lang

Graceful degradation
────────────────────
If Tesseract is not installed, ``OCRHandler`` raises a clear
``ConfigurationError`` on first use rather than a cryptic import error.
If opencv is not installed, the pre-processing step is silently skipped.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import fitz  # PyMuPDF

from bond_rag.core.config import OCRSettings, get_settings
from bond_rag.core.exceptions import ConfigurationError
from bond_rag.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class OCRHandler:
    """
    Renders a PDF page to an image and extracts text via Tesseract OCR.

    Usage::

        handler = OCRHandler()
        text = handler.extract(fitz_page)
    """

    def __init__(self, config: OCRSettings | None = None) -> None:
        self.cfg = config or get_settings().ocr
        self._tesseract_ok: bool | None = None   # lazy check

    # ── Public API ─────────────────────────────────────────────────────────────

    def extract(self, page: fitz.Page) -> str:
        """
        OCR a single fitz.Page and return the extracted text.

        Parameters
        ----------
        page : an open fitz.Page object

        Returns
        -------
        str — OCR'd text (may be empty if the page is blank or unreadable)

        Raises
        ------
        ConfigurationError if Tesseract is not installed / not on PATH.
        """
        self._ensure_tesseract()

        # ── Render to image ───────────────────────────────────────────────────
        pil_image = self._render_page(page)

        # ── Pre-process ───────────────────────────────────────────────────────
        if self.cfg.preprocess:
            pil_image = self._preprocess(pil_image)

        # ── OCR ───────────────────────────────────────────────────────────────
        import pytesseract  # deferred — only needed when OCR is used

        config_str = "--oem 3 --psm 6"   # LSTM engine, assume uniform block of text
        try:
            text = pytesseract.image_to_string(
                pil_image,
                lang   = self.cfg.language,
                config = config_str,
            )
        except Exception as exc:
            logger.warning(
                "Tesseract OCR failed on page",
                page=getattr(page, "number", "?") + 1,
                error=str(exc),
            )
            return ""

        cleaned = text.strip()
        logger.debug(
            "OCR complete",
            page=getattr(page, "number", "?") + 1,
            chars=len(cleaned),
        )
        return cleaned

    def is_scanned_page(self, text: str) -> bool:
        """
        Return True if the page is likely scanned (digital text extraction
        returned too little text).
        """
        return len(text.strip()) < self.cfg.min_text_length

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _render_page(self, page: fitz.Page):  # type: ignore[return]
        """Render fitz.Page to a PIL Image at configured DPI."""
        from PIL import Image  # deferred

        scale = self.cfg.dpi / 72.0          # 72 DPI is fitz's native resolution
        mat   = fitz.Matrix(scale, scale)
        pix   = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)

        # Convert pixmap bytes → PIL Image without touching disk
        img_bytes = pix.tobytes("png")
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def _preprocess(self, image):  # type: ignore[no-untyped-def]
        """
        Apply a sequence of image-processing steps that improve Tesseract
        accuracy on scanned financial documents:

        1. Greyscale   — remove colour noise; Tesseract works best on greyscale.
        2. Deskew      — correct page rotation (scanner tilt).
        3. Adaptive threshold — binarise; improves contrast on aged documents.
        4. Median blur — remove salt-and-pepper noise without blurring edges.

        Falls back to returning the original image if opencv is not installed.
        """
        try:
            import cv2
            import numpy as np
        except ImportError:
            logger.debug("opencv not installed — skipping OCR pre-processing")
            return image

        from PIL import Image

        # PIL → numpy array
        img = np.array(image)

        # 1. Greyscale
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 2. Deskew via Hough line detection
        grey = self._deskew(grey)

        # 3. Adaptive threshold (handles uneven lighting / shadows)
        binary = cv2.adaptiveThreshold(
            grey, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=8,
        )

        # 4. Median blur to remove isolated noise pixels
        denoised = cv2.medianBlur(binary, 3)

        return Image.fromarray(denoised)

    @staticmethod
    def _deskew(grey_img):  # type: ignore[no-untyped-def]
        """
        Estimate and correct rotation angle using Hough line transform.
        Limits correction to ±10° to avoid over-rotating pages with no
        horizontal lines (e.g. tables or figures only).
        """
        try:
            import cv2
            import numpy as np

            edges = cv2.Canny(grey_img, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
            if lines is None:
                return grey_img

            angles = []
            for line in lines[:20]:          # use top 20 lines only
                rho, theta = line[0]
                angle = (theta - np.pi / 2) * 180 / np.pi
                if abs(angle) < 10:          # only correct small tilts
                    angles.append(angle)

            if not angles:
                return grey_img

            median_angle = float(np.median(angles))
            if abs(median_angle) < 0.5:      # skip tiny corrections
                return grey_img

            h, w = grey_img.shape
            centre = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(centre, median_angle, 1.0)
            rotated = cv2.warpAffine(
                grey_img, M, (w, h),
                flags        = cv2.INTER_CUBIC,
                borderMode   = cv2.BORDER_REPLICATE,
            )
            logger.debug("Deskewed page", angle=f"{median_angle:.2f}°")
            return rotated

        except Exception:
            return grey_img   # deskew is best-effort; never fail the pipeline

    def _ensure_tesseract(self) -> None:
        """
        Check that Tesseract is reachable.  Raises ConfigurationError with
        installation instructions if it is not.

        Result is cached after the first successful check.
        """
        if self._tesseract_ok is True:
            return

        try:
            import pytesseract

            # Override command path if configured
            if self.cfg.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.cfg.tesseract_cmd

            version = pytesseract.get_tesseract_version()
            logger.info("Tesseract available", version=str(version))
            self._tesseract_ok = True

        except ImportError as exc:
            raise ConfigurationError(
                "pytesseract is not installed. "
                "Install it with: pip install pytesseract Pillow"
            ) from exc
        except pytesseract.TesseractNotFoundError as exc:  # type: ignore[possibly-undefined]
            raise ConfigurationError(
                "Tesseract OCR binary not found. Install it:\n"
                "  macOS  : brew install tesseract\n"
                "  Ubuntu : sudo apt install tesseract-ocr\n"
                "  Windows: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "Then restart your terminal, or set BOND_RAG__OCR__TESSERACT_CMD "
                "to the full path of the tesseract binary."
            ) from exc

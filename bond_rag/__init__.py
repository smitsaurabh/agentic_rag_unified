"""
bond_rag — ISIN-aware RAG system for financial bond PDFs.

Quick start::

    from bond_rag.rag.pipeline import BondRAGPipeline

    rag = BondRAGPipeline()
    rag.ingest("data/prospectus.pdf")
    answer = rag.query("What is the coupon rate for XS1234567890?")
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bond-rag")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]

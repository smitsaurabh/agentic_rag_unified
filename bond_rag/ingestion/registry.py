"""
SQLite-backed ingestion registry.

Replaces the fragile JSON registry with a proper relational store that
supports concurrent access (WAL mode), atomic transactions, and rich
query capability (list by ISIN, list by date, etc.).

Schema
──────
files
    hash        TEXT PRIMARY KEY   — SHA-256 of the PDF content
    filename    TEXT NOT NULL
    filepath    TEXT NOT NULL
    ingested_at TEXT NOT NULL      — ISO-8601 UTC timestamp
    total_pages INTEGER NOT NULL
    chunks      INTEGER NOT NULL
    status      TEXT NOT NULL      — 'loaded' | 'failed'

file_isins
    hash        TEXT NOT NULL REFERENCES files(hash) ON DELETE CASCADE
    isin        TEXT NOT NULL
    PRIMARY KEY (hash, isin)
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from bond_rag.core.exceptions import RegistryError
from bond_rag.core.logging import get_logger

logger = get_logger(__name__)

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
    hash        TEXT    PRIMARY KEY,
    filename    TEXT    NOT NULL,
    filepath    TEXT    NOT NULL,
    ingested_at TEXT    NOT NULL,
    total_pages INTEGER NOT NULL DEFAULT 0,
    chunks      INTEGER NOT NULL DEFAULT 0,
    status      TEXT    NOT NULL DEFAULT 'loaded'
);

CREATE TABLE IF NOT EXISTS file_isins (
    hash TEXT NOT NULL REFERENCES files(hash) ON DELETE CASCADE,
    isin TEXT NOT NULL,
    PRIMARY KEY (hash, isin)
);

CREATE INDEX IF NOT EXISTS idx_file_isins_isin ON file_isins(isin);
"""


class IngestionRegistry:
    """
    Thread-safe SQLite registry for tracking ingested PDFs.

    Each ``IngestionRegistry`` instance opens its own connection; use a
    single shared instance per process (e.g. via the pipeline singleton).
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._connect()
        self._apply_schema()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _apply_schema(self) -> None:
        try:
            with self._transaction() as cur:
                cur.executescript(_DDL)
        except sqlite3.Error as exc:
            raise RegistryError(
                f"Failed to initialise registry at {self._db_path}"
            ) from exc

    @contextmanager
    def _transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # ── Public API ─────────────────────────────────────────────────────────────

    def is_loaded(self, file_hash: str) -> bool:
        """Return True if this file hash is already registered."""
        cur = self._conn.execute(
            "SELECT 1 FROM files WHERE hash = ? AND status = 'loaded'",
            (file_hash,),
        )
        return cur.fetchone() is not None

    def register(
        self,
        file_hash: str,
        filename: str,
        filepath: str,
        total_pages: int,
        chunks: int,
        isins: list[str],
        status: str = "loaded",
    ) -> None:
        """
        Insert or replace a file record with its ISINs.
        Safe to call multiple times with the same hash (upsert).
        """
        now = datetime.now(tz=timezone.utc).isoformat()
        try:
            with self._transaction() as cur:
                cur.execute(
                    """
                    INSERT INTO files (hash, filename, filepath, ingested_at,
                                       total_pages, chunks, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(hash) DO UPDATE SET
                        filename    = excluded.filename,
                        filepath    = excluded.filepath,
                        ingested_at = excluded.ingested_at,
                        total_pages = excluded.total_pages,
                        chunks      = excluded.chunks,
                        status      = excluded.status
                    """,
                    (file_hash, filename, filepath, now, total_pages, chunks, status),
                )
                # Delete old ISIN associations then re-insert
                cur.execute("DELETE FROM file_isins WHERE hash = ?", (file_hash,))
                cur.executemany(
                    "INSERT OR IGNORE INTO file_isins (hash, isin) VALUES (?, ?)",
                    [(file_hash, isin) for isin in isins],
                )
        except sqlite3.Error as exc:
            raise RegistryError(
                f"Failed to register file '{filename}'"
            ) from exc

        logger.debug("Registry: registered", filename=filename, chunks=chunks, isins=isins)

    def unregister(self, file_hash: str) -> bool:
        """Remove a file record. Returns True if a row was deleted."""
        try:
            with self._transaction() as cur:
                cur.execute("DELETE FROM files WHERE hash = ?", (file_hash,))
                deleted = cur.rowcount > 0
        except sqlite3.Error as exc:
            raise RegistryError("Failed to unregister file") from exc
        return deleted

    def unregister_by_filename(self, filename: str) -> int:
        """Remove all records for the given filename. Returns deleted count."""
        try:
            with self._transaction() as cur:
                cur.execute(
                    "SELECT hash FROM files WHERE filename = ?", (filename,)
                )
                hashes = [row["hash"] for row in cur.fetchall()]
                if hashes:
                    cur.executemany(
                        "DELETE FROM files WHERE hash = ?",
                        [(h,) for h in hashes],
                    )
                return len(hashes)
        except sqlite3.Error as exc:
            raise RegistryError(f"Failed to unregister '{filename}'") from exc

    def list_files(self) -> list[dict]:
        """Return all registered files with their ISINs."""
        cur = self._conn.execute(
            """
            SELECT f.*, GROUP_CONCAT(fi.isin, ',') AS isins
            FROM files f
            LEFT JOIN file_isins fi ON fi.hash = f.hash
            GROUP BY f.hash
            ORDER BY f.ingested_at DESC
            """
        )
        rows = cur.fetchall()
        return [
            {
                **dict(row),
                "isins": [i for i in (row["isins"] or "").split(",") if i],
            }
            for row in rows
        ]

    def list_isins(self) -> list[str]:
        """Return all distinct ISINs across all registered files."""
        cur = self._conn.execute(
            "SELECT DISTINCT isin FROM file_isins ORDER BY isin"
        )
        return [row["isin"] for row in cur.fetchall()]

    def find_by_isin(self, isin: str) -> list[dict]:
        """Return files that contain the given ISIN."""
        cur = self._conn.execute(
            """
            SELECT f.filename, f.filepath, f.total_pages, f.ingested_at
            FROM files f
            JOIN file_isins fi ON fi.hash = f.hash
            WHERE fi.isin = ?
            """,
            (isin,),
        )
        return [dict(row) for row in cur.fetchall()]

    def stats(self) -> dict:
        """Return summary statistics."""
        row = self._conn.execute(
            """
            SELECT
                COUNT(*)                      AS total_files,
                COALESCE(SUM(chunks), 0)      AS total_chunks,
                COALESCE(SUM(total_pages), 0) AS total_pages
            FROM files
            WHERE status = 'loaded'
            """
        ).fetchone()
        n_isins = self._conn.execute(
            "SELECT COUNT(DISTINCT isin) FROM file_isins"
        ).fetchone()[0]
        return {
            "total_files":  row["total_files"],
            "total_chunks": row["total_chunks"],
            "total_pages":  row["total_pages"],
            "total_isins":  n_isins,
        }

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

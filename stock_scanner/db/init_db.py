"""SQLite schema bootstrap for the self-improving signal system.

See docs/SELF_IMPROVING_ARCHITECTURE.md for the design. This module only
creates/opens the database — backfilling existing data and the daily
insert/upsert steps live in scripts/init_db_and_backfill.py and the
pipeline modules that call into this, respectively.
"""
import hashlib
import sqlite3
from pathlib import Path

_DB_PATH = Path(__file__).parent.parent.parent / "data" / "db" / "signals.db"
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def signal_id(ticker: str, signal_date: str, strategy: str) -> str:
    """Deterministic primary key — same inputs always produce the same id,
    so backfills and daily inserts are naturally idempotent."""
    raw = f"{ticker}|{signal_date}|{strategy}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    db_path = db_path or _DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection | None = None) -> None:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        conn.executescript(_SCHEMA_PATH.read_text())
        conn.commit()
    finally:
        if own_conn:
            conn.close()

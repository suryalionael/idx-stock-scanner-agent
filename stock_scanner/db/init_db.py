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


def _migrate_legacy_knowledge_entries(conn: sqlite3.Connection) -> None:
    """One-time, idempotent column migration for a knowledge_entries table
    created before the promotion_status pass added
    promotion_status/promoted_at/promoted_by/promotion_reason.

    Root cause: `CREATE TABLE IF NOT EXISTS` in schema.sql is a no-op
    against an ALREADY-EXISTING table — SQLite only checks whether the
    table exists, it never diffs or adds columns. A knowledge_entries
    table persisted before this pass (data/db/signals.db is normally
    rebuilt fresh per CI run, but survives on a local machine / this
    session's sandbox across runs) therefore keeps its old 20-column
    shape forever, and the very next statement in schema.sql — `CREATE
    INDEX ... ON knowledge_entries(promotion_status)` — then fails with
    "no such column: promotion_status", aborting the whole executescript
    call before anything after it runs. This must run BEFORE
    conn.executescript(schema.sql), not after: executescript stops dead
    at its first failing statement, so nothing past the failing CREATE
    INDEX would ever execute if this ran afterward instead.

    No-op on a fresh database (table doesn't exist yet — PRAGMA
    table_info returns nothing, and schema.sql's own CREATE TABLE creates
    every column correctly from scratch) and no-op on an already-migrated
    one (each column is added only if missing). Column defs here are
    byte-identical to schema.sql's knowledge_entries block, including the
    CHECK constraint (SQLite supports CHECK in ALTER TABLE ADD COLUMN as
    long as it doesn't reference other columns, which this one doesn't) —
    a migrated table ends up in exactly the same state, CHECK included,
    as a freshly-created one. Update both together if that block ever
    changes."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(knowledge_entries)")}
    if not cols:
        return  # table doesn't exist yet — schema.sql creates it fresh, nothing to migrate
    if "promotion_status" not in cols:
        conn.execute(
            """ALTER TABLE knowledge_entries ADD COLUMN promotion_status TEXT NOT NULL
               DEFAULT 'candidate'
               CHECK (promotion_status IN ('candidate', 'promoted', 'rejected', 'archived'))"""
        )
    if "promoted_at" not in cols:
        conn.execute("ALTER TABLE knowledge_entries ADD COLUMN promoted_at TIMESTAMP")
    if "promoted_by" not in cols:
        conn.execute("ALTER TABLE knowledge_entries ADD COLUMN promoted_by TEXT")
    if "promotion_reason" not in cols:
        conn.execute("ALTER TABLE knowledge_entries ADD COLUMN promotion_reason TEXT")
    conn.commit()


def create_schema(conn: sqlite3.Connection | None = None) -> None:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        _migrate_legacy_knowledge_entries(conn)
        conn.executescript(_SCHEMA_PATH.read_text())
        conn.commit()
    finally:
        if own_conn:
            conn.close()

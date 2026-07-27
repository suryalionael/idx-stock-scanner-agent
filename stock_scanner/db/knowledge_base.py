"""Read/write helpers for the knowledge_base table — the only table
stock_scanner/learning/ ever writes to. See
docs/LEARNING_AGENT_ARCHITECTURE.md and stock_scanner/db/schema.sql.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_scanner.learning.hypothesis_agent import Hypothesis

_DEFAULT_MIRROR_PATH = (
    Path(__file__).parent.parent.parent / "data" / "published" / "knowledge_base.json"
)
_COLS = [
    "hypothesis_id",
    "generated_at",
    "hypothesis",
    "confidence",
    "supporting_trades",
    "affected_sector",
    "affected_dimension",
    "pattern_json",
    "expected_effect",
    "status",
    "reviewed_by",
    "linked_model_version_id",
    "source_run_id",
]


def _hypothesis_id(hypothesis: Hypothesis) -> str:
    """Deterministic on (source_cluster_id, hypothesis text) ONLY — never on
    a timestamp. Re-running this script against unchanged input must
    dedupe via INSERT OR IGNORE, not insert a fresh "duplicate" batch every
    time; a genuinely different hypothesis text for the same cluster (e.g.
    the LLM's answer evolves) correctly gets treated as new."""
    raw = f"{hypothesis.source_cluster_id}|{hypothesis.hypothesis}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def write_hypotheses(
    conn: sqlite3.Connection,
    hypotheses: list[Hypothesis],
    source_run_id: str,
    pattern_json_by_cluster_id: dict[str, dict] | None = None,
) -> int:
    """Insert hypotheses as new knowledge_base rows (status='candidate').
    Idempotent per (source_cluster_id, hypothesis text) via INSERT OR IGNORE
    — re-running against unchanged input inserts nothing new. Returns the
    number of rows actually inserted."""
    pattern_json_by_cluster_id = pattern_json_by_cluster_id or {}
    generated_at = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    inserted = 0
    for h in hypotheses:
        hid = _hypothesis_id(h)
        pattern_json = json.dumps(pattern_json_by_cluster_id.get(h.source_cluster_id, {}))
        cur.execute(
            """INSERT OR IGNORE INTO knowledge_base
               (hypothesis_id, generated_at, hypothesis, confidence, supporting_trades,
                affected_sector, pattern_json, expected_effect, status, source_run_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                hid,
                generated_at,
                h.hypothesis,
                h.confidence,
                h.supporting_trades,
                h.affected_sector,
                pattern_json,
                h.expected_effect,
                h.status,
                source_run_id,
            ),
        )
        inserted += cur.rowcount
    conn.commit()
    return inserted


def load_knowledge_base(conn: sqlite3.Connection, status: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM knowledge_base"
    params: tuple = ()
    if status is not None:
        query += " WHERE status = ?"
        params = (status,)
    return pd.read_sql(query, conn, params=params)


VALID_STATUSES = {
    "candidate",
    "reviewed",
    "testing",
    "tested_passed",
    "tested_failed",
    "promoted",
    "archived",
}


def update_status(
    conn: sqlite3.Connection,
    hypothesis_id: str,
    new_status: str,
    reviewed_by: str | None = None,
    linked_model_version_id: str | None = None,
) -> int:
    """Record a human review decision. `linked_model_version_id`, if given,
    must reference a real model_registry row — the DB's own foreign-key
    constraint (PRAGMA foreign_keys=ON, see init_db.get_connection) rejects
    a nonexistent one rather than silently accepting it. Setting
    status='promoted' here is a record-keeping annotation only — it never
    causes and is never caused by an actual promotion; that remains
    model_registry.status, changed only by scripts/promote_challenger.py.

    Returns the number of rows actually updated (0 if hypothesis_id doesn't
    exist) — an UPDATE with no matching WHERE row is not an error in SQL, so
    callers must check this rather than assume a call "succeeding" means a
    row was found."""
    if new_status not in VALID_STATUSES:
        raise ValueError(f"Unknown status {new_status!r}; must be one of {sorted(VALID_STATUSES)}")
    cur = conn.execute(
        """UPDATE knowledge_base
           SET status = ?, reviewed_by = COALESCE(?, reviewed_by),
               linked_model_version_id = COALESCE(?, linked_model_version_id)
           WHERE hypothesis_id = ?""",
        (new_status, reviewed_by, linked_model_version_id, hypothesis_id),
    )
    conn.commit()
    return cur.rowcount


def export_knowledge_base(conn: sqlite3.Connection, path: Path | None = None) -> Path:
    """Mirror knowledge_base to committed JSON — same rationale as
    stock_scanner/db/registry_io.py: data/db/signals.db is gitignored and
    rebuilt fresh from committed sources on every CI run, so anything not
    derivable from those sources (like LLM-generated hypotheses) needs its
    own durable mirror.

    Publishes top-level `generated_at`/`summary` fields, mirroring the
    shape stock_scanner.db.knowledge_entries.export_knowledge_report()
    already uses — so the dashboard can read this payload's status
    without computing it client-side, same as its AI Lab sibling."""
    path = path or _DEFAULT_MIRROR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = conn.cursor()
    rows = [
        dict(zip(_COLS, row))
        for row in cur.execute(f"SELECT {','.join(_COLS)} FROM knowledge_base").fetchall()
    ]
    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {"total_entries": len(rows), "by_status": by_status},
        "knowledge_base": rows,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def import_knowledge_base(conn: sqlite3.Connection, path: Path | None = None) -> int:
    """Idempotent — INSERT OR IGNORE on the primary key, safe to call
    against a DB that already has some/all of these rows."""
    path = path or _DEFAULT_MIRROR_PATH
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    cur = conn.cursor()
    n = 0
    for row in data.get("knowledge_base", []):
        cur.execute(
            f"""INSERT OR IGNORE INTO knowledge_base ({",".join(_COLS)})
                VALUES ({",".join("?" * len(_COLS))})""",
            tuple(row.get(c) for c in _COLS),
        )
        n += cur.rowcount
    conn.commit()
    return n

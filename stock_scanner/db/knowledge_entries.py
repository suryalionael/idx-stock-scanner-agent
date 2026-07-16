"""Read/write helpers for knowledge_entries — see
stock_scanner/ai_lab/knowledge_base_engine.py for how entries are curated
(pure code, deterministic, no LLM involved) and
scripts/run_knowledge_base_engine.py for the orchestrator that calls
these.

Named knowledge_entries.py, not knowledge_base.py — that path is already
taken by the unrelated production knowledge_base table (Learning Agent
Phase 1's human-reviewed hypothesis store). See schema.sql's
knowledge_entries comment block for the full distinction.

Standalone, experimental: nothing in stock_scanner/pipeline/ or
stock_scanner/alerts/ reads or writes this table. data/db/signals.db is
gitignored and rebuilt fresh on every CI runner (same rationale as
stock_scanner/db/hypotheses.py), so history lives in the committed
data/published/knowledge_report.json mirror.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from stock_scanner.ai_lab.schemas import KnowledgeEntry, KnowledgePromotionStatus

_MIRROR_PATH = Path(__file__).parent.parent.parent / "data" / "published" / "knowledge_report.json"

_ENTRY_COLS = [
    "knowledge_id", "created_at", "title", "description", "conditions",
    "originating_hypotheses", "evidence_count", "cumulative_sample_size",
    "cumulative_successes", "cumulative_failures", "average_win_rate", "shrunk_win_rate",
    "confidence_interval", "first_seen", "last_confirmed", "confirmation_count",
    "contradiction_count", "evidence_strength", "lifecycle_status",
    "previous_lifecycle_status", "llm_note",
    # Deployment gate — see schema.sql's knowledge_entries comment block /
    # stock_scanner.ai_lab.schemas.KnowledgePromotionStatus. Plain scalars,
    # no JSON encode/decode needed unlike conditions/originating_hypotheses.
    "promotion_status", "promoted_at", "promoted_by", "promotion_reason",
]

_DEFAULT_PROMOTION_STATUS = KnowledgePromotionStatus.CANDIDATE.value
_VALID_PROMOTION_STATUSES = {s.value for s in KnowledgePromotionStatus}


def upsert_knowledge_entries(conn: sqlite3.Connection, entries: list[KnowledgeEntry]) -> int:
    """Append-only insert keyed on the deterministic knowledge_id (unique
    per run — it hashes created_at, see knowledge_base_engine.py) — plain
    INSERT OR IGNORE, no conflict-update branch needed: entries are
    immutable curation snapshots, not mutable positions."""
    if not entries:
        return 0
    cur = conn.cursor()
    n = 0
    for e in entries:
        cur.execute(
            """INSERT OR IGNORE INTO knowledge_entries
               (knowledge_id, created_at, title, description, conditions,
                originating_hypotheses, evidence_count, cumulative_sample_size,
                cumulative_successes, cumulative_failures, average_win_rate, shrunk_win_rate,
                confidence_interval, first_seen, last_confirmed, confirmation_count,
                contradiction_count, evidence_strength, lifecycle_status,
                previous_lifecycle_status, llm_note,
                promotion_status, promoted_at, promoted_by, promotion_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                e.knowledge_id, e.created_at, e.title, e.description, json.dumps(e.conditions),
                json.dumps(e.originating_hypotheses), e.evidence_count, e.cumulative_sample_size,
                e.cumulative_successes, e.cumulative_failures, e.average_win_rate, e.shrunk_win_rate,
                json.dumps(e.confidence_interval), e.first_seen, e.last_confirmed, e.confirmation_count,
                e.contradiction_count, e.evidence_strength.value if e.evidence_strength else None,
                e.lifecycle_status.value,
                e.previous_lifecycle_status.value if e.previous_lifecycle_status else None,
                e.llm_note,
                e.promotion_status.value, e.promoted_at, e.promoted_by, e.promotion_reason,
            ),
        )
        n += cur.rowcount
    conn.commit()
    return n


def load_knowledge_entries(
    conn: sqlite3.Connection, lifecycle_status: str | None = None, limit: int | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM knowledge_entries"
    clauses, params = [], []
    if lifecycle_status is not None:
        clauses.append("lifecycle_status = ?")
        params.append(lifecycle_status)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC, confirmation_count DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    return pd.read_sql(query, conn, params=tuple(params))


def export_knowledge_report(
    conn: sqlite3.Connection,
    path: Path | None = None,
    narrative: dict | None = None,
    resolved_trade_count: int = 0,
) -> Path:
    """Write data/published/knowledge_report.json. `resolved_trade_count`
    is the size of the resolved-recommendation population as of this run
    (passed in by the caller) — same convention as
    export_reflection_report/export_hypotheses_report. `narrative` is the
    LLM's KnowledgeNarrativeOutput as a dict, or None if that call
    failed/was skipped — the report is always written either way."""
    path = path or _MIRROR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = conn.cursor()
    rows = [
        dict(zip(_ENTRY_COLS, row))
        for row in cur.execute(
            f"SELECT {','.join(_ENTRY_COLS)} FROM knowledge_entries ORDER BY created_at DESC, confirmation_count DESC"
        ).fetchall()
    ]
    for row in rows:
        row["conditions"] = json.loads(row["conditions"]) if row["conditions"] else []
        row["originating_hypotheses"] = json.loads(row["originating_hypotheses"]) if row["originating_hypotheses"] else []
        row["confidence_interval"] = json.loads(row["confidence_interval"]) if row["confidence_interval"] else []

    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row["lifecycle_status"]] = by_status.get(row["lifecycle_status"], 0) + 1

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_entries": len(rows),
            "by_lifecycle_status": by_status,
            "resolved_trade_count": resolved_trade_count,
        },
        "entries": rows,
        "narrative": narrative,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def import_knowledge_entries(conn: sqlite3.Connection, path: Path | None = None) -> int:
    """Idempotent — INSERT OR IGNORE on the primary key, same rehydration
    role as stock_scanner.db.hypotheses.import_hypotheses: repopulate a
    fresh ephemeral-runner DB from the committed mirror before a new run,
    without duplicating history already recorded there.

    Backward compatible with knowledge_report.json mirrors written before
    promotion_status existed: a row missing that key, or carrying any
    value the database's CHECK constraint wouldn't accept (unrecognized/
    corrupt data — never expected from a real export, but this import
    path reads foreign files), is normalized to 'candidate' here before
    it ever reaches the INSERT — never NULL (the column is NOT NULL), and
    never passed through unvalidated. This keeps the row correctly
    excluded from production either way, since
    stock_scanner.pipeline.knowledge_application only ever applies an
    exact 'promoted' match — and keeps this the only place a value gets
    coerced, so the database CHECK constraint is a backstop that should
    never actually reject anything coming through this function."""
    path = path or _MIRROR_PATH
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    cur = conn.cursor()
    n = 0
    for row in data.get("entries", []):
        row = dict(row)
        row["conditions"] = json.dumps(row.get("conditions") or [])
        row["originating_hypotheses"] = json.dumps(row.get("originating_hypotheses") or [])
        row["confidence_interval"] = json.dumps(row.get("confidence_interval") or [])
        promotion_status = row.get("promotion_status")
        if promotion_status not in _VALID_PROMOTION_STATUSES:
            if promotion_status is not None:  # present but not recognized — worth a log, not just missing
                logger.warning(
                    "knowledge_entries import: {} has unrecognized promotion_status={!r} — "
                    "treating as {!r} (fail closed)",
                    row.get("knowledge_id"), promotion_status, _DEFAULT_PROMOTION_STATUS,
                )
            promotion_status = _DEFAULT_PROMOTION_STATUS
        row["promotion_status"] = promotion_status
        cur.execute(
            f"""INSERT OR IGNORE INTO knowledge_entries ({",".join(_ENTRY_COLS)})
                VALUES ({",".join("?" * len(_ENTRY_COLS))})""",
            tuple(row.get(c) for c in _ENTRY_COLS),
        )
        n += cur.rowcount
    conn.commit()
    return n

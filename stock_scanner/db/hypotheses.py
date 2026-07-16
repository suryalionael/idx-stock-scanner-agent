"""Read/write helpers for validated_hypotheses — see
stock_scanner/ai_lab/hypothesis_engine.py (candidate generation) and
stock_scanner/ai_lab/statistical_validation.py (validation) for how
Hypothesis rows are computed (pure code, deterministic) and
scripts/run_hypothesis_engine.py for the orchestrator that calls these.

Standalone, experimental: nothing in stock_scanner/pipeline/ or
stock_scanner/alerts/ reads or writes this table. data/db/signals.db is
gitignored and rebuilt fresh on every CI runner (same rationale as
stock_scanner/db/reflection.py), so history lives in the committed
data/published/hypotheses_report.json mirror.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_scanner.ai_lab.schemas import Hypothesis

_MIRROR_PATH = Path(__file__).parent.parent.parent / "data" / "published" / "hypotheses_report.json"

_HYP_COLS = [
    "hypothesis_id", "created_at", "description", "conditions", "sample_size",
    "successes", "failures", "win_rate", "shrunk_win_rate", "wilson_lower", "wilson_upper",
    "fisher_p", "bh_adjusted_p", "evidence_strength", "status", "rejection_reason",
    "failed_gate", "source_reflection_ids", "metadata_json", "llm_note",
]


def upsert_hypotheses(conn: sqlite3.Connection, hypotheses: list[Hypothesis]) -> int:
    """Append-only insert keyed on the deterministic hypothesis_id (unique
    per run — it hashes created_at, see statistical_validation.py) — plain
    INSERT OR IGNORE, no conflict-update branch needed: hypotheses are
    immutable validation snapshots, not mutable positions like
    ai_recommendations, so there's nothing to preserve-on-conflict."""
    if not hypotheses:
        return 0
    cur = conn.cursor()
    n = 0
    for h in hypotheses:
        cur.execute(
            """INSERT OR IGNORE INTO validated_hypotheses
               (hypothesis_id, created_at, description, conditions, sample_size,
                successes, failures, win_rate, shrunk_win_rate, wilson_lower, wilson_upper,
                fisher_p, bh_adjusted_p, evidence_strength, status, rejection_reason,
                failed_gate, source_reflection_ids, metadata_json, llm_note)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                h.hypothesis_id, h.created_at, h.description, json.dumps(h.conditions),
                h.sample_size, h.successes, h.failures, h.win_rate, h.shrunk_win_rate,
                h.wilson_lower, h.wilson_upper, h.fisher_p, h.bh_adjusted_p,
                h.evidence_strength.value if h.evidence_strength else None,
                h.status.value, h.rejection_reason, h.failed_gate,
                json.dumps(h.source_reflection_ids), json.dumps(h.metadata_json), h.llm_note,
            ),
        )
        n += cur.rowcount
    conn.commit()
    return n


def load_hypotheses(
    conn: sqlite3.Connection, status: str | None = None, limit: int | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM validated_hypotheses"
    clauses, params = [], []
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC, bh_adjusted_p ASC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    return pd.read_sql(query, conn, params=tuple(params))


def export_hypotheses_report(
    conn: sqlite3.Connection,
    path: Path | None = None,
    narrative: dict | None = None,
    resolved_trade_count: int = 0,
) -> Path:
    """Write data/published/hypotheses_report.json. `resolved_trade_count`
    is the size of the resolved-recommendation population the run
    analyzed (passed in by the caller, the only place that knows it) —
    same convention as export_reflection_report. `narrative` is the LLM's
    HypothesisNarrativeOutput as a dict, or None if that call failed/was
    skipped — the report is always written either way."""
    path = path or _MIRROR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = conn.cursor()
    rows = [
        dict(zip(_HYP_COLS, row))
        for row in cur.execute(
            f"SELECT {','.join(_HYP_COLS)} FROM validated_hypotheses ORDER BY created_at DESC, bh_adjusted_p ASC"
        ).fetchall()
    ]
    for row in rows:
        row["conditions"] = json.loads(row["conditions"]) if row["conditions"] else []
        row["source_reflection_ids"] = json.loads(row["source_reflection_ids"]) if row["source_reflection_ids"] else []
        row["metadata_json"] = json.loads(row["metadata_json"]) if row["metadata_json"] else {}

    by_status: dict[str, int] = {}
    for row in rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_hypotheses": len(rows),
            "by_status": by_status,
            "resolved_trade_count": resolved_trade_count,
        },
        "hypotheses": rows,
        "narrative": narrative,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def import_hypotheses(conn: sqlite3.Connection, path: Path | None = None) -> int:
    """Idempotent — INSERT OR IGNORE on the primary key, same rehydration
    role as stock_scanner.db.reflection.import_reflection_observations:
    repopulate a fresh ephemeral-runner DB from the committed mirror
    before a new run, without duplicating history already recorded
    there."""
    path = path or _MIRROR_PATH
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    cur = conn.cursor()
    n = 0
    for row in data.get("hypotheses", []):
        row = dict(row)
        row["conditions"] = json.dumps(row.get("conditions") or [])
        row["source_reflection_ids"] = json.dumps(row.get("source_reflection_ids") or [])
        row["metadata_json"] = json.dumps(row.get("metadata_json") or {})
        cur.execute(
            f"""INSERT OR IGNORE INTO validated_hypotheses ({",".join(_HYP_COLS)})
                VALUES ({",".join("?" * len(_HYP_COLS))})""",
            tuple(row.get(c) for c in _HYP_COLS),
        )
        n += cur.rowcount
    conn.commit()
    return n

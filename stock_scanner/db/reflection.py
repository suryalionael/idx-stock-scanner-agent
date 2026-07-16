"""Read/write helpers for reflection_observations — see
stock_scanner/ai_lab/reflection_engine.py for how observations are
computed (pure code, deterministic) and scripts/run_reflection_engine.py
for the orchestrator that calls these.

Standalone, experimental: nothing in stock_scanner/pipeline/ or
stock_scanner/alerts/ reads or writes this table. data/db/signals.db is
gitignored and rebuilt fresh on every CI runner (same rationale as
stock_scanner/db/ai_lab.py), so history lives in the committed
data/published/reflection_report.json mirror.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_scanner.ai_lab.schemas import ReflectionObservation

_MIRROR_PATH = Path(__file__).parent.parent.parent / "data" / "published" / "reflection_report.json"

_OBS_COLS = [
    "observation_id", "category", "title", "description", "supporting_statistics",
    "affected_trade_count", "confidence", "llm_note", "generated_at",
]


def upsert_observations(conn: sqlite3.Connection, observations: list[ReflectionObservation]) -> int:
    """Append-only insert keyed on the deterministic observation_id (unique
    per run — it hashes generated_at, see reflection_engine.py) — plain
    INSERT OR IGNORE, no conflict-update branch needed: observations are
    immutable analysis snapshots, not mutable positions like
    ai_recommendations, so there's nothing to preserve-on-conflict."""
    if not observations:
        return 0
    cur = conn.cursor()
    n = 0
    for obs in observations:
        cur.execute(
            """INSERT OR IGNORE INTO reflection_observations
               (observation_id, category, title, description, supporting_statistics,
                affected_trade_count, confidence, llm_note, generated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                obs.observation_id, obs.category.value, obs.title, obs.description,
                json.dumps(obs.supporting_statistics, default=str),
                obs.affected_trade_count, obs.confidence, obs.llm_note, obs.generated_at,
            ),
        )
        n += cur.rowcount
    conn.commit()
    return n


def load_observations(
    conn: sqlite3.Connection, category: str | None = None, limit: int | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM reflection_observations"
    clauses, params = [], []
    if category is not None:
        clauses.append("category = ?")
        params.append(category)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY generated_at DESC, confidence DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    return pd.read_sql(query, conn, params=tuple(params))


def export_reflection_report(
    conn: sqlite3.Connection,
    path: Path | None = None,
    narrative: dict | None = None,
    resolved_trade_count: int = 0,
) -> Path:
    """Write data/published/reflection_report.json. `resolved_trade_count`
    is the size of the resolved-recommendation population the run analyzed
    (passed in by the caller, which is the only place that knows it) — not
    derived from the observations table, since the largest single slice's
    affected_trade_count would misleadingly understate the true
    population. `narrative` is the LLM's ReflectionNarrativeOutput as a
    dict, or None if that call failed/was skipped — the report is always
    written either way (see stock_scanner.ai_lab.agents.reflection_agent's
    docstring for why)."""
    path = path or _MIRROR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = conn.cursor()
    rows = [
        dict(zip(_OBS_COLS, row))
        for row in cur.execute(
            f"SELECT {','.join(_OBS_COLS)} FROM reflection_observations ORDER BY generated_at DESC, confidence DESC"
        ).fetchall()
    ]
    for row in rows:
        row["supporting_statistics"] = json.loads(row["supporting_statistics"]) if row["supporting_statistics"] else {}

    by_category: dict[str, int] = {}
    for row in rows:
        by_category[row["category"]] = by_category.get(row["category"], 0) + 1

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_observations": len(rows),
            "by_category": by_category,
            "resolved_trade_count": resolved_trade_count,
        },
        "observations": rows,
        "narrative": narrative,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def import_reflection_observations(conn: sqlite3.Connection, path: Path | None = None) -> int:
    """Idempotent — INSERT OR IGNORE on the primary key, same rehydration
    role as stock_scanner.db.ai_lab.import_ai_recommendations: repopulate
    a fresh ephemeral-runner DB from the committed mirror before a new run,
    without duplicating history already recorded there."""
    path = path or _MIRROR_PATH
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    cur = conn.cursor()
    n = 0
    for row in data.get("observations", []):
        row = dict(row)
        row["supporting_statistics"] = json.dumps(row.get("supporting_statistics") or {}, default=str)
        cur.execute(
            f"""INSERT OR IGNORE INTO reflection_observations ({",".join(_OBS_COLS)})
                VALUES ({",".join("?" * len(_OBS_COLS))})""",
            tuple(row.get(c) for c in _OBS_COLS),
        )
        n += cur.rowcount
    conn.commit()
    return n

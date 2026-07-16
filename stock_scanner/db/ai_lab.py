"""Read/write helpers for AI Lab's two tables (ai_recommendations,
ai_learning_events) — see stock_scanner/ai_lab/ for the agents that
produce recommendations and dashboard/ai_lab_view.py for the read-only
render.

Standalone, experimental: nothing in stock_scanner/pipeline/ or
stock_scanner/alerts/ reads or writes these tables. data/db/signals.db is
gitignored and rebuilt fresh on every CI runner (same rationale as
stock_scanner/db/knowledge_base.py / daily_movers.py / top_signals.py), so
history lives in the committed data/published/ai_recommendations.json and
data/published/ai_learning_events.json mirrors.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_scanner.ai_lab.schemas import AIRecommendation, RecommendationStatus

_RECS_MIRROR_PATH = Path(__file__).parent.parent.parent / "data" / "published" / "ai_recommendations.json"
_EVENTS_MIRROR_PATH = Path(__file__).parent.parent.parent / "data" / "published" / "ai_learning_events.json"

_REC_COLS = [
    "id", "ticker", "ai_model", "score", "confidence", "recommendation", "reasoning",
    "decision_trace", "confidence_breakdown", "historical_comparison",
    "expected_return", "risk_level", "generated_date", "created_at", "updated_at",
    "status", "entry_price", "exit_price", "return_percentage",
    "highest_price", "lowest_price", "max_runup_pct", "max_drawdown_pct",
    "holding_days", "trade_outcome", "model",
]
_JSON_COLS = ("reasoning", "decision_trace", "confidence_breakdown", "historical_comparison")
_EVENT_COLS = ["event_id", "ai_model", "event_type", "description", "metadata_json", "created_at"]

VALID_STATUSES = {s.value for s in RecommendationStatus}


# ---------------------------------------------------------------------------
# ai_recommendations
# ---------------------------------------------------------------------------

def upsert_recommendations(conn: sqlite3.Connection, recommendations: list[AIRecommendation]) -> int:
    """UPSERT recommendations keyed on the deterministic `id`. Idempotent —
    re-running the pipeline for the same (ticker, ai_model, generated_date)
    overwrites the score/reasoning/etc in place rather than duplicating,
    but never touches `status`/`entry_price`/`exit_price`/`return_percentage`
    or the tracking fields (`highest_price`/`lowest_price`/`max_runup_pct`/
    `max_drawdown_pct`/`holding_days`/`trade_outcome`) on conflict — those
    are lifecycle fields owned exclusively by update_recommendation_status()
    / stock_scanner.ai_lab.resolver, never by a re-run of the generation
    pipeline. This is what makes "never overwrite history" true: a
    recommendation's content can be regenerated the same day, but its
    lifecycle progress never gets silently reset."""
    if not recommendations:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    n = 0
    for rec in recommendations:
        cur.execute(
            """INSERT INTO ai_recommendations
               (id, ticker, ai_model, score, confidence, recommendation, reasoning,
                decision_trace, confidence_breakdown, historical_comparison,
                expected_return, risk_level, generated_date, created_at, updated_at,
                status, entry_price, exit_price, return_percentage,
                highest_price, lowest_price, max_runup_pct, max_drawdown_pct,
                holding_days, trade_outcome, model)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                 score=excluded.score, confidence=excluded.confidence,
                 recommendation=excluded.recommendation, reasoning=excluded.reasoning,
                 decision_trace=excluded.decision_trace,
                 confidence_breakdown=excluded.confidence_breakdown,
                 historical_comparison=excluded.historical_comparison,
                 expected_return=excluded.expected_return, risk_level=excluded.risk_level,
                 model=excluded.model, updated_at=excluded.updated_at""",
            (
                rec.id, rec.ticker, rec.ai_model, rec.score, rec.confidence,
                rec.recommendation.value, json.dumps(rec.reasoning),
                rec.decision_trace.model_dump_json(), rec.confidence_breakdown.model_dump_json(),
                rec.historical_comparison.model_dump_json(),
                rec.expected_return, rec.risk_level.value if rec.risk_level else None,
                rec.generated_date, now, now, rec.status.value,
                rec.entry_price, rec.exit_price, rec.return_percentage,
                rec.highest_price, rec.lowest_price, rec.max_runup_pct, rec.max_drawdown_pct,
                rec.holding_days, rec.trade_outcome.value if rec.trade_outcome else None, rec.model,
            ),
        )
        n += 1
    conn.commit()
    return n


def update_recommendation_status(
    conn: sqlite3.Connection,
    recommendation_id: str,
    new_status: str,
    entry_price: float | None = None,
    exit_price: float | None = None,
    return_percentage: float | None = None,
    highest_price: float | None = None,
    lowest_price: float | None = None,
    max_runup_pct: float | None = None,
    max_drawdown_pct: float | None = None,
    holding_days: int | None = None,
    trade_outcome: str | None = None,
) -> int:
    """Advance one recommendation's lifecycle, or refresh its tracking
    fields in place (pass the row's current status as `new_status` to
    update highest_price/lowest_price/max_runup_pct/max_drawdown_pct/
    holding_days without a real transition — see
    stock_scanner.ai_lab.resolver, which calls this every run for each
    ACTIVE row whether or not it resolves this time). Returns the number of
    rows updated (0 if recommendation_id doesn't exist) — an UPDATE with no
    matching WHERE row is not a SQL error, so callers must check this
    rather than assume success (same guardrail as
    stock_scanner.db.knowledge_base.update_status)."""
    if new_status not in VALID_STATUSES:
        raise ValueError(f"Unknown status {new_status!r}; must be one of {sorted(VALID_STATUSES)}")
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """UPDATE ai_recommendations
           SET status = ?, entry_price = COALESCE(?, entry_price),
               exit_price = COALESCE(?, exit_price),
               return_percentage = COALESCE(?, return_percentage),
               highest_price = COALESCE(?, highest_price),
               lowest_price = COALESCE(?, lowest_price),
               max_runup_pct = COALESCE(?, max_runup_pct),
               max_drawdown_pct = COALESCE(?, max_drawdown_pct),
               holding_days = COALESCE(?, holding_days),
               trade_outcome = COALESCE(?, trade_outcome),
               updated_at = ?
           WHERE id = ?""",
        (
            new_status, entry_price, exit_price, return_percentage,
            highest_price, lowest_price, max_runup_pct, max_drawdown_pct,
            holding_days, trade_outcome, now, recommendation_id,
        ),
    )
    conn.commit()
    return cur.rowcount


def load_recommendations(
    conn: sqlite3.Connection, ai_model: str | None = None, status: str | None = None,
) -> pd.DataFrame:
    query = "SELECT * FROM ai_recommendations"
    clauses, params = [], []
    if ai_model is not None:
        clauses.append("ai_model = ?")
        params.append(ai_model)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY generated_date DESC, score DESC"
    return pd.read_sql(query, conn, params=tuple(params))


def export_ai_recommendations(conn: sqlite3.Connection, path: Path | None = None) -> Path:
    path = path or _RECS_MIRROR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = conn.cursor()
    rows = [
        dict(zip(_REC_COLS, row))
        for row in cur.execute(
            f"SELECT {','.join(_REC_COLS)} FROM ai_recommendations ORDER BY generated_date DESC, score DESC"
        ).fetchall()
    ]
    for row in rows:
        for col in _JSON_COLS:
            row[col] = json.loads(row[col]) if row[col] else {}

    distinct_dates = sorted({row["generated_date"] for row in rows}, reverse=True)
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    payload = {
        "as_of_date": distinct_dates[0] if distinct_dates else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {"total_rows": len(rows), "distinct_dates": len(distinct_dates), "by_status": status_counts},
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def import_ai_recommendations(conn: sqlite3.Connection, path: Path | None = None) -> int:
    """Idempotent — INSERT OR IGNORE on the primary key, safe against a DB
    that already has some/all of these rows. Deliberately does NOT
    overwrite existing rows (unlike upsert_recommendations) — this is only
    for rehydrating a fresh ephemeral-runner DB from the committed mirror
    before a new day's generation run, so any lifecycle progress already
    recorded in the mirror is preserved exactly as committed."""
    path = path or _RECS_MIRROR_PATH
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    cur = conn.cursor()
    n = 0
    for row in data.get("rows", []):
        row = dict(row)
        for col in _JSON_COLS:
            row[col] = json.dumps(row.get(col) or {})
        cur.execute(
            f"""INSERT OR IGNORE INTO ai_recommendations ({",".join(_REC_COLS)})
                VALUES ({",".join("?" * len(_REC_COLS))})""",
            tuple(row.get(c) for c in _REC_COLS),
        )
        n += cur.rowcount
    conn.commit()
    return n


# ---------------------------------------------------------------------------
# ai_learning_events
# ---------------------------------------------------------------------------

def log_learning_event(
    conn: sqlite3.Connection, event_type: str, description: str,
    ai_model: str | None = None, metadata: dict | None = None,
) -> str:
    """Append one learning-timeline event. Append-only by construction —
    there is no update/delete helper for this table, matching the
    dashboard's "AI Learning Timeline" being a pure historical log."""
    now = datetime.now(timezone.utc).isoformat()
    raw = f"{ai_model}|{event_type}|{description}|{now}"
    event_id = hashlib.sha1(raw.encode()).hexdigest()[:16]
    conn.execute(
        """INSERT INTO ai_learning_events (event_id, ai_model, event_type, description, metadata_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(event_id) DO NOTHING""",
        (event_id, ai_model, event_type, description, json.dumps(metadata or {}), now),
    )
    conn.commit()
    return event_id


def load_learning_events(conn: sqlite3.Connection, limit: int = 50) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT * FROM ai_learning_events ORDER BY created_at DESC LIMIT ?", conn, params=(limit,),
    )


def export_learning_events(conn: sqlite3.Connection, path: Path | None = None) -> Path:
    path = path or _EVENTS_MIRROR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = conn.cursor()
    rows = [
        dict(zip(_EVENT_COLS, row))
        for row in cur.execute(
            f"SELECT {','.join(_EVENT_COLS)} FROM ai_learning_events ORDER BY created_at DESC"
        ).fetchall()
    ]
    for row in rows:
        row["metadata_json"] = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {"total_events": len(rows)},
        "events": rows,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def import_learning_events(conn: sqlite3.Connection, path: Path | None = None) -> int:
    path = path or _EVENTS_MIRROR_PATH
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    cur = conn.cursor()
    n = 0
    for row in data.get("events", []):
        row = dict(row)
        row["metadata_json"] = json.dumps(row.get("metadata_json") or {})
        cur.execute(
            f"""INSERT OR IGNORE INTO ai_learning_events ({",".join(_EVENT_COLS)})
                VALUES ({",".join("?" * len(_EVENT_COLS))})""",
            tuple(row.get(c) for c in _EVENT_COLS),
        )
        n += cur.rowcount
    conn.commit()
    return n

"""Read/write helpers for the top_signals table — see
stock_scanner/pipeline/top_signals.py for filtering/ranking and
scripts/build_top_signals.py for the orchestrator.

Non-production, standalone daily persistence feature — deliberately NOT the
knowledge_base table (Learning Agent Phase 1). Not read by signal_engine.py,
ml_ranker.py, or any promotion path. data/db/signals.db is gitignored and
rebuilt fresh on every CI runner (same rationale as
stock_scanner/db/daily_movers.py / registry_io.py / knowledge_base.py), so
history lives in the committed data/published/top_signals.json mirror —
import it into a fresh DB before upserting a new day's batch, export the
full table back to JSON after.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_DEFAULT_MIRROR_PATH = Path(__file__).parent.parent.parent / "data" / "published" / "top_signals.json"

_COLS = [
    "signal_id", "ticker", "strategy", "signal_date", "eval_date", "signal_label",
    "prev_close", "eval_close", "eval_high", "pct_close", "pct_high",
    "forward_return_pct", "quality_adjusted_score", "total_score",
    "enhanced_total_score", "ml_prob", "quality_source", "rank_in_day",
    "filter_threshold_pct", "source_run_id", "computed_at",
]


def upsert_top_signals(
    conn: sqlite3.Connection, rows: pd.DataFrame, source_run_id: str, threshold_pct: float = 10.0,
) -> int:
    """UPSERT rows into top_signals, keyed on the deterministic signal_id.

    Idempotent: re-running against unchanged signal_results.csv content
    overwrites each row with the same values (ON CONFLICT DO UPDATE) rather
    than duplicating. A later re-run that finds a previously-unavailable
    ranked CSV (see enrich_with_quality_scores) correctly fills in the
    quality columns retroactively for the same signal_id.
    """
    if rows.empty:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    n = 0
    for _, r in rows.iterrows():
        cur.execute(
            """INSERT INTO top_signals
               (signal_id, ticker, strategy, signal_date, eval_date, signal_label,
                prev_close, eval_close, eval_high, pct_close, pct_high,
                forward_return_pct, quality_adjusted_score, total_score,
                enhanced_total_score, ml_prob, quality_source, rank_in_day,
                filter_threshold_pct, source_run_id, computed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(signal_id) DO UPDATE SET
                 prev_close=excluded.prev_close, eval_close=excluded.eval_close,
                 eval_high=excluded.eval_high, pct_close=excluded.pct_close,
                 pct_high=excluded.pct_high, forward_return_pct=excluded.forward_return_pct,
                 quality_adjusted_score=excluded.quality_adjusted_score,
                 total_score=excluded.total_score,
                 enhanced_total_score=excluded.enhanced_total_score,
                 ml_prob=excluded.ml_prob, quality_source=excluded.quality_source,
                 rank_in_day=excluded.rank_in_day, source_run_id=excluded.source_run_id,
                 computed_at=excluded.computed_at""",
            (
                r["signal_id"], r["ticker"], r["strategy"], str(r["signal_date"])[:10],
                str(r["eval_date"])[:10], r.get("signal_label"),
                _f(r.get("prev_close")), _f(r.get("eval_close")), _f(r.get("eval_high")),
                _f(r.get("pct_close")), _f(r.get("pct_high")), float(r["forward_return_pct"]),
                _f(r.get("quality_adjusted_score")), _f(r.get("total_score")),
                _f(r.get("enhanced_total_score")), _f(r.get("ml_prob")),
                r.get("quality_source", "unavailable"), int(r["rank_in_day"]),
                threshold_pct, source_run_id, now,
            ),
        )
        n += 1
    conn.commit()
    return n


def _f(value) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def load_top_signals(conn: sqlite3.Connection, eval_date: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM top_signals"
    params: tuple = ()
    if eval_date is not None:
        query += " WHERE eval_date = ?"
        params = (str(eval_date)[:10],)
    query += " ORDER BY eval_date DESC, rank_in_day ASC"
    return pd.read_sql(query, conn, params=params)


def export_top_signals(conn: sqlite3.Connection, path: Path | None = None) -> Path:
    """Mirror the full top_signals table to committed JSON — same rationale
    as daily_movers.export_daily_movers()."""
    path = path or _DEFAULT_MIRROR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = conn.cursor()
    rows = [
        dict(zip(_COLS, row))
        for row in cur.execute(
            f"SELECT {','.join(_COLS)} FROM top_signals ORDER BY eval_date DESC, rank_in_day ASC"
        ).fetchall()
    ]
    distinct_dates = sorted({row["eval_date"] for row in rows}, reverse=True)
    payload = {
        "as_of_date": distinct_dates[0] if distinct_dates else None,
        "filter_rule": "forward_return_pct > 0.10 (pct_close / 100, strict)",
        "rank_method": "eval_date cohort: forward_return_pct desc, quality_adjusted_score desc (nulls last), ticker asc",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_rows": len(rows),
            "distinct_dates": len(distinct_dates),
            "quality_enriched_count": sum(1 for row in rows if row["quality_source"] == "ranked_csv"),
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def import_top_signals(conn: sqlite3.Connection, path: Path | None = None) -> int:
    """Idempotent — INSERT OR IGNORE on the primary key, safe to call
    against a DB that already has some/all of these rows."""
    path = path or _DEFAULT_MIRROR_PATH
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    cur = conn.cursor()
    n = 0
    for row in data.get("rows", []):
        cur.execute(
            f"""INSERT OR IGNORE INTO top_signals ({','.join(_COLS)})
                VALUES ({','.join('?' * len(_COLS))})""",
            tuple(row.get(c) for c in _COLS),
        )
        n += cur.rowcount
    conn.commit()
    return n

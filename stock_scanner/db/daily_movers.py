"""Read/write helpers for the daily_movers table — see
stock_scanner/pipeline/daily_movers.py for metric computation and
scripts/build_daily_movers.py for the orchestrator.

Non-production, standalone feature: not read by signal_engine.py,
ml_ranker.py, or any promotion path. data/db/signals.db is gitignored and
rebuilt fresh on every CI runner (same rationale as
stock_scanner/db/registry_io.py and stock_scanner/db/knowledge_base.py), so
history lives in the committed data/published/daily_movers.json mirror —
import it into a fresh DB before upserting a new day, export the full table
back to JSON after.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_DEFAULT_MIRROR_PATH = Path(__file__).parent.parent.parent / "data" / "published" / "daily_movers.json"

_COLS = [
    "trade_date", "ticker", "prev_close", "open", "high", "low", "close", "volume",
    "pct_change_close", "pct_change_high", "hit_10pct_close", "hit_10pct_intraday",
    "source", "inserted_at", "updated_at",
]


def upsert_daily_movers(conn: sqlite3.Connection, rows: pd.DataFrame, source: str = "yfinance") -> int:
    """UPSERT rows into daily_movers, keyed on (trade_date, ticker).

    Idempotent: re-running the same trade_date/ticker overwrites the metrics
    in place (ON CONFLICT DO UPDATE) rather than inserting a duplicate row.
    `inserted_at` is set only on first insert; `updated_at` always reflects
    the most recent write. Returns the number of rows processed (0 if
    `rows` is empty).
    """
    if rows.empty:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    n = 0
    for _, r in rows.iterrows():
        cur.execute(
            """INSERT INTO daily_movers
               (trade_date, ticker, prev_close, open, high, low, close, volume,
                pct_change_close, pct_change_high, hit_10pct_close, hit_10pct_intraday,
                source, inserted_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(trade_date, ticker) DO UPDATE SET
                 prev_close=excluded.prev_close, open=excluded.open, high=excluded.high,
                 low=excluded.low, close=excluded.close, volume=excluded.volume,
                 pct_change_close=excluded.pct_change_close,
                 pct_change_high=excluded.pct_change_high,
                 hit_10pct_close=excluded.hit_10pct_close,
                 hit_10pct_intraday=excluded.hit_10pct_intraday,
                 source=excluded.source, updated_at=excluded.updated_at""",
            (
                str(r["trade_date"])[:10], r["ticker"],
                float(r["prev_close"]), float(r["open"]), float(r["high"]),
                float(r["low"]), float(r["close"]),
                float(r["volume"]) if pd.notna(r.get("volume")) else None,
                float(r["pct_change_close"]), float(r["pct_change_high"]),
                int(bool(r["hit_10pct_close"])), int(bool(r["hit_10pct_intraday"])),
                source, now, now,
            ),
        )
        n += 1
    conn.commit()
    return n


def load_daily_movers(conn: sqlite3.Connection, trade_date: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM daily_movers"
    params: tuple = ()
    if trade_date is not None:
        query += " WHERE trade_date = ?"
        params = (str(trade_date)[:10],)
    query += " ORDER BY trade_date DESC, pct_change_close DESC"
    return pd.read_sql(query, conn, params=params)


def export_daily_movers(conn: sqlite3.Connection, path: Path | None = None) -> Path:
    """Mirror the full daily_movers table to committed JSON — same rationale
    as knowledge_base.export_knowledge_base(): data/db/signals.db is
    gitignored, so this JSON is the only durable, cross-run record."""
    path = path or _DEFAULT_MIRROR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = conn.cursor()
    rows = [
        dict(zip(_COLS, row))
        for row in cur.execute(
            f"SELECT {','.join(_COLS)} FROM daily_movers ORDER BY trade_date DESC, ticker"
        ).fetchall()
    ]
    for row in rows:
        row["hit_10pct_close"] = bool(row["hit_10pct_close"])
        row["hit_10pct_intraday"] = bool(row["hit_10pct_intraday"])

    distinct_dates = sorted({row["trade_date"] for row in rows}, reverse=True)
    payload = {
        "as_of_date": distinct_dates[0] if distinct_dates else None,
        "source": "yfinance",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_rows": len(rows),
            "distinct_dates": len(distinct_dates),
            "hit_10pct_close_count": sum(1 for row in rows if row["hit_10pct_close"]),
            "hit_10pct_intraday_count": sum(1 for row in rows if row["hit_10pct_intraday"]),
        },
        "rows": rows,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def import_daily_movers(conn: sqlite3.Connection, path: Path | None = None) -> int:
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
            f"""INSERT OR IGNORE INTO daily_movers ({','.join(_COLS)})
                VALUES ({','.join('?' * len(_COLS))})""",
            tuple(row.get(c) for c in _COLS),
        )
        n += cur.rowcount
    conn.commit()
    return n

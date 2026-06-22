#!/usr/bin/env python3
"""Week-1 starter: create the SQLite DB and backfill it from existing files.

Does NOT touch the live pipeline (scan.yml/performance.yml) or the screener.
This is purely a mirror of data that already exists, keyed properly by
signal_id so future joins don't repeat the bugs the leakage audit found
(zero-volume date mismatches, snapshot-file misalignment).

Sources (all already validated this session):
  data/performance/signal_results.csv     -> signals + outcomes
  data/signals/{signal_date}.parquet       -> feature_snapshots
  data/published/ihsg_recent.parquet       -> market_context
  stock_scanner/configs/issuers.csv        -> sector_reference

Usage:
    python scripts/init_db_and_backfill.py
"""
import json
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import pandas as pd
from loguru import logger

from stock_scanner.db.init_db import create_schema, get_connection, signal_id

_RESULTS_CSV = repo_root / "data" / "performance" / "signal_results.csv"
_SIGNALS_DIR = repo_root / "data" / "signals"
_IHSG_PATH = repo_root / "data" / "published" / "ihsg_recent.parquet"
_ISSUERS_CSV = repo_root / "stock_scanner" / "configs" / "issuers.csv"

_LABEL_THRESHOLD_PCT = 10.0   # matches the validated work this session

_FEATURE_COLS = [
    "ma5", "ma20", "ma50", "ma200", "ma_full_alignment", "ma_partial_alignment",
    "slope_ma20", "golden_cross", "price_vs_ma200",
    "rsi14", "macd", "macd_signal", "macd_histogram", "roc5", "roc20",
    "high_52w", "pct_from_52w_high", "atr14", "atr_breakout",
    "vol_ratio_20d", "vol_spike", "obv_trend",
    "atr_pct", "bb_width", "hist_vol_20d",
    "supertrend_bullish", "stoch_rsi_k", "stoch_rsi_d", "adx", "adx_pos", "adx_neg",
    "squeeze_on", "squeeze_release", "vwap_20d", "price_vs_vwap",
    "total_score", "quality_adjusted_score", "enhanced_total_score", "ml_prob",
]


def backfill_signals_and_outcomes(conn) -> int:
    df = pd.read_csv(_RESULTS_CSV)
    rows = []
    for _, r in df.iterrows():
        sid = signal_id(r["ticker"], r["signal_date"], r["strategy"])
        label_success = None
        if r["status"] == "evaluated" and pd.notna(r["pct_close"]):
            label_success = int(r["pct_close"] > _LABEL_THRESHOLD_PCT)
        rows.append({
            "signal_id": sid, "ticker": r["ticker"], "signal_date": r["signal_date"],
            "strategy": r["strategy"], "signal_label": r["signal"],
            "eval_date": r.get("eval_date"), "status": r["status"],
            "prev_close": r.get("prev"), "eval_high": r.get("high"), "eval_close": r.get("close"),
            "pct_high": r.get("pct_high"), "pct_close": r.get("pct_close"), "wl": r.get("wl"),
            "label_success": label_success,
        })

    cur = conn.cursor()
    for row in rows:
        cur.execute(
            """INSERT OR IGNORE INTO signals
               (signal_id, ticker, signal_date, strategy, signal_label)
               VALUES (?, ?, ?, ?, ?)""",
            (row["signal_id"], row["ticker"], row["signal_date"], row["strategy"], row["signal_label"]),
        )
        cur.execute(
            """INSERT INTO outcomes
               (signal_id, eval_date, status, prev_close, eval_high, eval_close,
                pct_high, pct_close, wl, label_success, labeled_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(signal_id) DO UPDATE SET
                 eval_date=excluded.eval_date, status=excluded.status,
                 prev_close=excluded.prev_close, eval_high=excluded.eval_high,
                 eval_close=excluded.eval_close, pct_high=excluded.pct_high,
                 pct_close=excluded.pct_close, wl=excluded.wl,
                 label_success=excluded.label_success, labeled_at=CURRENT_TIMESTAMP""",
            (row["signal_id"], row["eval_date"], row["status"], row["prev_close"],
             row["eval_high"], row["eval_close"], row["pct_high"], row["pct_close"],
             row["wl"], row["label_success"]),
        )
    conn.commit()
    return len(rows)


def backfill_feature_snapshots(conn) -> int:
    n = 0
    cur = conn.cursor()
    for f in sorted(_SIGNALS_DIR.glob("*.parquet")):
        signal_date = f.stem
        try:
            snap = pd.read_parquet(f)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skip {}: {}", f.name, exc)
            continue
        # Only need to snapshot tickers that actually have a signals row for
        # this date — found via the signals table itself, not by re-deriving
        # strategy membership here.
        sig_rows = cur.execute(
            "SELECT signal_id, ticker, strategy FROM signals WHERE signal_date = ?",
            (signal_date,),
        ).fetchall()
        if not sig_rows:
            continue
        snap_idx = snap.set_index("ticker")
        for sid, ticker, _strategy in sig_rows:
            if ticker not in snap_idx.index:
                continue
            row = snap_idx.loc[ticker]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            features = {c: (None if pd.isna(row.get(c)) else
                            bool(row[c]) if isinstance(row.get(c), (bool,))
                            else float(row[c]) if isinstance(row.get(c), (int, float))
                            else str(row[c]))
                        for c in _FEATURE_COLS if c in row.index}
            cur.execute(
                """INSERT INTO feature_snapshots
                   (signal_id, feature_set_version, features_json, raw_close,
                    raw_open, raw_volume, snapshot_source_path)
                   VALUES (?, 'fb_v1_backfill', ?, ?, ?, ?, ?)
                   ON CONFLICT(signal_id) DO NOTHING""",
                (sid, json.dumps(features), float(row.get("close")) if pd.notna(row.get("close")) else None,
                 None, float(row.get("volume")) if pd.notna(row.get("volume")) else None, str(f)),
            )
            n += 1
    conn.commit()
    return n


def backfill_market_context(conn) -> int:
    if not _IHSG_PATH.exists():
        logger.warning("IHSG bundle not found at {} — skipping market_context", _IHSG_PATH)
        return 0
    df = pd.read_parquet(_IHSG_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)
    df["pct_change"] = df["close"].pct_change() * 100
    df["trend_5d"] = df["close"].pct_change(5) * 100
    df["trend_20d"] = df["close"].pct_change(20) * 100

    cur = conn.cursor()
    n = 0
    for _, r in df.iterrows():
        regime = "neutral"
        if pd.notna(r["trend_5d"]):
            regime = "risk_on" if r["trend_5d"] > 1 else "risk_off" if r["trend_5d"] < -1 else "neutral"
        cur.execute(
            """INSERT INTO market_context
               (context_date, ihsg_close, ihsg_pct_change, ihsg_trend_5d, ihsg_trend_20d, regime_label)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(context_date) DO UPDATE SET
                 ihsg_close=excluded.ihsg_close, ihsg_pct_change=excluded.ihsg_pct_change,
                 ihsg_trend_5d=excluded.ihsg_trend_5d, ihsg_trend_20d=excluded.ihsg_trend_20d,
                 regime_label=excluded.regime_label""",
            (r["date"], float(r["close"]),
             None if pd.isna(r["pct_change"]) else float(r["pct_change"]),
             None if pd.isna(r["trend_5d"]) else float(r["trend_5d"]),
             None if pd.isna(r["trend_20d"]) else float(r["trend_20d"]), regime),
        )
        n += 1
    conn.commit()
    return n


def backfill_sector_reference(conn) -> int:
    if not _ISSUERS_CSV.exists():
        return 0
    df = pd.read_csv(_ISSUERS_CSV)
    cur = conn.cursor()
    for _, r in df.iterrows():
        cur.execute(
            """INSERT INTO sector_reference (ticker, company_name, sector) VALUES (?, ?, ?)
               ON CONFLICT(ticker) DO UPDATE SET
                 company_name=excluded.company_name, sector=excluded.sector""",
            (r["ticker"].strip(), r.get("company_name"), r.get("sector")),
        )
    conn.commit()
    return len(df)


def main() -> None:
    conn = get_connection()
    create_schema(conn)
    logger.info("Schema created/verified.")

    n_sig = backfill_signals_and_outcomes(conn)
    logger.info("signals + outcomes: {} rows from signal_results.csv", n_sig)

    n_feat = backfill_feature_snapshots(conn)
    logger.info("feature_snapshots: {} rows from data/signals/*.parquet", n_feat)

    n_mkt = backfill_market_context(conn)
    logger.info("market_context: {} rows from ihsg_recent.parquet", n_mkt)

    n_sec = backfill_sector_reference(conn)
    logger.info("sector_reference: {} rows from issuers.csv", n_sec)

    cur = conn.cursor()
    for table in ["signals", "feature_snapshots", "outcomes", "market_context", "sector_reference"]:
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info("  {} total rows in {}", count, table)
    conn.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the daily movers >10% table + published JSON artifact.

Standalone, non-production feature: does NOT touch signal_engine.py,
scanner_config.yaml, ml_ranker.py, or any promotion path, and is not read by
the live morning scan. See stock_scanner/pipeline/daily_movers.py (metric
computation) and stock_scanner/db/daily_movers.py (persistence).

Usage:
    python scripts/build_daily_movers.py
    python scripts/build_daily_movers.py --date 2026-07-10
    python scripts/build_daily_movers.py --universe stock_scanner/configs/idx_universe.csv
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from stock_scanner.db.daily_movers import (  # noqa: E402
    export_daily_movers,
    import_daily_movers,
    upsert_daily_movers,
)
from stock_scanner.db.init_db import create_schema, get_connection  # noqa: E402
from stock_scanner.pipeline.daily_movers import compute_daily_movers, fetch_raw_ohlc  # noqa: E402
from stock_scanner.utils.trading_calendar import is_trading_day, last_trading_day  # noqa: E402

_DEFAULT_UNIVERSE = repo_root / "stock_scanner" / "configs" / "idx_universe.csv"
_LOOKBACK_DAYS = 10  # calendar days of history fetched per ticker — comfortably
                      # covers IDX long-weekend/holiday gaps so a valid prev_close
                      # is always available for the target trading day


def _load_universe(universe_path: Path) -> list[str]:
    if not universe_path.exists():
        logger.error(f"Universe file not found: {universe_path}")
        return []
    df = pd.read_csv(universe_path)
    if "is_active" in df.columns:
        df = df[df["is_active"].astype(str).str.lower().isin(["true", "1", "yes"])]
    return df["ticker"].tolist()


def main(target_date: date, universe_path: Path) -> None:
    tickers = _load_universe(universe_path)
    if not tickers:
        logger.error("Universe empty or not found — aborting.")
        return
    logger.info(f"daily_movers: {len(tickers)} tickers, target_date={target_date}")

    start = (target_date - timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end = (target_date + timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance `end` is exclusive
    raw = fetch_raw_ohlc(tickers, start=start, end=end)
    if raw.empty:
        logger.warning("daily_movers: no OHLC data fetched — nothing to do.")
        return

    movers = compute_daily_movers(raw)
    target_ts = pd.Timestamp(target_date)
    movers = movers[movers["trade_date"] == target_ts]
    logger.info(f"daily_movers: {len(movers)} ticker(s) hit >=10% on {target_date}")

    conn = get_connection()
    create_schema(conn)   # idempotent — CREATE TABLE IF NOT EXISTS, adds daily_movers if missing
    n_imported = import_daily_movers(conn)
    logger.info(f"daily_movers: imported {n_imported} historical row(s) from published mirror")

    upsert_daily_movers(conn, movers)
    export_path = export_daily_movers(conn)
    logger.info(f"daily_movers: exported published artifact → {export_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=str, default=None,
                        help="Target trade date YYYY-MM-DD (default: last IDX trading day)")
    parser.add_argument("--universe", type=Path, default=_DEFAULT_UNIVERSE,
                        help="Path to idx_universe.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.date:
        target = date.fromisoformat(args.date)
    else:
        today = date.today()
        target = today if is_trading_day(today) else last_trading_day(today)
    main(target, args.universe)

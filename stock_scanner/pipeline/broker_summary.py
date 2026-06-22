"""Broker Summary — cache I/O helpers for per-ticker/per-date broker activity.

Real data source: Index Alpha API (stock_scanner/pipeline/fetch_indexalpha.py
— IndexAlphaFetcher). This module only reads/writes the local parquet cache;
it has no fetcher of its own and no mock-data fallback. A previous version
of this file had a PlaceholderBrokerFetcher that generated synthetic broker
numbers when the cache was empty — removed (2026-06) because it had zero
callers in the live app, but its existence was a silent-fake-data risk: any
future caller of get_broker_summary(use_mock_if_empty=True) would have
received random numbers indistinguishable in shape from real broker data.
If a fetcher abstraction is needed again, build it explicitly against
IndexAlphaFetcher — never reintroduce a mock-data branch in this module.

Format: data/broker/{ticker}_{YYYY-MM-DD}.parquet
Columns: broker_code, broker_name, buy_lot, sell_lot, net_lot
"""
from pathlib import Path

import pandas as pd
from loguru import logger

BROKER_SCHEMA = {
    "broker_code":  "str   — kode broker, mis. 'YP', 'CC', 'ZP'",
    "broker_name":  "str   — nama sekuritas, mis. 'Indo Premier Sekuritas'",
    "buy_lot":      "float — total lot beli pada tanggal tersebut",
    "sell_lot":     "float — total lot jual pada tanggal tersebut",
    "net_lot":      "float — buy_lot - sell_lot (positif = net buyer)",
}


def _broker_path(ticker: str, date: str, broker_dir: Path) -> Path:
    return broker_dir / f"{ticker}_{date}.parquet"


def save_broker_summary(ticker: str, date: str, df: pd.DataFrame, broker_dir: Path) -> None:
    broker_dir.mkdir(parents=True, exist_ok=True)
    path = _broker_path(ticker, date, broker_dir)
    df.to_parquet(path, index=False)
    logger.debug(f"Broker summary saved → {path}")


def load_broker_summary(ticker: str, date: str, broker_dir: Path) -> pd.DataFrame:
    """Load broker summary untuk ticker + tanggal. Return kosong jika tidak ada."""
    path = _broker_path(ticker, date, broker_dir)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

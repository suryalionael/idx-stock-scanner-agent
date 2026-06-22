from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from loguru import logger

from stock_scanner.pipeline.fetch_base import OHLCV_COLS, BaseFetcher

_DEFAULT_LOOKBACK_YEARS = 3
_DEFAULT_BATCH_SIZE = 20


class YFinanceFetcher(BaseFetcher):
    """Fetcher berbasis yfinance dengan batch download dan single-ticker fallback."""

    def __init__(self, batch_size: int = _DEFAULT_BATCH_SIZE):
        self.batch_size = batch_size

    def fetch(self, tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        results = {}
        batches = [tickers[i:i + self.batch_size] for i in range(0, len(tickers), self.batch_size)]

        for batch in batches:
            logger.info(f"Downloading batch ({len(batch)} tickers): {batch[:5]}{'...' if len(batch) > 5 else ''}")
            try:
                raw = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
                for ticker in batch:
                    df = self._extract_ticker(raw, ticker, len(batch))
                    if df is not None and not df.empty:
                        results[ticker] = df
                    else:
                        logger.warning(f"{ticker}: no data returned from batch download")
            except Exception as e:
                logger.error(f"Batch download failed ({batch}): {e} — falling back to single fetch")
                for ticker in batch:
                    try:
                        df = self.fetch_single(ticker, start, end)
                        if not df.empty:
                            results[ticker] = df
                    except Exception as e2:
                        logger.error(f"{ticker}: single fetch fallback also failed — {e2}")

        return results

    def fetch_single(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        return self._normalize(raw, ticker)

    def _extract_ticker(self, raw: pd.DataFrame, ticker: str, n_tickers: int) -> pd.DataFrame | None:
        try:
            df = raw.copy() if n_tickers == 1 else raw[ticker].copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty:
                return None
            return self._normalize(df, ticker)
        except KeyError:
            return None

    def _normalize(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index.name = "date"
        df = df.reset_index()

        # Selalu Timestamp tz-naive normalized ke midnight
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df["ticker"] = ticker

        df = df.rename(columns={"adj close": "adj_close"})
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]

        keep = [c for c in OHLCV_COLS if c in df.columns]
        return df[keep].dropna(subset=["close"])


def default_start_date(lookback_years: int = _DEFAULT_LOOKBACK_YEARS) -> str:
    return (datetime.today() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")


def default_end_date() -> str:
    return datetime.today().strftime("%Y-%m-%d")


def save_raw(ticker: str, df: pd.DataFrame, raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{ticker}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"{ticker}: saved {len(df)} rows → {path}")


def load_raw(ticker: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{ticker}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def incremental_update(
    ticker: str,
    fetcher: BaseFetcher,
    raw_dir: Path,
    lookback_years: int = _DEFAULT_LOOKBACK_YEARS,
) -> pd.DataFrame:
    """Fetch hanya data baru sejak baris terakhir yang tersimpan.

    Jika tidak ada data tersimpan, download penuh sejak `lookback_years` ke belakang.
    Range kosong (mis. libur bursa) → kembalikan existing tanpa error.
    """
    existing = load_raw(ticker, raw_dir)

    if existing.empty:
        start = default_start_date(lookback_years)
    else:
        last_date = pd.to_datetime(existing["date"]).max()
        start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

    end = default_end_date()

    if start >= end:
        logger.debug(f"{ticker}: already up to date (last={start})")
        return existing

    new_data = fetcher.fetch_single(ticker, start, end)
    if new_data.empty:
        # Bisa hari libur atau provider tidak punya data — bukan error
        logger.debug(f"{ticker}: no new data {start} → {end}, keeping existing")
        return existing

    combined = pd.concat([existing, new_data], ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None).dt.normalize()
    combined = (
        combined
        .drop_duplicates(subset=["date", "ticker"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    save_raw(ticker, combined, raw_dir)
    logger.info(f"{ticker}: +{len(new_data)} new rows (total={len(combined)})")
    return combined

"""Daily movers >10% — non-production, standalone feature.

Detects trading sessions where a ticker moved >=10% against its previous
close, using RAW (not split/dividend-adjusted) OHLC — adjusted prices would
distort the single-session high/prev_close and close/prev_close ratios this
feature is built to detect. Not wired into signal_engine.py,
scanner_config.yaml, ml_ranker.py, or any promotion path — see
scripts/build_daily_movers.py for the orchestrator and
stock_scanner/db/daily_movers.py for persistence.
"""
from __future__ import annotations

import pandas as pd
from loguru import logger

HIT_THRESHOLD = 0.10  # >=10% move; a plain constant since this is a standalone
                       # feature with a single, spec-defined threshold — not
                       # routed through scanner_config.yaml (production config)

_OUTPUT_COLS = [
    "trade_date", "ticker", "prev_close", "open", "high", "low", "close", "volume",
    "pct_change_close", "pct_change_high", "hit_10pct_close", "hit_10pct_intraday",
]


def compute_daily_movers(df: pd.DataFrame, threshold: float = HIT_THRESHOLD) -> pd.DataFrame:
    """Compute per-ticker daily move metrics from raw OHLCV rows.

    Args:
        df: columns date, ticker, open, high, low, close, volume (one row per
            ticker per trading day, raw i.e. not split/dividend-adjusted).
        threshold: hit threshold as a fraction (default 0.10 = 10%).

    Rows with no usable previous close (first observation for a ticker, or a
    non-positive/missing prev_close) are skipped gracefully rather than
    raising — a new listing or a data gap is a normal condition here, not an
    error.

    Returns:
        DataFrame restricted to rows where hit_10pct_close OR
        hit_10pct_intraday is True, with columns _OUTPUT_COLS.
    """
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values(["ticker", "date"])
    df["prev_close"] = df.groupby("ticker")["close"].shift(1)

    before = len(df)
    df = df[df["prev_close"].notna() & (df["prev_close"] > 0)]
    skipped = before - len(df)
    if skipped:
        logger.debug(f"daily_movers: skipped {skipped} row(s) with missing/invalid prev_close")

    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df["pct_change_close"] = df["close"] / df["prev_close"] - 1
    df["pct_change_high"] = df["high"] / df["prev_close"] - 1
    df["hit_10pct_close"] = df["pct_change_close"] >= threshold
    df["hit_10pct_intraday"] = df["pct_change_high"] >= threshold

    movers = df[df["hit_10pct_close"] | df["hit_10pct_intraday"]].copy()
    movers = movers.rename(columns={"date": "trade_date"})
    return movers[_OUTPUT_COLS].reset_index(drop=True)


def fetch_raw_ohlc(tickers: list[str], start: str, end: str, batch_size: int = 20) -> pd.DataFrame:
    """Fetch RAW (auto_adjust=False) daily OHLCV for `tickers` via yfinance.

    Deliberately does NOT reuse stock_scanner.pipeline.fetch_yfinance —
    that fetcher uses auto_adjust=True (correct for technical indicators,
    wrong for this feature's raw single-session event detection). Batched
    like fetch_yfinance for the same reason: yfinance handles large
    multi-ticker requests unreliably.

    A ticker with no data returned (holiday, delisted, bad symbol) is
    skipped with a warning — never raises, so one bad ticker can't crash
    the whole build.
    """
    import yfinance as yf

    frames = []
    batches = [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]

    for batch in batches:
        try:
            raw = yf.download(
                batch, start=start, end=end, auto_adjust=False,
                progress=False, group_by="ticker", threads=True,
            )
        except Exception as e:
            logger.warning(f"daily_movers: batch download failed ({batch[:3]}...): {e}")
            continue

        for ticker in batch:
            try:
                sub = raw.copy() if len(batch) == 1 else raw[ticker].copy()
            except KeyError:
                logger.warning(f"{ticker}: no data returned from yfinance")
                continue
            if isinstance(sub.columns, pd.MultiIndex):
                sub.columns = sub.columns.get_level_values(0)
            sub.columns = [c.lower() for c in sub.columns]
            sub = sub.dropna(subset=["close"]) if "close" in sub.columns else sub
            if sub.empty:
                continue
            sub.index.name = "date"
            sub = sub.reset_index()
            sub["ticker"] = ticker
            keep = [c for c in ["date", "ticker", "open", "high", "low", "close", "volume"] if c in sub.columns]
            frames.append(sub[keep])

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])
    return pd.concat(frames, ignore_index=True)

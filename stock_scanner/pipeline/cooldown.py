"""Cooldown filter for IDX Stock Scanner ranked output.

Prevents the same ticker from re-appearing in ranked output within a cooldown
window. This avoids "re-recommending" a stock that was already flagged recently
and may have already been acted upon (or failed).

Cooldown logic:
    A ticker is in cooldown if it appeared in ANY ranked file within the last
    cooldown_days TRADING days with the same or higher signal tier.

    Signal tier order (highest first): BREAKOUT > PRE_MARKUP > WATCH

Cooldown is NOT applied to:
    - BREAKOUT signals (always show, it's a genuine event)
    - Tickers that upgraded tier (e.g., WATCH → PRE_MARKUP — always show)

Public API
----------
apply_cooldown(df, ranked_dir, cooldown_days, scan_date) -> pd.DataFrame
    Filter df to remove tickers in cooldown. Returns filtered DataFrame.

get_cooldown_set(ranked_dir, scan_date, cooldown_days) -> set[tuple[str, str]]
    Return set of (ticker, min_signal_tier) pairs in cooldown.

check_ticker_cooldown(ticker, signal, ranked_dir, scan_date, cooldown_days) -> bool
    Return True if ticker is in cooldown for the given signal tier.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Signal tier ordering (higher index = higher tier)
# ---------------------------------------------------------------------------

_TIER_ORDER = {"WATCH": 0, "PRE_MARKUP": 1, "BREAKOUT": 2}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_cooldown(
    df: pd.DataFrame,
    ranked_dir: Path,
    cooldown_days: int = 5,
    scan_date: str | None = None,
    exclude_breakout: bool = True,
) -> pd.DataFrame:
    """Remove tickers in cooldown from the ranked DataFrame.

    Args:
        df             : Ranked signals DataFrame (must have 'ticker' and 'signal').
        ranked_dir     : Path to data/ranked/ — used to scan recent history.
        cooldown_days  : Number of trading days to look back for duplicates.
        scan_date      : Date string "YYYY-MM-DD" for today. Defaults to latest ranked date.
        exclude_breakout: If True, BREAKOUT signals bypass cooldown (always shown).

    Returns:
        Filtered DataFrame with cooldown rows removed.
        Adds '_cooldown_skip' column (True = would have been filtered) for audit.
    """
    if df.empty or "ticker" not in df.columns or "signal" not in df.columns:
        return df

    ranked_dir = Path(ranked_dir)
    cooldown_set = get_cooldown_set(ranked_dir, scan_date, cooldown_days)

    if not cooldown_set:
        df = df.copy()
        df["_cooldown_skip"] = False
        return df

    def _is_cooled(row: pd.Series) -> bool:
        ticker = str(row.get("ticker", ""))
        signal = str(row.get("signal", ""))

        # BREAKOUT always passes (if exclude_breakout=True)
        if exclude_breakout and signal == "BREAKOUT":
            return False

        current_tier = _TIER_ORDER.get(signal, -1)

        # In cooldown if a SAME or HIGHER tier appeared recently
        for (cd_ticker, cd_signal) in cooldown_set:
            if cd_ticker == ticker:
                past_tier = _TIER_ORDER.get(cd_signal, -1)
                if past_tier >= current_tier:
                    return True   # same or better signal appeared recently → cooldown
        return False

    df = df.copy()
    df["_cooldown_skip"] = df.apply(_is_cooled, axis=1)

    n_filtered = df["_cooldown_skip"].sum()
    if n_filtered > 0:
        logger.info(
            f"Cooldown filter: {n_filtered} tickers suppressed "
            f"(cooldown_days={cooldown_days})"
        )

    return df[~df["_cooldown_skip"]].copy()


def get_cooldown_set(
    ranked_dir: Path,
    scan_date: str | None = None,
    cooldown_days: int = 5,
) -> set[tuple[str, str]]:
    """Return set of (ticker, signal) pairs that appeared in the cooldown window.

    Scans ranked files from (scan_date - cooldown_days business days) up to
    (scan_date - 1 business day). Today's file is excluded.

    Args:
        ranked_dir   : Path to data/ranked/
        scan_date    : Date string "YYYY-MM-DD". Defaults to most recent ranked date.
        cooldown_days: Number of trading days to look back.

    Returns:
        Set of (ticker, signal) tuples seen in the cooldown window.
    """
    ranked_dir = Path(ranked_dir)
    files = sorted(ranked_dir.glob("ranked_*.csv"))
    if not files:
        return set()

    # Determine scan_date
    if scan_date is None:
        # Use most recent ranked file's date
        scan_date = files[-1].stem.replace("ranked_", "")

    try:
        today = pd.Timestamp(scan_date)
    except Exception:
        logger.warning(f"Cannot parse scan_date: {scan_date} — no cooldown applied")
        return set()

    # Build the cooldown window: [today - cooldown_days bdays, today - 1 bday]
    bdate_range = pd.bdate_range(end=today - pd.Timedelta(days=1), periods=cooldown_days)
    window_start = bdate_range[0] if len(bdate_range) > 0 else today

    cooldown_set: set[tuple[str, str]] = set()

    for f in files:
        date_str = f.stem.replace("ranked_", "")
        try:
            file_date = pd.Timestamp(date_str)
        except Exception:
            continue

        # Only files strictly within cooldown window (not today itself)
        if not (window_start <= file_date < today):
            continue

        try:
            df = pd.read_csv(f, usecols=["ticker", "signal"])
            for _, row in df.iterrows():
                cooldown_set.add((str(row["ticker"]), str(row["signal"])))
        except Exception as e:
            logger.debug(f"Cannot read {f.name}: {e}")

    return cooldown_set


def check_ticker_cooldown(
    ticker: str,
    signal: str,
    ranked_dir: Path,
    scan_date: str | None = None,
    cooldown_days: int = 5,
) -> bool:
    """Return True if this ticker/signal combo is in cooldown.

    Convenience function for checking a single ticker.

    Args:
        ticker       : IDX ticker string (e.g., "BBCA.JK")
        signal       : Signal tier ("BREAKOUT", "PRE_MARKUP", "WATCH")
        ranked_dir   : Path to data/ranked/
        scan_date    : Today's date string
        cooldown_days: Lookback window in trading days

    Returns:
        True if the ticker appeared at the same or higher tier within cooldown window.
    """
    cooldown_set = get_cooldown_set(ranked_dir, scan_date, cooldown_days)
    current_tier = _TIER_ORDER.get(signal, -1)

    for (cd_ticker, cd_signal) in cooldown_set:
        if cd_ticker == ticker:
            past_tier = _TIER_ORDER.get(cd_signal, -1)
            if past_tier >= current_tier:
                return True
    return False

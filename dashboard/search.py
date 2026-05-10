"""Search & ticker context loading for the dashboard search feature.

Public API:
    normalize_ticker(ticker)             → canonical ticker string ("BBCA" → "BBCA.JK")
    get_search_universe(scan_date)       → sorted list of all known tickers
    load_ticker_context(ticker, date)    → dict with all data needed for detail panel
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Ticker normalisation
# ---------------------------------------------------------------------------

def normalize_ticker(ticker: str) -> str:
    """Normalise user input to the system's canonical ticker format.

    Rules:
        - Strip whitespace, uppercase
        - Append '.JK' if not already present

    Examples:
        "bbca"   → "BBCA.JK"
        "GOTO.JK"→ "GOTO.JK"
    """
    t = ticker.strip().upper()
    if t and not t.endswith(".JK"):
        t += ".JK"
    return t


# ---------------------------------------------------------------------------
# Search universe (for autocomplete)
# ---------------------------------------------------------------------------

def get_search_universe(scan_date: str | None = None) -> list[str]:
    """Return a sorted list of all known tickers for search autocomplete.

    Combines:
        1. Tickers from stock_scanner/configs/issuers.csv (static reference)
        2. Tickers found in the signals file for scan_date (dynamic)
    """
    from stock_scanner.reference.issuers import all_tickers as _issuer_tickers
    from dashboard.data_loader import load_all_tickers_for_date, latest_ranked_date

    universe: set[str] = set(_issuer_tickers())

    # Add tickers from the most recent scan so freshly-scanned tickers appear
    date = scan_date or latest_ranked_date()
    if date:
        try:
            df = load_all_tickers_for_date(date)
            if not df.empty and "ticker" in df.columns:
                universe.update(df["ticker"].dropna().tolist())
        except Exception:
            pass

    return sorted(universe)


def format_ticker_option(ticker: str) -> str:
    """Return 'TICKER — Company Name' for selectbox display."""
    from stock_scanner.reference.issuers import ticker_display
    return ticker_display(ticker)


# ---------------------------------------------------------------------------
# Ticker context (all data needed for detail panel)
# ---------------------------------------------------------------------------

def load_ticker_context(ticker: str, scan_date: str | None = None) -> dict:
    """Load every data source needed to render the full ticker detail panel.

    Args:
        ticker    : canonical ticker, e.g. "BBCA.JK"
        scan_date : "YYYY-MM-DD"; if None, uses latest available date

    Returns a dict with keys:
        ticker          (str)
        scan_date       (str)
        found_in_scan   (bool)  — True if ticker appeared in the scan signals
        signal_row      (pd.Series | None) — row from signals DataFrame
        raw_ohlcv       (pd.DataFrame)
        broker          (pd.DataFrame)
        composition     (pd.DataFrame)   — shareholder composition
        monthly_holders (pd.DataFrame)   — monthly count + growth %
    """
    from dashboard.data_loader import (
        load_all_tickers_for_date,
        load_broker_for_ticker,
        load_raw,
        latest_ranked_date,
    )
    from dashboard.shareholders import (
        get_monthly_shareholder_stats,
        get_shareholder_composition,
    )

    date = scan_date or latest_ranked_date() or ""

    ctx: dict = {
        "ticker": ticker,
        "scan_date": date,
        "found_in_scan": False,
        "signal_row": None,
        "raw_ohlcv": pd.DataFrame(),
        "broker": pd.DataFrame(),
        "composition": pd.DataFrame(),
        "monthly_holders": pd.DataFrame(),
    }

    # --- Signal data ---
    if date:
        try:
            df_all = load_all_tickers_for_date(date)
            if not df_all.empty and "ticker" in df_all.columns:
                rows = df_all[df_all["ticker"] == ticker]
                if not rows.empty:
                    ctx["signal_row"] = rows.iloc[0]
                    ctx["found_in_scan"] = True
        except Exception:
            pass

    # --- Raw OHLCV ---
    try:
        ctx["raw_ohlcv"] = load_raw(ticker)
    except Exception:
        pass

    # --- Broker summary ---
    if date:
        try:
            ctx["broker"] = load_broker_for_ticker(ticker, date, use_mock=True)
        except Exception:
            pass

    # --- Shareholder composition ---
    try:
        ctx["composition"] = get_shareholder_composition(ticker, date or "")
    except Exception:
        pass

    # --- Monthly holders ---
    try:
        ctx["monthly_holders"] = get_monthly_shareholder_stats(ticker)
    except Exception:
        pass

    return ctx

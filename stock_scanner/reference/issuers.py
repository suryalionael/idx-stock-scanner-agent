"""Issuer reference data — ticker to company name / sector mapping.

Data source: stock_scanner/configs/issuers.csv
Columns   : ticker, company_name, sector

To extend the mapping, edit issuers.csv directly — no code changes needed.
The file is cached in memory after the first call.
"""
from functools import lru_cache
from pathlib import Path

import pandas as pd

_CSV_PATH = Path(__file__).parent.parent / "configs" / "issuers.csv"


@lru_cache(maxsize=1)
def _load() -> pd.DataFrame:
    """Load issuers.csv once and keep it in memory."""
    if not _CSV_PATH.exists():
        return pd.DataFrame(columns=["ticker", "company_name", "sector"])
    df = pd.read_csv(_CSV_PATH)
    df["ticker"] = df["ticker"].str.strip()
    return df


def _lookup(ticker: str) -> pd.Series | None:
    """Return first matching row for ticker (tries exact, .JK suffix, without suffix)."""
    df = _load()
    if df.empty:
        return None
    t = ticker.strip()
    row = df[df["ticker"] == t]
    if row.empty and not t.endswith(".JK"):
        row = df[df["ticker"] == t + ".JK"]
    if row.empty and t.endswith(".JK"):
        row = df[df["ticker"] == t[:-3]]
    return row.iloc[0] if not row.empty else None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_company_name(ticker: str) -> str:
    """Return company name for ticker, or ticker itself if not found.

    Args:
        ticker: e.g. "BBCA.JK" or "BBCA"

    Returns:
        e.g. "Bank Central Asia Tbk PT"
    """
    hit = _lookup(ticker)
    return str(hit["company_name"]) if hit is not None else ticker


def get_sector(ticker: str) -> str:
    """Return sector label for ticker, or empty string if not found."""
    hit = _lookup(ticker)
    if hit is None or "sector" not in hit.index:
        return ""
    return str(hit["sector"])


def ticker_display(ticker: str) -> str:
    """Format 'TICKER — Company Name' for UI display."""
    name = get_company_name(ticker)
    return ticker if name == ticker else f"{ticker} — {name}"


def all_tickers() -> list[str]:
    """Return all tickers present in the issuers reference CSV."""
    df = _load()
    return df["ticker"].dropna().tolist() if not df.empty else []


def search_tickers(query: str, max_results: int = 20) -> list[str]:
    """Return tickers whose code or name partially matches query (case-insensitive).

    Useful for autocomplete in the dashboard search bar.
    """
    df = _load()
    if df.empty or not query:
        return []
    q = query.strip().upper()
    # Match on ticker code or company name
    mask = (
        df["ticker"].str.upper().str.contains(q, na=False)
        | df["company_name"].str.upper().str.contains(q, na=False)
    )
    return df[mask]["ticker"].head(max_results).tolist()

"""Shareholder data helpers for dashboard.

Provides two main features:

1. **Shareholder composition** (local vs foreign breakdown) per emiten.
   - get_shareholder_composition(ticker, as_of_date) → DataFrame
   - Columns: category, shares, percentage

2. **Monthly shareholder count + MoM growth %** per emiten.
   - get_monthly_shareholder_stats(ticker) → DataFrame
   - Columns: month (YYYY-MM), shareholder_count, growth_pct

Data sources (current = mock / placeholder):
    Composition   : data/shareholders/{ticker}_{YYYYMM}.csv
    Monthly count : data/shareholders/monthly_holders/{ticker}.csv

TODO: Replace mock generators with real loaders once data is available.
      Real sources: KSEI C-BEST, IDX disclosure, RTI, Stockbit API.
"""
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
_SH_DIR = _ROOT / "data" / "shareholders"
_MONTHLY_DIR = _SH_DIR / "monthly_holders"

# Categories shown in the composition table / pie chart.
# Edit here to match your real data provider's category labels.
COMPOSITION_CATEGORIES = [
    "Local Institution",
    "Local Retail",
    "Foreign Institution",
    "Foreign Retail",
]


# ---------------------------------------------------------------------------
# Shareholder composition
# ---------------------------------------------------------------------------

def get_shareholder_composition(ticker: str, as_of_date: str) -> pd.DataFrame:
    """Return shareholder composition for ticker on a given date.

    Tries to load from CSV first; falls back to deterministic mock data if the
    file does not exist.

    Args:
        ticker     : e.g. "BBCA.JK"
        as_of_date : "YYYY-MM-DD" or "YYYY-MM"

    Returns:
        DataFrame with columns: category (str), shares (int), percentage (float)
        Empty DataFrame signals a hard failure (not just missing data).
    """
    yyyymm = _to_yyyymm(as_of_date)

    # TODO: load from KSEI / IDX / RTI
    # Real file: _SH_DIR / f"{ticker}_{yyyymm}.csv"
    path = _SH_DIR / f"{ticker}_{yyyymm}.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            _validate_composition(df)
            return df[["category", "shares", "percentage"]]
        except Exception:
            pass  # fall through to mock

    # Deterministic mock (seed = ticker + month) — remove once real data arrives
    return _mock_composition(ticker, yyyymm)


def _mock_composition(ticker: str, yyyymm: str) -> pd.DataFrame:
    """Generate realistic-looking deterministic mock ownership breakdown.

    TODO: delete this function and update get_shareholder_composition() to
          return an empty DataFrame when no real data file exists.
    """
    rng = np.random.default_rng(abs(hash(ticker + yyyymm)) % (2 ** 32))
    # Simulate realistic IDX mix: foreign typically 20–60 % for large caps
    raw = rng.dirichlet([3.5, 4.0, 2.0, 0.8]) * 100
    percentages = raw.round(2)
    total_shares = int(rng.integers(500_000_000, 5_000_000_000))
    shares = (percentages / 100 * total_shares).round(0).astype(int)
    return pd.DataFrame({
        "category": COMPOSITION_CATEGORIES,
        "shares": shares,
        "percentage": percentages,
    })


def _validate_composition(df: pd.DataFrame) -> None:
    required = {"category", "shares", "percentage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Composition CSV missing columns: {missing}")


# ---------------------------------------------------------------------------
# Monthly shareholder count
# ---------------------------------------------------------------------------

def get_monthly_shareholder_stats(ticker: str) -> pd.DataFrame:
    """Return monthly shareholder count + MoM growth % for a ticker.

    Tries to load from CSV first; falls back to deterministic mock.

    Returns:
        DataFrame with columns:
            month (str, YYYY-MM),
            shareholder_count (int),
            growth_pct (float — NaN for the first row)
    """
    _MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: load from KSEI monthly report / your data provider
    # Real file: _MONTHLY_DIR / f"{ticker}.csv"
    path = _MONTHLY_DIR / f"{ticker}.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            df["month"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m")
            df["shareholder_count"] = pd.to_numeric(df["shareholder_count"], errors="coerce")
            df = df.sort_values("month").reset_index(drop=True)
        except Exception:
            df = _mock_monthly(ticker)
    else:
        # Deterministic mock — remove once real data arrives
        df = _mock_monthly(ticker)

    # Compute MoM growth regardless of source
    df["growth_pct"] = (df["shareholder_count"].pct_change() * 100).round(2)
    return df[["month", "shareholder_count", "growth_pct"]]


def _mock_monthly(ticker: str) -> pd.DataFrame:
    """Generate 18 months of synthetic shareholder counts.

    TODO: delete this function once real data is available.
    """
    rng = np.random.default_rng(abs(hash(ticker)) % (2 ** 32))
    base = int(rng.integers(5_000, 100_000))
    months = (
        pd.date_range(end=pd.Timestamp.now(), periods=18, freq="MS")
        .strftime("%Y-%m")
        .tolist()
    )
    counts = [base]
    for _ in range(17):
        delta = int(rng.integers(-800, 2_500))
        counts.append(max(1_000, counts[-1] + delta))
    return pd.DataFrame({"month": months, "shareholder_count": counts})


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_yyyymm(date_str: str) -> str:
    """Coerce 'YYYY-MM-DD' or 'YYYY-MM' to 'YYYYMM' key."""
    if not date_str:
        return pd.Timestamp.now().strftime("%Y%m")
    try:
        return pd.to_datetime(date_str).strftime("%Y%m")
    except Exception:
        return pd.Timestamp.now().strftime("%Y%m")

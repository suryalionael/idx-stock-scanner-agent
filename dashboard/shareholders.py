"""Shareholder data helpers for dashboard.

Provides two features:

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

Safety contract
---------------
Every public function in this module is guaranteed to:
- Never raise an exception to the caller.
- Return an empty (but schema-correct) DataFrame on any failure.
- Log a warning when the fallback path is taken.
"""
from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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

# Empty schemas — returned on any unrecoverable failure
_EMPTY_COMPOSITION = pd.DataFrame(columns=["category", "shares", "percentage"])
_EMPTY_MONTHLY = pd.DataFrame(columns=["month", "shareholder_count", "growth_pct"])


# ---------------------------------------------------------------------------
# Stable seed helper (avoids Python's PYTHONHASHSEED randomisation)
# ---------------------------------------------------------------------------

def _stable_seed(key: str) -> int:
    """Return a uint32 seed derived from key via MD5 — PYTHONHASHSEED-independent."""
    digest = hashlib.md5(key.encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:8], 16)  # first 32-bit hex chunk → 0 … 4_294_967_295


# ---------------------------------------------------------------------------
# Explicit month-list builder (avoids pd.date_range alignment edge cases)
# ---------------------------------------------------------------------------

def _last_n_months(n: int, reference: datetime | None = None) -> list[str]:
    """Return n consecutive YYYY-MM strings ending at reference month (inclusive).

    Uses pure-Python integer arithmetic — no pandas, no dateutil, no edge cases.
    """
    if reference is None:
        reference = datetime.now()
    year, month = reference.year, reference.month

    result: list[str] = []
    for i in range(n - 1, -1, -1):
        # Total months from year-0 offset (0-indexed), then back-compute y/m
        total = year * 12 + (month - 1) - i
        y = total // 12
        m = total % 12 + 1
        result.append(f"{y}-{m:02d}")

    # Sanity guard — should never fire, but protects against future refactors
    if len(result) != n:
        logger.error("_last_n_months: expected %d items, got %d — returning empty", n, len(result))
        return []
    return result


# ---------------------------------------------------------------------------
# Shareholder composition
# ---------------------------------------------------------------------------

def get_shareholder_composition(ticker: str, as_of_date: str) -> pd.DataFrame:
    """Return shareholder composition for ticker on a given date.

    Tries to load from CSV first; falls back to deterministic mock data if the
    file does not exist. Returns empty DataFrame on any failure.

    Args:
        ticker     : e.g. "BBCA.JK"
        as_of_date : "YYYY-MM-DD" or "YYYY-MM"

    Returns:
        DataFrame with columns: category (str), shares (int), percentage (float)
    """
    try:
        yyyymm = _to_yyyymm(as_of_date)

        # TODO: load from KSEI / IDX / RTI
        # Real file: _SH_DIR / f"{ticker}_{yyyymm}.csv"
        path = _SH_DIR / f"{ticker}_{yyyymm}.csv"
        if path.exists():
            df = pd.read_csv(path)
            _validate_composition(df)
            return df[["category", "shares", "percentage"]].reset_index(drop=True)

        # Deterministic mock (remove once real data arrives)
        return _mock_composition(ticker, yyyymm)

    except Exception as exc:
        logger.warning("get_shareholder_composition(%s): %s — returning empty", ticker, exc)
        return _EMPTY_COMPOSITION.copy()


def _mock_composition(ticker: str, yyyymm: str) -> pd.DataFrame:
    """Generate realistic-looking deterministic mock ownership breakdown.

    TODO: delete this function and return _EMPTY_COMPOSITION once real data is available.
    """
    n_cats = len(COMPOSITION_CATEGORIES)
    rng = np.random.default_rng(_stable_seed(ticker + yyyymm))

    # dirichlet always returns exactly n_cats values
    alpha = [3.5, 4.0, 2.0, 0.8][:n_cats]
    raw = rng.dirichlet(alpha) * 100
    percentages = raw.round(2).tolist()          # Python list — explicit length
    total_shares = int(rng.integers(500_000_000, 5_000_000_000))
    shares = [round(p / 100 * total_shares) for p in percentages]  # Python list

    # Guard before DataFrame construction
    if len(COMPOSITION_CATEGORIES) != len(shares) != len(percentages):
        logger.error("_mock_composition: length mismatch cats=%d shares=%d pct=%d",
                     len(COMPOSITION_CATEGORIES), len(shares), len(percentages))
        return _EMPTY_COMPOSITION.copy()

    return pd.DataFrame({
        "category":   list(COMPOSITION_CATEGORIES),  # explicit copy
        "shares":     shares,
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
        On any failure: empty schema-correct DataFrame.
    """
    try:
        _MONTHLY_DIR.mkdir(parents=True, exist_ok=True)

        # TODO: load from KSEI monthly report / your data provider
        # Real file: _MONTHLY_DIR / f"{ticker}.csv"
        path = _MONTHLY_DIR / f"{ticker}.csv"
        df: pd.DataFrame

        if path.exists():
            df = _load_monthly_csv(path)
        else:
            df = _mock_monthly(ticker)

        if df.empty:
            return _EMPTY_MONTHLY.copy()

        # Compute MoM growth regardless of source
        df = df.sort_values("month").reset_index(drop=True)
        df["growth_pct"] = (df["shareholder_count"].pct_change() * 100).round(2)
        return df[["month", "shareholder_count", "growth_pct"]]

    except Exception as exc:
        logger.warning("get_monthly_shareholder_stats(%s): %s — returning empty", ticker, exc)
        return _EMPTY_MONTHLY.copy()


def _load_monthly_csv(path: Path) -> pd.DataFrame:
    """Load and normalise a monthly shareholders CSV. Raises on bad data."""
    df = pd.read_csv(path)
    if "month" not in df.columns or "shareholder_count" not in df.columns:
        raise ValueError(f"CSV missing required columns: {path}")
    df["month"] = pd.to_datetime(df["month"]).dt.strftime("%Y-%m")
    df["shareholder_count"] = pd.to_numeric(df["shareholder_count"], errors="coerce")
    return df.dropna(subset=["shareholder_count"]).reset_index(drop=True)


def _mock_monthly(ticker: str) -> pd.DataFrame:
    """Generate 18 months of synthetic shareholder counts.

    Design principles:
    - Seed via hashlib.md5 → stable across Python restarts (no PYTHONHASHSEED issue)
    - Month list via explicit integer arithmetic → no pd.date_range edge cases
    - Explicit length assertions before DataFrame construction

    TODO: delete this function once real data is available.
    """
    N = 18
    rng = np.random.default_rng(_stable_seed(ticker))
    base = int(rng.integers(5_000, 100_000))

    # Pure-Python month generation — guaranteed exactly N items
    months: list[str] = _last_n_months(N)
    if len(months) != N:
        # _last_n_months already logged the error
        return _EMPTY_MONTHLY.copy()

    # Build counts — exactly N items: 1 seed + (N-1) loop steps
    counts: list[int] = [base]
    for _ in range(N - 1):
        delta = int(rng.integers(-800, 2_500))
        counts.append(max(1_000, counts[-1] + delta))

    # Explicit length guard before DataFrame construction
    if len(months) != len(counts):
        logger.error(
            "_mock_monthly(%s): length mismatch months=%d counts=%d",
            ticker, len(months), len(counts),
        )
        return _EMPTY_MONTHLY.copy()

    return pd.DataFrame({"month": months, "shareholder_count": counts})


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_yyyymm(date_str: str) -> str:
    """Coerce 'YYYY-MM-DD' or 'YYYY-MM' to 'YYYYMM' key."""
    if not date_str:
        return datetime.now().strftime("%Y%m")
    try:
        return pd.to_datetime(date_str).strftime("%Y%m")
    except Exception:
        return datetime.now().strftime("%Y%m")

"""Fundamental data fetcher for IDX tickers.

Architecture
------------
Providers (fetch raw fundamental dict)
    BaseFundamentalProvider  ← abstract
    YFinanceFundamentalProvider  — yfinance Ticker.info (free, no API key)

FundamentalAggregator
    Tries providers in order; returns status.

Output columns added to feature DataFrame
    eps                float — trailing EPS (in IDR)
    pe_ratio           float — trailing P/E
    pbv                float — price-to-book value
    roe_pct            float — return on equity in % (e.g. 22.97)
    der                float — debt-to-equity RATIO (×), e.g. 1.5 means 150%
                                ⚠️  yfinance debtToEquity is in % — we divide by 100.
    div_yield_pct      float — dividend yield in % (e.g. 5.44)
    market_cap         float — market cap in IDR
    revenue_growth_pct float — YoY revenue growth in %
    profit_growth_pct  float — YoY earnings growth in %
    ebitda             float — EBITDA in IDR (positive = profitable at operating level)
    public_float_pct   float — public float as % of shares outstanding
    fundamental_status  str — "ok" | "partial" | "missing"

DER scale note
    yfinance `debtToEquity` returns values in PERCENT (e.g. 15.234 for ACES).
    We store DER as a ratio (÷100), so ACES → der = 0.15.
    Threshold "DER ≤ 4.0" in scanner_config.yaml is in RATIO units.

Status semantics
    "ok"      : ≥3 key metrics available (PE/PBV/ROE at minimum)
    "partial"  : 1–2 metrics available
    "missing"  : fetch failed or truly no data available
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class FundamentalProviderError(Exception):
    """Raised when a provider fails to fetch fundamental data."""


# ---------------------------------------------------------------------------
# Provider base
# ---------------------------------------------------------------------------

class BaseFundamentalProvider(ABC):
    """Interface for a fundamental data source."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs."""

    @abstractmethod
    def fetch(self, ticker: str) -> dict:
        """Fetch fundamental data for ticker.

        Returns:
            Dict with any subset of the standard keys listed at module top.
            May return partial dict (only available fields).

        Raises:
            FundamentalProviderError: on network failure, malformed response.
        """


# ---------------------------------------------------------------------------
# yfinance provider
# ---------------------------------------------------------------------------

# Fields to extract from yfinance .info with their output column names
_YFINANCE_FIELD_MAP = {
    "trailingEps":      "eps",
    "trailingPE":       "pe_ratio",
    "priceToBook":      "pbv",
    "returnOnEquity":   "roe_raw",              # needs × 100
    "debtToEquity":     "der_raw",              # yfinance is in %; needs ÷ 100 → ratio
    "dividendYield":    "div_yield_raw",        # needs normalization
    "marketCap":        "market_cap",
    "revenueGrowth":    "revenue_growth_raw",   # needs × 100
    "earningsGrowth":   "profit_growth_raw",    # needs × 100
    "ebitda":           "ebitda",               # absolute IDR amount (new)
    "floatShares":      "_float_shares",        # for public_float_pct (new)
    "sharesOutstanding": "_shares_outstanding", # for public_float_pct (new)
}

# Key metrics used to determine status
_KEY_METRICS = ("pe_ratio", "pbv", "roe_pct", "eps", "der", "div_yield_pct")


class YFinanceFundamentalProvider(BaseFundamentalProvider):
    """Fetches fundamental data via yfinance Ticker.info."""

    @property
    def name(self) -> str:
        return "yfinance"

    def fetch(self, ticker: str) -> dict:
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
        except Exception as exc:
            raise FundamentalProviderError(f"yfinance import/call failed: {exc}") from exc

        if not info or not isinstance(info, dict):
            raise FundamentalProviderError("yfinance.info returned empty or non-dict")

        # Guard: yfinance sometimes returns a minimal stub dict with just 'symbol'
        # when it can't find the ticker
        if len(info) < 5:
            raise FundamentalProviderError(
                f"yfinance.info returned stub dict with {len(info)} keys only"
            )

        result: dict = {}

        for yf_key, out_key in _YFINANCE_FIELD_MAP.items():
            val = info.get(yf_key)
            if val is not None:
                try:
                    result[out_key] = float(val)
                except (TypeError, ValueError):
                    pass

        # ── Normalize percentages ──────────────────────────────────────────

        # roe: stored as decimal fraction (0.229 = 22.9%)
        if "roe_raw" in result:
            result["roe_pct"] = round(result.pop("roe_raw") * 100, 2)

        # revenue/profit growth: also decimal fraction
        if "revenue_growth_raw" in result:
            result["revenue_growth_pct"] = round(result.pop("revenue_growth_raw") * 100, 2)
        if "profit_growth_raw" in result:
            result["profit_growth_pct"] = round(result.pop("profit_growth_raw") * 100, 2)

        # dividendYield: yfinance is inconsistent — sometimes decimal (0.054),
        # sometimes already percentage (5.44). Heuristic: if > 1.0, assume %.
        if "div_yield_raw" in result:
            dy = result.pop("div_yield_raw")
            result["div_yield_pct"] = round(dy if dy > 1.0 else dy * 100, 2)

        # ── DER: yfinance debtToEquity is in % → convert to ratio ────────
        # Example: yfinance returns 15.234 for ACES, which is 15.234% = 0.15×
        # We store as ratio so threshold config (e.g. der_max: 4.0) is intuitive.
        if "der_raw" in result:
            result["der"] = round(result.pop("der_raw") / 100.0, 4)

        # ── Public float % ───────────────────────────────────────────────
        float_shares = result.pop("_float_shares", None)
        total_shares = result.pop("_shares_outstanding", None)
        if float_shares and total_shares and total_shares > 0:
            result["public_float_pct"] = round(float_shares / total_shares * 100, 2)

        return result


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class FundamentalAggregator:
    """Run providers in order; stop at first success."""

    def __init__(self, providers: list[BaseFundamentalProvider] | None = None) -> None:
        if providers is None:
            providers = [YFinanceFundamentalProvider()]
        self.providers = providers

    def fetch_with_status(self, ticker: str) -> tuple[dict, str, str]:
        """
        Returns:
            (data_dict, fundamental_status, error_message)
        """
        errors: list[str] = []

        for provider in self.providers:
            try:
                data = provider.fetch(ticker)
                status = _determine_status(data)
                logger.debug(
                    "%s: fundamental via %s → status=%s (%d fields)",
                    ticker, provider.name, status, len(data),
                )
                return data, status, ""
            except FundamentalProviderError as exc:
                msg = f"{provider.name}: {exc}"
                logger.debug("%s: fundamental provider failed — %s", ticker, msg)
                errors.append(msg)

        combined = " | ".join(errors)
        logger.warning("%s: all fundamental providers failed — %s", ticker, combined)
        return {}, "missing", combined


def _determine_status(data: dict) -> str:
    """Determine status from available key metrics."""
    available = sum(1 for k in _KEY_METRICS if k in data and data[k] is not None)
    if available >= 3:
        return "ok"
    if available >= 1:
        return "partial"
    return "missing"


# ---------------------------------------------------------------------------
# Public API: per-ticker
# ---------------------------------------------------------------------------

_DEFAULT_AGGREGATOR = FundamentalAggregator()

# All columns that enrich_with_fundamentals adds to the DataFrame
# DER is stored as RATIO (×), not percent. ebitda is absolute IDR.
FUNDAMENTAL_COLS = [
    "eps", "pe_ratio", "pbv", "roe_pct", "der",
    "div_yield_pct", "market_cap",
    "revenue_growth_pct", "profit_growth_pct",
    "ebitda",               # new: EBITDA in IDR (positive = profitable operating level)
    "public_float_pct",     # new: float shares / total shares × 100
    "fundamental_status",
]

_EMPTY_ROW = {col: float("nan") for col in FUNDAMENTAL_COLS if col != "fundamental_status"}
_EMPTY_ROW["fundamental_status"] = "missing"


def get_fundamental_snapshot(
    ticker: str,
    as_of_date: str | None = None,
    fundamentals_dir: Path | None = None,
    aggregator: FundamentalAggregator | None = None,
) -> dict:
    """Return fundamental snapshot for one ticker.

    Args:
        ticker          : e.g. "BBCA.JK"
        as_of_date      : "YYYY-MM-DD" — used to look up cached parquet first
        fundamentals_dir: directory where {scan_date}.parquet files are saved
        aggregator      : override default aggregator (useful for testing)

    Returns:
        Dict with keys from FUNDAMENTAL_COLS (always present; NaN if missing).
    """
    # ── Try loading from cached parquet first ──────────────────────────────
    if fundamentals_dir and as_of_date:
        path = Path(fundamentals_dir) / f"{as_of_date}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                row = df[df["ticker"] == ticker]
                if not row.empty:
                    return row.iloc[0].to_dict()
            except Exception as exc:
                logger.debug("%s: could not load cached fundamental — %s", ticker, exc)

    # ── Live fetch ─────────────────────────────────────────────────────────
    agg = aggregator or _DEFAULT_AGGREGATOR
    data, status, _ = agg.fetch_with_status(ticker)
    result = {**_EMPTY_ROW.copy(), **data, "fundamental_status": status}
    return result


# ---------------------------------------------------------------------------
# Batch enrichment
# ---------------------------------------------------------------------------

def enrich_with_fundamentals(
    df_features: pd.DataFrame,
    fundamentals_dir: Path | None = None,
    save: bool = True,
    scan_date: str | None = None,
    aggregator: FundamentalAggregator | None = None,
    delay_between_tickers: float = 0.3,
) -> pd.DataFrame:
    """Add fundamental columns to the feature DataFrame.

    NaN policy:
        All numeric fundamental columns are NaN when status="missing".
        Never fill NaN with fake zeros.

    Args:
        df_features         : must have "ticker" column
        fundamentals_dir    : where to save/load parquet cache
        save                : whether to persist results
        scan_date           : "YYYY-MM-DD" for file naming
        aggregator          : override default (useful for testing)
        delay_between_tickers: seconds to sleep between yfinance calls (rate limit)
    """
    df = df_features.copy()
    results: list[dict] = []
    t0 = time.monotonic()
    status_counts = {"ok": 0, "partial": 0, "missing": 0}

    agg = aggregator or _DEFAULT_AGGREGATOR

    for ticker in df["ticker"].unique():
        try:
            data, status, error = agg.fetch_with_status(ticker)
            row = {**_EMPTY_ROW.copy(), **data, "fundamental_status": status, "ticker": ticker}
            results.append(row)
            status_counts[status] = status_counts.get(status, 0) + 1

            if status == "missing" and error:
                logger.debug("%s: fundamental missing — %s", ticker, error)

        except Exception as exc:
            logger.error("%s: unexpected error in fundamental fetch — %s", ticker, exc)
            results.append({**_EMPTY_ROW.copy(), "fundamental_status": "missing", "ticker": ticker})
            status_counts["missing"] = status_counts.get("missing", 0) + 1

        # Gentle rate limiting for yfinance
        if delay_between_tickers > 0:
            time.sleep(delay_between_tickers)

    fund_df = pd.DataFrame(results)
    df = df.merge(fund_df[["ticker"] + FUNDAMENTAL_COLS], on="ticker", how="left")

    # Status default for any ticker not in results
    df["fundamental_status"] = df["fundamental_status"].fillna("missing")

    elapsed = time.monotonic() - t0
    total = len(df["ticker"].unique())
    logger.info(
        "Fundamental enrichment done in %.1fs — ok=%d partial=%d missing=%d / %d tickers",
        elapsed,
        status_counts.get("ok", 0),
        status_counts.get("partial", 0),
        status_counts.get("missing", 0),
        total,
    )

    if save and fundamentals_dir and scan_date:
        Path(fundamentals_dir).mkdir(parents=True, exist_ok=True)
        path = Path(fundamentals_dir) / f"{scan_date}.parquet"
        fund_df.to_parquet(path, index=False)
        logger.info("Fundamental data saved → %s", path)

    return df

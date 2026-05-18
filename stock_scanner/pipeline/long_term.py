"""Long-term investment scorer for IDX stocks.

Assesses business quality, financial fundamentals, valuation,
and structural characteristics for buy-and-hold suitability.

Three-layer analysis:
  A. Fundamental quality   — ROE, DER, margin, growth trend
  B. Valuation estimate    — PE/PBV vs sector, Graham Number, intrinsic value
  C. Cyclicality           — defensive vs cyclical sector classification

Data sources:
  - Existing fundamental columns from signals parquet (fast, no I/O)
  - yfinance `.financials` / `.balance_sheet` / `.cashflow` for multi-period
    comparison (lazy, per-ticker, cached at call site with @st.cache_data)

Public API
----------
compute_long_term_score(row) -> dict
    Score a single ticker row from existing columns. Fast, no I/O.

enrich_df_with_long_term(df) -> pd.DataFrame
    Batch-enrich a signals DataFrame. Adds long_term_score, label, reason.

compare_financial_statements(ticker) -> dict
    Fetch multi-period P&L, balance sheet, cash flow from yfinance.
    Slow (~1–3 s) — call lazily per selected ticker, not in bulk.

compute_graham_number(eps, bvps) -> float | None
    Classic Graham Number: sqrt(22.5 × EPS × BVPS).

classify_cyclicality(sector) -> str
    "defensive" | "cyclical" | "mixed" | "unknown"
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Sector cyclicality lookup
# ---------------------------------------------------------------------------

_DEFENSIVE_SECTORS = {
    "Consumer Staples", "Healthcare", "Utilities",
    "Telecommunication Services", "Financials",
}
_CYCLICAL_SECTORS = {
    "Energy", "Materials", "Industrials",
    "Consumer Discretionary", "Real Estate",
}
_MIXED_SECTORS = {
    "Information Technology", "Communication Services",
}


def classify_cyclicality(sector: str) -> str:
    """Return cyclicality classification for a sector string.

    Returns:
        "defensive" | "cyclical" | "mixed" | "unknown"
    """
    if not sector:
        return "unknown"
    s = str(sector).strip()
    if s in _DEFENSIVE_SECTORS:
        return "defensive"
    if s in _CYCLICAL_SECTORS:
        return "cyclical"
    if s in _MIXED_SECTORS:
        return "mixed"
    # Fuzzy match
    sl = s.lower()
    if any(d.lower() in sl for d in ["staple", "health", "util", "teleco", "bank", "financ"]):
        return "defensive"
    if any(c.lower() in sl for c in ["energy", "material", "mining", "coal", "palm", "plantation",
                                      "industrial", "cement", "steel", "property", "real estate"]):
        return "cyclical"
    return "unknown"


# ---------------------------------------------------------------------------
# Valuation helpers
# ---------------------------------------------------------------------------

def compute_graham_number(eps: float | None, bvps: float | None) -> float | None:
    """Classic Graham Number: sqrt(22.5 × EPS × BVPS).

    Valid only when EPS > 0 AND BVPS > 0.
    Returns None if inputs are missing or non-positive.

    Args:
        eps : Trailing twelve-month EPS (Rupiah).
        bvps: Book value per share (Rupiah). Approximated from PBV if needed:
              bvps ≈ close / pbv.
    """
    if eps is None or bvps is None:
        return None
    try:
        eps_f, bvps_f = float(eps), float(bvps)
    except (TypeError, ValueError):
        return None
    if eps_f <= 0 or bvps_f <= 0:
        return None
    return math.sqrt(22.5 * eps_f * bvps_f)


def compute_intrinsic_value(row: dict | pd.Series) -> dict:
    """Estimate intrinsic value and margin of safety.

    Primary method: Graham Number (works for profitable, asset-backed companies).
    Fallback: PE-based (EPS × fair PE for sector).

    Returns:
        {
            "intrinsic_value":   float | None,
            "margin_of_safety":  float | None,  # pct, negative = overvalued
            "valuation_status":  "UNDERVALUED" | "FAIR" | "OVERVALUED" | "UNKNOWN",
            "valuation_method":  str,
        }
    """
    close  = _f(row.get("close"))
    eps    = _f(row.get("eps"))
    pe     = _f(row.get("pe_ratio"))
    pbv    = _f(row.get("pbv"))

    iv = None
    method = "none"

    # Method 1: Graham Number
    bvps = None
    if close and pbv and pbv > 0:
        bvps = close / pbv  # back-calculate BVPS from PBV

    graham = compute_graham_number(eps, bvps)
    if graham and graham > 0:
        iv = graham
        method = "graham_number"

    # Method 2: Fallback — PE-based with a conservative fair PE
    if iv is None and eps and eps > 0:
        fair_pe = 15.0  # conservative average for IDX mid/large cap
        iv = eps * fair_pe
        method = "pe_based_fair15x"

    # Margin of safety
    mos = None
    if iv and close and close > 0:
        mos = (iv - close) / close * 100  # positive = undervalued

    # Status
    if mos is None:
        status = "UNKNOWN"
    elif mos >= 20:
        status = "UNDERVALUED"
    elif mos >= -10:
        status = "FAIR"
    else:
        status = "OVERVALUED"

    return {
        "intrinsic_value":  round(iv, 0) if iv else None,
        "margin_of_safety": round(mos, 1) if mos is not None else None,
        "valuation_status": status,
        "valuation_method": method,
    }


# ---------------------------------------------------------------------------
# Long-term scoring
# ---------------------------------------------------------------------------

def compute_long_term_score(row: dict | pd.Series, sector: str = "") -> dict:
    """Score a ticker for long-term investment suitability.

    Scoring rubric (max 10 pts):
    ┌────────────────────────────────┬────────────────────────────────────┐
    │ Factor                         │ Max pts                            │
    ├────────────────────────────────┼────────────────────────────────────┤
    │ ROE (profitability)            │ 2.5                                │
    │ DER (financial leverage)       │ 2.0                                │
    │ Revenue growth (top-line)      │ 1.5                                │
    │ Net profit growth              │ 1.5                                │
    │ Valuation (PE/PBV + Graham)    │ 2.0                                │
    │ Dividend yield                 │ 0.5                                │
    └────────────────────────────────┴────────────────────────────────────┘

    Args:
        row    : Dict or pd.Series with fundamental columns.
        sector : Sector string (used for cyclicality classification and
                 sector-aware thresholds). Leave empty to skip.

    Returns:
        {
            "long_term_score":   float (0–10),
            "long_term_label":   "LONG_TERM_CORE" | "LONG_TERM_WATCHLIST" | "NOT_LONG_TERM",
            "valuation_status":  "UNDERVALUED" | "FAIR" | "OVERVALUED" | "UNKNOWN",
            "intrinsic_value":   float | None,
            "margin_of_safety":  float | None,
            "valuation_method":  str,
            "cyclicality":       "defensive" | "cyclical" | "mixed" | "unknown",
            "long_term_reason":  str,
            "strengths":         list[str],
            "red_flags":         list[str],
        }
    """
    score = 0.0
    strengths:  list[str] = []
    red_flags:  list[str] = []

    roe      = _f(row.get("roe_pct"))
    der      = _f(row.get("der"))
    pe       = _f(row.get("pe_ratio"))
    pbv      = _f(row.get("pbv"))
    rev_g    = _f(row.get("revenue_growth_pct"))
    prof_g   = _f(row.get("profit_growth_pct"))
    div_y    = _f(row.get("div_yield_pct"))
    fund_st  = str(row.get("fundamental_status", "")).lower()
    cyclical = classify_cyclicality(sector)

    # --- Adjust thresholds by cyclicality ---
    if cyclical == "cyclical":
        roe_great, roe_good  = 20.0, 12.0  # lower bar for cyclicals
        der_safe,  der_warn  = 2.0, 4.0    # cyclicals naturally have more debt
        pe_cheap,  pe_fair   = 8.0, 15.0   # cyclicals trade cheaper
    elif cyclical == "defensive":
        roe_great, roe_good  = 20.0, 15.0
        der_safe,  der_warn  = 1.0, 2.5
        pe_cheap,  pe_fair   = 12.0, 22.0  # defensive premiums are acceptable
    else:  # mixed / unknown
        roe_great, roe_good  = 20.0, 15.0
        der_safe,  der_warn  = 1.5, 3.0
        pe_cheap,  pe_fair   = 10.0, 20.0

    # ── A. ROE (up to 2.5 pts) ────────────────────────────────────────────
    if roe is not None:
        if roe >= roe_great:
            score += 2.5
            strengths.append(f"ROE sangat kuat {roe:.1f}%")
        elif roe >= roe_good:
            score += 1.5
            strengths.append(f"ROE baik {roe:.1f}%")
        elif roe >= 8:
            score += 0.5
        elif roe < 5:
            red_flags.append(f"ROE lemah {roe:.1f}%")
            score -= 0.5
        if roe < 0:
            red_flags.append(f"ROE negatif — sedang merugi")
            score -= 1.0
    else:
        red_flags.append("Data ROE tidak tersedia")

    # ── B. DER / Leverage (up to 2.0 pts) ────────────────────────────────
    if der is not None:
        if der <= der_safe:
            score += 2.0
            strengths.append(f"Leverage aman (DER {der:.2f}×)")
        elif der <= der_warn:
            score += 1.0
        elif der <= 5.0:
            score += 0.0
            red_flags.append(f"DER tinggi {der:.2f}× — perhatikan beban utang")
        else:
            score -= 0.5
            red_flags.append(f"DER sangat tinggi {der:.2f}×")
    else:
        # For banks/financials DER is expected to be high — don't penalize
        if cyclical == "defensive" and "financ" in sector.lower():
            pass  # skip DER check for banks
        else:
            red_flags.append("Data DER tidak tersedia")

    # ── C. Revenue growth (up to 1.5 pts) ─────────────────────────────────
    if rev_g is not None:
        if rev_g >= 15:
            score += 1.5
            strengths.append(f"Revenue tumbuh kuat +{rev_g:.1f}%")
        elif rev_g >= 5:
            score += 1.0
            strengths.append(f"Revenue tumbuh +{rev_g:.1f}%")
        elif rev_g >= 0:
            score += 0.4
        else:
            red_flags.append(f"Revenue turun {rev_g:.1f}%")
            score -= 0.3
    else:
        red_flags.append("Data pertumbuhan revenue tidak tersedia")

    # ── D. Net profit growth (up to 1.5 pts) ──────────────────────────────
    if prof_g is not None:
        if prof_g >= 20:
            score += 1.5
            strengths.append(f"Laba tumbuh kuat +{prof_g:.1f}%")
        elif prof_g >= 8:
            score += 1.0
            strengths.append(f"Laba tumbuh +{prof_g:.1f}%")
        elif prof_g >= 0:
            score += 0.3
        else:
            red_flags.append(f"Laba turun {prof_g:.1f}%")
            score -= 0.3
    else:
        red_flags.append("Data pertumbuhan laba tidak tersedia")

    # ── E. Valuation (up to 2.0 pts) ──────────────────────────────────────
    val = compute_intrinsic_value(row)
    mos = val["margin_of_safety"]
    v_status = val["valuation_status"]

    if v_status == "UNDERVALUED":
        score += 2.0
        strengths.append(f"Undervalued — MoS {mos:+.0f}%")
    elif v_status == "FAIR":
        score += 1.0
    elif v_status == "OVERVALUED":
        score += 0.0
        if mos is not None and mos < -30:
            red_flags.append(f"Valuasi mahal — {abs(mos):.0f}% premium vs intrinsik")

    # Sector-aware PE check
    if pe is not None:
        if pe < pe_cheap:
            score += 0.3
            if "Valuasi" not in " ".join(strengths):
                strengths.append(f"PE murah ({pe:.1f}×)")
        elif pe > pe_fair * 2:
            red_flags.append(f"PE sangat mahal ({pe:.1f}×)")

    # ── F. Dividend yield (up to 0.5 pts) ─────────────────────────────────
    if div_y is not None and div_y >= 3:
        score += 0.5
        strengths.append(f"Dividen {div_y:.2f}% — pendapatan pasif")

    # ── Cyclicality warning ────────────────────────────────────────────────
    if cyclical == "cyclical":
        red_flags.append(
            "Saham siklikal — cocok hold saat siklus positif, bukan buy-and-forget"
        )

    # ── Final score ────────────────────────────────────────────────────────
    score = round(max(0.0, min(10.0, score)), 2)

    if score >= 6.5:
        label = "LONG_TERM_CORE"
    elif score >= 4.0:
        label = "LONG_TERM_WATCHLIST"
    else:
        label = "NOT_LONG_TERM"

    # Build reason string
    top_strength = strengths[0] if strengths else "—"
    top_flag     = red_flags[0]  if red_flags  else "—"
    reason = f"✅ {top_strength}" if strengths else ""
    if red_flags:
        reason += ("; " if reason else "") + f"⚠️ {top_flag}"

    return {
        "long_term_score":   score,
        "long_term_label":   label,
        "valuation_status":  v_status,
        "intrinsic_value":   val["intrinsic_value"],
        "margin_of_safety":  val["margin_of_safety"],
        "valuation_method":  val["valuation_method"],
        "cyclicality":       cyclical,
        "long_term_reason":  reason,
        "strengths":         strengths,
        "red_flags":         red_flags,
    }


def enrich_df_with_long_term(
    df: pd.DataFrame,
    sector_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Batch-enrich a signals DataFrame with long-term scores.

    Adds columns:
        long_term_score, long_term_label, valuation_status,
        intrinsic_value, margin_of_safety, valuation_method,
        cyclicality, long_term_reason

    Args:
        df         : Signals DataFrame.
        sector_map : {ticker: sector_str} mapping (from issuers.csv). If None,
                     sector-aware thresholds are skipped.

    Returns:
        DataFrame with long-term scoring columns.
    """
    if df.empty:
        for col in ["long_term_score", "long_term_label", "valuation_status",
                    "intrinsic_value", "margin_of_safety", "cyclicality", "long_term_reason"]:
            df[col] = None
        return df

    sector_map = sector_map or {}

    def _score_row(r: pd.Series) -> pd.Series:
        ticker = str(r.get("ticker", ""))
        sector = sector_map.get(ticker, sector_map.get(ticker.replace(".JK", ""), ""))
        result = compute_long_term_score(r, sector=sector)
        # Flatten list fields to strings for storage
        result["strengths"]  = " | ".join(result["strengths"][:3])
        result["red_flags"]  = " | ".join(result["red_flags"][:3])
        return pd.Series(result)

    lt_rows = df.apply(_score_row, axis=1)
    drop = [c for c in lt_rows.columns if c in df.columns]
    df   = df.drop(columns=drop)
    return pd.concat([df, lt_rows], axis=1)


# ---------------------------------------------------------------------------
# Multi-period financial statement comparison (lazy, per-ticker)
# ---------------------------------------------------------------------------

def compare_financial_statements(ticker: str) -> dict:
    """Fetch and compare multi-period financial data from yfinance.

    Retrieves annual income statement, balance sheet, and cash flow for
    the last 2–4 periods.  For quarterly, compares the latest quarter
    vs the same quarter last year.

    Args:
        ticker: IDX ticker string (with or without .JK suffix).

    Returns:
        {
            "status":   "ok" | "partial" | "failed",
            "error":    str | None,
            "annual":   {
                "revenue":   [{"year": "2024", "value": float}, ...],
                "net_income":[...],
                "gross_profit": [...],
                "total_assets": [...],
                "total_equity": [...],
                "total_debt":   [...],
                "op_cash_flow": [...],
                "free_cash_flow": [...],
                "net_margin": [...],
            },
            "yoy": {   # Year-over-year change % for most recent vs prior year
                "revenue_chg":    float | None,
                "net_income_chg": float | None,
                "asset_chg":      float | None,
                "equity_chg":     float | None,
            },
        }

    ⚠️ This function makes network calls to yfinance (~1–3 seconds).
    Cache the result in the calling layer (e.g. @st.cache_data in Streamlit).
    """
    import yfinance as yf

    clean = ticker if ticker.endswith(".JK") else f"{ticker}.JK"
    result: dict = {
        "status": "failed", "error": None,
        "annual": {}, "yoy": {},
    }

    try:
        yt = yf.Ticker(clean)
        fin = yt.financials          # Annual income statement
        bs  = yt.balance_sheet       # Annual balance sheet
        cf  = yt.cashflow            # Annual cash flow

        if fin is None or fin.empty:
            result["error"] = "No financial data returned by yfinance"
            return result

        def _extract(df: pd.DataFrame, label: str, default_labels: list[str]) -> list[dict]:
            """Try to find a row by name (case-insensitive partial match)."""
            if df is None or df.empty:
                return []
            df_t = df if df.index.dtype == object else df.T
            for candidate in default_labels:
                matches = [i for i in df_t.index if candidate.lower() in str(i).lower()]
                if matches:
                    row = df_t.loc[matches[0]]
                    out = []
                    for col in sorted(row.index, reverse=True)[:4]:  # last 4 years
                        try:
                            val = float(row[col]) / 1e9  # convert to billions IDR
                            year = str(col)[:4] if hasattr(col, "__str__") else str(col)
                            out.append({"year": year, "value": round(val, 2)})
                        except (TypeError, ValueError):
                            pass
                    return out
            return []

        annual = {
            "revenue":      _extract(fin, "revenue", ["Total Revenue", "Revenue"]),
            "net_income":   _extract(fin, "net_income", ["Net Income", "Net Income From Continuing Operation"]),
            "gross_profit": _extract(fin, "gross_profit", ["Gross Profit"]),
            "total_assets": _extract(bs, "total_assets", ["Total Assets"]),
            "total_equity": _extract(bs, "total_equity", ["Total Equity", "Stockholders Equity",
                                                            "Common Stock Equity"]),
            "total_debt":   _extract(bs, "total_debt", ["Total Debt", "Long Term Debt",
                                                          "Net Debt"]),
            "op_cash_flow": _extract(cf, "op_cash_flow", ["Operating Cash Flow",
                                                            "Cash Flow From Continuing Operating"]),
            "free_cash_flow": _extract(cf, "free_cash_flow", ["Free Cash Flow"]),
        }

        # Net margin = net_income / revenue
        rev_list = annual.get("revenue", [])
        ni_list  = annual.get("net_income", [])
        margins  = []
        for rev_item in rev_list:
            yr = rev_item["year"]
            ni_item = next((n for n in ni_list if n["year"] == yr), None)
            if ni_item and rev_item["value"] and rev_item["value"] != 0:
                margin = ni_item["value"] / rev_item["value"] * 100
                margins.append({"year": yr, "value": round(margin, 2)})
        annual["net_margin"] = margins

        result["annual"] = annual

        # YoY changes (most recent vs prior year)
        def _yoy(series: list[dict]) -> float | None:
            if len(series) >= 2:
                a, b = series[0]["value"], series[1]["value"]
                if b and b != 0:
                    return round((a - b) / abs(b) * 100, 1)
            return None

        result["yoy"] = {
            "revenue_chg":    _yoy(annual.get("revenue", [])),
            "net_income_chg": _yoy(annual.get("net_income", [])),
            "asset_chg":      _yoy(annual.get("total_assets", [])),
            "equity_chg":     _yoy(annual.get("total_equity", [])),
            "ocf_chg":        _yoy(annual.get("op_cash_flow", [])),
        }

        result["status"] = "ok" if any(v for v in annual.values()) else "partial"

    except Exception as exc:
        logger.warning("compare_financial_statements({}) failed: {}", ticker, exc)
        result["error"] = str(exc)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _f(val: Any) -> float | None:
    """Safe float; returns None for NaN/None/non-numeric."""
    if val is None:
        return None
    try:
        v = float(val)
        return None if v != v else v
    except (TypeError, ValueError):
        return None

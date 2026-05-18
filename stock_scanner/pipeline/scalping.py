"""Scalping candidate scorer for IDX stocks.

Identifies tickers with strong intraday/daily momentum that could become
top gainers or strong movers on the day.

⚠️ Limitation: No real-time / intraday OHLC data is available.
All signals are derived from daily OHLCV + pre-computed technical indicators
in the signals DataFrame.  "Scalping" here means:
    - strong daily momentum setup,
    - high volume participation,
    - price near key breakout levels.

For true tick-by-tick scalping you would need Level-2 order book data
(not implemented).

Public API
----------
compute_scalping_score(row) -> dict
    Score a single row (dict or pd.Series). Fast — all inputs from existing
    feature columns, no I/O.

enrich_df_with_scalping(df) -> pd.DataFrame
    Apply compute_scalping_score to every row in a signals DataFrame.
    Adds: scalping_score, scalping_label, scalping_reason.
"""
from __future__ import annotations

from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Scoring weights & thresholds
# ---------------------------------------------------------------------------

_LABEL_HIGH  = "SCALPING_HIGH"
_LABEL_WATCH = "SCALPING_WATCH"
_LABEL_NO    = "NOT_SCALPING"

_SCORE_HIGH  = 6.5   # threshold for SCALPING_HIGH
_SCORE_WATCH = 4.0   # threshold for SCALPING_WATCH


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_scalping_score(row: dict | pd.Series) -> dict:
    """Score a ticker row for scalping / momentum-day-trading potential.

    Scoring rubric (max 10 pts):
    ┌─────────────────────────────┬──────────────────────────────────┐
    │ Factor                      │ Max pts                          │
    ├─────────────────────────────┼──────────────────────────────────┤
    │ Volume spike / vol ratio    │ 2.5  (strongest intraday signal) │
    │ Short-term price momentum   │ 2.5  (ROC5 / roc20 direction)    │
    │ ATR breakout                │ 1.0  (range expansion)           │
    │ Momentum score (composite)  │ 1.5  (signal_engine component)   │
    │ RSI sweet-spot (50–72)      │ 1.0  (trending, not overbought)  │
    │ Near 52-week high           │ 1.0  (price already strong)      │
    │ News catalyst               │ 0.5  (positive catalyst present) │
    │ Squeeze release             │ 0.5  (Bollinger squeeze off)     │
    └─────────────────────────────┴──────────────────────────────────┘

    Args:
        row: Dict or pd.Series with signal_engine output columns.

    Returns:
        {
            "scalping_score":  float (0–10),
            "scalping_label":  str ("SCALPING_HIGH" | "SCALPING_WATCH" | "NOT_SCALPING"),
            "scalping_reason": str (comma-separated key reasons),
        }
    """
    score = 0.0
    reasons: list[str] = []

    vol_ratio  = _f(row.get("vol_ratio_20d")) or 0.0   # often 0 for IDX (yfinance gap)
    vol_score  = _f(row.get("volume_score"))    or 0.0  # signal_engine composite (reliable)
    vol_spike  = _bool(row.get("vol_spike"))
    hist_vol   = _f(row.get("hist_vol_20d"))    or 0.0  # historical 20d vol %
    roc5       = _f(row.get("roc5")) or 0.0
    roc20      = _f(row.get("roc20")) or 0.0
    atr_brk    = _bool(row.get("atr_breakout"))
    mom_score  = _f(row.get("momentum_score"))
    rsi        = _f(row.get("rsi14"))
    pct52      = _f(row.get("pct_from_52w_high"))
    adx        = _f(row.get("adx"))
    news_s     = _f(row.get("news_score"))
    news_st    = str(row.get("news_data_status", "")).lower()
    squeeze    = _bool(row.get("squeeze_on"))
    supertrend = _bool(row.get("supertrend_bullish"))
    close      = _f(row.get("close")) or 0.0

    # ── 1. Volume / liquidity participation (up to 2.5 pts) ──────────────
    # Primary: use signal_engine's volume_score (reliable even when vol_ratio=0)
    # Fallback: vol_ratio if available, hist_vol as volatility proxy
    if vol_spike or (vol_ratio > 0 and vol_ratio >= 4):
        score += 2.5
        reasons.append(f"Volume spike {vol_ratio:.1f}×")
    elif vol_score >= 7:
        score += 2.0
        reasons.append(f"Volume score tinggi ({vol_score:.0f}/10)")
    elif vol_score >= 5:
        score += 1.2
        reasons.append(f"Volume score baik ({vol_score:.0f}/10)")
    elif vol_score >= 3:
        score += 0.5
    # Bonus if hist_vol is high (volatile day = scalping opportunity)
    if hist_vol >= 50:
        score += 0.3
        reasons.append(f"Volatilitas tinggi ({hist_vol:.0f}%)")
    elif hist_vol >= 30:
        score += 0.1

    # ── 2. Price momentum — ROC5 (up to 2.5 pts) ──────────────────────────
    if roc5 >= 8:
        score += 2.5
        reasons.append(f"Momentum kuat +{roc5:.1f}% (5d)")
    elif roc5 >= 5:
        score += 2.0
        reasons.append(f"Momentum +{roc5:.1f}% (5d)")
    elif roc5 >= 3:
        score += 1.5
        reasons.append(f"Naik +{roc5:.1f}% (5d)")
    elif roc5 >= 1:
        score += 0.8
        reasons.append(f"+{roc5:.1f}% (5d)")
    elif roc5 < -3:
        score -= 0.5  # slight penalty for downward momentum

    # Bonus: short-term & medium-term aligned upward
    if roc5 > 0 and roc20 and roc20 > 0:
        score += 0.3

    # ── 3. ATR Breakout (1.0 pts) ─────────────────────────────────────────
    if atr_brk:
        score += 1.0
        reasons.append("ATR breakout")

    # ── 4. Momentum score composite (up to 1.5 pts) ───────────────────────
    if mom_score is not None:
        if mom_score >= 8:
            score += 1.5
        elif mom_score >= 6.5:
            score += 1.0
        elif mom_score >= 5:
            score += 0.4

    # ── 5. RSI sweet-spot (up to 1.0 pts) ────────────────────────────────
    # 52–72: uptrend momentum without hard overbought
    if rsi is not None:
        if 52 <= rsi <= 68:
            score += 1.0
            reasons.append(f"RSI ideal ({rsi:.0f})")
        elif 45 <= rsi < 52 or 68 < rsi <= 75:
            score += 0.5
        elif rsi > 80:
            score -= 0.5  # overbought penalty
        elif rsi < 35:
            score -= 0.3

    # ── 6. Near 52-week high (up to 1.0 pts) ──────────────────────────────
    if pct52 is not None:
        if pct52 >= -2:
            score += 1.0
            reasons.append("Dekat 52w high")
        elif pct52 >= -5:
            score += 0.7
        elif pct52 >= -10:
            score += 0.3

    # ── 7. News catalyst (up to 0.5 pts) ──────────────────────────────────
    if news_st == "ok" and news_s is not None and news_s >= 7:
        score += 0.5
        reasons.append("Katalis berita positif")

    # ── 8. Squeeze release (0.5 pts) ──────────────────────────────────────
    # Squeeze releasing = Bollinger/Keltner compression breaking out → volatility burst
    if not squeeze and atr_brk:   # squeeze just released + ATR breakout
        score += 0.5
        reasons.append("Squeeze release")

    # ── 9. ADX / trend confirmation (optional) ────────────────────────────
    if adx is not None and adx >= 30 and supertrend:
        score += 0.3

    # ── Liquidity sanity check ────────────────────────────────────────────
    # For sub-Rp50 stocks (extremely illiquid/pump-prone), discount the score
    if close is not None and close < 50:
        score *= 0.7

    # Clip to [0, 10]
    score = round(max(0.0, min(10.0, score)), 2)

    # Determine label
    if score >= _SCORE_HIGH:
        label = _LABEL_HIGH
    elif score >= _SCORE_WATCH:
        label = _LABEL_WATCH
    else:
        label = _LABEL_NO

    reason_str = ", ".join(reasons[:3]) if reasons else "Tidak ada sinyal kuat"

    return {
        "scalping_score":  score,
        "scalping_label":  label,
        "scalping_reason": reason_str,
    }


def enrich_df_with_scalping(df: pd.DataFrame) -> pd.DataFrame:
    """Apply compute_scalping_score to every row and add 3 columns.

    New columns: scalping_score, scalping_label, scalping_reason.
    Existing level columns are overwritten if present.
    Safe to call on an empty DataFrame.

    Args:
        df: Signals DataFrame (must have feature columns from feature_builder).

    Returns:
        DataFrame with 3 new scalping columns.
    """
    if df.empty:
        for col in ["scalping_score", "scalping_label", "scalping_reason"]:
            df[col] = None
        return df

    scalp_rows = df.apply(
        lambda r: pd.Series(compute_scalping_score(r)),
        axis=1,
    )
    # Drop existing scalping cols to avoid duplicates
    drop = [c for c in scalp_rows.columns if c in df.columns]
    df = df.drop(columns=drop)
    return pd.concat([df, scalp_rows], axis=1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _f(val: Any) -> float | None:
    """Safe float conversion; returns None on NaN / error."""
    if val is None:
        return None
    try:
        v = float(val)
        return None if v != v else v  # NaN guard
    except (TypeError, ValueError):
        return None


def _bool(val: Any) -> bool:
    """Safe bool conversion (handles string 'True'/'1')."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).lower() in ("true", "1", "yes")

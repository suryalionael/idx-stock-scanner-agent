"""Trading level calculator for IDX stocks.

Computes entry zone, take-profit targets, and cutloss levels
from technical indicators already present in the signals DataFrame.

IDX Tick Size Reference (BEI regulations):
    Price < 200        → tick = 1
    200  ≤ price < 500 → tick = 2
    500  ≤ price < 2000 → tick = 5
    2000 ≤ price < 5000 → tick = 10
    price ≥ 5000       → tick = 25

Level logic per signal type:
    BREAKOUT   : entry at current close ± 0.5 ATR (stock has triggered breakout)
    PRE_MARKUP : entry at support zone (MA20 area) up to current price
    WATCH/other: entry at pullback zone below current price

Risk-reward minimum target: 2:1 (tp_low = entry + 2×risk)
"""
from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# IDX Tick helpers
# ---------------------------------------------------------------------------

def get_tick_size(price: float) -> int:
    """Return IDX lot tick size for the given price."""
    if price < 200:
        return 1
    if price < 500:
        return 2
    if price < 2_000:
        return 5
    if price < 5_000:
        return 10
    return 25


def round_to_tick(price: float, tick: int, down: bool = False) -> int:
    """Round price to nearest tick. If down=True, round toward lower price."""
    if tick <= 0:
        return int(price)
    if down:
        return int(price // tick * tick)
    return int((price + tick / 2) // tick * tick)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_trading_levels(row: pd.Series) -> dict:
    """Compute entry/TP/cutloss from a signal row.

    Returns:
        {
            "entry_low":  int,
            "entry_high": int,
            "tp_low":     int,
            "tp_high":    int,
            "cutloss":    int,
        }
        All values are 0 if close price is missing or zero.
    """
    close = _safe_float(row.get("close"))
    if not close or close <= 0:
        return {"entry_low": 0, "entry_high": 0, "tp_low": 0, "tp_high": 0, "cutloss": 0}

    signal = str(row.get("signal", "")).upper()
    tick = get_tick_size(close)

    # ATR: use atr14 column, fallback to atr_pct * close / 100, then 2.5%
    atr = _safe_float(row.get("atr14")) or 0.0
    if atr <= 0:
        atr_pct = _safe_float(row.get("atr_pct")) or 0.0
        atr = close * atr_pct / 100 if atr_pct > 0 else close * 0.025

    # Support references
    ma20 = _safe_float(row.get("ma20")) or 0.0
    ma50 = _safe_float(row.get("ma50")) or 0.0

    # Find nearest support below close
    candidates = [s for s in [ma20, ma50] if 0 < s < close]
    support = max(candidates) if candidates else close * 0.95

    # ── Entry zone ────────────────────────────────────────────────────────
    if signal == "BREAKOUT":
        # Stock already breaking out — buy at current or chase slightly
        entry_low  = close
        entry_high = close + atr * 0.5
    elif signal == "PRE_MARKUP":
        # Building up — entry from support to current
        entry_low  = max(support * 1.005, close * 0.97)
        entry_high = close
    else:
        # WATCH / others — entry on pullback
        entry_low  = max(support, close * 0.95)
        entry_high = close * 0.99

    # ── Cutloss ───────────────────────────────────────────────────────────
    # Below support by ~1 ATR, but never below 5% of close
    cutloss_raw = support - atr * 0.5
    cutloss_raw = max(cutloss_raw, close * 0.92)  # hard cap at -8%

    # ── TP: risk-reward ratio ─────────────────────────────────────────────
    risk = entry_low - cutloss_raw
    if risk <= 0:
        risk = close * 0.03  # fallback 3% risk
    tp_low_raw  = entry_high + risk * 1.8
    tp_high_raw = entry_high + risk * 3.0

    # ── Round everything to tick ──────────────────────────────────────────
    return {
        "entry_low":  round_to_tick(entry_low,  tick),
        "entry_high": round_to_tick(entry_high, tick),
        "tp_low":     round_to_tick(tp_low_raw,  tick),
        "tp_high":    round_to_tick(tp_high_raw, tick),
        "cutloss":    round_to_tick(cutloss_raw, tick, down=True),
    }


def _safe_float(val) -> float:
    """Convert value to float, return 0.0 on failure or NaN."""
    try:
        v = float(val)
        return 0.0 if (v != v) else v  # NaN check without importing math
    except (TypeError, ValueError):
        return 0.0

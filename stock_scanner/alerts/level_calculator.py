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
    BREAKOUT   : entry at current close to close + 0.5 ATR
    PRE_MARKUP : entry at support zone (MA20 area) up to current price
    WATCH      : entry on pullback below current price (narrower range)
    AVOID/NONE : trade_setup_status = "inactive", all price fields = 0

Risk-reward minimum target: 1.8:1 (tp_low = entry + 1.8×risk)

trade_setup_status:
    "active"   — BREAKOUT, PRE_MARKUP, WATCH
    "inactive" — AVOID, NONE, missing close
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
# Internal constants
# ---------------------------------------------------------------------------

_ACTIVE_SIGNALS  = {"BREAKOUT", "PRE_MARKUP", "WATCH"}

_INACTIVE_RESULT = {
    "entry_low":          0,
    "entry_high":         0,
    "tp_low":             0,
    "tp_high":            0,
    "cutloss":            0,
    "trade_setup_status": "inactive",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_trading_levels(row: pd.Series) -> dict:
    """Compute entry/TP/cutloss from a signal row.

    Returns:
        {
            "entry_low":           int,
            "entry_high":          int,
            "tp_low":              int,
            "tp_high":             int,
            "cutloss":             int,
            "trade_setup_status":  "active" | "inactive",
        }

    trade_setup_status:
        "active"   — BREAKOUT, PRE_MARKUP, WATCH (levels are meaningful)
        "inactive" — AVOID, NONE, or missing close (all price fields = 0)
    """
    close  = _safe_float(row.get("close"))
    signal = str(row.get("signal", "")).upper()

    # Bail early for inactive signals or missing price
    if not close or close <= 0 or signal not in _ACTIVE_SIGNALS:
        return dict(_INACTIVE_RESULT)

    tick = get_tick_size(close)

    # ATR: prefer atr14, fallback to atr_pct * close / 100, then 2.5%
    atr = _safe_float(row.get("atr14")) or 0.0
    if atr <= 0:
        atr_pct = _safe_float(row.get("atr_pct")) or 0.0
        atr = close * atr_pct / 100 if atr_pct > 0 else close * 0.025

    # Support references
    ma20 = _safe_float(row.get("ma20")) or 0.0
    ma50 = _safe_float(row.get("ma50")) or 0.0

    # Nearest support below close
    candidates = [s for s in [ma20, ma50] if 0 < s < close]
    support = max(candidates) if candidates else close * 0.95

    # ── Entry zone ────────────────────────────────────────────────────────
    if signal == "BREAKOUT":
        # Stock already breaking out — buy at current or slight chase
        entry_low  = close
        entry_high = close + atr * 0.5

    elif signal == "PRE_MARKUP":
        # Building up — entry from near support to current price
        # Cap entry_low at close so we never have entry_low > entry_high
        entry_low  = min(max(support * 1.005, close * 0.97), close)
        entry_high = close

    else:  # WATCH — wait for pullback, tighter range
        entry_low  = max(support, close * 0.95)
        entry_high = close * 0.99

    # ── Cutloss ───────────────────────────────────────────────────────────
    # Below support by ~0.5 ATR, hard floor at -8% from close
    cutloss_raw = support - atr * 0.5
    cutloss_raw = max(cutloss_raw, close * 0.92)

    # ── TP: risk-reward ratio ─────────────────────────────────────────────
    risk = entry_low - cutloss_raw
    if risk <= 0:
        risk = close * 0.03  # fallback 3% risk
    tp_low_raw  = entry_high + risk * 1.8
    tp_high_raw = entry_high + risk * 3.0

    # ── Round everything to tick ──────────────────────────────────────────
    return {
        "entry_low":          round_to_tick(entry_low,  tick),
        "entry_high":         round_to_tick(entry_high, tick),
        "tp_low":             round_to_tick(tp_low_raw,  tick),
        "tp_high":            round_to_tick(tp_high_raw, tick),
        "cutloss":            round_to_tick(cutloss_raw, tick, down=True),
        "trade_setup_status": "active",
    }


def enrich_df_with_levels(df: pd.DataFrame, min_rr: float = 1.5) -> pd.DataFrame:
    """Apply compute_trading_levels to every row in a signals DataFrame.

    Adds columns:
        entry_low, entry_high, tp_low, tp_high, cutloss, trade_setup_status

    After tick rounding, validates Risk:Reward ratio.
    If realized R:R (using entry_low vs tp_low vs cutloss) < min_rr,
    trade_setup_status is set to "low_rr" instead of "active".

    Existing level columns are overwritten.  All other columns are preserved.
    Safe to call on an empty DataFrame (returns it unchanged).

    Args:
        df     : Signals DataFrame (must have 'close', 'signal', optionally atr14/ma20/ma50).
        min_rr : Minimum Risk:Reward ratio after tick rounding (default 1.5).
                 Set to 0.0 to disable R:R validation.

    Returns:
        DataFrame with 6 new columns appended.
    """
    if df.empty:
        for col in ["entry_low", "entry_high", "tp_low", "tp_high", "cutloss", "trade_setup_status"]:
            df[col] = None
        return df

    level_rows = df.apply(compute_trading_levels, axis=1, result_type="expand")

    # ── R:R validation after tick rounding ───────────────────────────────
    if min_rr > 0:
        def _check_rr(row: pd.Series) -> str:
            if row.get("trade_setup_status") != "active":
                return row.get("trade_setup_status", "inactive")
            try:
                el  = int(row["entry_low"])
                tl  = int(row["tp_low"])
                cl  = int(row["cutloss"])
                if el <= cl or cl <= 0 or tl <= el:
                    return "active"    # can't validate, keep as-is
                rr = (tl - el) / (el - cl)
                return "active" if rr >= min_rr else "low_rr"
            except (TypeError, ValueError, ZeroDivisionError):
                return row.get("trade_setup_status", "inactive")

        level_rows["trade_setup_status"] = level_rows.apply(_check_rr, axis=1)

    # Drop existing level cols to avoid duplicates, then concat
    drop_cols = [c for c in level_rows.columns if c in df.columns]
    df = df.drop(columns=drop_cols)
    return pd.concat([df, level_rows], axis=1)


def _safe_float(val) -> float:
    """Convert value to float, return 0.0 on failure or NaN."""
    try:
        v = float(val)
        return 0.0 if (v != v) else v  # NaN check without importing math
    except (TypeError, ValueError):
        return 0.0

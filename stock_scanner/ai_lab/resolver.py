"""Performance Tracker automation — resolves AI Lab recommendations against
forward OHLCV data instead of relying on manual
stock_scanner.db.ai_lab.update_recommendation_status() calls (see the
runbook in docs/AI_LAB_ARCHITECTURE.md).

Pure logic only — no DB/network I/O beyond reading raw parquet, mirroring
the hypothesis_agent.py/decision_agent.py split (agents/resolver compute,
scripts/ orchestrate). Both functions take already-loaded DataFrames (from
stock_scanner.db.ai_lab.load_recommendations) and raw_dir, and return plain
dicts ready for update_recommendation_status(**dict).

Exit policy deliberately mirrors stock_scanner.pipeline.evaluator's
fallback-only branch (AI Lab recommendations never carry entry_low/tp_low/
cutloss columns — Evidence produces a score/confidence/recommendation
tier, not price levels): TP/SL checked in the same order, same default
horizon/risk-pct, so return_percentage is directly comparable to the
production scanner's own realized win rate (see "Future Ready — Auto
Promotion Engine" in docs/AI_LAB_ARCHITECTURE.md).

trade_outcome is deliberately a separate axis from status: an EXPIRED
mark-to-market row that happened to be profitable is still WIN, never
inferred as a loss just because it never hit its target.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

_DEFAULT_HORIZON = 10  # trading days — same default as evaluator._DEFAULT_HORIZON
_DEFAULT_RISK_PCT = 3.0  # % — same default as evaluator._DEFAULT_RISK_PCT


def _load_raw(ticker: str, raw_dir: Path) -> pd.DataFrame | None:
    raw_path = raw_dir / f"{ticker}.parquet"
    if not raw_path.exists():
        raw_path = raw_dir / f"{ticker.replace('.JK', '')}.parquet"
    if not raw_path.exists():
        return None
    try:
        raw_df = pd.read_parquet(raw_path)
        raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.tz_localize(None).dt.normalize()
        return raw_df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        logger.debug(f"{ticker}: cannot load raw — {e}")
        return None


def activate_pending(df_pending: pd.DataFrame, raw_dir: Path) -> list[dict]:
    """PENDING -> ACTIVE: entry_price = the close on generated_date, once
    that day's raw OHLCV row exists. Rows for tickers whose raw data
    doesn't cover generated_date yet are left out of the result (still
    PENDING) — safe to re-run tomorrow."""
    results = []
    for _, row in df_pending.iterrows():
        ticker = str(row["ticker"])
        generated_date = pd.Timestamp(row["generated_date"]).normalize()

        raw_df = _load_raw(ticker, raw_dir)
        if raw_df is None:
            continue

        entry_row = raw_df[raw_df["date"] == generated_date]
        if entry_row.empty:
            continue

        entry_price = float(entry_row["close"].iloc[0])
        if entry_price <= 0:
            logger.debug(f"{ticker}: non-positive close on {generated_date.date()} — skip activation")
            continue

        results.append(
            {
                "id": row["id"],
                "status": "ACTIVE",
                "entry_price": entry_price,
                "highest_price": entry_price,
                "lowest_price": entry_price,
                "holding_days": 0,
            }
        )
    return results


def resolve_active(
    df_active: pd.DataFrame,
    raw_dir: Path,
    horizon_days: int = _DEFAULT_HORIZON,
    risk_pct: float = _DEFAULT_RISK_PCT,
) -> list[dict]:
    """ACTIVE -> CLOSED | EXPIRED, plus a running MFE/MAE/holding_days
    refresh on every call regardless of whether this row resolves this run.

    Fallback-only exit rule (no explicit tp/cutloss on AIRecommendation):
    tp = entry * (1 + risk_pct/100 * 1.8), cl = entry * (1 - risk_pct/100),
    checked TP-then-SL day by day, capped at horizon_days. Rows with no new
    forward data since the last run (0 rows past generated_date) are
    omitted from the result — nothing changed, no-op write avoided.
    """
    results = []
    for _, row in df_active.iterrows():
        ticker = str(row["ticker"])
        entry_price = float(row["entry_price"])
        generated_date = pd.Timestamp(row["generated_date"]).normalize()

        raw_df = _load_raw(ticker, raw_dir)
        if raw_df is None:
            continue

        fwd = raw_df[raw_df["date"] > generated_date].head(horizon_days)
        if fwd.empty:
            continue

        tp = entry_price * (1 + risk_pct / 100 * 1.8)
        cl = entry_price * (1 - risk_pct / 100)

        highest_price = entry_price
        lowest_price = entry_price
        exit_price: float | None = None
        outcome_day: int | None = None

        for i, (h, lo) in enumerate(zip(fwd["high"].values, fwd["low"].values)):
            highest_price = max(highest_price, float(h))
            lowest_price = min(lowest_price, float(lo))
            if h >= tp:
                exit_price = tp
                outcome_day = i + 1
                break
            if lo <= cl:
                exit_price = cl
                outcome_day = i + 1
                break

        result: dict = {
            "id": row["id"],
            "highest_price": highest_price,
            "lowest_price": lowest_price,
            "max_runup_pct": max(0.0, (highest_price - entry_price) / entry_price * 100),
            "max_drawdown_pct": max(0.0, (entry_price - lowest_price) / entry_price * 100),
        }

        if exit_price is not None:
            result["status"] = "CLOSED"
            result["holding_days"] = outcome_day
        elif len(fwd) >= horizon_days:
            exit_price = float(fwd["close"].values[-1])
            result["status"] = "EXPIRED"
            result["holding_days"] = len(fwd)
        else:
            # Still open, not enough time elapsed yet — re-check next run.
            result["holding_days"] = len(fwd)
            results.append(result)
            continue

        return_percentage = (exit_price - entry_price) / entry_price * 100
        result["exit_price"] = exit_price
        result["return_percentage"] = return_percentage
        result["trade_outcome"] = (
            "WIN" if return_percentage > 0 else "LOSS" if return_percentage < 0 else "BREAKEVEN"
        )
        results.append(result)

    return results

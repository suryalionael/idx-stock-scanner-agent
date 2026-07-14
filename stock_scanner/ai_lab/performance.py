"""AI Lab performance metrics — dashboard section 3 ("AI Performance" cards).

Computed entirely from ai_recommendations rows (never re-derived from the
LLM's own claims). Grouping by `ai_model` is free — since ai_model is just
a column value with no hardcoded list anywhere (see models.py), a new
plug-and-play model appears in per-model performance automatically the
first time it produces a recommendation.

Metric definitions mirror the existing realized-R metrics in
stock_scanner.pipeline.evaluator.summarize_evaluations (profit_factor =
sum(wins)/abs(sum(losses)), expectancy = mean(all returns)) for
consistency across the codebase, applied here to return_percentage instead
of realized_R.
"""
from __future__ import annotations

import pandas as pd

_RESOLVED_STATUSES = {"CLOSED", "EXPIRED"}


def compute_performance(df: pd.DataFrame, ai_model: str | None = None) -> dict:
    """Compute the AI Performance card metrics.

    Args:
        df: ai_recommendations rows (as loaded by stock_scanner.db.ai_lab.load_recommendations).
        ai_model: if given, restrict to this model's rows only; otherwise
            aggregate across all models.

    Returns a dict with win_rate, average_return, average_loss,
    profit_factor, expected_value, sharpe_ratio (all None if there isn't
    enough resolved data yet — not 0 or NaN, so the dashboard can render
    "not enough data" instead of a misleading zero), plus
    number_of_recommendations, active_positions, closed_positions (always
    integers, never None).
    """
    if ai_model is not None and not df.empty:
        df = df[df["ai_model"] == ai_model]

    n_total = len(df)
    active_positions = int((df["status"] == "ACTIVE").sum()) if n_total else 0
    closed_positions = int(df["status"].isin(_RESOLVED_STATUSES).sum()) if n_total else 0

    resolved = df[df["status"].isin(_RESOLVED_STATUSES) & df["return_percentage"].notna()]
    metrics = {
        "win_rate": None, "average_return": None, "average_loss": None,
        "profit_factor": None, "expected_value": None, "sharpe_ratio": None,
    }

    if not resolved.empty:
        returns = resolved["return_percentage"]
        wins = returns[returns > 0]
        losses = returns[returns <= 0]

        metrics["win_rate"] = round(len(wins) / len(resolved), 4)
        metrics["average_return"] = round(float(wins.mean()), 4) if not wins.empty else None
        metrics["average_loss"] = round(float(losses.mean()), 4) if not losses.empty else None

        losses_sum_abs = abs(losses.sum())
        metrics["profit_factor"] = round(float(wins.sum() / losses_sum_abs), 4) if losses_sum_abs > 0 else None

        metrics["expected_value"] = round(float(returns.mean()), 4)

        std = returns.std()
        # Per-trade Sharpe-like ratio (mean / std of realized return per
        # recommendation) — NOT time-annualized, since recommendations
        # don't share a fixed holding period yet.
        metrics["sharpe_ratio"] = round(float(returns.mean() / std), 4) if std and std > 0 else None

    return {
        **metrics,
        "number_of_recommendations": n_total,
        "active_positions": active_positions,
        "closed_positions": closed_positions,
    }

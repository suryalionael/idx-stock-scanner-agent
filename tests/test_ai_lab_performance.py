"""Tests for stock_scanner/ai_lab/performance.py — AI Performance card metrics."""
import pandas as pd

from stock_scanner.ai_lab.performance import compute_performance


def _df(rows: list[dict]) -> pd.DataFrame:
    defaults = {"ai_model": "momentum_ai", "status": "CLOSED", "return_percentage": 0.0}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_empty_dataframe_returns_none_metrics_not_crash():
    result = compute_performance(pd.DataFrame(columns=["ai_model", "status", "return_percentage"]))
    assert result["win_rate"] is None
    assert result["number_of_recommendations"] == 0
    assert result["active_positions"] == 0
    assert result["closed_positions"] == 0


def test_win_rate_and_counts():
    df = _df([
        {"status": "CLOSED", "return_percentage": 0.1},
        {"status": "CLOSED", "return_percentage": -0.05},
        {"status": "ACTIVE", "return_percentage": None},
        {"status": "PENDING", "return_percentage": None},
    ])
    result = compute_performance(df)
    assert result["win_rate"] == 0.5
    assert result["number_of_recommendations"] == 4
    assert result["active_positions"] == 1
    assert result["closed_positions"] == 2  # both CLOSED rows; PENDING/ACTIVE don't count


def test_average_return_and_average_loss():
    df = _df([
        {"status": "CLOSED", "return_percentage": 0.10},
        {"status": "CLOSED", "return_percentage": 0.20},
        {"status": "CLOSED", "return_percentage": -0.05},
    ])
    result = compute_performance(df)
    assert result["average_return"] == 0.15   # mean of wins: (0.10+0.20)/2
    assert result["average_loss"] == -0.05


def test_profit_factor_matches_repo_convention():
    # profit_factor = sum(wins) / abs(sum(losses)) — same formula as
    # stock_scanner.pipeline.evaluator.summarize_evaluations.
    df = _df([
        {"status": "CLOSED", "return_percentage": 0.30},
        {"status": "CLOSED", "return_percentage": -0.10},
        {"status": "CLOSED", "return_percentage": -0.10},
    ])
    result = compute_performance(df)
    assert result["profit_factor"] == 1.5  # 0.30 / (0.10+0.10)


def test_profit_factor_none_when_no_losses():
    df = _df([{"status": "CLOSED", "return_percentage": 0.10}])
    result = compute_performance(df)
    assert result["profit_factor"] is None


def test_expected_value_is_mean_of_all_resolved_returns():
    df = _df([
        {"status": "CLOSED", "return_percentage": 0.10},
        {"status": "EXPIRED", "return_percentage": -0.20},
    ])
    result = compute_performance(df)
    assert result["expected_value"] == -0.05


def test_sharpe_ratio_none_when_zero_variance():
    df = _df([
        {"status": "CLOSED", "return_percentage": 0.10},
        {"status": "CLOSED", "return_percentage": 0.10},
    ])
    result = compute_performance(df)
    assert result["sharpe_ratio"] is None  # std == 0


def test_sharpe_ratio_computed_when_variance_present():
    df = _df([
        {"status": "CLOSED", "return_percentage": 0.10},
        {"status": "CLOSED", "return_percentage": -0.05},
        {"status": "CLOSED", "return_percentage": 0.20},
    ])
    result = compute_performance(df)
    assert result["sharpe_ratio"] is not None


def test_filters_by_ai_model():
    df = _df([
        {"ai_model": "momentum_ai", "status": "CLOSED", "return_percentage": 0.50},
        {"ai_model": "breakout_ai", "status": "CLOSED", "return_percentage": -0.50},
    ])
    momentum_only = compute_performance(df, ai_model="momentum_ai")
    assert momentum_only["number_of_recommendations"] == 1
    assert momentum_only["expected_value"] == 0.50


def test_unresolved_returns_excluded_from_ratio_metrics():
    df = _df([
        {"status": "PENDING", "return_percentage": None},
        {"status": "ACTIVE", "return_percentage": None},
    ])
    result = compute_performance(df)
    assert result["win_rate"] is None
    assert result["expected_value"] is None

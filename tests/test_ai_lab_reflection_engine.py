"""Tests for stock_scanner/ai_lab/reflection_engine.py — pure statistical
engine, no DB/LLM involved. Mirrors tests/test_pattern_dedup.py's
convention of engineering a known synthetic scenario and asserting the
exact expected outcome."""
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from stock_scanner.ai_lab import reflection_engine
from stock_scanner.ai_lab.reflection_engine import generate_observations
from stock_scanner.ai_lab.schemas import ObservationCategory


def _row(
    id_, ticker, ai_model, trade_outcome, *, confidence=0.5, recommendation="BUY",
    verdict="similar", golden_cross=True, ma_full_alignment=True, return_percentage=1.0,
    holding_days=3,
) -> dict:
    return dict(
        id=id_, ticker=ticker, ai_model=ai_model, confidence=confidence, recommendation=recommendation,
        reasoning=json.dumps({"technical_indicators": {"golden_cross": golden_cross, "ma_full_alignment": ma_full_alignment}}),
        historical_comparison=json.dumps({"verdict": verdict}),
        trade_outcome=trade_outcome, return_percentage=return_percentage, holding_days=holding_days,
    )


def _mixed_population(n_success_model_a=16, n_model_a=20, n_success_model_b=4, n_model_b=20) -> pd.DataFrame:
    """momentum_ai: strong success pattern; breakout_ai: strong failure
    pattern — same fixture shape used for the manual verification during
    implementation."""
    rows = []
    for i in range(n_model_a):
        rows.append(_row(
            f"a{i}", "BBCA.JK", "momentum_ai", "WIN" if i < n_success_model_a else "LOSS",
            recommendation="BUY", verdict="stronger", golden_cross=True, ma_full_alignment=True,
        ))
    for i in range(n_model_b):
        rows.append(_row(
            f"b{i}", "TLKM.JK", "breakout_ai", "WIN" if i < n_success_model_b else "LOSS",
            recommendation="WATCH", verdict="weaker", golden_cross=False, ma_full_alignment=False,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_dataframe_returns_no_observations():
    assert generate_observations(pd.DataFrame()) == []


def test_too_few_trades_returns_no_observations():
    df = _mixed_population(n_success_model_a=3, n_model_a=4, n_success_model_b=1, n_model_b=4)
    assert generate_observations(df) == []


def test_no_signal_no_observations():
    """Every group has the same win rate as the baseline (50%) — nothing
    should clear the gate; a flat population is not a finding."""
    rows = []
    for i in range(20):
        rows.append(_row(f"x{i}", "BBCA.JK", "momentum_ai", "WIN" if i % 2 == 0 else "LOSS"))
    df = pd.DataFrame(rows)
    assert generate_observations(df, min_n_success=3, alpha=0.05) == []


# ---------------------------------------------------------------------------
# Categorical dimensions — success + failure patterns, both directions
# ---------------------------------------------------------------------------

def test_detects_model_success_and_failure_patterns():
    df = _mixed_population()
    obs = generate_observations(df, min_n_success=3, alpha=0.05)
    by_category = {o.category for o in obs}
    assert ObservationCategory.MODEL_PERFORMANCE in by_category

    model_obs = {o.supporting_statistics["value"]: o for o in obs if o.category == ObservationCategory.MODEL_PERFORMANCE}
    assert "momentum_ai" in model_obs and "breakout_ai" in model_obs
    assert model_obs["momentum_ai"].supporting_statistics["win_rate"] == pytest.approx(0.8)
    assert model_obs["breakout_ai"].supporting_statistics["win_rate"] == pytest.approx(0.2)
    assert "consistently succeeds" in model_obs["momentum_ai"].title
    assert "consistently fails" in model_obs["breakout_ai"].title


def test_detects_sector_and_recommendation_and_verdict_patterns():
    df = _mixed_population()
    obs = generate_observations(df, min_n_success=3, alpha=0.05)
    categories = {o.category for o in obs}
    assert ObservationCategory.SECTOR_PERFORMANCE in categories
    assert ObservationCategory.RECOMMENDATION_LEVEL_PERFORMANCE in categories
    assert ObservationCategory.HISTORICAL_VERDICT_ACCURACY in categories


def test_technical_indicator_pattern_detected():
    df = _mixed_population()
    obs = generate_observations(df, min_n_success=3, alpha=0.05)
    tech = [o for o in obs if o.category == ObservationCategory.TECHNICAL_PATTERN]
    assert any(o.supporting_statistics["dimension"] == "golden_cross" for o in tech)


def test_supporting_statistics_are_real_numbers_not_invented():
    df = _mixed_population()
    obs = generate_observations(df, min_n_success=3, alpha=0.05)
    momentum = next(o for o in obs if o.category == ObservationCategory.MODEL_PERFORMANCE
                     and o.supporting_statistics["value"] == "momentum_ai")
    stats = momentum.supporting_statistics
    assert stats["n"] == 20
    assert stats["n_success"] == 16
    assert stats["win_rate"] == pytest.approx(16 / 20)
    assert 0.0 <= stats["ci_lower"] <= stats["win_rate"] <= stats["ci_upper"] <= 1.0
    assert stats["p_value_adjusted"] < 0.05
    assert momentum.affected_trade_count == 20
    assert 0.0 <= momentum.confidence <= 1.0


# ---------------------------------------------------------------------------
# Failure-pattern gate uses (n - n_success), not n_success — regression
# guard for the asymmetry described in reflection_engine.py's module
# docstring: gating a failure pattern on n_success (its rare, low count)
# would make failure patterns nearly undiscoverable.
# ---------------------------------------------------------------------------

def test_failure_pattern_gates_on_non_success_count_not_success_count():
    # breakout_ai: only 1 win out of 20 (n_success=1, below min_n_success=3)
    # but 19 losses (well above min_n_success=3) — should still gate as a
    # failure pattern since (n - n_success) >= min_n_success.
    df = _mixed_population(n_success_model_a=16, n_model_a=20, n_success_model_b=1, n_model_b=20)
    obs = generate_observations(df, min_n_success=3, alpha=0.05)
    model_obs = {o.supporting_statistics["value"]: o for o in obs if o.category == ObservationCategory.MODEL_PERFORMANCE}
    assert "breakout_ai" in model_obs
    assert "consistently fails" in model_obs["breakout_ai"].title


# ---------------------------------------------------------------------------
# Confidence calibration
# ---------------------------------------------------------------------------

def test_confidence_calibration_detects_overconfidence():
    # Stated confidence high (0.9) but realized win rate low (~20%).
    rows = []
    for i in range(20):
        rows.append(_row(f"c{i}", "BBCA.JK", "momentum_ai", "WIN" if i < 4 else "LOSS", confidence=0.9))
    # Add a low-confidence, low-win-rate control group so qcut has spread.
    for i in range(20):
        rows.append(_row(f"d{i}", "TLKM.JK", "breakout_ai", "WIN" if i < 10 else "LOSS", confidence=0.2))
    df = pd.DataFrame(rows)
    obs = generate_observations(df, min_n_success=3, alpha=0.05)
    calibration = [o for o in obs if o.category == ObservationCategory.CONFIDENCE_CALIBRATION]
    assert any(o.supporting_statistics.get("calibration_issue") == "overconfident" for o in calibration)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_observation_id_is_deterministic_for_fixed_generated_at(monkeypatch):
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, tzinfo=tz)

    monkeypatch.setattr(reflection_engine, "datetime", _FrozenDatetime)
    df = _mixed_population()
    obs_a = generate_observations(df, min_n_success=3, alpha=0.05)
    obs_b = generate_observations(df, min_n_success=3, alpha=0.05)
    ids_a = sorted(o.observation_id for o in obs_a)
    ids_b = sorted(o.observation_id for o in obs_b)
    assert ids_a == ids_b
    assert len(ids_a) == len(set(ids_a))  # no collisions within one run


def test_generated_at_is_real_utc_timestamp():
    df = _mixed_population()
    obs = generate_observations(df, min_n_success=3, alpha=0.05)
    for o in obs:
        parsed = datetime.fromisoformat(o.generated_at)
        assert parsed.tzinfo is not None

"""Tests for AI Lab's Pydantic contracts (stock_scanner/ai_lab/schemas.py)."""
import pytest
from pydantic import ValidationError

from stock_scanner.ai_lab.schemas import (
    AIRecommendation,
    ConfidenceBreakdown,
    DecisionOutput,
    DecisionTrace,
    Evidence,
    HistoricalComparison,
    HistoricalComparisonVerdict,
    HypothesisOutput,
    RecommendationLevel,
    RecommendationStatus,
    RiskLevel,
)


def _trace(**overrides) -> DecisionTrace:
    base = dict(technical_score=80.0, statistical_score=20.0, pattern_similarity_score=90.0,
                risk_score=60.0, final_score=55.0)
    base.update(overrides)
    return DecisionTrace(**base)


def _confidence(**overrides) -> ConfidenceBreakdown:
    base = dict(technical=0.8, statistical=0.2, pattern_similarity=0.9,
                risk_adjustment=-0.18, final_confidence=0.45)
    base.update(overrides)
    return ConfidenceBreakdown(**base)


def _comparison(**overrides) -> HistoricalComparison:
    base = dict(pattern_description="MA Alignment + ATR Breakout", sample_size=115, win_rate=0.13,
                ci_lower=0.081, ci_upper=0.187, verdict=HistoricalComparisonVerdict.SIMILAR,
                explanation="test")
    base.update(overrides)
    return HistoricalComparison(**base)


def test_hypothesis_output_valid():
    h = HypothesisOutput(why="test", strengths=["a"], weaknesses=["b"], risks=["c"])
    assert h.strengths == ["a"]


def test_hypothesis_output_caps_strengths_at_five():
    with pytest.raises(ValidationError):
        HypothesisOutput(why="test", strengths=["a", "b", "c", "d", "e", "f"])


def test_decision_output_valid():
    d = DecisionOutput(reasoning_summary="test", historical_comparison_explanation="test",
                       confidence_explanation="test")
    assert d.reasoning_summary == "test"


def test_decision_trace_score_out_of_range_rejected():
    with pytest.raises(ValidationError):
        _trace(technical_score=150.0)


def test_confidence_breakdown_risk_adjustment_must_be_non_positive():
    with pytest.raises(ValidationError):
        _confidence(risk_adjustment=0.1)  # a "penalty" that's actually a bonus is invalid


def test_confidence_breakdown_valid_negative_risk_adjustment():
    c = _confidence(risk_adjustment=-0.05)
    assert c.risk_adjustment == -0.05


def test_historical_comparison_no_data_verdict_allows_null_stats():
    hc = HistoricalComparison(verdict=HistoricalComparisonVerdict.NO_DATA)
    assert hc.sample_size is None
    assert hc.win_rate is None


def test_recommendation_level_rejects_old_sell_value():
    # SELL was removed as part of the recommendation-level upgrade —
    # STRONG_BUY/BUY/WATCH/AVOID only.
    with pytest.raises(ValueError):
        RecommendationLevel("SELL")


def test_recommendation_level_accepts_strong_buy():
    assert RecommendationLevel("STRONG_BUY") == RecommendationLevel.STRONG_BUY


def test_evidence_is_frozen():
    ev = Evidence(ticker="BBCA.JK", technical_indicators={"rsi14": 55.0})
    with pytest.raises(ValidationError):
        ev.ticker = "OTHER.JK"


def test_evidence_best_pattern_similarity_defaults_to_zero():
    ev = Evidence(ticker="BBCA.JK")
    assert ev.best_pattern_similarity_pct == 0.0


def test_ai_recommendation_full_round_trip():
    rec = AIRecommendation(
        id="abc123", ticker="BBCA.JK", ai_model="momentum_ai", score=55.0, confidence=0.45,
        recommendation=RecommendationLevel.WATCH, reasoning={"why": "test"},
        decision_trace=_trace(), confidence_breakdown=_confidence(), historical_comparison=_comparison(),
        expected_return=None, risk_level=RiskLevel.MEDIUM, generated_date="2026-07-14",
        status=RecommendationStatus.PENDING, model="oc/hy3-free",
    )
    dumped = rec.model_dump()
    restored = AIRecommendation.model_validate(dumped)
    assert restored == rec


def test_ai_recommendation_expected_return_can_be_null():
    rec = AIRecommendation(
        id="abc123", ticker="BBCA.JK", ai_model="momentum_ai", score=55.0, confidence=0.45,
        recommendation=RecommendationLevel.WATCH, decision_trace=_trace(),
        confidence_breakdown=_confidence(), historical_comparison=_comparison(),
        expected_return=None, generated_date="2026-07-14", model="oc/hy3-free",
    )
    assert rec.expected_return is None

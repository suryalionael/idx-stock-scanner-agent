"""Tests for AI Lab's Pydantic contracts (stock_scanner/ai_lab/schemas.py)."""
import pytest
from pydantic import ValidationError

from stock_scanner.ai_lab.schemas import (
    AIRecommendation,
    DecisionOutput,
    Evidence,
    HypothesisOutput,
    RecommendationAction,
    RecommendationStatus,
    RiskLevel,
)


def test_hypothesis_output_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        HypothesisOutput(
            why="test", confidence=1.5, confidence_explanation="test",
            strengths=[], weaknesses=[], risks=[],
        )


def test_hypothesis_output_valid_confidence_accepted():
    h = HypothesisOutput(
        why="test", confidence=0.75, confidence_explanation="test",
        strengths=["a"], weaknesses=["b"], risks=["c"],
    )
    assert h.confidence == 0.75


def test_decision_output_score_out_of_range_rejected():
    with pytest.raises(ValidationError):
        DecisionOutput(
            score=150.0, confidence=0.5, recommendation=RecommendationAction.BUY,
            expected_return=0.05, risk_level=RiskLevel.LOW, reasoning_summary="test",
        )


def test_decision_output_rejects_unknown_recommendation_string():
    with pytest.raises(ValidationError):
        DecisionOutput(
            score=80.0, confidence=0.5, recommendation="STRONG_BUY",  # not a valid enum value
            expected_return=0.05, risk_level=RiskLevel.LOW, reasoning_summary="test",
        )


def test_evidence_is_frozen():
    ev = Evidence(ticker="BBCA.JK", technical_indicators={"rsi14": 55.0})
    with pytest.raises(ValidationError):
        ev.ticker = "OTHER.JK"


def test_ai_recommendation_full_round_trip():
    rec = AIRecommendation(
        id="abc123", ticker="BBCA.JK", ai_model="momentum_ai", score=92.0, confidence=0.88,
        recommendation=RecommendationAction.BUY, reasoning={"why": "test"},
        expected_return=0.08, risk_level=RiskLevel.MEDIUM, generated_date="2026-07-14",
        status=RecommendationStatus.PENDING, model="deepseek-v4-flash-free",
    )
    dumped = rec.model_dump()
    restored = AIRecommendation.model_validate(dumped)
    assert restored == rec

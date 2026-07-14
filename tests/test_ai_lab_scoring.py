"""Tests for stock_scanner/ai_lab/scoring.py — the deterministic scoring
system behind AI Lab's explainability upgrade (decision trace, confidence
breakdown, recommendation level, risk level, historical comparison,
evidence highlights). Every function here must be a pure function of its
inputs — same evidence in, same numbers out, always."""
import pandas as pd

from stock_scanner.ai_lab.models import AI_MODEL_REGISTRY
from stock_scanner.ai_lab.schemas import (
    Evidence,
    HistoricalComparisonVerdict,
    RecommendationLevel,
    RiskLevel,
)
from stock_scanner.ai_lab.scoring import (
    classify_recommendation_level,
    classify_risk_level,
    compute_confidence_breakdown,
    compute_decision_trace,
    compute_expected_return,
    compute_historical_comparison,
    compute_pattern_similarity_score,
    compute_risk_score,
    compute_statistical_score,
    compute_technical_score,
    generate_evidence_highlights,
)


def _row(**overrides) -> pd.Series:
    base = {"trend_score": 5.0, "momentum_score": 5.0, "breakout_score": 5.0, "volume_score": 5.0,
            "quality_penalty_total": 0.0, "atr_pct": 2.0, "is_uma": False, "is_special_monitoring": False}
    base.update(overrides)
    return pd.Series(base)


def _evidence(**overrides) -> Evidence:
    base = dict(ticker="BBCA.JK", technical_indicators={}, statistical_evidence=[],
                similar_patterns=[], best_pattern_similarity_pct=0.0)
    base.update(overrides)
    return Evidence(**base)


# ---------------------------------------------------------------------------
# compute_technical_score — reuses production component scores
# ---------------------------------------------------------------------------

def test_technical_score_momentum_reuses_trend_and_momentum_scores():
    row = _row(trend_score=8.0, momentum_score=6.0)
    score = compute_technical_score(AI_MODEL_REGISTRY["momentum_ai"], row, {})
    assert score == 70.0  # (8+6)/2 * 10


def test_technical_score_breakout_reuses_breakout_score():
    row = _row(breakout_score=9.0)
    score = compute_technical_score(AI_MODEL_REGISTRY["breakout_ai"], row, {})
    assert score == 90.0


def test_technical_score_volume_reuses_volume_score():
    row = _row(volume_score=3.0)
    score = compute_technical_score(AI_MODEL_REGISTRY["volume_ai"], row, {})
    assert score == 30.0


def test_technical_score_reversal_uses_oversold_indicators():
    row = _row()
    indicators = {"rsi14": 20.0, "stoch_rsi_k": 15.0, "macd_histogram": 1.0}
    score = compute_technical_score(AI_MODEL_REGISTRY["reversal_ai"], row, indicators)
    # rsi: 100-20=80, stoch: 100-15=85, macd: 50+10=60 -> mean=75
    assert score == 75.0


def test_technical_score_reversal_neutral_when_no_indicators():
    score = compute_technical_score(AI_MODEL_REGISTRY["reversal_ai"], _row(), {})
    assert score == 50.0


def test_technical_score_missing_production_scores_defaults_to_zero_contribution():
    row = pd.Series({})  # no trend_score/momentum_score at all
    score = compute_technical_score(AI_MODEL_REGISTRY["momentum_ai"], row, {})
    assert score == 0.0


# ---------------------------------------------------------------------------
# compute_statistical_score
# ---------------------------------------------------------------------------

def test_statistical_score_zero_when_no_evidence():
    assert compute_statistical_score(_evidence()) == 0.0


def test_statistical_score_uses_win_rate_shrunk():
    ev = _evidence(statistical_evidence=[{"win_rate_shrunk": 0.65}])
    assert compute_statistical_score(ev) == 65.0


def test_statistical_score_averages_multiple_patterns():
    ev = _evidence(statistical_evidence=[{"win_rate_shrunk": 0.6}, {"win_rate_shrunk": 0.4}])
    assert compute_statistical_score(ev) == 50.0


# ---------------------------------------------------------------------------
# compute_pattern_similarity_score
# ---------------------------------------------------------------------------

def test_pattern_similarity_score_mirrors_evidence_field():
    ev = _evidence(best_pattern_similarity_pct=73.5)
    assert compute_pattern_similarity_score(ev) == 73.5


# ---------------------------------------------------------------------------
# compute_risk_score
# ---------------------------------------------------------------------------

def test_risk_score_zero_for_clean_ticker():
    row = _row(quality_penalty_total=0.0, atr_pct=0.0, is_uma=False, is_special_monitoring=False)
    assert compute_risk_score(row) == 0.0


def test_risk_score_uma_flag_adds_forty():
    row = _row(is_uma=True)
    assert compute_risk_score(row) >= 40.0


def test_risk_score_special_monitoring_adds_thirty():
    row = _row(is_special_monitoring=True, is_uma=False, atr_pct=0.0, quality_penalty_total=0.0)
    assert compute_risk_score(row) == 30.0


def test_risk_score_capped_at_100():
    row = _row(is_uma=True, is_special_monitoring=True, quality_penalty_total=100.0, atr_pct=100.0)
    assert compute_risk_score(row) == 100.0


def test_risk_score_quality_penalty_contributes():
    row = _row(quality_penalty_total=5.0, atr_pct=0.0)
    assert compute_risk_score(row) == 15.0  # min(30, 5*3)


# ---------------------------------------------------------------------------
# compute_decision_trace — weighted composite
# ---------------------------------------------------------------------------

def test_decision_trace_final_score_matches_documented_weights():
    row = _row(trend_score=8.0, momentum_score=8.0, quality_penalty_total=0.0, atr_pct=0.0)
    ev = _evidence(statistical_evidence=[{"win_rate_shrunk": 0.5}], best_pattern_similarity_pct=100.0)
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    assert trace.technical_score == 80.0
    assert trace.statistical_score == 50.0
    assert trace.pattern_similarity_score == 100.0
    assert trace.risk_score == 0.0
    expected = 80.0 * 0.35 + 50.0 * 0.35 + 100.0 * 0.15 + 100.0 * 0.15
    assert trace.final_score == round(expected, 2)


def test_decision_trace_high_risk_pulls_final_score_down():
    row_low_risk = _row(is_uma=False, quality_penalty_total=0.0, atr_pct=0.0)
    row_high_risk = _row(is_uma=True, quality_penalty_total=10.0, atr_pct=20.0)
    ev = _evidence()
    trace_low = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row_low_risk, ev)
    trace_high = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row_high_risk, ev)
    assert trace_high.final_score < trace_low.final_score


def test_decision_trace_all_fields_within_bounds():
    row = _row(trend_score=10.0, momentum_score=10.0)
    ev = _evidence(statistical_evidence=[{"win_rate_shrunk": 1.0}], best_pattern_similarity_pct=100.0)
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    assert 0.0 <= trace.final_score <= 100.0


# ---------------------------------------------------------------------------
# compute_confidence_breakdown
# ---------------------------------------------------------------------------

def test_confidence_breakdown_risk_adjustment_always_non_positive():
    row = _row(is_uma=True)
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, _evidence())
    confidence = compute_confidence_breakdown(trace)
    assert confidence.risk_adjustment <= 0.0


def test_confidence_breakdown_zero_risk_means_zero_adjustment():
    row = _row(is_uma=False, is_special_monitoring=False, quality_penalty_total=0.0, atr_pct=0.0)
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, _evidence())
    confidence = compute_confidence_breakdown(trace)
    assert confidence.risk_adjustment == 0.0


def test_confidence_breakdown_final_confidence_bounded_0_to_1():
    row = _row(trend_score=10.0, momentum_score=10.0, is_uma=True, quality_penalty_total=10.0, atr_pct=20.0)
    ev = _evidence(statistical_evidence=[{"win_rate_shrunk": 1.0}], best_pattern_similarity_pct=100.0)
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    confidence = compute_confidence_breakdown(trace)
    assert 0.0 <= confidence.final_confidence <= 1.0


# ---------------------------------------------------------------------------
# classify_risk_level / classify_recommendation_level — rule-based
# ---------------------------------------------------------------------------

def test_classify_risk_level_thresholds():
    from stock_scanner.ai_lab.schemas import DecisionTrace

    def trace_with_risk(risk):
        return DecisionTrace(technical_score=50, statistical_score=50,
                             pattern_similarity_score=50, risk_score=risk, final_score=50)

    assert classify_risk_level(trace_with_risk(10)) == RiskLevel.LOW
    assert classify_risk_level(trace_with_risk(50)) == RiskLevel.MEDIUM
    assert classify_risk_level(trace_with_risk(80)) == RiskLevel.HIGH


def test_classify_recommendation_level_is_deterministic_and_reproducible():
    row = _row(trend_score=9.0, momentum_score=9.0)
    ev = _evidence(statistical_evidence=[{"win_rate_shrunk": 0.8}], best_pattern_similarity_pct=90.0)
    trace1 = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    conf1 = compute_confidence_breakdown(trace1)
    level1 = classify_recommendation_level(trace1, conf1)

    trace2 = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    conf2 = compute_confidence_breakdown(trace2)
    level2 = classify_recommendation_level(trace2, conf2)

    assert level1 == level2  # same inputs -> same output, every time


def test_classify_recommendation_level_high_risk_forces_avoid():
    row = _row(is_uma=True, quality_penalty_total=20.0, atr_pct=30.0)  # drives risk_score >= 80
    ev = _evidence(statistical_evidence=[{"win_rate_shrunk": 0.9}], best_pattern_similarity_pct=100.0)
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    assert trace.risk_score >= 80.0
    confidence = compute_confidence_breakdown(trace)
    assert classify_recommendation_level(trace, confidence) == RecommendationLevel.AVOID


def test_classify_recommendation_level_strong_setup_yields_strong_buy():
    row = _row(trend_score=10.0, momentum_score=10.0, is_uma=False, quality_penalty_total=0.0, atr_pct=0.0)
    ev = _evidence(statistical_evidence=[{"win_rate_shrunk": 0.9}], best_pattern_similarity_pct=100.0)
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    confidence = compute_confidence_breakdown(trace)
    assert classify_recommendation_level(trace, confidence) == RecommendationLevel.STRONG_BUY


def test_classify_recommendation_level_weak_setup_yields_avoid():
    row = _row(trend_score=0.0, momentum_score=0.0, is_uma=False, quality_penalty_total=0.0, atr_pct=0.0)
    ev = _evidence()
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    confidence = compute_confidence_breakdown(trace)
    assert classify_recommendation_level(trace, confidence) == RecommendationLevel.AVOID


# ---------------------------------------------------------------------------
# compute_expected_return — always None (no return-magnitude data available)
# ---------------------------------------------------------------------------

def test_expected_return_always_none_even_with_rich_evidence():
    ev = _evidence(statistical_evidence=[{"win_rate_shrunk": 0.9, "n": 500}])
    assert compute_expected_return(ev) is None


def test_expected_return_none_with_no_evidence():
    assert compute_expected_return(_evidence()) is None


# ---------------------------------------------------------------------------
# compute_historical_comparison
# ---------------------------------------------------------------------------

def test_historical_comparison_no_data_when_no_evidence():
    row = _row()
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, _evidence())
    comparison = compute_historical_comparison(_evidence(), trace)
    assert comparison.verdict == HistoricalComparisonVerdict.NO_DATA
    assert comparison.sample_size is None


def test_historical_comparison_carries_real_stats_not_fabricated():
    ev = _evidence(
        statistical_evidence=[{"n": 115, "n_success": 15, "win_rate": 0.13, "win_rate_shrunk": 0.1166,
                               "ci_lower": 0.081, "ci_upper": 0.187}],
        similar_patterns=["ma_full_alignment=True, atr_breakout=True"],
        best_pattern_similarity_pct=100.0,
    )
    row = _row(trend_score=1.0, momentum_score=1.0)
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    comparison = compute_historical_comparison(ev, trace)
    assert comparison.sample_size == 115
    assert comparison.win_rate == 0.13
    assert comparison.ci_lower == 0.081
    assert comparison.ci_upper == 0.187
    assert comparison.pattern_description == "ma_full_alignment=True, atr_breakout=True"


def test_historical_comparison_verdict_stronger_when_technical_and_similarity_high():
    ev = _evidence(statistical_evidence=[{"win_rate_shrunk": 0.5}], best_pattern_similarity_pct=90.0)
    row = _row(trend_score=9.0, momentum_score=9.0)
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    comparison = compute_historical_comparison(ev, trace)
    assert comparison.verdict == HistoricalComparisonVerdict.STRONGER


def test_historical_comparison_verdict_weaker_when_technical_low():
    ev = _evidence(statistical_evidence=[{"win_rate_shrunk": 0.5}], best_pattern_similarity_pct=100.0)
    row = _row(trend_score=0.0, momentum_score=0.0)
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    comparison = compute_historical_comparison(ev, trace)
    assert comparison.verdict == HistoricalComparisonVerdict.WEAKER


# ---------------------------------------------------------------------------
# generate_evidence_highlights — grounded-only, no fabrication
# ---------------------------------------------------------------------------

def test_highlights_only_reference_supplied_indicator_values():
    row = _row()
    ev = _evidence(technical_indicators={"ma_full_alignment": True, "adx": 45.0, "golden_cross": False})
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    highlights = generate_evidence_highlights(row, ev, trace)
    joined = " ".join(highlights["strengths"] + highlights["weaknesses"])
    assert "45.0" in joined  # the real ADX value appears, not an invented one
    assert any("golden cross" in w.lower() for w in highlights["weaknesses"])


def test_highlights_flag_low_win_rate_as_weakness_and_risk():
    row = _row()
    ev = _evidence(statistical_evidence=[{"win_rate": 0.13, "win_rate_shrunk": 0.1166}])
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    highlights = generate_evidence_highlights(row, ev, trace)
    assert any("13%" in w for w in highlights["weaknesses"])
    assert any("follow-through" in r for r in highlights["risks"])


def test_highlights_flag_uma_as_risk():
    row = _row(is_uma=True)
    ev = _evidence()
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    highlights = generate_evidence_highlights(row, ev, trace)
    assert any("UMA" in r for r in highlights["risks"])


def test_highlights_capped_at_five_each():
    row = _row(is_uma=True, is_special_monitoring=True)
    ev = _evidence(
        technical_indicators={"ma_full_alignment": True, "adx": 45.0, "golden_cross": True,
                              "roc5": 5.0, "roc20": 10.0, "atr_breakout": True, "squeeze_release": True,
                              "vol_spike": True, "rsi14": 20.0},
        statistical_evidence=[{"win_rate": 0.6}],
    )
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    highlights = generate_evidence_highlights(row, ev, trace)
    assert len(highlights["strengths"]) <= 5
    assert len(highlights["weaknesses"]) <= 5
    assert len(highlights["risks"]) <= 5


def test_highlights_no_statistical_evidence_flags_weakness_and_risk():
    row = _row()
    ev = _evidence()
    trace = compute_decision_trace(AI_MODEL_REGISTRY["momentum_ai"], row, ev)
    highlights = generate_evidence_highlights(row, ev, trace)
    assert any("no statistically validated" in w.lower() for w in highlights["weaknesses"])
    assert any("no historical track record" in r.lower() for r in highlights["risks"])

"""Tests for AI Lab agents: evidence building (hypothesis_agent) and
recommendation assembly (decision_agent)."""
import asyncio
import json

import pandas as pd

from stock_scanner.ai_lab.agents.decision_agent import (
    assemble_recommendation,
    generate_decision,
    recommendation_id,
)
from stock_scanner.ai_lab.agents.hypothesis_agent import build_evidence, generate_hypothesis
from stock_scanner.ai_lab.client import MockNineRouterClient
from stock_scanner.ai_lab.models import AI_MODEL_REGISTRY
from stock_scanner.ai_lab.schemas import RecommendationStatus
from stock_scanner.ai_lab.scoring import (
    compute_confidence_breakdown,
    compute_decision_trace,
    compute_historical_comparison,
    generate_evidence_highlights,
)


def _feature_row(**overrides) -> pd.Series:
    base = {
        "ticker": "BBCA.JK", "ma_full_alignment": True, "ma_partial_alignment": False,
        "slope_ma20": 0.5, "roc5": 2.1, "roc20": 5.0, "adx": 30.0, "adx_pos": 25.0,
        "adx_neg": 10.0, "golden_cross": True, "rsi14": 55.0,
        "trend_score": 7.0, "momentum_score": 6.0, "quality_penalty_total": 0.0,
    }
    base.update(overrides)
    return pd.Series(base)


def _kb_df(*pattern_dicts) -> pd.DataFrame:
    rows = []
    for i, pattern in enumerate(pattern_dicts):
        rows.append({"hypothesis_id": f"h{i}", "pattern_json": json.dumps({"representative": pattern})})
    return pd.DataFrame(rows)


def _full_pipeline_precompute(model_spec, feature_row, evidence):
    trace = compute_decision_trace(model_spec, feature_row, evidence)
    confidence = compute_confidence_breakdown(trace)
    comparison = compute_historical_comparison(evidence, trace)
    highlights = generate_evidence_highlights(feature_row, evidence, trace)
    return trace, confidence, comparison, highlights


# ---------------------------------------------------------------------------
# build_evidence
# ---------------------------------------------------------------------------

def test_build_evidence_restricts_to_focus_features():
    row = _feature_row()
    model_spec = AI_MODEL_REGISTRY["momentum_ai"]
    evidence = build_evidence("BBCA.JK", row, None, model_spec)
    assert set(evidence.technical_indicators) == set(model_spec.focus_features)
    assert evidence.technical_indicators["ma_full_alignment"] is True


def test_build_evidence_matches_pattern_with_equal_slice():
    row = _feature_row(ma_full_alignment=True, golden_cross=True)
    kb_df = _kb_df({
        "slice_definition": {"ma_full_alignment": True, "golden_cross": True},
        "n": 40, "n_success": 28, "win_rate": 0.7, "win_rate_shrunk": 0.65,
        "ci_lower": 0.55, "ci_upper": 0.75, "p_value_adjusted": 0.01,
    })
    evidence = build_evidence("BBCA.JK", row, kb_df, AI_MODEL_REGISTRY["momentum_ai"])
    assert len(evidence.statistical_evidence) == 1
    assert evidence.statistical_evidence[0]["win_rate"] == 0.7
    assert evidence.statistical_evidence[0]["ci_upper"] == 0.75
    assert "ma_full_alignment=True" in evidence.similar_patterns[0]
    assert evidence.best_pattern_similarity_pct == 100.0


def test_build_evidence_skips_non_matching_pattern_but_tracks_partial_similarity():
    row = _feature_row(ma_full_alignment=False, golden_cross=True)
    kb_df = _kb_df({
        "slice_definition": {"ma_full_alignment": True, "golden_cross": True},
        "n": 10, "n_success": 8, "win_rate": 0.8, "win_rate_shrunk": 0.7,
        "ci_lower": 0.5, "ci_upper": 0.9, "p_value_adjusted": 0.02,
    })
    evidence = build_evidence("BBCA.JK", row, kb_df, AI_MODEL_REGISTRY["momentum_ai"])
    # Not an exact match (ma_full_alignment differs) — excluded from the
    # "confirmed" evidence lists...
    assert evidence.statistical_evidence == []
    assert evidence.similar_patterns == []
    # ...but the partial match (1 of 2 keys) is still tracked continuously.
    assert evidence.best_pattern_similarity_pct == 50.0


def test_build_evidence_handles_empty_knowledge_base():
    row = _feature_row()
    evidence = build_evidence("BBCA.JK", row, pd.DataFrame(), AI_MODEL_REGISTRY["momentum_ai"])
    assert evidence.statistical_evidence == []
    assert evidence.best_pattern_similarity_pct == 0.0


def test_build_evidence_handles_malformed_pattern_json_gracefully():
    row = _feature_row()
    kb_df = pd.DataFrame([{"hypothesis_id": "h1", "pattern_json": "not valid json"}])
    evidence = build_evidence("BBCA.JK", row, kb_df, AI_MODEL_REGISTRY["momentum_ai"])
    assert evidence.statistical_evidence == []


# ---------------------------------------------------------------------------
# generate_hypothesis / generate_decision (via MockNineRouterClient)
# ---------------------------------------------------------------------------

def test_generate_hypothesis_returns_output_from_mock_client():
    client = MockNineRouterClient()
    model_spec = AI_MODEL_REGISTRY["momentum_ai"]
    row = _feature_row()
    evidence = build_evidence("BBCA.JK", row, None, model_spec)
    _, _, _, highlights = _full_pipeline_precompute(model_spec, row, evidence)
    result = asyncio.run(generate_hypothesis(client, evidence, model_spec, highlights))
    assert result is not None
    assert result.why


def test_generate_decision_returns_output_from_mock_client():
    client = MockNineRouterClient()
    model_spec = AI_MODEL_REGISTRY["momentum_ai"]
    row = _feature_row()
    evidence = build_evidence("BBCA.JK", row, None, model_spec)
    trace, confidence, comparison, highlights = _full_pipeline_precompute(model_spec, row, evidence)
    hypothesis = asyncio.run(generate_hypothesis(client, evidence, model_spec, highlights))
    decision = asyncio.run(generate_decision(client, evidence, hypothesis, model_spec, trace, confidence, comparison))
    assert decision is not None


# ---------------------------------------------------------------------------
# recommendation_id / assemble_recommendation
# ---------------------------------------------------------------------------

def test_recommendation_id_deterministic():
    id1 = recommendation_id("BBCA.JK", "momentum_ai", "2026-07-14")
    id2 = recommendation_id("BBCA.JK", "momentum_ai", "2026-07-14")
    assert id1 == id2


def test_recommendation_id_differs_by_model():
    id1 = recommendation_id("BBCA.JK", "momentum_ai", "2026-07-14")
    id2 = recommendation_id("BBCA.JK", "breakout_ai", "2026-07-14")
    assert id1 != id2


def test_assemble_recommendation_full_pipeline():
    client = MockNineRouterClient()
    model_spec = AI_MODEL_REGISTRY["momentum_ai"]
    row = _feature_row()
    evidence = build_evidence("BBCA.JK", row, None, model_spec)
    trace, confidence, comparison, highlights = _full_pipeline_precompute(model_spec, row, evidence)
    hypothesis = asyncio.run(generate_hypothesis(client, evidence, model_spec, highlights))
    decision = asyncio.run(generate_decision(client, evidence, hypothesis, model_spec, trace, confidence, comparison))

    rec = assemble_recommendation(
        evidence, hypothesis, decision, model_spec, "2026-07-14", "mock",
        trace, confidence, comparison, highlights,
    )

    assert rec.ticker == "BBCA.JK"
    assert rec.ai_model == "momentum_ai"
    assert rec.status == RecommendationStatus.PENDING
    assert rec.model == "mock"
    assert rec.reasoning["why"] == hypothesis.why
    assert rec.reasoning["technical_indicators"] == evidence.technical_indicators
    assert rec.id == recommendation_id("BBCA.JK", "momentum_ai", "2026-07-14")
    # Backward-compat invariants: score/confidence always mirror the trace/breakdown.
    assert rec.score == trace.final_score
    assert rec.confidence == confidence.final_confidence
    assert rec.decision_trace == trace
    assert rec.confidence_breakdown == confidence
    assert rec.expected_return is None  # never estimated — see scoring.compute_expected_return


def test_assemble_recommendation_falls_back_to_highlights_when_llm_returns_empty_lists():
    from stock_scanner.ai_lab.schemas import HypothesisOutput

    client = MockNineRouterClient()
    model_spec = AI_MODEL_REGISTRY["momentum_ai"]
    row = _feature_row()
    evidence = build_evidence("BBCA.JK", row, None, model_spec)
    trace, confidence, comparison, highlights = _full_pipeline_precompute(model_spec, row, evidence)
    assert highlights["strengths"]  # sanity: this feature_row does generate candidates

    empty_hypothesis = HypothesisOutput(why="test", strengths=[], weaknesses=[], risks=[])
    decision = asyncio.run(generate_decision(client, evidence, empty_hypothesis, model_spec, trace, confidence, comparison))

    rec = assemble_recommendation(
        evidence, empty_hypothesis, decision, model_spec, "2026-07-14", "mock",
        trace, confidence, comparison, highlights,
    )
    assert rec.reasoning["strengths"] == highlights["strengths"]

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


def _feature_row(**overrides) -> pd.Series:
    base = {
        "ticker": "BBCA.JK", "ma_full_alignment": True, "ma_partial_alignment": False,
        "slope_ma20": 0.5, "roc5": 2.1, "roc20": 5.0, "adx": 30.0, "adx_pos": 25.0,
        "adx_neg": 10.0, "golden_cross": True, "rsi14": 55.0,
    }
    base.update(overrides)
    return pd.Series(base)


def _kb_df(*pattern_dicts) -> pd.DataFrame:
    rows = []
    for i, pattern in enumerate(pattern_dicts):
        rows.append({"hypothesis_id": f"h{i}", "pattern_json": json.dumps({"representative": pattern})})
    return pd.DataFrame(rows)


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
        "ci_lower": 0.55, "p_value_adjusted": 0.01,
    })
    evidence = build_evidence("BBCA.JK", row, kb_df, AI_MODEL_REGISTRY["momentum_ai"])
    assert len(evidence.statistical_evidence) == 1
    assert evidence.statistical_evidence[0]["win_rate"] == 0.7
    assert "ma_full_alignment=True" in evidence.similar_patterns[0]


def test_build_evidence_skips_non_matching_pattern():
    row = _feature_row(ma_full_alignment=False)
    kb_df = _kb_df({
        "slice_definition": {"ma_full_alignment": True},
        "n": 10, "n_success": 8, "win_rate": 0.8, "win_rate_shrunk": 0.7,
        "ci_lower": 0.5, "p_value_adjusted": 0.02,
    })
    evidence = build_evidence("BBCA.JK", row, kb_df, AI_MODEL_REGISTRY["momentum_ai"])
    assert evidence.statistical_evidence == []
    assert evidence.similar_patterns == []


def test_build_evidence_handles_empty_knowledge_base():
    row = _feature_row()
    evidence = build_evidence("BBCA.JK", row, pd.DataFrame(), AI_MODEL_REGISTRY["momentum_ai"])
    assert evidence.statistical_evidence == []


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
    evidence = build_evidence("BBCA.JK", _feature_row(), None, AI_MODEL_REGISTRY["momentum_ai"])
    result = asyncio.run(generate_hypothesis(client, evidence, AI_MODEL_REGISTRY["momentum_ai"]))
    assert result is not None
    assert 0.0 <= result.confidence <= 1.0


def test_generate_decision_returns_output_from_mock_client():
    client = MockNineRouterClient()
    evidence = build_evidence("BBCA.JK", _feature_row(), None, AI_MODEL_REGISTRY["momentum_ai"])
    hypothesis = asyncio.run(generate_hypothesis(client, evidence, AI_MODEL_REGISTRY["momentum_ai"]))
    decision = asyncio.run(generate_decision(client, evidence, hypothesis, AI_MODEL_REGISTRY["momentum_ai"]))
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
    evidence = build_evidence("BBCA.JK", _feature_row(), None, model_spec)
    hypothesis = asyncio.run(generate_hypothesis(client, evidence, model_spec))
    decision = asyncio.run(generate_decision(client, evidence, hypothesis, model_spec))

    rec = assemble_recommendation(evidence, hypothesis, decision, model_spec, "2026-07-14", "mock")

    assert rec.ticker == "BBCA.JK"
    assert rec.ai_model == "momentum_ai"
    assert rec.status == RecommendationStatus.PENDING
    assert rec.model == "mock"
    assert rec.reasoning["why"] == hypothesis.why
    assert rec.reasoning["technical_indicators"] == evidence.technical_indicators
    assert rec.id == recommendation_id("BBCA.JK", "momentum_ai", "2026-07-14")

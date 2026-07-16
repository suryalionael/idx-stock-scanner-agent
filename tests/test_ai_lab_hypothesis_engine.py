"""Tests for stock_scanner/ai_lab/hypothesis_engine.py — pure candidate
generation, no stats testing, no DB/LLM involved. Mirrors
tests/test_ai_lab_reflection_engine.py's convention of engineering a known
synthetic scenario and asserting the exact expected outcome."""
import json

import pandas as pd
import pytest

from stock_scanner.ai_lab.hypothesis_engine import generate_candidate_hypotheses
from stock_scanner.ai_lab.reflection_engine import generate_observations


def _row(
    id_, ticker, ai_model, trade_outcome, *, confidence=0.5, recommendation="BUY",
    verdict="similar", golden_cross=True, ma_full_alignment=True, rsi14=50.0,
    return_percentage=1.0, holding_days=3,
) -> dict:
    return dict(
        id=id_, ticker=ticker, ai_model=ai_model, confidence=confidence, recommendation=recommendation,
        reasoning=json.dumps({"technical_indicators": {
            "golden_cross": golden_cross, "ma_full_alignment": ma_full_alignment, "rsi14": rsi14,
        }}),
        historical_comparison=json.dumps({"verdict": verdict}),
        trade_outcome=trade_outcome, return_percentage=return_percentage, holding_days=holding_days,
    )


def _mixed_population() -> pd.DataFrame:
    """momentum_ai: strong success pattern with high RSI; breakout_ai:
    strong failure pattern with low RSI — same fixture shape used during
    implementation, extended with a numeric indicator spread so tercile
    bucketing has something real to detect."""
    rows = []
    for i in range(20):
        rows.append(_row(
            f"a{i}", "BBCA.JK", "momentum_ai", "WIN" if i < 16 else "LOSS",
            recommendation="BUY", verdict="stronger", golden_cross=True, ma_full_alignment=True,
            rsi14=70.0 + i,
        ))
    for i in range(20):
        rows.append(_row(
            f"b{i}", "TLKM.JK", "breakout_ai", "WIN" if i < 4 else "LOSS",
            recommendation="WATCH", verdict="weaker", golden_cross=False, ma_full_alignment=False,
            rsi14=10.0 + i,
        ))
    return pd.DataFrame(rows)


def _reflection_seeds(df: pd.DataFrame, **kwargs) -> list[dict]:
    obs = generate_observations(df, min_n_success=kwargs.get("min_n_success", 3), alpha=kwargs.get("alpha", 0.05))
    return [o.model_dump() for o in obs]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_dataframe_returns_no_candidates():
    df = _mixed_population()
    seeds = _reflection_seeds(df)
    assert generate_candidate_hypotheses(pd.DataFrame(), seeds) == []


def test_no_reflection_observations_returns_no_candidates():
    df = _mixed_population()
    assert generate_candidate_hypotheses(df, []) == []


# ---------------------------------------------------------------------------
# Seeding from each reflection category
# ---------------------------------------------------------------------------

def test_seeds_from_categorical_and_technical_pattern_observations():
    df = _mixed_population()
    seeds = _reflection_seeds(df)
    categories_seen = {s["category"] for s in seeds}
    # Sanity: the fixture should produce at least model/sector/recommendation/
    # verdict/technical_pattern observations to seed from (same as the
    # reflection engine's own test suite already proves).
    assert "model_performance" in categories_seen
    assert "technical_pattern" in categories_seen

    candidates = generate_candidate_hypotheses(df, seeds, max_order=3)
    assert candidates  # some candidates must be generated from real seeds
    orders = {c.order for c in candidates}
    assert orders <= {2, 3}
    # every candidate must trace back to at least one real seed observation
    assert all(c.source_reflection_ids for c in candidates)


def test_confidence_calibration_observations_are_never_seeds():
    """CONFIDENCE_CALIBRATION describes confidence buckets, not a
    recommendation-attribute condition — must be excluded from seeding."""
    from stock_scanner.ai_lab.hypothesis_engine import _seed_from_reflection

    fake_obs = [{
        "category": "confidence_calibration", "observation_id": "obsX",
        "supporting_statistics": {"dimension": "confidence_bucket", "value": "(0.5, 0.75]"},
    }]
    assert _seed_from_reflection(fake_obs) == []


# ---------------------------------------------------------------------------
# Numeric bucketing
# ---------------------------------------------------------------------------

def test_numeric_indicator_bucketed_and_discoverable():
    df = _mixed_population()
    seeds = _reflection_seeds(df)
    candidates = generate_candidate_hypotheses(df, seeds, max_order=3)
    rsi_conditions = {
        tuple(cond) for c in candidates for cond in c.conditions if cond[0] == "rsi14"
    }
    assert rsi_conditions, "expected at least one rsi14=<bucket> condition among candidates"
    assert rsi_conditions <= {("rsi14", "Low"), ("rsi14", "Mid"), ("rsi14", "High")}


# ---------------------------------------------------------------------------
# Duplicate prevention
# ---------------------------------------------------------------------------

def test_no_duplicate_condition_sets_generated():
    df = _mixed_population()
    seeds = _reflection_seeds(df)
    candidates = generate_candidate_hypotheses(df, seeds, max_order=3)
    seen = set()
    for c in candidates:
        key = frozenset(tuple(cond) for cond in c.conditions)
        assert key not in seen, f"duplicate condition set generated: {c.conditions}"
        seen.add(key)


# ---------------------------------------------------------------------------
# Order cap
# ---------------------------------------------------------------------------

def test_order_never_exceeds_max_order():
    df = _mixed_population()
    seeds = _reflection_seeds(df)
    for max_order in (2, 3):
        candidates = generate_candidate_hypotheses(df, seeds, max_order=max_order)
        assert all(c.order <= max_order for c in candidates)
        assert all(c.order >= 2 for c in candidates)  # order-1 is Reflection's own job, never re-emitted


def test_order_3_candidates_exist_when_max_order_3():
    df = _mixed_population()
    seeds = _reflection_seeds(df)
    candidates = generate_candidate_hypotheses(df, seeds, max_order=3)
    assert any(c.order == 3 for c in candidates)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_candidate_generation_is_deterministic():
    df = _mixed_population()
    seeds = _reflection_seeds(df)
    first = generate_candidate_hypotheses(df, seeds, max_order=3)
    second = generate_candidate_hypotheses(df, seeds, max_order=3)
    assert [c.conditions for c in first] == [c.conditions for c in second]
    assert [c.n for c in first] == [c.n for c in second]


def test_candidates_are_sorted_by_order_then_conditions():
    df = _mixed_population()
    seeds = _reflection_seeds(df)
    candidates = generate_candidate_hypotheses(df, seeds, max_order=3)
    keys = [(c.order, c.conditions) for c in candidates]
    assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Basic counts are real, not invented
# ---------------------------------------------------------------------------

def test_candidate_counts_match_real_data():
    df = _mixed_population()
    seeds = _reflection_seeds(df)
    candidates = generate_candidate_hypotheses(df, seeds, max_order=3)
    momentum_buy = next(
        c for c in candidates
        if sorted(c.conditions) == sorted([["ai_model", "momentum_ai"], ["recommendation", "BUY"]])
    )
    assert momentum_buy.n == 20
    assert momentum_buy.n_success == 16
    assert momentum_buy.n_failure == 4
    assert momentum_buy.win_rate == pytest.approx(0.8)

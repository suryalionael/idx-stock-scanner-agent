"""Tests for stock_scanner/pipeline/knowledge_application.py — Knowledge
Application Engine v1. Pure function, no LLM/SQLite involved; fixtures
mirror data/published/knowledge_report.json's "entries" shape directly.
"""
import json

import pandas as pd
import pytest

from stock_scanner.ai_lab.schemas import is_entry_promoted
from stock_scanner.pipeline.knowledge_application import (
    _condition_matches,
    _condition_supported,
    _entry_bonus,
    apply_knowledge_ranking,
    filter_applicable_entries,
)


_UNSET = object()


def _entry(
    knowledge_id, conditions, lifecycle_status="strong", shrunk_win_rate=0.70,
    promotion_status="promoted",
):
    # promotion_status defaults to "promoted" here so every pre-existing
    # test below (written before the promotion gate existed) keeps
    # isolating the gate it was actually testing — lifecycle/dimension
    # filtering, matching, scoring — rather than incidentally also
    # exercising the promotion gate. The promotion gate itself gets its
    # own dedicated section further down, which explicitly overrides this
    # (including passing promotion_status=_UNSET to simulate a real
    # pre-this-change knowledge_report.json row that has no such key).
    entry = {
        "knowledge_id": knowledge_id,
        "conditions": conditions,
        "lifecycle_status": lifecycle_status,
        "shrunk_win_rate": shrunk_win_rate,
    }
    if promotion_status is not _UNSET:
        entry["promotion_status"] = promotion_status
    return entry


def _candidates_df(**overrides) -> pd.DataFrame:
    df = pd.DataFrame({
        "ticker": ["AAAA.JK", "BBBB.JK", "CCCC.JK"],
        "signal": ["BREAKOUT", "PRE_MARKUP", "WATCH"],
        "final_status": ["eligible", "eligible", "eligible"],
        "total_score": [8.0, 6.5, 5.2],
        "vol_spike": [True, False, True],
        "ma_full_alignment": [True, True, False],
    })
    for k, v in overrides.items():
        df[k] = v
    return df


# ---------------------------------------------------------------------------
# Condition classification
# ---------------------------------------------------------------------------

def test_boolean_condition_supported():
    assert _condition_supported("vol_spike", "True") is True
    assert _condition_supported("vol_spike", "False") is True


def test_sector_condition_supported():
    assert _condition_supported("sector", "Technology") is True


def test_numeric_tercile_condition_unsupported():
    assert _condition_supported("rsi14", "High") is False
    assert _condition_supported("atr_pct", "Low") is False


def test_ai_only_dimension_unsupported():
    assert _condition_supported("ai_model", "momentum_ai") is False
    assert _condition_supported("recommendation", "BUY") is False
    assert _condition_supported("historical_verdict", "stronger") is False


def test_unknown_value_shape_unsupported_by_default():
    assert _condition_supported("some_future_dimension", "Weird") is False


# ---------------------------------------------------------------------------
# Entry filtering
# ---------------------------------------------------------------------------

def test_filter_drops_emerging_and_weakening_lifecycle():
    entries = [
        _entry("k1", [["vol_spike", "True"]], lifecycle_status="emerging"),
        _entry("k2", [["vol_spike", "True"]], lifecycle_status="weakening"),
        _entry("k3", [["vol_spike", "True"]], lifecycle_status="contradicted"),
        _entry("k4", [["vol_spike", "True"]], lifecycle_status="archived"),
    ]
    assert filter_applicable_entries(entries) == []


def test_filter_keeps_confirmed_and_strong():
    entries = [
        _entry("k1", [["vol_spike", "True"]], lifecycle_status="confirmed"),
        _entry("k2", [["vol_spike", "True"]], lifecycle_status="strong"),
    ]
    kept_ids = {e["knowledge_id"] for e in filter_applicable_entries(entries)}
    assert kept_ids == {"k1", "k2"}


def test_filter_drops_whole_entry_when_any_condition_unsupported():
    # Mixed entry: one supported boolean condition + one unsupported numeric
    # tercile condition. The whole entry must be dropped, not partially
    # applied — applying only the boolean half would extrapolate beyond
    # what the underlying statistics actually validated jointly.
    entries = [
        _entry("k1", [["vol_spike", "True"], ["rsi14", "High"]], lifecycle_status="strong"),
    ]
    assert filter_applicable_entries(entries) == []


def test_filter_drops_entry_with_ai_only_dimension():
    entries = [
        _entry("k1", [["vol_spike", "True"], ["ai_model", "momentum_ai"]], lifecycle_status="strong"),
    ]
    assert filter_applicable_entries(entries) == []


def test_filter_drops_entry_with_no_conditions():
    entries = [_entry("k1", [], lifecycle_status="strong")]
    assert filter_applicable_entries(entries) == []


# ---------------------------------------------------------------------------
# Promotion gate — orthogonal to lifecycle_status, fail-closed by default.
# A statistically confirmed/strong entry must still be excluded unless
# promotion_status is exactly "promoted".
# ---------------------------------------------------------------------------

def test_is_promoted_true_only_for_exact_promoted_string():
    assert is_entry_promoted({"promotion_status": "promoted"}) is True


@pytest.mark.parametrize("promotion_status", [
    "candidate",       # statistically confirmed but never reviewed
    "rejected",        # a human explicitly said no
    "archived",        # was promoted once, no longer trusted
    "PROMOTED",        # wrong case is not an exact match — never guess
    "approved",        # a plausible-sounding but wrong value
    "",                # empty string
    None,              # explicit null
])
def test_is_promoted_false_for_anything_other_than_exact_promoted(promotion_status):
    assert is_entry_promoted({"promotion_status": promotion_status}) is False


def test_is_promoted_false_when_key_entirely_missing():
    # Simulates a real pre-this-change knowledge_report.json row, which
    # has no promotion_status key at all.
    assert is_entry_promoted({}) is False


def test_filter_drops_confirmed_strong_entries_that_are_not_promoted():
    entries = [
        _entry("k1", [["vol_spike", "True"]], lifecycle_status="strong", promotion_status="candidate"),
        _entry("k2", [["vol_spike", "True"]], lifecycle_status="confirmed", promotion_status="rejected"),
        _entry("k3", [["vol_spike", "True"]], lifecycle_status="strong", promotion_status=_UNSET),
    ]
    assert filter_applicable_entries(entries) == []


def test_filter_keeps_only_the_promoted_entry_among_mixed_statuses():
    entries = [
        _entry("k1", [["vol_spike", "True"]], lifecycle_status="strong", promotion_status="candidate"),
        _entry("k2", [["vol_spike", "True"]], lifecycle_status="strong", promotion_status="promoted"),
        _entry("k3", [["vol_spike", "True"]], lifecycle_status="strong", promotion_status="rejected"),
    ]
    kept_ids = {e["knowledge_id"] for e in filter_applicable_entries(entries)}
    assert kept_ids == {"k2"}


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def test_boolean_condition_matches_true_and_false():
    row = pd.Series({"vol_spike": True})
    assert _condition_matches("vol_spike", "True", row, sector="") is True
    assert _condition_matches("vol_spike", "False", row, sector="") is False

    row_false = pd.Series({"vol_spike": False})
    assert _condition_matches("vol_spike", "False", row_false, sector="") is True


def test_missing_or_nan_column_never_matches():
    row = pd.Series({"other_col": True})
    assert _condition_matches("vol_spike", "True", row, sector="") is False

    row_nan = pd.Series({"vol_spike": float("nan")})
    assert _condition_matches("vol_spike", "True", row_nan, sector="") is False


def test_sector_condition_matches_exact_string():
    row = pd.Series({"vol_spike": True})
    assert _condition_matches("sector", "Technology", row, sector="Technology") is True
    assert _condition_matches("sector", "Technology", row, sector="Energy") is False


# ---------------------------------------------------------------------------
# Bonus formula
# ---------------------------------------------------------------------------

def test_strong_entry_gets_full_weight_confirmed_gets_half():
    strong = _entry("k1", [["vol_spike", "True"]], lifecycle_status="strong", shrunk_win_rate=0.70)
    confirmed = _entry("k2", [["vol_spike", "True"]], lifecycle_status="confirmed", shrunk_win_rate=0.70)
    strong_bonus = _entry_bonus(strong, bonus_scale=1.0)
    confirmed_bonus = _entry_bonus(confirmed, bonus_scale=1.0)
    assert strong_bonus == pytest.approx(0.20)
    assert confirmed_bonus == pytest.approx(0.10)
    assert confirmed_bonus == pytest.approx(strong_bonus / 2)


def test_win_rate_below_half_gives_negative_bonus():
    losing = _entry("k1", [["vol_spike", "True"]], lifecycle_status="strong", shrunk_win_rate=0.30)
    assert _entry_bonus(losing, bonus_scale=1.0) == pytest.approx(-0.20)


# ---------------------------------------------------------------------------
# apply_knowledge_ranking — the byte-identical guarantee
# ---------------------------------------------------------------------------

def test_byte_identical_when_knowledge_file_missing(tmp_path):
    df = _candidates_df()
    result = apply_knowledge_ranking(df, knowledge_path=tmp_path / "does_not_exist.json")
    assert result is df  # same object — not even a copy was made
    assert "knowledge_bonus" not in result.columns


def test_byte_identical_when_disabled(tmp_path):
    knowledge_path = tmp_path / "knowledge_report.json"
    knowledge_path.write_text(json.dumps({"entries": [
        _entry("k1", [["vol_spike", "True"]]),
    ]}))
    df = _candidates_df()
    result = apply_knowledge_ranking(df, knowledge_path=knowledge_path, config={"enabled": False})
    assert result is df


def test_byte_identical_when_no_entries_applicable(tmp_path):
    knowledge_path = tmp_path / "knowledge_report.json"
    knowledge_path.write_text(json.dumps({"entries": [
        _entry("k1", [["rsi14", "High"]]),  # numeric — unsupported
        _entry("k2", [["vol_spike", "True"]], lifecycle_status="emerging"),  # wrong lifecycle
        _entry("k3", [["vol_spike", "True"]], promotion_status="candidate"),  # statistically fine, not promoted
        _entry("k4", [["vol_spike", "True"]], promotion_status=_UNSET),  # pre-this-change row shape
    ]}))
    df = _candidates_df()
    result = apply_knowledge_ranking(df, knowledge_path=knowledge_path)
    assert result is df


def test_byte_identical_end_to_end_for_confirmed_but_unpromoted_knowledge(tmp_path):
    # An entry that is statistically confirmed, condition-supported, and
    # would fully match candidate 0 and 2 (vol_spike=True) — the ONLY
    # thing wrong with it is that no human has promoted it yet. Ranking
    # must be completely untouched: this is the exact scenario the
    # promotion gate exists to guard against — "confirmed" must never be
    # read as "safe to deploy."
    knowledge_path = tmp_path / "knowledge_report.json"
    knowledge_path.write_text(json.dumps({"entries": [
        _entry("k1", [["vol_spike", "True"]], lifecycle_status="strong",
               shrunk_win_rate=0.95, promotion_status="candidate"),
    ]}))
    df = _candidates_df()
    result = apply_knowledge_ranking(df, knowledge_path=knowledge_path)
    assert result is df
    assert "knowledge_bonus" not in result.columns


def test_byte_identical_when_no_base_score_column():
    df = pd.DataFrame({"ticker": ["AAAA.JK"], "signal": ["BREAKOUT"]})  # no total_score etc.
    result = apply_knowledge_ranking(df, knowledge_path=None)
    assert result is df


def test_malformed_knowledge_file_is_a_no_op(tmp_path):
    knowledge_path = tmp_path / "knowledge_report.json"
    knowledge_path.write_text("{not valid json")
    df = _candidates_df()
    result = apply_knowledge_ranking(df, knowledge_path=knowledge_path)
    assert result is df


# ---------------------------------------------------------------------------
# apply_knowledge_ranking — real application
# ---------------------------------------------------------------------------

def test_applies_bonus_only_to_matching_candidates(tmp_path):
    knowledge_path = tmp_path / "knowledge_report.json"
    knowledge_path.write_text(json.dumps({"entries": [
        _entry("k1", [["vol_spike", "True"]], lifecycle_status="strong", shrunk_win_rate=0.70),
    ]}))
    df = _candidates_df()  # vol_spike: [True, False, True]
    result = apply_knowledge_ranking(df, knowledge_path=knowledge_path)

    assert list(result["knowledge_bonus"]) == pytest.approx([0.20, 0.0, 0.20])
    assert list(result["knowledge_adjusted_score"]) == pytest.approx([8.2, 6.5, 5.4])
    assert json.loads(result["knowledge_matched_ids"].iloc[0]) == ["k1"]
    assert json.loads(result["knowledge_matched_ids"].iloc[1]) == []

    rules = json.loads(result["knowledge_applied_rules"].iloc[0])
    assert rules[0]["knowledge_id"] == "k1"
    assert rules[0]["individual_bonus"] == pytest.approx(0.20)


def test_multiple_matches_sum_and_clip_to_max_total_bonus(tmp_path):
    knowledge_path = tmp_path / "knowledge_report.json"
    knowledge_path.write_text(json.dumps({"entries": [
        _entry("k1", [["vol_spike", "True"]], lifecycle_status="strong", shrunk_win_rate=0.95),
        _entry("k2", [["ma_full_alignment", "True"]], lifecycle_status="strong", shrunk_win_rate=0.95),
    ]}))
    df = _candidates_df()  # row 0: vol_spike=True AND ma_full_alignment=True — both match
    result = apply_knowledge_ranking(
        df, knowledge_path=knowledge_path, config={"max_total_bonus": 0.5},
    )
    # Unclipped sum would be ~0.45*2=0.90 — must be clipped to 0.5.
    assert result["knowledge_bonus"].iloc[0] == pytest.approx(0.5)


def test_never_modifies_signal_or_final_status(tmp_path):
    knowledge_path = tmp_path / "knowledge_report.json"
    knowledge_path.write_text(json.dumps({"entries": [
        _entry("k1", [["vol_spike", "True"]], lifecycle_status="strong", shrunk_win_rate=0.90),
    ]}))
    df = _candidates_df()
    original_signal = df["signal"].tolist()
    original_status = df["final_status"].tolist()
    result = apply_knowledge_ranking(df, knowledge_path=knowledge_path)
    assert result["signal"].tolist() == original_signal
    assert result["final_status"].tolist() == original_status
    assert result["total_score"].tolist() == df["total_score"].tolist()  # base column untouched


def test_prefers_quality_adjusted_score_as_base_when_present(tmp_path):
    knowledge_path = tmp_path / "knowledge_report.json"
    knowledge_path.write_text(json.dumps({"entries": [
        _entry("k1", [["vol_spike", "True"]], lifecycle_status="strong", shrunk_win_rate=0.70),
    ]}))
    df = _candidates_df(quality_adjusted_score=[7.5, 6.0, 4.9])
    result = apply_knowledge_ranking(df, knowledge_path=knowledge_path)
    assert list(result["knowledge_adjusted_score"]) == pytest.approx([7.7, 6.0, 5.1])

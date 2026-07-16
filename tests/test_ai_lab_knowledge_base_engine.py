"""Tests for stock_scanner/ai_lab/knowledge_base_engine.py — pure
curation engine, no DB/LLM involved. Mirrors
tests/test_ai_lab_reflection_engine.py's/test_ai_lab_statistical_validation.py's
convention of engineering a known synthetic scenario and asserting the
exact expected outcome."""
from datetime import datetime

import pytest

from stock_scanner.ai_lab import knowledge_base_engine
from stock_scanner.ai_lab.knowledge_base_engine import generate_knowledge_entries
from stock_scanner.ai_lab.schemas import KnowledgeLifecycleStatus, KnowledgePromotionStatus


def _hyp(
    hid, conditions, created_at, status="validated", win_rate=0.8, baseline_rate=0.5,
    sample_size=20, successes=16, evidence_strength="STRONG",
) -> dict:
    return dict(
        hypothesis_id=hid, created_at=created_at, conditions=conditions, status=status,
        win_rate=win_rate, sample_size=sample_size, successes=successes, failures=sample_size - successes,
        shrunk_win_rate=round(win_rate * 0.85, 4), wilson_lower=max(0.0, win_rate - 0.15),
        wilson_upper=min(1.0, win_rate + 0.15),
        evidence_strength=evidence_strength if status == "validated" else None,
        metadata_json={"baseline_rate": baseline_rate},
    )


_TECH_TECH = [["sector", "Technology"], ["rsi14", "High"]]
_ENERGY_BREAKOUT = [["pattern", "Breakout"], ["sector", "Energy"]]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_input_returns_no_entries():
    assert generate_knowledge_entries([]) == []


def test_only_rejected_rows_produce_no_entry():
    rows = [_hyp("r1", _TECH_TECH, "2026-07-10T00:00:00+00:00", status="rejected")]
    assert generate_knowledge_entries(rows) == []


def test_row_missing_baseline_rate_is_skipped_not_crashed_on():
    row = _hyp("m1", _TECH_TECH, "2026-07-10T00:00:00+00:00")
    row["metadata_json"] = {}  # legacy row, no baseline_rate
    assert generate_knowledge_entries([row]) == []


# ---------------------------------------------------------------------------
# Deterministic merging — exact normalized condition-set equality
# ---------------------------------------------------------------------------

def test_same_condition_set_across_runs_merges_into_one_entry():
    rows = [_hyp(f"a{i}", _TECH_TECH, f"2026-07-{10+i:02d}T00:00:00+00:00") for i in range(3)]
    entries = generate_knowledge_entries(rows)
    assert len(entries) == 1
    assert entries[0].confirmation_count == 3


def test_new_entries_always_get_candidate_promotion_status():
    # No automatic promotion exists anywhere in this engine — every entry
    # this engine produces, regardless of lifecycle_status (even STRONG,
    # even with a high confirmation_count), starts at CANDIDATE. Only a
    # future human-run promotion tool may advance it.
    rows = [_hyp(f"a{i}", _TECH_TECH, f"2026-07-{10+i:02d}T00:00:00+00:00") for i in range(6)]
    entries = generate_knowledge_entries(rows)
    assert entries[0].lifecycle_status == KnowledgeLifecycleStatus.STRONG
    assert entries[0].promotion_status == KnowledgePromotionStatus.CANDIDATE


def test_different_condition_set_ordering_still_merges():
    """Defensive: even if two rows' `conditions` lists arrive in a
    different order (hypothesis_engine.py always sorts them, but this
    engine's grouping key must not silently depend on that)."""
    row_a = _hyp("a1", [["sector", "Technology"], ["rsi14", "High"]], "2026-07-10T00:00:00+00:00")
    row_b = _hyp("b1", [["rsi14", "High"], ["sector", "Technology"]], "2026-07-11T00:00:00+00:00")
    entries = generate_knowledge_entries([row_a, row_b])
    assert len(entries) == 1
    assert entries[0].confirmation_count == 2


def test_distinct_condition_sets_produce_distinct_entries():
    rows = [
        _hyp("a1", _TECH_TECH, "2026-07-10T00:00:00+00:00"),
        _hyp("b1", _ENERGY_BREAKOUT, "2026-07-10T00:00:00+00:00"),
    ]
    entries = generate_knowledge_entries(rows)
    assert len(entries) == 2


# ---------------------------------------------------------------------------
# Contradiction handling — spec's own "Energy + Breakout: Positive -> Negative"
# ---------------------------------------------------------------------------

def test_opposite_direction_validation_increments_contradiction_without_flipping_direction():
    rows = [
        _hyp("b1", _ENERGY_BREAKOUT, "2026-07-10T00:00:00+00:00", win_rate=0.75, baseline_rate=0.5),
        _hyp("b2", _ENERGY_BREAKOUT, "2026-07-11T00:00:00+00:00", win_rate=0.20, baseline_rate=0.5),
    ]
    entries = generate_knowledge_entries(rows)
    assert len(entries) == 1
    e = entries[0]
    assert e.confirmation_count == 1
    assert e.contradiction_count == 1
    # cumulative/average stats reflect only the CONFIRMING (first) row, not the contradiction
    assert e.average_win_rate == pytest.approx(0.75)
    assert e.cumulative_successes == 16


def test_contradiction_does_not_move_last_confirmed():
    rows = [
        _hyp("b1", _ENERGY_BREAKOUT, "2026-07-10T00:00:00+00:00", win_rate=0.75, baseline_rate=0.5),
        _hyp("b2", _ENERGY_BREAKOUT, "2026-07-11T00:00:00+00:00", win_rate=0.20, baseline_rate=0.5),
    ]
    entries = generate_knowledge_entries(rows)
    assert entries[0].last_confirmed == "2026-07-10T00:00:00+00:00"


# ---------------------------------------------------------------------------
# Lifecycle ladder — every rung, spec-exact thresholds
# ---------------------------------------------------------------------------

def _run(n_confirm: int, n_contradict: int, strong_threshold=5, archive_margin=3):
    rows = []
    for i in range(n_confirm):
        rows.append(_hyp(f"c{i}", _TECH_TECH, f"2026-07-{10+i:02d}T00:00:00+00:00", win_rate=0.8, baseline_rate=0.5))
    for i in range(n_contradict):
        rows.append(_hyp(f"x{i}", _TECH_TECH, f"2026-08-{10+i:02d}T00:00:00+00:00", win_rate=0.2, baseline_rate=0.5))
    entries = generate_knowledge_entries(rows, strong_threshold=strong_threshold, archive_margin=archive_margin)
    assert len(entries) == 1
    return entries[0]


def test_emerging_at_exactly_one_confirmation():
    e = _run(n_confirm=1, n_contradict=0)
    assert e.lifecycle_status == KnowledgeLifecycleStatus.EMERGING


def test_confirmed_at_two_to_four_confirmations():
    for n in (2, 3, 4):
        e = _run(n_confirm=n, n_contradict=0)
        assert e.lifecycle_status == KnowledgeLifecycleStatus.CONFIRMED, f"n={n}"


def test_strong_at_five_confirmations_spec_exact():
    e = _run(n_confirm=5, n_contradict=0)
    assert e.lifecycle_status == KnowledgeLifecycleStatus.STRONG


def test_weakening_when_contradictions_exist_but_trail_confirmations():
    e = _run(n_confirm=5, n_contradict=2)
    assert e.lifecycle_status == KnowledgeLifecycleStatus.WEAKENING


def test_contradicted_when_contradictions_catch_up_to_confirmations():
    e = _run(n_confirm=3, n_contradict=3)
    assert e.lifecycle_status == KnowledgeLifecycleStatus.CONTRADICTED


def test_archived_when_contradictions_exceed_confirmations_by_margin():
    e = _run(n_confirm=2, n_contradict=5, archive_margin=3)  # 5 >= 2 + 3
    assert e.lifecycle_status == KnowledgeLifecycleStatus.ARCHIVED


def test_thresholds_are_configurable():
    e = _run(n_confirm=3, n_contradict=0, strong_threshold=3)
    assert e.lifecycle_status == KnowledgeLifecycleStatus.STRONG


# ---------------------------------------------------------------------------
# previous_lifecycle_status
# ---------------------------------------------------------------------------

def test_previous_lifecycle_status_is_none_on_first_appearance():
    e = _run(n_confirm=1, n_contradict=0)
    assert e.previous_lifecycle_status is None


def test_previous_lifecycle_status_tracks_the_step_before():
    e = _run(n_confirm=2, n_contradict=0)
    assert e.previous_lifecycle_status == KnowledgeLifecycleStatus.EMERGING
    assert e.lifecycle_status == KnowledgeLifecycleStatus.CONFIRMED


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_identical_input_produces_identical_content(monkeypatch):
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, tzinfo=tz)

    monkeypatch.setattr(knowledge_base_engine, "datetime", _FrozenDatetime)
    rows = [_hyp(f"a{i}", _TECH_TECH, f"2026-07-{10+i:02d}T00:00:00+00:00") for i in range(3)]
    first = generate_knowledge_entries(rows)
    second = generate_knowledge_entries(rows)
    assert [e.knowledge_id for e in first] == [e.knowledge_id for e in second]
    assert [e.lifecycle_status for e in first] == [e.lifecycle_status for e in second]
    assert [e.confirmation_count for e in first] == [e.confirmation_count for e in second]


def test_entries_sorted_deterministically():
    rows = [
        _hyp("a1", _TECH_TECH, "2026-07-10T00:00:00+00:00"),
        _hyp("a2", _TECH_TECH, "2026-07-11T00:00:00+00:00"),
        _hyp("b1", _ENERGY_BREAKOUT, "2026-07-10T00:00:00+00:00"),
    ]
    entries = generate_knowledge_entries(rows)
    confirmation_counts = [e.confirmation_count for e in entries]
    assert confirmation_counts == sorted(confirmation_counts, reverse=True)

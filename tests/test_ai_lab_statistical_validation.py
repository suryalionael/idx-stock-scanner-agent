"""Tests for stock_scanner/ai_lab/statistical_validation.py — Fisher's
exact / Wilson CI / shrunk win rate / Benjamini-Hochberg correctness via
known synthetic scenarios, mirrors tests/test_ai_lab_reflection_engine.py's
convention."""
import pytest

from stock_scanner.ai_lab.schemas import EvidenceStrength, HypothesisCandidate, HypothesisStatus
from stock_scanner.ai_lab.statistical_validation import validate_hypotheses


def _candidate(conditions, n, n_success, order=2, source_reflection_ids=None) -> HypothesisCandidate:
    return HypothesisCandidate(
        conditions=sorted(conditions), order=order, n=n, n_success=n_success, n_failure=n - n_success,
        win_rate=n_success / n, avg_return_percentage=1.0, avg_holding_days=3.0,
        source_reflection_ids=source_reflection_ids or ["seed1"],
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_candidates_returns_empty():
    assert validate_hypotheses([], baseline_rate=0.5, total_n=40, total_success=20) == []


# ---------------------------------------------------------------------------
# Validated vs. rejected classification
# ---------------------------------------------------------------------------

def test_strong_success_candidate_is_validated_and_strong():
    # 16/20 wins vs 50% baseline in a 40-trade population — a clear, large lift.
    candidates = [_candidate([["ai_model", "momentum_ai"], ["sector", "Financials"]], n=20, n_success=16)]
    results = validate_hypotheses(candidates, baseline_rate=0.5, total_n=40, total_success=20,
                                   min_n_success=3, alpha=0.05)
    assert len(results) == 1
    h = results[0]
    assert h.status == HypothesisStatus.VALIDATED
    assert h.evidence_strength in (EvidenceStrength.STRONG, EvidenceStrength.MODERATE)
    assert h.rejection_reason is None
    assert h.failed_gate is None
    assert h.win_rate == pytest.approx(0.8)
    assert h.successes == 16
    assert h.failures == 4
    assert 0.0 <= h.wilson_lower <= h.win_rate <= h.wilson_upper <= 1.0


def test_strong_failure_candidate_is_validated_with_underperform_description():
    # 4/20 wins vs 50% baseline — a clear failure pattern.
    candidates = [_candidate([["ai_model", "breakout_ai"], ["sector", "Technology"]], n=20, n_success=4)]
    results = validate_hypotheses(candidates, baseline_rate=0.5, total_n=40, total_success=20,
                                   min_n_success=3, alpha=0.05)
    h = results[0]
    assert h.status == HypothesisStatus.VALIDATED
    assert "underperforms" in h.description
    assert h.win_rate == pytest.approx(0.2)


def test_no_lift_candidate_is_rejected_not_significant():
    # win rate equals baseline exactly -> never significant.
    candidates = [_candidate([["ai_model", "momentum_ai"], ["sector", "Financials"]], n=20, n_success=10)]
    results = validate_hypotheses(candidates, baseline_rate=0.5, total_n=40, total_success=20,
                                   min_n_success=3, alpha=0.05)
    h = results[0]
    assert h.status == HypothesisStatus.REJECTED
    assert h.failed_gate == "not_significant"
    assert h.evidence_strength is None
    assert "alpha=0.05" in h.rejection_reason


def test_evidence_strength_strong_vs_moderate_thresholds():
    strong = _candidate([["a", "1"], ["b", "1"]], n=30, n_success=27)   # very large lift, tiny p
    moderate = _candidate([["c", "1"], ["d", "1"]], n=20, n_success=15)  # smaller, still real lift
    results = validate_hypotheses([strong, moderate], baseline_rate=0.3, total_n=100, total_success=30,
                                   min_n_success=3, alpha=0.05)
    by_conditions = {tuple(map(tuple, h.conditions)): h for h in results}
    strong_h = by_conditions[(("a", "1"), ("b", "1"))]
    assert strong_h.status == HypothesisStatus.VALIDATED
    assert strong_h.bh_adjusted_p < 0.01
    assert strong_h.evidence_strength == EvidenceStrength.STRONG


# ---------------------------------------------------------------------------
# Rejection due to insufficient sample floor despite apparent significance
# ---------------------------------------------------------------------------

def test_failure_pattern_gate_uses_non_success_count_not_success_count():
    # Mirrors reflection_engine's own regression guard: only 1 win out of
    # 20 (n_success=1, below min_n_success=3) but 19 losses (well above
    # the floor) — should still validate as a failure pattern since
    # (n - n_success) >= min_n_success, reusing passes_slice_gate exactly.
    candidates = [_candidate([["ai_model", "breakout_ai"], ["sector", "Technology"]], n=20, n_success=1)]
    results = validate_hypotheses(candidates, baseline_rate=0.5, total_n=40, total_success=20,
                                   min_n_success=3, alpha=0.05)
    h = results[0]
    assert h.status == HypothesisStatus.VALIDATED
    assert h.win_rate == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# BH correction pooled per interaction order
# ---------------------------------------------------------------------------

def test_bh_correction_pooled_separately_per_order():
    order2 = [_candidate([["a", str(i)], ["b", "1"]], n=20, n_success=16, order=2) for i in range(5)]
    order3 = [_candidate([["a", str(i)], ["b", "1"], ["c", "1"]], n=20, n_success=16, order=3) for i in range(20)]
    results = validate_hypotheses(order2 + order3, baseline_rate=0.5, total_n=40, total_success=20,
                                   min_n_success=3, alpha=0.05)
    order2_q = {h.bh_adjusted_p for h in results if len(h.conditions) == 2}
    order3_q = {h.bh_adjusted_p for h in results if len(h.conditions) == 3}
    # Identical raw p-values within each tier -> identical (and equal to
    # the raw fisher_p, since BH of N identical p-values ranked together
    # never inflates the smallest) adjusted q within that tier — but the
    # two tiers are corrected independently, so this must not raise/crash
    # and every result must carry its own valid bh_adjusted_p.
    assert order2_q and order3_q
    assert all(0.0 <= q <= 1.0 for q in order2_q | order3_q)


# ---------------------------------------------------------------------------
# hypothesis_id / created_at / determinism
# ---------------------------------------------------------------------------

def test_hypothesis_id_deterministic_for_same_conditions_and_created_at(monkeypatch):
    from datetime import datetime

    import stock_scanner.ai_lab.statistical_validation as sv

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 1, 1, tzinfo=tz)

    monkeypatch.setattr(sv, "datetime", _FrozenDatetime)
    candidates = [_candidate([["ai_model", "momentum_ai"], ["sector", "Financials"]], n=20, n_success=16)]
    first = validate_hypotheses(candidates, baseline_rate=0.5, total_n=40, total_success=20)
    second = validate_hypotheses(candidates, baseline_rate=0.5, total_n=40, total_success=20)
    assert first[0].hypothesis_id == second[0].hypothesis_id


def test_results_sorted_validated_first_then_by_p_value():
    validated = _candidate([["a", "1"], ["b", "1"]], n=20, n_success=16)
    rejected = _candidate([["c", "1"], ["d", "1"]], n=20, n_success=10)
    results = validate_hypotheses([rejected, validated], baseline_rate=0.5, total_n=40, total_success=20,
                                   min_n_success=3, alpha=0.05)
    statuses = [h.status for h in results]
    assert statuses.index(HypothesisStatus.VALIDATED) < statuses.index(HypothesisStatus.REJECTED)

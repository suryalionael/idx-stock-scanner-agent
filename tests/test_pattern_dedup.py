"""Tests for pattern de-duplication — see docs/LEARNING_AGENT_ARCHITECTURE.md.

Includes a regression test for a real bug found while manually validating
this module against the actual Phase 1 report: single-linkage (union-find)
clustering collapsed 224 candidates into one 223-member cluster via
transitive chaining. Fixed with greedy complete-linkage — a candidate joins
a cluster only if it's a duplicate of EVERY existing member, not just one.
test_cluster_patterns_does_not_chain_through_intermediate below guards
against that regression directly.
"""
import pytest

from stock_scanner.learning.pattern_dedup import (
    cluster_patterns,
    jaccard,
    overlap_coefficient,
)
from stock_scanner.learning.pattern_miner import PatternCandidate


def _candidate(slice_definition: dict, signal_ids: set, p_value_adjusted: float = 0.01, n_success: int | None = None) -> PatternCandidate:
    n_success = n_success if n_success is not None else len(signal_ids)
    return PatternCandidate(
        dimensions=tuple(slice_definition.keys()), slice_definition=slice_definition,
        interaction_order=2, n=n_success * 10, n_success=n_success, win_rate=0.1,
        win_rate_shrunk=0.1, baseline_win_rate=0.04, ci_lower=0.05, ci_upper=0.15,
        p_value=p_value_adjusted, p_value_adjusted=p_value_adjusted, passed_gate=True,
        signal_ids=frozenset(signal_ids),
    )


# ---------------------------------------------------------------------------
# jaccard / overlap_coefficient
# ---------------------------------------------------------------------------

def test_jaccard_identical_sets_is_one():
    a = frozenset({"1", "2", "3"})
    assert jaccard(a, a) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    assert jaccard(frozenset({"1"}), frozenset({"2"})) == 0.0


def test_overlap_coefficient_full_containment_is_one_despite_low_jaccard():
    # The exact scenario flagged for the fallback: unequal support, full
    # containment of the smaller set — Jaccard understates it, overlap
    # coefficient correctly reports 1.0.
    big = frozenset(str(i) for i in range(50))
    small = frozenset(str(i) for i in range(8))   # fully contained in `big`
    assert jaccard(big, small) == pytest.approx(8 / 50)
    assert overlap_coefficient(big, small) == 1.0


# ---------------------------------------------------------------------------
# cluster_patterns
# ---------------------------------------------------------------------------

def test_cluster_patterns_merges_high_jaccard_pair():
    a = _candidate({"vol_spike": True}, {"s1", "s2", "s3", "s4", "s5"})
    b = _candidate({"atr_breakout": True}, {"s1", "s2", "s3", "s4", "s6"})  # jaccard=4/6=0.67
    clusters = cluster_patterns([a, b], jaccard_threshold=0.6, overlap_threshold=0.8)
    assert len(clusters) == 1
    assert clusters[0].member_count == 2


def test_cluster_patterns_keeps_disjoint_patterns_separate():
    a = _candidate({"vol_spike": True}, {"s1", "s2", "s3"})
    b = _candidate({"rsi14": "Q5"}, {"s4", "s5", "s6"})
    clusters = cluster_patterns([a, b], jaccard_threshold=0.6, overlap_threshold=0.8)
    assert len(clusters) == 2
    assert all(c.member_count == 1 for c in clusters)


def test_cluster_patterns_catches_unequal_support_containment():
    big = _candidate({"vol_ratio_20d": "Q5"}, set(str(i) for i in range(50)))
    small = _candidate({"vol_ratio_20d": "Q5", "atr_breakout": True}, set(str(i) for i in range(8)))
    clusters = cluster_patterns([big, small], jaccard_threshold=0.6, overlap_threshold=0.8)
    assert len(clusters) == 1   # merged via overlap_coefficient, not jaccard


def test_cluster_patterns_does_not_chain_through_intermediate():
    # Regression test for the real bug found while validating this module:
    # a is fully contained in b (overlap_coefficient=1.0) and c is fully
    # contained in b (overlap_coefficient=1.0), but a and c share nothing at
    # all. Single-linkage (union-find) would merge all three transitively
    # via b; complete-linkage must keep a and c in different clusters since
    # a-c alone never qualifies under either measure.
    a = _candidate({"dim_a": True}, {"x1", "x2", "x3", "x4"})
    b = _candidate({"dim_b": True}, {"x1", "x2", "x3", "x4", "y1", "y2", "y3", "y4"})
    c = _candidate({"dim_c": True}, {"y1", "y2", "y3", "y4"})
    assert jaccard(a.signal_ids, c.signal_ids) == 0.0
    assert overlap_coefficient(a.signal_ids, c.signal_ids) == 0.0

    clusters = cluster_patterns([a, b, c], jaccard_threshold=0.6, overlap_threshold=0.8)

    def _cluster_of(candidate) -> str:
        return next(cl.cluster_id for cl in clusters for m in cl.members if m is candidate)

    assert _cluster_of(a) != _cluster_of(c)


def test_cluster_patterns_representative_is_lowest_adjusted_p_value():
    weak = _candidate({"vol_spike": True}, {"s1", "s2", "s3", "s4", "s5"}, p_value_adjusted=0.04)
    strong = _candidate({"atr_breakout": True}, {"s1", "s2", "s3", "s4", "s6"}, p_value_adjusted=0.0001)
    clusters = cluster_patterns([weak, strong], jaccard_threshold=0.6, overlap_threshold=0.8)
    assert len(clusters) == 1
    assert clusters[0].representative.p_value_adjusted == 0.0001


def test_cluster_patterns_empty_input():
    assert cluster_patterns([]) == []


def test_cluster_id_is_deterministic_across_repeated_runs():
    a = _candidate({"vol_spike": True}, {"s1", "s2", "s3", "s4", "s5"})
    b = _candidate({"atr_breakout": True}, {"s1", "s2", "s3", "s4", "s6"})
    clusters_1 = cluster_patterns([a, b], jaccard_threshold=0.6, overlap_threshold=0.8)
    clusters_2 = cluster_patterns([a, b], jaccard_threshold=0.6, overlap_threshold=0.8)
    assert clusters_1[0].cluster_id == clusters_2[0].cluster_id

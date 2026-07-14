"""Tests for the Phase 1 (statistics-only) Pattern Miner — see
docs/LEARNING_AGENT_ARCHITECTURE.md. Covers the three hand-implemented
stats primitives (Wilson CI, Benjamini-Hochberg, Beta-Binomial shrinkage)
against known reference behavior, the hard n_success floor (must reject a
technically-significant-but-tiny-sample slice), and an end-to-end
mine_patterns() run against a synthetic DataFrame with a known injected
pattern.
"""
import numpy as np
import pandas as pd
import pytest

from stock_scanner.learning.pattern_miner import (
    benjamini_hochberg,
    mine_patterns,
    shrunk_win_rate,
    wilson_ci,
)


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------

def test_wilson_ci_zero_n_returns_zero_zero():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_contains_point_estimate():
    lower, upper = wilson_ci(30, 100)
    assert lower < 0.30 < upper


def test_wilson_ci_symmetric_at_p_half():
    # Algebraically, the Wilson center is exactly phat when phat=0.5,
    # regardless of n — a useful exact check rather than a hand-derived
    # approximate bound.
    lower, upper = wilson_ci(10, 20)
    assert (lower + upper) / 2 == pytest.approx(0.5, abs=1e-9)
    assert lower == pytest.approx(1 - upper, abs=1e-9)


def test_wilson_ci_narrows_with_more_data():
    lower_small, upper_small = wilson_ci(5, 10)
    lower_large, upper_large = wilson_ci(500, 1000)
    assert (upper_small - lower_small) > (upper_large - lower_large)


# ---------------------------------------------------------------------------
# benjamini_hochberg
# ---------------------------------------------------------------------------

def test_benjamini_hochberg_textbook_example():
    p_values = [0.01, 0.02, 0.03, 0.04, 0.20]
    q_values = benjamini_hochberg(p_values)
    expected = [0.05, 0.05, 0.05, 0.05, 0.20]
    for q, e in zip(q_values, expected):
        assert q == pytest.approx(e, abs=1e-9)


def test_benjamini_hochberg_empty_input():
    assert benjamini_hochberg([]) == []


def test_benjamini_hochberg_adjusted_never_below_raw_p_value_at_smallest_rank():
    # q-value for the single smallest p-value can only be reduced by the
    # min-running correction, never increased beyond raw*n/1... this checks
    # the procedure doesn't invert ordering.
    p_values = [0.5, 0.001, 0.3, 0.01, 0.2]
    q_values = benjamini_hochberg(p_values)
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    sorted_q = [q_values[i] for i in order]
    assert sorted_q == sorted(sorted_q)


# ---------------------------------------------------------------------------
# shrunk_win_rate
# ---------------------------------------------------------------------------

def test_shrunk_win_rate_pulls_small_sample_toward_baseline():
    baseline = 0.05
    raw = shrunk_win_rate(2, 2, baseline)  # 2/2 = 100% raw win rate
    assert 0.0 < raw < 1.0
    assert raw > baseline   # still pulled up — both observations were wins
    assert raw < 0.5        # but nowhere near the misleading raw 100%


def test_shrunk_win_rate_converges_to_raw_with_large_n():
    baseline = 0.05
    raw_rate = 0.40
    n = 100_000
    n_success = int(raw_rate * n)
    shrunk = shrunk_win_rate(n_success, n, baseline)
    assert shrunk == pytest.approx(raw_rate, abs=1e-3)


# ---------------------------------------------------------------------------
# mine_patterns — end-to-end on synthetic data with a known injected pattern
# ---------------------------------------------------------------------------

def _synthetic_df() -> pd.DataFrame:
    n_pattern, succ_pattern = 60, 40      # vol_spike=True: 40/60 = 66.7% win rate
    n_base, succ_base = 240, 20           # vol_spike=False: 20/240 = 8.3% win rate
    total = n_pattern + n_base

    label_success = (
        [1] * succ_pattern + [0] * (n_pattern - succ_pattern)
        + [1] * succ_base + [0] * (n_base - succ_base)
    )
    vol_spike = [True] * n_pattern + [False] * n_base

    # squeeze_on: a tiny, "perfect" 5-row pattern (all wins) — should be
    # statistically extreme but must be rejected by the n_success>=8 floor.
    squeeze_on = [True] * 5 + [False] * (total - 5)

    return pd.DataFrame({
        "ticker": [f"TICK{i % 20}.JK" for i in range(total)],
        "signal_date": pd.date_range("2026-01-01", periods=total, freq="D"),
        "strategy": ["swing"] * total,
        "signal_label": ["BREAKOUT"] * total,
        "label_success": label_success,
        "vol_spike": vol_spike,
        "squeeze_on": squeeze_on,
        "ihsg_pct_change_eval": np.tile([0.5, -0.5], total // 2 + 1)[:total],
    })


def test_mine_patterns_detects_injected_single_feature_pattern():
    df = _synthetic_df()
    result = mine_patterns(df, sector_df=None, min_n_success=8, alpha=0.05, max_order=1)

    order1 = result.candidates_by_order[1]
    vol_spike_true = [
        c for c in order1
        if c.dimensions == ("vol_spike",) and c.slice_definition.get("vol_spike") is True
    ]
    assert len(vol_spike_true) == 1
    candidate = vol_spike_true[0]
    assert candidate.n == 60
    assert candidate.n_success == 40
    assert candidate.passed_gate is True
    assert candidate.ci_lower > candidate.baseline_win_rate


def test_mine_patterns_rejects_tiny_perfect_slice_below_n_success_floor():
    df = _synthetic_df()
    result = mine_patterns(df, sector_df=None, min_n_success=8, alpha=0.05, max_order=1)

    order1 = result.candidates_by_order[1]
    squeeze_true = [
        c for c in order1
        if c.dimensions == ("squeeze_on",) and c.slice_definition.get("squeeze_on") is True
    ]
    assert len(squeeze_true) == 1
    candidate = squeeze_true[0]
    assert candidate.n_success == 5          # below the floor
    assert candidate.win_rate == 1.0          # "perfect" — would look extremely significant
    assert candidate.passed_gate is False     # must still be rejected


def test_mine_patterns_summary_counts_are_consistent():
    df = _synthetic_df()
    result = mine_patterns(df, sector_df=None, min_n_success=8, alpha=0.05, max_order=1)
    assert result.total_n == len(df)
    assert result.total_success == int(df["label_success"].sum())
    assert result.baseline_win_rate == pytest.approx(df["label_success"].mean())


def test_mine_patterns_empty_dataframe_returns_empty_result():
    result = mine_patterns(pd.DataFrame(), sector_df=None)
    assert result.total_n == 0
    assert result.candidates_by_order == {1: [], 2: [], 3: []}

"""_save_ranked() sort-priority tests — verifies promoted_rule_score slots
in ahead of ml_prob when present, and that output is unchanged (falls back
to ml_prob) when no model is currently promoted. See
stock_scanner/pipeline/run_daily_scan.py::_apply_promoted_challenger_score
and docs/SELF_IMPROVING_ARCHITECTURE.md.
"""
import pandas as pd

from stock_scanner.pipeline.run_daily_scan import _save_ranked


def _base_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "signal": ["BREAKOUT", "BREAKOUT", "BREAKOUT"],
        "total_score": [8.0, 8.0, 8.0],
        "ml_prob": [0.9, 0.1, 0.5],
    })


def test_ranked_sort_uses_promoted_rule_score_over_ml_prob(tmp_path):
    df = _base_df()
    df["promoted_rule_score"] = [1, 3, 2]   # would rank B last on ml_prob alone
    _save_ranked(df, tmp_path, "2026-07-13", config={})
    out = pd.read_csv(tmp_path / "ranked_2026-07-13.csv")
    assert list(out["ticker"]) == ["B", "C", "A"]


def test_ranked_sort_falls_back_to_ml_prob_when_no_promoted_score(tmp_path):
    df = _base_df()
    _save_ranked(df, tmp_path, "2026-07-13", config={})
    out = pd.read_csv(tmp_path / "ranked_2026-07-13.csv")
    assert list(out["ticker"]) == ["A", "C", "B"]


def test_ranked_sort_quality_adjusted_score_still_takes_priority(tmp_path):
    # quality_adjusted_score must still win over promoted_rule_score — the
    # promoted challenger only slots in between quality_adjusted_score and
    # ml_prob, it does not replace quality filtering.
    df = _base_df()
    df["promoted_rule_score"] = [1, 3, 2]
    df["quality_adjusted_score"] = [5.0, 1.0, 9.0]
    _save_ranked(df, tmp_path, "2026-07-13", config={})
    out = pd.read_csv(tmp_path / "ranked_2026-07-13.csv")
    assert list(out["ticker"]) == ["C", "A", "B"]

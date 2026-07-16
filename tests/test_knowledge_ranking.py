"""_save_ranked() sort-priority tests for knowledge_adjusted_score — mirrors
tests/test_promoted_ranking.py's convention. Verifies knowledge_adjusted_score
slots in ahead of quality_adjusted_score/promoted_rule_score/ml_prob when
present, and that output is unchanged when the column is absent (no
applicable knowledge). See stock_scanner/pipeline/run_daily_scan.py::_save_ranked
and stock_scanner/pipeline/knowledge_application.py.
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


def test_ranked_sort_uses_knowledge_adjusted_score_over_everything_else(tmp_path):
    df = _base_df()
    df["promoted_rule_score"] = [1, 3, 2]
    df["quality_adjusted_score"] = [5.0, 1.0, 9.0]
    df["knowledge_adjusted_score"] = [1.0, 3.0, 2.0]  # would rank B first despite losing on all others
    _save_ranked(df, tmp_path, "2026-07-13", config={})
    out = pd.read_csv(tmp_path / "ranked_2026-07-13.csv")
    assert list(out["ticker"]) == ["B", "C", "A"]


def test_ranked_sort_falls_back_when_no_knowledge_adjusted_score(tmp_path):
    # No applicable knowledge → column absent entirely → falls back to the
    # pre-existing cascade unchanged (quality_adjusted_score here).
    df = _base_df()
    df["quality_adjusted_score"] = [5.0, 1.0, 9.0]
    _save_ranked(df, tmp_path, "2026-07-13", config={})
    out = pd.read_csv(tmp_path / "ranked_2026-07-13.csv")
    assert list(out["ticker"]) == ["C", "A", "B"]


def test_knowledge_adjusted_score_equal_to_base_is_a_true_no_op(tmp_path):
    # knowledge_adjusted_score == quality_adjusted_score for every row (bonus
    # was 0 everywhere but the column still exists) must sort identically to
    # sorting by quality_adjusted_score alone.
    df = _base_df()
    df["quality_adjusted_score"] = [5.0, 1.0, 9.0]
    df["knowledge_adjusted_score"] = df["quality_adjusted_score"]
    _save_ranked(df, tmp_path, "2026-07-13", config={})
    out = pd.read_csv(tmp_path / "ranked_2026-07-13.csv")
    assert list(out["ticker"]) == ["C", "A", "B"]

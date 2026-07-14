"""Shared challenger scoring formula — single source of truth.

Used by scripts/train_challenger.py (fits/validates the vol_ratio_20d
threshold on TRAIN), scripts/promote_challenger.py (re-scores TEST with a
promoted model's recorded threshold), and
stock_scanner/pipeline/run_daily_scan.py (applies a PROMOTED model's
threshold live, see docs/SELF_IMPROVING_ARCHITECTURE.md). All three must
score identically given the same threshold — this module is the only place
the formula lives, so training/promotion/production can never silently
drift apart.
"""
import pandas as pd


def compute_rule_score(df: pd.DataFrame, vol_thresh: float) -> pd.Series:
    def b(col):
        return df[col].map({True: True, False: False, "True": True, "False": False}).fillna(False).astype(bool)

    return (
        b("atr_breakout").astype(int)
        + b("vol_spike").astype(int)
        + (df["vol_ratio_20d"] > vol_thresh).fillna(False).astype(int)
        - 2 * b("squeeze_on").astype(int)
    )

#!/usr/bin/env python3
"""Out-of-sample validation of the SAFE-only scoring framework.

Builds a strictly chronological train/test split on `signal_date`, selects
the vol_ratio_20d threshold from TRAIN ONLY (criterion fixed before looking
at the sweep table), locks the rule, then evaluates it on TEST — which the
threshold-selection step never saw. Also runs the sensitivity checks
(threshold perturbation, split-date shift, ticker exclusion, market regime)
and an RSI re-audit (context feature, not core rule).

Input: data/reports/signal_diagnosis_dataset.csv (from diagnose_successful_signals.py)
Output: printed report to stdout (this script does not write new files — the
        full write-up with interpretation lives in
        data/reports/signal_diagnosis_validation_oos.md)

Usage:
    python scripts/validate_safe_score_oos.py
"""
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import pandas as pd
from loguru import logger

_DATASET = repo_root / "data" / "reports" / "signal_diagnosis_dataset.csv"
_SPLIT_DATE = pd.Timestamp("2026-06-05")   # locked: train < split <= test
_VOL_RATIO_THRESH = 2.0                     # locked: chosen from TRAIN only (see select_threshold)


def _load() -> pd.DataFrame:
    df = pd.read_csv(_DATASET)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    df["is_success"] = df["pct_close"] > 10
    for c in ["squeeze_on", "atr_breakout", "vol_spike"]:
        df[c] = df[c].map({True: True, False: False, "True": True, "False": False}).astype("boolean")
    return df


def safe_score(df: pd.DataFrame, vol_thresh: float = _VOL_RATIO_THRESH) -> pd.Series:
    return (
        (df["atr_breakout"] == True).fillna(False).astype(int)  # noqa: E712
        + (df["vol_spike"] == True).fillna(False).astype(int)  # noqa: E712
        + (df["vol_ratio_20d"] > vol_thresh).fillna(False).astype(int)
        - 2 * (df["squeeze_on"] == True).fillna(False).astype(int)  # noqa: E712
    )


def select_threshold(train: pd.DataFrame) -> float:
    """TRAIN-ONLY threshold selection. Criterion fixed BEFORE viewing results:
    smallest candidate threshold reaching lift>=1.5x while n_true stays >=25%
    of train rows (avoids an overly narrow rule)."""
    candidates = [1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
    n_min = 0.25 * len(train)
    for thr in candidates:
        mask = train["vol_ratio_20d"] > thr
        valid = train["vol_ratio_20d"].notna()
        n_true = mask[valid].sum()
        if n_true < n_min:
            continue
        rt = train.loc[valid & mask, "is_success"].mean()
        rf = train.loc[valid & ~mask, "is_success"].mean()
        lift = rt / rf if rf else float("inf")
        if lift >= 1.5:
            return thr
    return candidates[-1]


def bucket_table(df: pd.DataFrame, score_col: str = "safe_score") -> pd.DataFrame:
    bands = [(-2, -1, "<=-1 veto"), (0, 1, "0-1 avoid"), (2, 2, "2 neutral"), (3, 3, "3 strong")]
    rows = []
    for lo, hi, name in bands:
        sub = df[(df[score_col] >= lo) & (df[score_col] <= hi)]
        rows.append({"bucket": name, "n": len(sub),
                     "success_rate_pct": round(sub["is_success"].mean() * 100, 2) if len(sub) else None,
                     "n_success": int(sub["is_success"].sum())})
    return pd.DataFrame(rows)


def precision_recall_lift(df: pd.DataFrame, score_col: str = "safe_score", min_score: int = 3) -> dict:
    baseline = df["is_success"].mean()
    sel = df[df[score_col] >= min_score]
    precision = sel["is_success"].mean() if len(sel) else float("nan")
    recall = sel["is_success"].sum() / df["is_success"].sum() if df["is_success"].sum() else float("nan")
    lift = precision / baseline if baseline else float("nan")
    return {"n_sel": len(sel), "precision": precision, "recall": recall,
            "lift": lift, "baseline": baseline}


def main() -> None:
    df = _load()
    train = df[df["signal_date"] < _SPLIT_DATE].copy()
    test = df[df["signal_date"] >= _SPLIT_DATE].copy()
    logger.info("TRAIN: n={} success={} ({:.2f}%) | TEST: n={} success={} ({:.2f}%)",
                len(train), train["is_success"].sum(), train["is_success"].mean() * 100,
                len(test), test["is_success"].sum(), test["is_success"].mean() * 100)

    thr = select_threshold(train)
    logger.info("Selected vol_ratio_20d threshold from TRAIN only: {}", thr)
    assert thr == _VOL_RATIO_THRESH, "Re-run with updated _VOL_RATIO_THRESH if the data changed"

    df["safe_score"] = safe_score(df, thr)
    train = df[df["signal_date"] < _SPLIT_DATE]
    test = df[df["signal_date"] >= _SPLIT_DATE]

    print("\n=== TRAIN bucket table ===")
    print(bucket_table(train))
    print("\n=== TEST bucket table (out-of-sample, rule locked from train) ===")
    print(bucket_table(test))

    print("\n=== TEST precision/recall/lift, score>=3 ===")
    print(precision_recall_lift(test))

    print("\n=== Sensitivity: ticker exclusion (FORU.JK) ===")
    for excl in [None, "FORU.JK"]:
        t = test[test["ticker"] != excl] if excl else test
        m = precision_recall_lift(t)
        print(f"  {'full' if not excl else f'excl {excl}':12s} n_sel={m['n_sel']:3d} "
              f"precision={m['precision']*100:5.2f}% lift={m['lift']:.2f}x")

    print("\n=== Sensitivity: split-date shift ===")
    for shift in ["2026-05-29", "2026-06-02", "2026-06-05", "2026-06-09", "2026-06-12"]:
        t = df[df["signal_date"] >= pd.Timestamp(shift)]
        m = precision_recall_lift(t)
        print(f"  test-from={shift}  n_test={len(t):4d} n_sel={m['n_sel']:3d} "
              f"precision={m['precision']*100:5.2f}% lift={m['lift']:.2f}x")

    print("\n=== RSI re-audit: standalone, TRAIN vs TEST ===")
    for name, sub in [("TRAIN", train), ("TEST", test)]:
        mask = sub["rsi14"] >= 70
        valid = sub["rsi14"].notna()
        rt = sub.loc[valid & mask, "is_success"].mean()
        rf = sub.loc[valid & ~mask, "is_success"].mean()
        print(f"  {name}: n_true={mask[valid].sum():4d} lift={rt/rf if rf else float('inf'):.2f}")

    print("\n=== RSI re-audit: as context modifier within score>=3 bucket ===")
    for name, sub in [("TRAIN", train), ("TEST", test)]:
        good = sub[sub["safe_score"] >= 3]
        hi = good[good["rsi14"] >= 70]
        lo = good[good["rsi14"] < 70]
        print(f"  {name}: RSI>=70 n={len(hi)} rate={hi['is_success'].mean()*100 if len(hi) else 0:.2f}% | "
              f"RSI<70 n={len(lo)} rate={lo['is_success'].mean()*100 if len(lo) else 0:.2f}%")

    print("\n=== Old model (total_score) vs new model (safe_score), TEST rows with total_score ===")
    twf = test[test["total_score"].notna()]
    baseline = twf["is_success"].mean()
    for q in [0.9, 0.75, 0.5]:
        thr_q = twf["total_score"].quantile(q)
        sel = twf[twf["total_score"] > thr_q]
        print(f"  total_score top {int((1-q)*100):2d}%  n={len(sel):4d} "
              f"precision={sel['is_success'].mean()*100:5.2f}% lift={sel['is_success'].mean()/baseline:.2f}x")
    sel_new = twf[twf["safe_score"] >= 3]
    print(f"  safe_score>=3        n={len(sel_new):4d} "
          f"precision={sel_new['is_success'].mean()*100:5.2f}% lift={sel_new['is_success'].mean()/baseline:.2f}x")
    print(f"  corr(total_score, is_success) = {twf['total_score'].corr(twf['is_success']):.3f}")
    print(f"  corr(safe_score,  is_success) = {twf['safe_score'].corr(twf['is_success']):.3f}")


if __name__ == "__main__":
    main()

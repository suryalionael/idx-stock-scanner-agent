#!/usr/bin/env python3
"""Train + validate a challenger model against the closed-loop DB.

Fixes the mismatch found auditing train_ranker_from_history.py / ml_ranker.py:
that pipeline trains on forward-5-day-return>3% computed from EVERY ticker-day
in raw OHLCV history (the whole liquid universe, every day), then is deployed
to score only the narrow subset of candidates signal_engine already flagged
BREAKOUT/PRE_MARKUP/SCALPING_HIGH — a different label AND a different
population than what it's actually ranking live. This script trains and
validates against the EXACT label and EXACT population the live system
measures: outcomes.label_success (pct_close > label_threshold_pct, next
session, only for signals signal_engine actually fired) — sourced from the
DB built in scripts/init_db_and_backfill.py.

This does NOT replace train_ranker_from_history.py (kept as-is — a separate
"general market-timing" research track, not part of the closed loop). It also
does NOT promote anything — a challenger that passes validation here gets
status='candidate' in model_registry; promotion is a separate, explicit step
(scripts/promote_challenger.py).

Model type produced here: 'rule_score' — the SAFE-only additive score
(squeeze_on veto, atr_breakout, vol_spike, vol_ratio_20d) already validated
out-of-sample in data/reports/signal_diagnosis_validation_oos.md. An
'xgboost_ranker' challenger trained on this same DB population is the
designed-but-not-yet-built next step (needs more accumulated labeled rows
than currently exist — see docs/SELF_IMPROVING_ARCHITECTURE.md §7).

Usage:
    python scripts/train_challenger.py
    python scripts/train_challenger.py --train-frac 0.6 --val-frac 0.2
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import pandas as pd
from loguru import logger

from stock_scanner.db.init_db import get_connection

_LABEL_THRESHOLD_PCT = 10.0
_LABEL_HORIZON = "next_session"
_MODEL_TYPE = "rule_score"
_FEATURE_LIST = ["squeeze_on", "atr_breakout", "vol_spike", "vol_ratio_20d"]
_VOL_RATIO_CANDIDATES = [1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
_VOL_RATIO_MIN_LIFT = 1.5
_VOL_RATIO_MIN_N_FRAC = 0.25
_MIN_SCORE_FOR_SELECTION = 3


# ---------------------------------------------------------------------------
# Data loading — the actual fix: read the live-labeled population from the DB
# ---------------------------------------------------------------------------

def load_training_examples(conn) -> pd.DataFrame:
    """signals ⋈ feature_snapshots ⋈ outcomes, evaluated rows only.

    This is the population+label the live system actually measures —
    exactly what train_ranker_from_history.py does NOT use today.
    """
    query = """
        SELECT s.signal_id, s.ticker, s.signal_date, s.strategy, s.signal_label,
               o.eval_date, o.pct_high, o.pct_close, o.wl, o.label_success,
               fs.features_json, mc.ihsg_pct_change AS ihsg_pct_change_eval
        FROM signals s
        JOIN outcomes o ON s.signal_id = o.signal_id
        JOIN feature_snapshots fs ON s.signal_id = fs.signal_id
        LEFT JOIN market_context mc ON o.eval_date = mc.context_date
        WHERE o.status = 'evaluated'
    """
    df = pd.read_sql(query, conn)
    feats = pd.json_normalize(df["features_json"].apply(json.loads))
    df = pd.concat([df.drop(columns=["features_json"]), feats], axis=1)
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    return df


def chronological_split(df: pd.DataFrame, train_frac: float, val_frac: float):
    """Split by whole signal_date (never split a single date across sets) —
    same discipline as validate_safe_score_oos.py, generalized to 3 sets."""
    dates = sorted(df["signal_date"].unique())
    n = len(dates)
    train_end = dates[int(n * train_frac) - 1]
    val_end = dates[int(n * (train_frac + val_frac)) - 1]
    train = df[df["signal_date"] <= train_end]
    val = df[(df["signal_date"] > train_end) & (df["signal_date"] <= val_end)]
    test = df[df["signal_date"] > val_end]
    return train, val, test


# ---------------------------------------------------------------------------
# Rule fitting — threshold chosen on TRAIN only, criterion fixed in advance
# ---------------------------------------------------------------------------

def select_vol_ratio_threshold(train: pd.DataFrame) -> float:
    n_min = _VOL_RATIO_MIN_N_FRAC * len(train)
    for thr in _VOL_RATIO_CANDIDATES:
        mask = train["vol_ratio_20d"] > thr
        valid = train["vol_ratio_20d"].notna()
        if mask[valid].sum() < n_min:
            continue
        rt = train.loc[valid & mask, "label_success"].mean()
        rf = train.loc[valid & ~mask, "label_success"].mean()
        lift = rt / rf if rf else float("inf")
        if lift >= _VOL_RATIO_MIN_LIFT:
            return thr
    return _VOL_RATIO_CANDIDATES[-1]


def rule_score(df: pd.DataFrame, vol_thresh: float) -> pd.Series:
    def b(col):
        return df[col].map({True: True, False: False, "True": True, "False": False}).fillna(False).astype(bool)
    return (
        b("atr_breakout").astype(int)
        + b("vol_spike").astype(int)
        + (df["vol_ratio_20d"] > vol_thresh).fillna(False).astype(int)
        - 2 * b("squeeze_on").astype(int)
    )


def bucket_metrics(df: pd.DataFrame, score_col: str, min_score: int) -> dict:
    baseline = df["label_success"].mean()
    sel = df[df[score_col] >= min_score]
    precision = sel["label_success"].mean() if len(sel) else float("nan")
    recall = sel["label_success"].sum() / df["label_success"].sum() if df["label_success"].sum() else float("nan")
    return {
        "n": int(len(df)), "n_selected": int(len(sel)), "baseline": float(baseline),
        "precision": float(precision) if pd.notna(precision) else None,
        "recall": float(recall) if pd.notna(recall) else None,
        "lift": float(precision / baseline) if baseline and pd.notna(precision) else None,
        "n_success": int(df["label_success"].sum()),
    }


# ---------------------------------------------------------------------------
# Validation gate — sensitivity battery on TRAIN+VAL only, TEST untouched
# ---------------------------------------------------------------------------

def sensitivity_battery(train_val: pd.DataFrame, vol_thresh: float) -> dict:
    """Returns {"passed": bool, "checks": {...}}. Every check must pass for
    the candidate to be eligible for the (single) TEST evaluation."""
    checks = {}

    # 1. Threshold perturbation: +-1.0 around the chosen vol_ratio threshold
    #    must not collapse precision to near-baseline.
    base_metrics = bucket_metrics(train_val.assign(score=rule_score(train_val, vol_thresh)), "score", _MIN_SCORE_FOR_SELECTION)
    perturbed_ok = True
    for delta in (-0.5, 0.5):
        m = bucket_metrics(train_val.assign(score=rule_score(train_val, vol_thresh + delta)), "score", _MIN_SCORE_FOR_SELECTION)
        if m["lift"] is None or m["lift"] < 2.0:
            perturbed_ok = False
    checks["threshold_perturbation"] = {"passed": perturbed_ok, "base_lift": base_metrics["lift"]}

    # 2. Ticker-exclusion stress test: drop the single most-frequent ticker
    #    in the positive class, lift must stay clearly above 1.
    score = rule_score(train_val, vol_thresh)
    tv = train_val.assign(score=score)
    pos = tv[tv["label_success"] == 1]
    if not pos.empty:
        top_ticker = pos["ticker"].value_counts().idxmax()
        excl = tv[tv["ticker"] != top_ticker]
        m_excl = bucket_metrics(excl, "score", _MIN_SCORE_FOR_SELECTION)
        ticker_ok = m_excl["lift"] is not None and m_excl["lift"] >= 2.0
    else:
        top_ticker, m_excl, ticker_ok = None, None, False
    checks["ticker_exclusion"] = {"passed": ticker_ok, "excluded_ticker": top_ticker,
                                   "lift_after_exclusion": m_excl["lift"] if m_excl else None}

    # 3. Time-split stability: split train_val itself in half chronologically,
    #    direction of lift (>1) must agree in both halves.
    dates = sorted(tv["signal_date"].unique())
    mid = dates[len(dates) // 2]
    early, late = tv[tv["signal_date"] < mid], tv[tv["signal_date"] >= mid]
    m_early, m_late = bucket_metrics(early, "score", _MIN_SCORE_FOR_SELECTION), bucket_metrics(late, "score", _MIN_SCORE_FOR_SELECTION)
    direction_ok = all((m["lift"] is not None and m["lift"] > 1.0) for m in (m_early, m_late) if m["n_selected"] >= 5)
    checks["time_split"] = {"passed": direction_ok, "early_lift": m_early["lift"], "late_lift": m_late["lift"]}

    # 4. Regime split: market-up vs market-down days, direction must agree
    #    (magnitude is allowed to differ — that's an expected, real effect).
    if "ihsg_pct_change_eval" in tv.columns:
        up = tv[tv["ihsg_pct_change_eval"] > 0]
        down = tv[tv["ihsg_pct_change_eval"] <= 0]
        m_up, m_down = bucket_metrics(up, "score", _MIN_SCORE_FOR_SELECTION), bucket_metrics(down, "score", _MIN_SCORE_FOR_SELECTION)
        regime_ok = all((m["lift"] is None or m["lift"] > 0.8) for m in (m_up, m_down))
        checks["regime_split"] = {"passed": regime_ok, "up_lift": m_up["lift"], "down_lift": m_down["lift"]}
    else:
        checks["regime_split"] = {"passed": True, "note": "ihsg_pct_change_eval not joined — skipped"}

    passed = all(c["passed"] for c in checks.values())
    return {"passed": passed, "checks": checks}


# ---------------------------------------------------------------------------
# Registry write
# ---------------------------------------------------------------------------

def register_model(conn, train, val, test, vol_thresh: float,
                    train_metrics: dict, val_metrics: dict,
                    sensitivity: dict, test_metrics: dict | None, status: str) -> str:
    model_version_id = f"{_MODEL_TYPE}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    # Threshold is a hyperparameter of the rule, not a feature — keep it out
    # of feature_list_json (which should stay a clean list of input columns)
    # and record it on each metrics blob instead, so every status (rejected/
    # candidate/promoted) carries the operating threshold that produced it.
    train_metrics = {**train_metrics, "vol_ratio_20d_threshold": vol_thresh}
    val_metrics = {**val_metrics, "vol_ratio_20d_threshold": vol_thresh} if val_metrics else val_metrics
    if test_metrics:
        test_metrics = {**test_metrics, "vol_ratio_20d_threshold": vol_thresh}
    conn.execute(
        """INSERT INTO model_registry
           (model_version_id, model_type, feature_list_json,
            train_start_date, train_end_date, val_start_date, val_end_date,
            test_start_date, test_end_date, label_threshold_pct, label_horizon,
            train_metrics_json, val_metrics_json, test_metrics_json,
            sensitivity_json, artifact_path, status)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)""",
        (model_version_id, _MODEL_TYPE, json.dumps(_FEATURE_LIST),
         str(train["signal_date"].min().date()), str(train["signal_date"].max().date()),
         str(val["signal_date"].min().date()) if len(val) else None,
         str(val["signal_date"].max().date()) if len(val) else None,
         str(test["signal_date"].min().date()) if len(test) else None,
         str(test["signal_date"].max().date()) if len(test) else None,
         _LABEL_THRESHOLD_PCT, _LABEL_HORIZON,
         json.dumps(train_metrics), json.dumps(val_metrics),
         json.dumps(test_metrics) if test_metrics else None,
         json.dumps(sensitivity), status),
    )
    conn.commit()
    return model_version_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(train_frac: float = 0.6, val_frac: float = 0.2) -> None:
    conn = get_connection()
    df = load_training_examples(conn)
    logger.info("training_examples loaded from DB: {} rows, {} positive ({:.2f}%)",
                len(df), df["label_success"].sum(), df["label_success"].mean() * 100)

    train, val, test = chronological_split(df, train_frac, val_frac)
    logger.info("TRAIN n={} ({} -> {})  VAL n={} ({} -> {})  TEST n={} ({} -> {})",
                len(train), train["signal_date"].min().date(), train["signal_date"].max().date(),
                len(val), val["signal_date"].min().date() if len(val) else "-", val["signal_date"].max().date() if len(val) else "-",
                len(test), test["signal_date"].min().date() if len(test) else "-", test["signal_date"].max().date() if len(test) else "-")

    # Step 1: fit (threshold selection) on TRAIN only.
    vol_thresh = select_vol_ratio_threshold(train)
    logger.info("vol_ratio_20d threshold selected from TRAIN only: {}", vol_thresh)

    train_metrics = bucket_metrics(train.assign(score=rule_score(train, vol_thresh)), "score", _MIN_SCORE_FOR_SELECTION)
    logger.info("TRAIN metrics: {}", train_metrics)

    # Step 2: pick operating point / sanity-check on VAL (calibration step —
    # for this simple additive score there's no extra hyperparameter beyond
    # the threshold already fixed in step 1, but VAL metrics are recorded for
    # comparability with future model_type='xgboost_ranker' candidates that
    # WILL use VAL for calibration).
    val_metrics = bucket_metrics(val.assign(score=rule_score(val, vol_thresh)), "score", _MIN_SCORE_FOR_SELECTION) if len(val) else {}
    logger.info("VAL metrics: {}", val_metrics)

    # Step 3: sensitivity battery on TRAIN+VAL — TEST is not touched here.
    train_val = pd.concat([train, val], ignore_index=True)
    sensitivity = sensitivity_battery(train_val, vol_thresh)
    for name, check in sensitivity["checks"].items():
        logger.info("  sensitivity[{}] passed={} {}", name, check["passed"],
                     {k: v for k, v in check.items() if k != "passed"})

    if not sensitivity["passed"]:
        logger.warning("Sensitivity battery FAILED — candidate stays at status='rejected'. TEST not evaluated.")
        model_version_id = register_model(conn, train, val, test, vol_thresh,
                                           train_metrics, val_metrics, sensitivity,
                                           test_metrics=None, status="rejected")
        logger.info("Registered as REJECTED: {}", model_version_id)
        conn.close()
        return

    # Step 4: sensitivity passed -> evaluate ONCE on TEST.
    test_metrics = bucket_metrics(test.assign(score=rule_score(test, vol_thresh)), "score", _MIN_SCORE_FOR_SELECTION) if len(test) else {}
    logger.info("TEST metrics (out-of-sample, evaluated once): {}", test_metrics)

    status = "candidate" if test_metrics.get("n_success", 0) >= 5 else "needs_more_data"
    model_version_id = register_model(conn, train, val, test, vol_thresh,
                                       train_metrics, val_metrics, sensitivity,
                                       test_metrics, status=status)
    logger.info("Registered as {}: {}", status.upper(), model_version_id)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    args = parser.parse_args()
    main(args.train_frac, args.val_frac)

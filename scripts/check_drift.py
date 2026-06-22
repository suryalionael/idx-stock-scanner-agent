#!/usr/bin/env python3
"""Daily drift monitoring — alert-only, never triggers a retrain by itself.

Three checks, each independent:
  1. Feature drift   — is the incoming feature distribution still consistent
                        with the trailing reference window? (could be a data
                        pipeline bug OR a genuine regime change — either way,
                        worth a human look before trusting the next retrain.)
  2. Prediction drift — is the rate of signals clearing the promoted model's
                        operating threshold consistent with its own history?
  3. Label drift      — is the REALIZED win rate within the selected bucket
                        still consistent with what the model's test_metrics
                        promised at promotion time?

Writes one row/day to live_monitoring. Every alert comes with a concrete
recommended action in the log output — this script never decides to retrain
or roll back on its own; a human reads the alert and runs
scripts/train_challenger.py / scripts/promote_challenger.py --rollback
deliberately.

Usage:
    python scripts/check_drift.py
    python scripts/check_drift.py --model-type rule_score --reference-days 30
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import pandas as pd
from loguru import logger

from stock_scanner.db.init_db import get_connection
from scripts.train_challenger import load_training_examples, rule_score, _MIN_SCORE_FOR_SELECTION

_FEATURE_DRIFT_REL_THRESHOLD = 0.5    # median shift >50% of reference -> flag
_SELECTION_RATE_REL_THRESHOLD = 0.5   # selected-rate shift >50% of reference -> flag
_LABEL_DRIFT_RATIO = 0.5              # realized precision < 50% of expected -> flag


def _latest_promoted(conn, model_type: str) -> dict | None:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT * FROM model_registry WHERE model_type=? AND status='promoted' "
        "ORDER BY promoted_at DESC LIMIT 1", (model_type,),
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def check_feature_drift(df: pd.DataFrame, reference_days: int) -> dict:
    """Compare the most recent signal_date's vol_ratio_20d median against the
    trailing `reference_days` window before it (excluding the most recent
    date itself)."""
    dates = sorted(df["signal_date"].unique())
    if len(dates) < 5:
        return {"flag": False, "note": "too few dates to assess drift"}
    latest_date = dates[-1]
    today = df[df["signal_date"] == latest_date]
    reference = df[(df["signal_date"] < latest_date)].tail(reference_days * 30)  # ~rows, generous

    results = {}
    flagged = False
    for col in ["vol_ratio_20d", "rsi14", "atr_pct"]:
        if col not in df.columns:
            continue
        ref_med = reference[col].median()
        today_med = today[col].median()
        if pd.isna(ref_med) or ref_med == 0 or pd.isna(today_med):
            continue
        rel_shift = abs(today_med - ref_med) / abs(ref_med)
        results[col] = {"reference_median": round(float(ref_med), 3),
                         "today_median": round(float(today_med), 3),
                         "relative_shift": round(float(rel_shift), 3)}
        if rel_shift > _FEATURE_DRIFT_REL_THRESHOLD:
            flagged = True
    return {"flag": flagged, "latest_date": str(latest_date.date()), "details": results}


def check_prediction_drift(df: pd.DataFrame, vol_thresh: float, reference_days: int) -> dict:
    """Selection rate (score>=3) on the most recent date(s) vs trailing
    reference — a sudden drop to ~0 or spike usually means a pipeline issue,
    not a real market shift, since the rule's inputs are technical and
    shouldn't vanish overnight."""
    df = df.assign(score=rule_score(df, vol_thresh))
    dates = sorted(df["signal_date"].unique())
    if len(dates) < 5:
        return {"flag": False, "note": "too few dates to assess drift"}
    latest_date = dates[-1]
    today = df[df["signal_date"] == latest_date]
    reference = df[df["signal_date"] < latest_date]

    today_rate = (today["score"] >= _MIN_SCORE_FOR_SELECTION).mean() if len(today) else None
    ref_rate = (reference["score"] >= _MIN_SCORE_FOR_SELECTION).mean() if len(reference) else None
    if today_rate is None or ref_rate is None or ref_rate == 0:
        return {"flag": False, "note": "insufficient data"}
    rel_shift = abs(today_rate - ref_rate) / ref_rate
    return {"flag": rel_shift > _SELECTION_RATE_REL_THRESHOLD,
            "latest_date": str(latest_date.date()),
            "today_selection_rate": round(float(today_rate), 4),
            "reference_selection_rate": round(float(ref_rate), 4),
            "relative_shift": round(float(rel_shift), 4)}


def check_label_drift(df: pd.DataFrame, vol_thresh: float, expected_precision: float,
                       reference_days: int) -> dict:
    """Realized win rate in the score>=3 bucket over the trailing window of
    EVALUATED outcomes vs what the promoted model's test_metrics promised."""
    df = df.assign(score=rule_score(df, vol_thresh))
    dates = sorted(df["signal_date"].unique())
    recent_dates = dates[-reference_days:] if len(dates) >= reference_days else dates
    recent = df[df["signal_date"].isin(recent_dates)]
    sel = recent[recent["score"] >= _MIN_SCORE_FOR_SELECTION]
    if len(sel) < 5:
        return {"flag": False, "note": f"only {len(sel)} selected rows in trailing window — too few to assess"}
    realized = sel["label_success"].mean()
    ratio = realized / expected_precision if expected_precision else None
    flag = ratio is not None and ratio < _LABEL_DRIFT_RATIO
    return {"flag": flag, "n_selected": int(len(sel)), "realized_precision": round(float(realized), 4),
            "expected_precision": round(float(expected_precision), 4),
            "ratio": round(float(ratio), 4) if ratio is not None else None}


def write_live_monitoring(conn, monitor_date: str, model_version_id: str | None,
                           n_signals: int, n_evaluated: int, realized_win_rate: float | None,
                           realized_precision: float | None, avg_predicted_prob: float | None,
                           feature_drift_flag: bool, alert_triggered: bool) -> None:
    conn.execute(
        """INSERT INTO live_monitoring
           (monitor_date, production_model_id, n_signals, n_evaluated,
            realized_win_rate, realized_precision_at_threshold, avg_predicted_prob,
            feature_drift_flag, alert_triggered)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(monitor_date) DO UPDATE SET
             production_model_id=excluded.production_model_id, n_signals=excluded.n_signals,
             n_evaluated=excluded.n_evaluated, realized_win_rate=excluded.realized_win_rate,
             realized_precision_at_threshold=excluded.realized_precision_at_threshold,
             avg_predicted_prob=excluded.avg_predicted_prob,
             feature_drift_flag=excluded.feature_drift_flag, alert_triggered=excluded.alert_triggered""",
        (monitor_date, model_version_id, n_signals, n_evaluated, realized_win_rate,
         realized_precision, avg_predicted_prob, int(feature_drift_flag), int(alert_triggered)),
    )
    conn.commit()


def main(model_type: str, reference_days: int) -> None:
    conn = get_connection()
    df = load_training_examples(conn)
    if df.empty:
        logger.warning("No training examples in DB — nothing to monitor.")
        conn.close()
        return

    monitor_date = str(df["signal_date"].max().date())
    n_signals = int((df["signal_date"] == df["signal_date"].max()).sum())
    n_evaluated = int(len(df))

    feature_drift = check_feature_drift(df, reference_days)
    logger.info("[feature_drift] flag={} {}", feature_drift["flag"],
                feature_drift.get("details", feature_drift.get("note")))
    if feature_drift["flag"]:
        logger.warning("ALERT (feature_drift): distribusi fitur teknikal bergeser >{}% dari baseline {} "
                        "hari terakhir. TINDAKAN: jangan ubah threshold/model otomatis. Cek dulu (a) log "
                        "validator.py untuk data-quality issue hari ini, (b) market_context apakah ini "
                        "genuinely regime baru. Kalau regime baru terkonfirmasi, catat di model_registry "
                        "saat retrain berikutnya — jangan retrain darurat hanya karena drift terdeteksi.",
                        int(_FEATURE_DRIFT_REL_THRESHOLD * 100), reference_days)

    champion = _latest_promoted(conn, model_type)
    prediction_drift = {"flag": False, "note": "no promoted model yet"}
    label_drift = {"flag": False, "note": "no promoted model yet"}
    avg_prob = None
    realized_win_rate = None
    realized_precision = None

    if champion is not None:
        metrics = json.loads(champion["test_metrics_json"] or champion["train_metrics_json"])
        vol_thresh = metrics.get("vol_ratio_20d_threshold", 2.0)
        expected_precision = metrics.get("precision", 0.1)

        prediction_drift = check_prediction_drift(df, vol_thresh, reference_days)
        logger.info("[prediction_drift] flag={} {}", prediction_drift["flag"], prediction_drift)
        if prediction_drift["flag"]:
            logger.warning("ALERT (prediction_drift): tingkat sinyal yang lolos skor>=3 berubah >{}% dari "
                            "baseline. TINDAKAN: cek log scan.yml hari ini — kemungkinan partial fetch "
                            "failure atau bug feature computation, BUKAN sinyal untuk retrain.",
                            int(_SELECTION_RATE_REL_THRESHOLD * 100))

        label_drift = check_label_drift(df, vol_thresh, expected_precision, reference_days)
        logger.info("[label_drift] flag={} {}", label_drift["flag"], label_drift)
        if label_drift["flag"]:
            logger.warning("ALERT (label_drift): win rate realized ({:.1%}) jauh di bawah ekspektasi model "
                            "saat promoted ({:.1%}) selama {} hari terakhir. TINDAKAN: jangan retrain "
                            "otomatis. (1) Cek market_context — apakah ini regime risk-off berkepanjangan "
                            "(sudah terbukti di validation report: hit rate bisa turun ke 12% saat market "
                            "lemah, vs 47% saat market kuat — itu wajar, bukan model rusak). (2) Kalau "
                            "berlanjut >2 minggu tanpa penjelasan regime, pertimbangkan "
                            "'python scripts/promote_challenger.py --rollback' ke model sebelumnya kalau "
                            "ada. (3) Jangan retrain ad-hoc di luar jadwal mingguan hanya karena alert ini.",
                            label_drift.get("realized_precision", 0), expected_precision, reference_days)

        sel = df.assign(score=rule_score(df, vol_thresh))
        sel = sel[sel["score"] >= _MIN_SCORE_FOR_SELECTION]
        avg_prob = float(sel["score"].mean()) if len(sel) else None
        recent = df[df["signal_date"] >= sorted(df["signal_date"].unique())[-reference_days:][0]]
        realized_win_rate = float(recent["label_success"].mean()) if len(recent) else None
        realized_precision = label_drift.get("realized_precision")
    else:
        logger.info("No promoted model for model_type={} yet — prediction/label drift skipped "
                    "(feature drift still checked above, independent of model status).", model_type)

    alert_triggered = feature_drift["flag"] or prediction_drift["flag"] or label_drift["flag"]
    write_live_monitoring(conn, monitor_date, champion["model_version_id"] if champion else None,
                           n_signals, n_evaluated, realized_win_rate, realized_precision, avg_prob,
                           feature_drift["flag"], alert_triggered)
    logger.info("live_monitoring row written for {} (alert_triggered={})", monitor_date, alert_triggered)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="rule_score")
    parser.add_argument("--reference-days", type=int, default=30)
    args = parser.parse_args()
    main(args.model_type, args.reference_days)

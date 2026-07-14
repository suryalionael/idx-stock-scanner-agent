#!/usr/bin/env python3
"""Promotion gate — champion vs challenger, with explicit, non-negotiable
criteria. This is a SEPARATE step from training (scripts/train_challenger.py)
on purpose: a candidate that passes the sensitivity battery does NOT get
promoted automatically. Promotion requires this script to be run and to
decide 'promoted' — never a side effect of training.

Champion  = the model_registry row with status='promoted' for a given
            model_type. There is at most one per model_type.
Challenger = the most recent status='candidate' row for that model_type.

Every run writes a promotion_decisions row — 'promoted', 'rejected', or
'needs_more_data' — even when the answer is no. Rejections are evidence,
not failures to hide.

Usage:
    python scripts/promote_challenger.py                  # evaluate latest candidate
    python scripts/promote_challenger.py --model-type rule_score
    python scripts/promote_challenger.py --rollback        # revert to the
                                                            # previously-promoted model
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
from stock_scanner.pipeline.challenger_score import compute_rule_score as rule_score
from scripts.train_challenger import load_training_examples, bucket_metrics, _MIN_SCORE_FOR_SELECTION

# --- Promotion criteria (see docs/SELF_IMPROVING_ARCHITECTURE.md §5) -------
_MIN_TEST_POSITIVES = 30          # below this -> 'needs_more_data', no promotion either way
_MIN_RECALL_RATIO = 0.70          # challenger recall must be >= 70% of champion's
_MIN_TICKER_EXCL_LIFT = 1.0       # challenger's precision-after-ticker-exclusion must clear this vs baseline


def _latest_row(conn, model_type: str, status: str) -> dict | None:
    cur = conn.cursor()
    row = cur.execute(
        "SELECT * FROM model_registry WHERE model_type=? AND status=? ORDER BY trained_at DESC LIMIT 1",
        (model_type, status),
    ).fetchone()
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def _score_with_recorded_threshold(df: pd.DataFrame, model_row: dict) -> pd.Series:
    metrics = json.loads(model_row["test_metrics_json"] or model_row["train_metrics_json"])
    thresh = metrics.get("vol_ratio_20d_threshold", 2.0)
    return rule_score(df, thresh)


def _ticker_exclusion_precision(df: pd.DataFrame, score: pd.Series) -> float | None:
    tv = df.assign(score=score)
    pos = tv[tv["label_success"] == 1]
    if pos.empty:
        return None
    top_ticker = pos["ticker"].value_counts().idxmax()
    excl = tv[tv["ticker"] != top_ticker]
    m = bucket_metrics(excl, "score", _MIN_SCORE_FOR_SELECTION)
    return m["precision"]


def evaluate_promotion(conn, model_type: str) -> dict:
    challenger = _latest_row(conn, model_type, "candidate")
    if challenger is None:
        return {"decision": "no_candidate", "reason": f"No status='candidate' row for model_type={model_type}"}

    champion = _latest_row(conn, model_type, "promoted")

    df = load_training_examples(conn)
    test_start = pd.Timestamp(challenger["test_start_date"])
    test_end = pd.Timestamp(challenger["test_end_date"])
    test_df = df[(df["signal_date"] >= test_start) & (df["signal_date"] <= test_end)]

    challenger_score = _score_with_recorded_threshold(test_df, challenger)
    challenger_metrics = bucket_metrics(test_df.assign(score=challenger_score), "score", _MIN_SCORE_FOR_SELECTION)
    challenger_metrics["precision_excl_top_ticker"] = _ticker_exclusion_precision(test_df, challenger_score)

    if champion is None:
        # First-ever model of this type — no champion to compare against.
        # Still gated: must independently clear a minimum bar, not promoted
        # just because it's the only candidate.
        if challenger_metrics["n_success"] < _MIN_TEST_POSITIVES:
            decision = "needs_more_data"
            reason = (f"No existing champion, but only {challenger_metrics['n_success']} positive test "
                      f"examples (< {_MIN_TEST_POSITIVES} minimum) — not enough evidence either way yet.")
        elif (challenger_metrics["lift"] or 0) > 2.0:
            decision = "promoted"
            reason = f"No existing champion; challenger lift={challenger_metrics['lift']:.2f} clears the minimum bar (2.0x)."
        else:
            decision = "rejected"
            reason = f"No existing champion; challenger lift={challenger_metrics['lift']:.2f} below minimum bar (2.0x)."
        champion_metrics = None
    else:
        champion_score = _score_with_recorded_threshold(test_df, champion)
        champion_metrics = bucket_metrics(test_df.assign(score=champion_score), "score", _MIN_SCORE_FOR_SELECTION)

        reasons = []
        ok = True
        if challenger_metrics["n_success"] < _MIN_TEST_POSITIVES:
            ok = False
            reasons.append(f"only {challenger_metrics['n_success']} positive test examples "
                           f"(< {_MIN_TEST_POSITIVES} minimum) — decision is needs_more_data regardless of metrics")
        else:
            precision_excl = challenger_metrics["precision_excl_top_ticker"]
            champ_precision = champion_metrics["precision"] or 0
            if precision_excl is None or precision_excl <= champ_precision:
                ok = False
                reasons.append(f"challenger precision-after-ticker-exclusion "
                               f"({precision_excl}) must exceed champion's point-estimate precision "
                               f"({champ_precision:.4f}) — it does not")
            recall_ratio = (challenger_metrics["recall"] or 0) / (champion_metrics["recall"] or 1e-9)
            if recall_ratio < _MIN_RECALL_RATIO:
                ok = False
                reasons.append(f"challenger recall ratio vs champion = {recall_ratio:.2f} "
                               f"(< {_MIN_RECALL_RATIO} minimum)")
        if challenger_metrics["n_success"] >= _MIN_TEST_POSITIVES and ok:
            decision = "promoted"
            reason = "All promotion criteria met: min sample size, precision-after-exclusion > champion, recall ratio ok."
        elif challenger_metrics["n_success"] < _MIN_TEST_POSITIVES:
            decision = "needs_more_data"
            reason = "; ".join(reasons)
        else:
            decision = "rejected"
            reason = "; ".join(reasons)

    return {
        "decision": decision, "reason": reason,
        "challenger_model_id": challenger["model_version_id"],
        "production_model_id": champion["model_version_id"] if champion else None,
        "challenger_metrics": challenger_metrics, "champion_metrics": champion_metrics,
    }


def apply_decision(conn, result: dict) -> None:
    conn.execute(
        """INSERT INTO promotion_decisions
           (challenger_model_id, production_model_id, decision,
            challenger_metrics_json, production_metrics_json, reason)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (result["challenger_model_id"], result["production_model_id"], result["decision"],
         json.dumps(result["challenger_metrics"]),
         json.dumps(result["champion_metrics"]) if result["champion_metrics"] else None,
         result["reason"]),
    )
    if result["decision"] == "promoted":
        now = datetime.now(timezone.utc).isoformat()
        if result["production_model_id"]:
            conn.execute(
                "UPDATE model_registry SET status='retired', retired_at=? WHERE model_version_id=?",
                (now, result["production_model_id"]),
            )
        conn.execute(
            "UPDATE model_registry SET status='promoted', promoted_at=? WHERE model_version_id=?",
            (now, result["challenger_model_id"]),
        )
    elif result["decision"] == "rejected":
        conn.execute(
            "UPDATE model_registry SET status='rejected' WHERE model_version_id=?",
            (result["challenger_model_id"],),
        )
    # 'needs_more_data' -> leave challenger status as 'candidate', re-evaluate later.
    conn.commit()


def rollback(conn, model_type: str) -> dict:
    """Revert to the most recently retired model of this type — a metadata
    flip, not a deploy. Requires a human to run this script; never automatic."""
    current = _latest_row(conn, model_type, "promoted")
    previous = _latest_row(conn, model_type, "retired")
    if previous is None:
        return {"decision": "rollback_failed", "reason": "No previously-retired model to roll back to."}
    now = datetime.now(timezone.utc).isoformat()
    if current:
        conn.execute("UPDATE model_registry SET status='retired', retired_at=? WHERE model_version_id=?",
                     (now, current["model_version_id"]))
    conn.execute("UPDATE model_registry SET status='promoted', promoted_at=?, retired_at=NULL WHERE model_version_id=?",
                 (now, previous["model_version_id"]))
    conn.execute(
        """INSERT INTO promotion_decisions
           (challenger_model_id, production_model_id, decision, reason)
           VALUES (?, ?, 'rollback', ?)""",
        (previous["model_version_id"], current["model_version_id"] if current else None,
         "Manual rollback triggered."),
    )
    conn.commit()
    return {"decision": "rolled_back", "restored_model_id": previous["model_version_id"],
            "retired_model_id": current["model_version_id"] if current else None}


def main(model_type: str, do_rollback: bool) -> None:
    conn = get_connection()
    if do_rollback:
        result = rollback(conn, model_type)
        logger.info("Rollback result: {}", result)
        conn.close()
        return

    result = evaluate_promotion(conn, model_type)
    logger.info("Decision: {}", result["decision"])
    logger.info("Reason: {}", result["reason"])
    logger.info("Challenger metrics: {}", result.get("challenger_metrics"))
    logger.info("Champion metrics: {}", result.get("champion_metrics"))

    if result["decision"] in ("promoted", "rejected", "needs_more_data"):
        apply_decision(conn, result)
        logger.info("promotion_decisions row written; model_registry status updated as applicable.")
    else:
        logger.warning("No action taken: {}", result.get("reason"))
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default="rule_score")
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    main(args.model_type, args.rollback)

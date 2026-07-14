#!/usr/bin/env python3
"""Manual review tool for knowledge_base hypotheses. This is where a human
reads a candidate and records a decision — it does NOT translate anything
into a train_challenger.py run, does NOT touch model_registry, and does NOT
promote anything. See docs/LEARNING_AGENT_RUNBOOK.md and
docs/KNOWLEDGE_BASE_POLICY.md.

Usage:
    python scripts/review_knowledge_base.py --list
    python scripts/review_knowledge_base.py --list --status candidate
    python scripts/review_knowledge_base.py --show <hypothesis_id>
    python scripts/review_knowledge_base.py --decide <hypothesis_id> --status reviewed --reviewed-by "your_name"
    python scripts/review_knowledge_base.py --decide <hypothesis_id> --status archived --reviewed-by "your_name"
    python scripts/review_knowledge_base.py --decide <hypothesis_id> --status tested_passed \\
        --reviewed-by "your_name" --linked-model-version-id rule_score_20260720T050000
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from stock_scanner.db.init_db import get_connection  # noqa: E402
from stock_scanner.db.knowledge_base import (  # noqa: E402
    VALID_STATUSES,
    export_knowledge_base,
    load_knowledge_base,
    update_status,
)

_NOT_INITIALIZED_MSG = (
    "knowledge_base table not found. Run the pipeline first:\n"
    "  python scripts/run_learning_agent.py\n"
    "  python scripts/run_pattern_dedup.py\n"
    "  python scripts/run_hypothesis_agent.py --mock\n"
    "See docs/LEARNING_AGENT_RUNBOOK.md."
)


def _safe_load(conn: sqlite3.Connection, status: str | None = None) -> pd.DataFrame | None:
    """load_knowledge_base(), but a fresh checkout where the pipeline has
    never run (table doesn't exist yet) gets a clear message instead of a
    raw pandas/sqlite traceback. Returns None on failure — callers must
    check for that, not assume a DataFrame."""
    try:
        return load_knowledge_base(conn, status=status)
    except (pd.errors.DatabaseError, sqlite3.OperationalError) as e:
        if "no such table" in str(e).lower():
            print(_NOT_INITIALIZED_MSG)
        else:
            print(f"Could not read knowledge_base: {e}")
        return None


def _validate_status_filter(status: str | None) -> bool:
    """--list --status is a read filter, not a mutation — a typo should not
    be silently reinterpreted as "zero matching rows" (which looks
    identical to a correctly-spelled status that just has no candidates
    yet). Returns False (after printing a warning) for an unrecognized
    status; True otherwise, including for None (no filter)."""
    if status is not None and status not in VALID_STATUSES:
        print(f"Warning: {status!r} is not a recognized status — this will always return zero rows.")
        print(f"Valid statuses: {sorted(VALID_STATUSES)}")
        return False
    return True


def cmd_list(status: str | None) -> None:
    if not _validate_status_filter(status):
        return
    conn = get_connection()
    df = _safe_load(conn, status=status)
    conn.close()
    if df is None:
        return
    if df.empty:
        print(f"No hypotheses{' with status=' + status if status else ''}.")
        return

    if status is None:
        counts = df["status"].value_counts().to_dict()
        summary = ", ".join(f"{s}={n}" for s, n in sorted(counts.items()))
        print(f"{len(df)} total ({summary})")
        print()

    for _, row in df.sort_values("generated_at", ascending=False).iterrows():
        text = row["hypothesis"][:80] + ("..." if len(row["hypothesis"]) > 80 else "")
        print(f"[{row['status']:14s}] {row['hypothesis_id']}  conf={row['confidence']:.2f}  "
              f"support={row['supporting_trades']:>3}  {text}")


def cmd_show(hypothesis_id: str) -> None:
    conn = get_connection()
    df = _safe_load(conn)
    conn.close()
    if df is None:
        return
    match = df[df["hypothesis_id"] == hypothesis_id]
    if match.empty:
        print(f"No hypothesis found with id {hypothesis_id!r}")
        return
    row = match.iloc[0]
    print("--- LLM-authored (qualitative — not statistical proof) ---")
    print(f"hypothesis:          {row['hypothesis']}")
    print(f"confidence:          {row['confidence']}  (LLM-assigned, qualitative — not a p-value)")
    print(f"affected_sector:     {row['affected_sector']}")
    print(f"expected_effect:     {row['expected_effect']}")
    print()
    print("--- Code-derived / structural (not LLM output) ---")
    print(f"hypothesis_id:       {row['hypothesis_id']}")
    print(f"status:              {row['status']}")
    print(f"generated_at:        {row['generated_at']}")
    print(f"supporting_trades:   {row['supporting_trades']}")
    print(f"reviewed_by:         {row['reviewed_by']}")
    print(f"linked_model_version_id: {row['linked_model_version_id']}")
    print(f"source_run_id:       {row['source_run_id']}")

    pattern = json.loads(row["pattern_json"] or "{}")
    rep = pattern.get("representative", {})
    if rep:
        print()
        print("Source statistical pattern (the actual numbers to judge trust from):")
        print(f"  slice_definition:          {rep.get('slice_definition')}")
        print(f"  n / n_success:             {rep.get('n')} / {rep.get('n_success')}")
        print(f"  win_rate (raw / shrunk):   {rep.get('win_rate')} / {rep.get('win_rate_shrunk')}")
        print(f"  ci_lower:                  {rep.get('ci_lower')}")
        print(f"  p_value_adjusted:          {rep.get('p_value_adjusted')}")
        print(f"  ticker_concentration_flag: {rep.get('ticker_concentration_flag')}")
        print(f"  time_split_stable:         {rep.get('time_split_stable')}")
        print(f"  cluster member_count:      {pattern.get('member_count')}")


def cmd_decide(hypothesis_id: str, status: str, reviewed_by: str | None, linked_model_version_id: str | None) -> None:
    if status not in VALID_STATUSES:
        print(f"Invalid status {status!r}. Must be one of: {sorted(VALID_STATUSES)}")
        return

    conn = get_connection()
    before = _safe_load(conn)
    if before is None:
        conn.close()
        return
    prior = before[before["hypothesis_id"] == hypothesis_id]
    old_status = prior.iloc[0]["status"] if not prior.empty else None

    try:
        n_updated = update_status(conn, hypothesis_id, status, reviewed_by=reviewed_by,
                                  linked_model_version_id=linked_model_version_id)
    except Exception as e:  # noqa: BLE001 — surface FK violations etc. plainly, don't silently swallow
        print(f"Could not update: {e}")
        conn.close()
        return
    if n_updated == 0:
        print(f"No hypothesis found with id {hypothesis_id!r} — nothing updated.")
        conn.close()
        return
    export_path = export_knowledge_base(conn)
    conn.close()
    print(f"{hypothesis_id}: {old_status} -> {status}"
          + (f"  (reviewed_by={reviewed_by})" if reviewed_by else "")
          + (f"  (linked_model_version_id={linked_model_version_id})" if linked_model_version_id else ""))
    print(f"Mirror re-exported -> {export_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual review tool for knowledge_base hypotheses")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--show", metavar="HYPOTHESIS_ID")
    parser.add_argument("--decide", metavar="HYPOTHESIS_ID")
    parser.add_argument("--status", help="For --list: filter. For --decide: the new status (required).")
    parser.add_argument("--reviewed-by", default=None)
    parser.add_argument("--linked-model-version-id", default=None)
    args = parser.parse_args()

    if args.list:
        cmd_list(args.status)
    elif args.show:
        cmd_show(args.show)
    elif args.decide:
        if not args.status:
            print("--decide requires --status")
            sys.exit(1)
        cmd_decide(args.decide, args.status, args.reviewed_by, args.linked_model_version_id)
    else:
        parser.print_help()

#!/usr/bin/env python3
"""Backfill evaluation script — compute realized outcomes for all historical ranked files.

Reads every ranked_YYYY-MM-DD.csv in data/ranked/, evaluates each against
forward OHLCV data in data/raw/, and writes eval_YYYY-MM-DD.csv to data/evaluation/.

Files with insufficient forward data (< min_forward_days) are skipped with a log message.

Usage:
    python scripts/backfill_evaluate.py
    python scripts/backfill_evaluate.py --horizon 10 --min-fwd 5 --overwrite
    python scripts/backfill_evaluate.py --ranked-dir data/ranked_v2  # evaluate v2 output
"""
import argparse
import sys
from pathlib import Path

# Ensure repo root is on path when running as a script
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import pandas as pd
from loguru import logger

from stock_scanner.pipeline.evaluator import (
    evaluate_all_files,
    summarize_evaluations,
    score_quintile_monotonicity,
)


def main(
    ranked_dir: Path,
    raw_dir: Path,
    eval_dir: Path,
    horizon_days: int,
    min_forward_days: int,
    overwrite: bool,
) -> None:
    logger.info("=" * 60)
    logger.info("IDX Scanner — Backfill Evaluation")
    logger.info(f"  ranked_dir   : {ranked_dir}")
    logger.info(f"  raw_dir      : {raw_dir}")
    logger.info(f"  eval_dir     : {eval_dir}")
    logger.info(f"  horizon_days : {horizon_days}")
    logger.info(f"  min_fwd_days : {min_forward_days}")
    logger.info(f"  overwrite    : {overwrite}")
    logger.info("=" * 60)

    # --- Run evaluations ---
    written = evaluate_all_files(
        ranked_dir=ranked_dir,
        raw_dir=raw_dir,
        eval_dir=eval_dir,
        horizon_days=horizon_days,
        min_forward_days=min_forward_days,
        overwrite=overwrite,
    )

    if not written:
        logger.warning(
            "No evaluation files were written. Possible reasons:\n"
            "  • All ranked files have insufficient forward data "
            f"(need ≥ {min_forward_days} trading days after signal date)\n"
            "  • All eval files already exist (use --overwrite to redo)\n"
            "  • data/raw/ files missing or empty"
        )
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation complete — {len(written)} files written:")
    for p in written:
        logger.info(f"  {p}")

    # --- Summary across all eval files ---
    logger.info(f"\n{'='*60}")
    logger.info("OVERALL SUMMARY (all evaluated dates)")
    logger.info("=" * 60)

    summary_df = summarize_evaluations(eval_dir)
    if summary_df.empty:
        logger.warning("No evaluation data to summarize.")
        return

    print("\n" + summary_df.to_string(index=False))

    # Overall aggregate
    all_evals_frames = []
    for f in sorted(eval_dir.glob("eval_*.csv")):
        df = pd.read_csv(f)
        if "realized_R" in df.columns:
            all_evals_frames.append(df)

    if all_evals_frames:
        all_evals = pd.concat(all_evals_frames, ignore_index=True)
        r = all_evals["realized_R"]
        wins   = r[r > 0]
        losses = r[r <= 0]
        pf     = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else float("inf")

        print(f"\n{'─'*60}")
        print(f"AGGREGATE (n={len(all_evals)} signals, {len(all_evals_frames)} dates)")
        print(f"  Win rate      : {len(wins)/len(all_evals)*100:.1f}%")
        print(f"  Avg realized R: {r.mean():.3f}")
        print(f"  Profit factor : {pf:.2f}")
        print(f"  Avg MFE       : {all_evals['MFE_pct'].mean():.2f}%")
        print(f"  Avg MAE       : {all_evals['MAE_pct'].mean():.2f}%")
        print(f"  TP hits       : {(all_evals['outcome']=='TP').sum()} ({(all_evals['outcome']=='TP').mean()*100:.1f}%)")
        print(f"  Stop hits     : {(all_evals['outcome']=='STOP').sum()} ({(all_evals['outcome']=='STOP').mean()*100:.1f}%)")
        print(f"  Still open    : {(all_evals['outcome']=='OPEN').sum()} ({(all_evals['outcome']=='OPEN').mean()*100:.1f}%)")

        # --- Per signal type breakdown ---
        if "signal" in all_evals.columns:
            print(f"\n{'─'*60}")
            print("PER SIGNAL TYPE breakdown")
            print(f"{'─'*60}")
            sig_grp = all_evals.groupby("signal")["realized_R"]
            for sig, grp in sig_grp:
                w = grp[grp > 0]
                l = grp[grp <= 0]
                pf_sig = w.sum() / abs(l.sum()) if abs(l.sum()) > 0 else float("inf")
                print(
                    f"  {sig:12s}: n={len(grp):4d} | "
                    f"WR={len(w)/len(grp)*100:5.1f}% | "
                    f"avgR={grp.mean():+.3f} | "
                    f"PF={pf_sig:.2f}"
                )

        # --- Score quintile monotonicity ---
        print(f"\n{'─'*60}")
        print("SCORE QUINTILE MONOTONICITY (total_score Q1=lowest, Q5=highest)")
        print(f"{'─'*60}")
        mono_df = score_quintile_monotonicity(eval_dir)
        if not mono_df.empty:
            print(mono_df.to_string(index=False))
            # Check monotonicity
            avg_Rs = mono_df["avg_R"].tolist()
            is_mono = all(avg_Rs[i] <= avg_Rs[i+1] for i in range(len(avg_Rs)-1))
            print(f"\n  Monotonically increasing: {'✅ YES' if is_mono else '❌ NO (signal quality concern)'}")
        else:
            print("  Not enough data for quintile analysis (need ≥20 evaluated signals).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill evaluation for ranked signals")
    parser.add_argument(
        "--ranked-dir", type=Path,
        default=repo_root / "data" / "ranked",
        help="Directory containing ranked_*.csv files",
    )
    parser.add_argument(
        "--raw-dir", type=Path,
        default=repo_root / "data" / "raw",
        help="Directory containing raw OHLCV parquets",
    )
    parser.add_argument(
        "--eval-dir", type=Path,
        default=repo_root / "data" / "evaluation",
        help="Output directory for eval_*.csv files",
    )
    parser.add_argument(
        "--horizon", type=int, default=10,
        help="Forward-looking window in trading days (default 10)",
    )
    parser.add_argument(
        "--min-fwd", type=int, default=5,
        help="Minimum forward days required to evaluate a file (default 5)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-evaluate even if output already exists",
    )
    args = parser.parse_args()

    main(
        ranked_dir=args.ranked_dir,
        raw_dir=args.raw_dir,
        eval_dir=args.eval_dir,
        horizon_days=args.horizon,
        min_forward_days=args.min_fwd,
        overwrite=args.overwrite,
    )

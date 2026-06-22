"""Signal evaluator — compute realized outcomes for historical ranked signals.

For each row in a ranked CSV file, looks up forward OHLCV data and computes:
    hit_tp       : bool — price reached TP within horizon
    hit_stop     : bool — price hit cutloss within horizon
    outcome      : "TP" | "STOP" | "OPEN" (still open at horizon end)
    exit_day     : int  — which forward trading day outcome occurred (NaN = OPEN)
    realized_pct : float — actual return % from entry to exit price
    realized_R   : float — (exit - entry) / (entry - cutloss) — R-multiple
    MFE_pct      : float — max favorable excursion % from entry within horizon
    MAE_pct      : float — max adverse excursion % from entry (always ≥ 0)
    forward_days_used: int — actual number of trading days evaluated

Evaluation mode (column: evaluation_method):
    "levels"       — used pre-computed entry_low / tp_low / cutloss from ranked file
    "fallback_3pct"— inferred from close ± default_risk_pct (for older files without levels)

Public API
----------
evaluate_signal_file(ranked_path, raw_dir, horizon_days, min_forward_days) -> pd.DataFrame
    Evaluate all signals in one ranked CSV. Returns DataFrame with evaluation columns.

evaluate_all_files(ranked_dir, raw_dir, eval_dir, ...) -> list[Path]
    Convenience wrapper: evaluate all ranked files, save to eval_dir.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_HORIZON    = 10   # trading days to look forward
_DEFAULT_MIN_FWD    = 5    # minimum forward days required to evaluate
_DEFAULT_RISK_PCT   = 3.0  # % risk / reward for fallback mode (no levels)
_LEVEL_COLS         = ["entry_low", "tp_low", "cutloss"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_signal_file(
    ranked_path: Path,
    raw_dir: Path,
    horizon_days: int = _DEFAULT_HORIZON,
    min_forward_days: int = _DEFAULT_MIN_FWD,
    default_risk_pct: float = _DEFAULT_RISK_PCT,
) -> pd.DataFrame:
    """Evaluate one ranked CSV file against forward OHLCV data.

    Args:
        ranked_path      : Path to ranked_{YYYY-MM-DD}.csv
        raw_dir          : Path to data/raw/ (contains {ticker}.parquet files)
        horizon_days     : Maximum trading days to look forward for outcome
        min_forward_days : Skip evaluation if fewer forward days available
        default_risk_pct : % used for TP/cutloss when levels are missing

    Returns:
        DataFrame with all original columns + evaluation columns.
        Empty DataFrame if the file has insufficient forward data.
    """
    ranked_path = Path(ranked_path)
    if not ranked_path.exists():
        logger.warning(f"Ranked file not found: {ranked_path}")
        return pd.DataFrame()

    # Extract signal date from filename: ranked_YYYY-MM-DD.csv
    signal_date_str = ranked_path.stem.replace("ranked_", "")
    try:
        signal_date = pd.Timestamp(signal_date_str)
    except Exception:
        logger.error(f"Cannot parse date from filename: {ranked_path.name}")
        return pd.DataFrame()

    df = pd.read_csv(ranked_path)
    if df.empty:
        logger.info(f"{signal_date_str}: ranked file is empty — skipping")
        return pd.DataFrame()

    has_levels = all(c in df.columns for c in _LEVEL_COLS)
    eval_method = "levels" if has_levels else f"fallback_{int(default_risk_pct)}pct"

    logger.info(
        f"Evaluating {signal_date_str} ({len(df)} signals, "
        f"method={eval_method}, horizon={horizon_days}d)"
    )

    results = []
    skipped = 0

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", ""))
        close  = _sf(row.get("close"))

        if not ticker or close is None or close <= 0:
            skipped += 1
            continue

        # Load forward OHLCV
        raw_path = raw_dir / f"{ticker}.parquet"
        if not raw_path.exists():
            raw_path = raw_dir / f"{ticker.replace('.JK', '')}.parquet"
        if not raw_path.exists():
            logger.debug(f"{ticker}: raw file not found — skip")
            skipped += 1
            continue

        try:
            raw_df = pd.read_parquet(raw_path)
            raw_df["date"] = pd.to_datetime(raw_df["date"])
            raw_df = raw_df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.debug(f"{ticker}: cannot load raw — {e}")
            skipped += 1
            continue

        # Slice forward rows (strictly after signal_date)
        fwd = raw_df[raw_df["date"] > signal_date].head(horizon_days)
        if len(fwd) < min_forward_days:
            skipped += 1
            continue

        # Determine entry / TP / cutloss
        if has_levels:
            entry  = _sf(row.get("entry_low")) or close
            tp     = _sf(row.get("tp_low"))
            cl     = _sf(row.get("cutloss"))
            # Fallback if stored levels are 0
            if not tp or tp <= entry:
                tp = entry * (1 + default_risk_pct / 100 * 1.8)
            if not cl or cl >= entry:
                cl = entry * (1 - default_risk_pct / 100)
        else:
            entry = close
            tp    = close * (1 + default_risk_pct / 100 * 1.8)
            cl    = close * (1 - default_risk_pct / 100)

        risk = entry - cl
        if risk <= 0:
            risk = entry * 0.03   # fallback 3%

        # Simulate trade through forward window
        outcome_day = None
        exit_price  = None
        outcome     = "OPEN"

        fwd_highs = fwd["high"].values
        fwd_lows  = fwd["low"].values
        fwd_closes= fwd["close"].values

        for i, (h, lo, c_fwd) in enumerate(zip(fwd_highs, fwd_lows, fwd_closes)):
            # Check TP first (more conservative), then stop
            if h >= tp:
                outcome     = "TP"
                outcome_day = i + 1
                exit_price  = tp
                break
            if lo <= cl:
                outcome     = "STOP"
                outcome_day = i + 1
                exit_price  = cl
                break

        if outcome == "OPEN":
            # Mark-to-market at horizon close
            exit_price = float(fwd_closes[-1])

        realized_pct = (exit_price - entry) / entry * 100 if entry > 0 else 0.0
        realized_R   = (exit_price - entry) / risk if risk > 0 else 0.0

        # MFE / MAE over the entire forward window evaluated
        all_highs = fwd["high"].values
        all_lows  = fwd["low"].values
        MFE_pct   = (max(all_highs) - entry) / entry * 100 if entry > 0 else 0.0
        MAE_pct   = (entry - min(all_lows))  / entry * 100 if entry > 0 else 0.0
        MFE_pct   = max(0.0, MFE_pct)
        MAE_pct   = max(0.0, MAE_pct)

        eval_row = dict(row)
        eval_row.update({
            "signal_date":        signal_date_str,
            "evaluation_method":  eval_method,
            "entry_used":         round(entry, 2),
            "tp_used":            round(tp, 2),
            "cl_used":            round(cl, 2),
            "forward_days_used":  len(fwd),
            "hit_tp":             outcome == "TP",
            "hit_stop":           outcome == "STOP",
            "outcome":            outcome,
            "exit_day":           outcome_day,
            "exit_price":         round(exit_price, 2) if exit_price else None,
            "realized_pct":       round(realized_pct, 3),
            "realized_R":         round(realized_R, 3),
            "MFE_pct":            round(MFE_pct, 3),
            "MAE_pct":            round(MAE_pct, 3),
        })
        results.append(eval_row)

    if not results:
        logger.info(f"{signal_date_str}: no evaluatable signals (skipped={skipped})")
        return pd.DataFrame()

    out = pd.DataFrame(results)
    logger.info(
        f"{signal_date_str}: {len(out)} evaluated, {skipped} skipped | "
        f"TP={out['hit_tp'].sum()}, STOP={out['hit_stop'].sum()}, "
        f"OPEN={(out['outcome']=='OPEN').sum()}"
    )
    return out


def evaluate_all_files(
    ranked_dir: Path,
    raw_dir: Path,
    eval_dir: Path,
    horizon_days: int = _DEFAULT_HORIZON,
    min_forward_days: int = _DEFAULT_MIN_FWD,
    overwrite: bool = False,
) -> list[Path]:
    """Evaluate all ranked_*.csv files in ranked_dir and save results to eval_dir.

    Args:
        ranked_dir       : Path to data/ranked/
        raw_dir          : Path to data/raw/
        eval_dir         : Output directory for eval_*.csv files
        horizon_days     : Forward-looking window in trading days
        min_forward_days : Skip files with fewer forward days
        overwrite        : If False, skip files that already have an eval output

    Returns:
        List of paths to evaluation CSV files that were written.
    """
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    ranked_files = sorted(Path(ranked_dir).glob("ranked_*.csv"))
    if not ranked_files:
        logger.warning(f"No ranked files found in {ranked_dir}")
        return []

    written: list[Path] = []
    for rp in ranked_files:
        date_str = rp.stem.replace("ranked_", "")
        out_path = eval_dir / f"eval_{date_str}.csv"

        if out_path.exists() and not overwrite:
            logger.info(f"{date_str}: eval already exists — skip (use overwrite=True to redo)")
            continue

        df_eval = evaluate_signal_file(
            rp, raw_dir,
            horizon_days=horizon_days,
            min_forward_days=min_forward_days,
        )
        if df_eval.empty:
            logger.info(f"{date_str}: no output (insufficient forward data or empty ranked)")
            continue

        df_eval.to_csv(out_path, index=False)
        logger.info(f"Saved → {out_path} ({len(df_eval)} rows)")
        written.append(out_path)

    return written


def summarize_evaluations(eval_dir: Path) -> pd.DataFrame:
    """Load all eval_*.csv files and return a summary table.

    Returns:
        DataFrame with one row per eval date, columns:
            date, n_signals, win_rate, avg_realized_R, profit_factor,
            expectancy_R, pct_TP, pct_STOP, pct_OPEN
        plus per-signal-type breakdown.
    """
    files = sorted(Path(eval_dir).glob("eval_*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty or "realized_R" not in df.columns:
            continue
        df["_file"] = f.stem.replace("eval_", "")
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    all_evals = pd.concat(frames, ignore_index=True)

    rows = []
    for date, grp in all_evals.groupby("_file"):
        r = grp["realized_R"]
        wins   = r[r > 0]
        losses = r[r <= 0]

        pf = (wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else float("inf")
        rows.append({
            "date":           date,
            "n_signals":      len(grp),
            "win_rate":       round(len(wins) / len(grp), 3) if len(grp) > 0 else 0,
            "avg_realized_R": round(r.mean(), 3),
            "profit_factor":  round(pf, 2),
            "expectancy_R":   round(r.mean(), 3),    # same as avg_R here
            "pct_TP":         round((grp["outcome"] == "TP").mean() * 100, 1),
            "pct_STOP":       round((grp["outcome"] == "STOP").mean() * 100, 1),
            "pct_OPEN":       round((grp["outcome"] == "OPEN").mean() * 100, 1),
            "avg_MFE_pct":    round(grp["MFE_pct"].mean(), 2),
            "avg_MAE_pct":    round(grp["MAE_pct"].mean(), 2),
        })

    return pd.DataFrame(rows).sort_values("date")


def score_quintile_monotonicity(eval_dir: Path) -> pd.DataFrame:
    """Check if higher total_score → better realized_R (monotonicity test).

    Returns a DataFrame with quintile breakdown of avg realized_R by score quintile.
    Monotonically increasing → good signal quality.
    """
    files = sorted(Path(eval_dir).glob("eval_*.csv"))
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_csv(f)
        if df.empty or "realized_R" not in df.columns or "total_score" not in df.columns:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    all_evals = pd.concat(frames, ignore_index=True)
    all_evals = all_evals.dropna(subset=["total_score", "realized_R"])
    if len(all_evals) < 20:
        return pd.DataFrame()

    all_evals["score_quintile"] = pd.qcut(
        all_evals["total_score"], q=5, labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]
    )
    tbl = (
        all_evals.groupby("score_quintile", observed=True)
        .agg(
            n         =("realized_R", "count"),
            avg_R     =("realized_R", "mean"),
            win_rate  =("hit_tp",     "mean"),
            avg_score =("total_score","mean"),
        )
        .round(3)
        .reset_index()
    )
    return tbl


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sf(val: Any) -> float | None:
    """Safe float conversion."""
    if val is None:
        return None
    try:
        v = float(val)
        return None if (v != v) else v   # NaN guard
    except (TypeError, ValueError):
        return None

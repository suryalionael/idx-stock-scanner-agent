"""Top signals >10% — non-production, standalone daily persistence.

Filters already-evaluated signal outcomes (data/performance/signal_results.csv,
produced by stock_scanner.pipeline.performance) down to the ones whose
realized close-to-close return exceeds the 10% threshold, ranks them, and
best-effort enriches with quality/ML scores when a same-day ranked file is
available. Not wired into signal_engine.py, scanner_config.yaml,
ml_ranker.py, or any promotion path — see scripts/build_top_signals.py for
the orchestrator and stock_scanner/db/top_signals.py for persistence.

Return definition: forward_return_pct = pct_close / 100, where pct_close is
signal_results.csv's existing close-to-close % return (PREV CLOSE at signal
date -> CLOSE at eval_date) — the same field stock_scanner.pipeline.
performance already uses to decide win/loss ("W" if close > prev). This is
NOT the intraday-high daily_movers.py metric; it is the realized outcome of
a signal that already went through signal_engine.py + evaluation.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger

RETURN_THRESHOLD = 0.10   # strict >10% — "above 10%", not >=10%

_QUALITY_COLS = ["quality_adjusted_score", "total_score", "enhanced_total_score", "ml_prob"]

_OUTPUT_COLS = [
    "signal_id", "ticker", "strategy", "signal_date", "eval_date", "signal_label",
    "prev_close", "eval_close", "eval_high", "pct_close", "pct_high",
    "forward_return_pct", "quality_adjusted_score", "total_score",
    "enhanced_total_score", "ml_prob", "quality_source", "rank_in_day",
]


def filter_top_signals(results: pd.DataFrame, threshold: float = RETURN_THRESHOLD) -> pd.DataFrame:
    """Filter evaluated signals to forward_return_pct > threshold (10% default).

    Args:
        results: signal_results.csv loaded as a DataFrame — columns
            signal_date, eval_date, strategy, ticker, signal, prev, close,
            high, pct_high, pct_close, wl, status.
        threshold: fraction, e.g. 0.10 for >10%.

    Rows with status != 'evaluated' (still pending) are excluded — not an
    error, just not yet eligible. Returns an empty frame (not a raise) if
    nothing qualifies.
    """
    if results.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = results[results["status"] == "evaluated"].copy()
    df["forward_return_pct"] = df["pct_close"] / 100.0
    df = df[df["forward_return_pct"] > threshold]
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    from stock_scanner.db.init_db import signal_id as _make_signal_id
    df["signal_id"] = [
        _make_signal_id(r["ticker"], r["signal_date"], r["strategy"]) for _, r in df.iterrows()
    ]
    df = df.rename(columns={"prev": "prev_close", "close": "eval_close",
                             "high": "eval_high", "signal": "signal_label"})
    for col in _QUALITY_COLS:
        df[col] = None
    df["quality_source"] = "unavailable"
    return df


def enrich_with_quality_scores(df: pd.DataFrame, ranked_dir: Path) -> pd.DataFrame:
    """Best-effort join against data/ranked/ranked_{signal_date}.csv for
    quality_adjusted_score / total_score / enhanced_total_score / ml_prob.

    data/ranked/ is gitignored and only survives via the GitHub Actions
    cache (rolled weekly) — a signal_date far enough in the past may no
    longer have its ranked file available. That is a normal, expected
    condition here (e.g. no promoted challenger model that day, or a cold
    cache), not an error: rows simply keep quality_source='unavailable' and
    NULL quality columns, and downstream ranking falls back to
    forward_return_pct alone for those rows.
    """
    if df.empty:
        return df

    df = df.copy()
    for signal_date, group_idx in df.groupby("signal_date").groups.items():
        ranked_path = ranked_dir / f"ranked_{signal_date}.csv"
        if not ranked_path.exists():
            continue
        try:
            ranked = pd.read_csv(ranked_path)
        except Exception as e:
            logger.debug(f"top_signals: cannot read {ranked_path.name}: {e}")
            continue
        available_cols = [c for c in _QUALITY_COLS if c in ranked.columns]
        if not available_cols or "ticker" not in ranked.columns:
            continue
        ranked_lookup = ranked.set_index("ticker")[available_cols]
        for idx in group_idx:
            ticker = df.at[idx, "ticker"]
            if ticker not in ranked_lookup.index:
                continue
            row = ranked_lookup.loc[ticker]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            for col in available_cols:
                val = row.get(col)
                df.at[idx, col] = None if pd.isna(val) else float(val)
            df.at[idx, "quality_source"] = "ranked_csv"

    return df


def rank_top_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Assign rank_in_day: rank within each eval_date cohort, ordered by
    forward_return_pct DESC, then quality_adjusted_score DESC (NULLs last),
    then ticker ASC as a final deterministic tie-break. Reproducible: same
    input always produces the same ranks, independent of row arrival order."""
    if df.empty:
        df = df.copy()
        df["rank_in_day"] = pd.Series(dtype="int64")
        return df

    df = df.copy()
    df["_quality_sort"] = df["quality_adjusted_score"].fillna(float("-inf"))
    df = df.sort_values(
        by=["eval_date", "forward_return_pct", "_quality_sort", "ticker"],
        ascending=[True, False, False, True],
    )
    df["rank_in_day"] = df.groupby("eval_date").cumcount() + 1
    return df.drop(columns="_quality_sort").reset_index(drop=True)


def build_top_signals(
    results: pd.DataFrame, ranked_dir: Path, threshold: float = RETURN_THRESHOLD,
) -> pd.DataFrame:
    """Full pipeline: filter -> enrich (best-effort) -> rank. See the three
    functions above for each step's contract."""
    df = filter_top_signals(results, threshold=threshold)
    df = enrich_with_quality_scores(df, ranked_dir)
    df = rank_top_signals(df)
    return df[_OUTPUT_COLS] if not df.empty else df

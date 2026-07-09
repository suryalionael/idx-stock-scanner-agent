#!/usr/bin/env python3
"""Diagnose why successful signals (>10% gain) succeeded, vs why others failed.

Read-only analysis — does NOT touch the screener. Joins the full evaluated
signal population (data/performance/daily/*.csv) with:
  - pre-signal technical features (data/signals/{signal_date}.parquet)
  - eval-date OHLCV + a trailing 20d volume baseline (data/raw/{ticker}.parquet)
  - market regime on eval-date (data/published/ihsg_recent.parquet)
  - sector mapping (stock_scanner/configs/issuers.csv, partial coverage)

For every evaluated signal it derives boolean/categorical condition flags
(gap up, breakout, volume spike, MA position, momentum, volatility expansion,
candle structure, market regime, foreign-flow accumulation) and computes the
>10%-close hit rate for each condition across the WHOLE population — not just
within the successful subset — so the comparison is success vs everything
else, not success described in isolation.

Outputs:
  data/reports/signal_diagnosis_dataset.csv  — one row per evaluated signal,
                                                all raw + derived columns
  data/reports/signal_diagnosis_report.md    — hit-rate tables, group
                                                comparison, scoring framework

Usage:
    python scripts/diagnose_successful_signals.py
"""
import glob
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from loguru import logger

from stock_scanner.reference.issuers import get_sector

_DAILY_DIR = repo_root / "data" / "performance" / "daily"
_SIGNALS_DIR = repo_root / "data" / "signals"
_RAW_DIR = repo_root / "data" / "raw"
_IHSG_PATH = repo_root / "data" / "published" / "ihsg_recent.parquet"
_OUT_DIR = repo_root / "data" / "reports"
_OUT_CSV = _OUT_DIR / "signal_diagnosis_dataset.csv"
_OUT_LIFT_CSV = _OUT_DIR / "signal_diagnosis_lift_table.csv"
_OUT_SCORE_CSV = _OUT_DIR / "signal_diagnosis_score_buckets.csv"
_OUT_MD = _OUT_DIR / "signal_diagnosis_report.md"

_SUCCESS_THRESH = 10.0   # pct_close > this = SUCCESS
_LOSS_THRESH = 0.0       # pct_close <= this = FAIL

# Pre-signal feature columns to carry over from data/signals/ snapshots.
_FEATURE_COLS = [
    "ma5", "ma20", "ma50", "ma200", "ma_full_alignment", "ma_partial_alignment",
    "slope_ma20", "golden_cross", "price_vs_ma200",
    "rsi14", "macd", "macd_signal", "macd_histogram", "roc5", "roc20",
    "pct_from_52w_high", "atr14", "atr_breakout", "atr_pct",
    "vol_ratio_20d", "vol_spike", "obv_trend", "bb_width", "hist_vol_20d",
    "supertrend_bullish", "stoch_rsi_k", "stoch_rsi_d", "adx", "adx_pos", "adx_neg",
    "squeeze_on", "squeeze_release", "vwap_20d", "price_vs_vwap",
    "foreign_net", "foreign_net_5d", "foreign_net_20d", "foreign_flow_score",
    "news_sentiment_score", "news_count_3d",
    "total_score", "quality_adjusted_score", "enhanced_total_score", "ml_prob",
    "trend_score", "momentum_score", "breakout_score", "volume_score",
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_population() -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(_DAILY_DIR / "*.csv"))):
        name = Path(f).name
        if not (name.startswith("scalping_") or name.startswith("swing_")):
            continue
        d = pd.read_csv(f)
        if not d.empty:
            frames.append(d)
    pop = pd.concat(frames, ignore_index=True)
    return pop[pop["status"] == "evaluated"].copy()


def _load_signal_snapshots() -> dict[str, pd.DataFrame]:
    """signal_date -> DataFrame indexed by ticker, pre-signal features only."""
    out = {}
    for f in sorted(_SIGNALS_DIR.glob("*.parquet")):
        d = f.stem
        try:
            df = pd.read_parquet(f)
            out[d] = df.set_index("ticker")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Skip signals snapshot {}: {}", f.name, exc)
    return out


_raw_cache: dict[str, pd.DataFrame] = {}


def _load_raw(ticker: str) -> pd.DataFrame:
    if ticker not in _raw_cache:
        path = _RAW_DIR / f"{ticker}.parquet"
        if not path.exists():
            _raw_cache[ticker] = pd.DataFrame()
        else:
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
            df = df.sort_values("date").reset_index(drop=True)
            _raw_cache[ticker] = df
    return _raw_cache[ticker]


def _load_ihsg() -> pd.DataFrame:
    if not _IHSG_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(_IHSG_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    non_zero = df["volume"].fillna(0) > 0
    if non_zero.any():
        df = df.loc[:non_zero[non_zero].index[-1]]
    df["ihsg_pct_change"] = df["close"].pct_change() * 100
    return df


# ---------------------------------------------------------------------------
# Per-row enrichment
# ---------------------------------------------------------------------------

def _eval_day_metrics(ticker: str, eval_date: str) -> dict:
    """Open/High/Low/Close/Volume on eval_date + a trailing 20d volume
    baseline ending the session BEFORE eval_date, plus derived candle/volume
    metrics. Returns {} keys as None if the raw bar is unavailable."""
    out = {"open": None, "volume": None, "low": None,
           "gap_pct": None, "vol_ratio_eval": None,
           "close_position": None, "open_position": None, "body_ratio": None,
           "breakout_20d_high": None, "prior_high_20d": None}
    df = _load_raw(ticker)
    if df.empty:
        return out
    ts = pd.Timestamp(eval_date)
    idx = df.index[df["date"] == ts]
    if len(idx) == 0:
        return out
    i = idx[0]
    row = df.loc[i]
    out["open"], out["volume"], out["low"] = row.get("open"), row.get("volume"), row.get("low")

    # Zero-volume rows (yfinance holiday/suspension artefact) must be excluded
    # from the baseline window — matches the filter performance.py already
    # applies when computing the official prev/high/close labels. Without
    # this, holiday artefacts drag the 20d average volume down and distort
    # vol_ratio_eval / the 20d-high reference.
    prior_all = df.loc[:i - 1] if i > 0 else df.iloc[0:0]
    prior = prior_all[prior_all["volume"].fillna(0) > 0]
    if not prior.empty and pd.notna(row.get("open")):
        prior_close = prior.iloc[-1]["close"]
        if pd.notna(prior_close) and prior_close > 0:
            out["gap_pct"] = (row["open"] - prior_close) / prior_close * 100

    window = prior.tail(20)
    if len(window) >= 5:
        avg_vol = window["volume"].mean()
        if pd.notna(avg_vol) and avg_vol > 0 and pd.notna(row.get("volume")):
            out["vol_ratio_eval"] = row["volume"] / avg_vol
        prior_high = window["high"].max()
        out["prior_high_20d"] = prior_high
        if pd.notna(prior_high) and pd.notna(row.get("high")):
            out["breakout_20d_high"] = bool(row["high"] > prior_high)

    h, l, c, o = row.get("high"), row.get("low"), row.get("close"), row.get("open")
    if all(pd.notna(v) for v in (h, l, c, o)) and (h - l) > 0:
        out["close_position"] = (c - l) / (h - l)
        out["open_position"] = (o - l) / (h - l)
        out["body_ratio"] = abs(c - o) / (h - l)
    return out


def build_dataset() -> pd.DataFrame:
    pop = _load_population()
    logger.info("Population: {} evaluated signals", len(pop))

    snapshots = _load_signal_snapshots()
    ihsg = _load_ihsg()
    ihsg_idx = ihsg.set_index(ihsg["date"].dt.strftime("%Y-%m-%d")) if not ihsg.empty else pd.DataFrame()

    rows = []
    for _, r in pop.iterrows():
        rec = r.to_dict()
        snap = snapshots.get(r["signal_date"])
        if snap is not None and r["ticker"] in snap.index:
            feat = snap.loc[r["ticker"]]
            if isinstance(feat, pd.DataFrame):  # duplicate ticker rows — take first
                feat = feat.iloc[0]
            for c in _FEATURE_COLS:
                rec[c] = feat.get(c)
            rec["sector"] = get_sector(r["ticker"])
        else:
            for c in _FEATURE_COLS:
                rec[c] = None
            rec["sector"] = get_sector(r["ticker"])

        rec.update(_eval_day_metrics(r["ticker"], r["eval_date"]))

        if not ihsg_idx.empty and r["eval_date"] in ihsg_idx.index:
            rec["ihsg_pct_change_eval"] = ihsg_idx.loc[r["eval_date"], "ihsg_pct_change"]
        else:
            rec["ihsg_pct_change_eval"] = None

        rows.append(rec)

    df = pd.DataFrame(rows)
    df["has_features"] = df["rsi14"].notna()

    df["group"] = "modest_win"
    df.loc[df["pct_close"] > _SUCCESS_THRESH, "group"] = "success"
    df.loc[df["pct_close"] <= _LOSS_THRESH, "group"] = "fail"
    df.loc[(df["pct_high"] > _SUCCESS_THRESH) & (df["pct_close"] <= _LOSS_THRESH), "group"] = "spike_fade"
    return df


def _tobool(s: pd.Series) -> pd.Series:
    return s.map({True: True, False: False, "True": True, "False": False}).astype("boolean")


# Condition flags tested for >10%-close hit rate across the WHOLE population.
# Split so the report can separate "known before the move" (usable as a
# screener filter) from "only observable on the move's own session" (useful
# for live confirmation / invalidation, not for pre-filtering).
# Each entry: (mask_fn, [source columns]). The source-column list drives
# validity (`valid = df[cols].notna().all()`) — NOT `mask.notna()`, because a
# plain pandas comparison (e.g. `series > 3`) silently collapses NaN to False
# rather than propagating it, so checking the mask's own null-ness after the
# fact finds nothing to exclude (verified empirically: `pd.Series([5,nan,2])
# > 3` → `[True, False, False]`, all non-null). Rows with missing source data
# must be excluded from BOTH buckets, not silently folded into "false".
_PRE_SIGNAL_FLAGS = {
    "golden_cross": (lambda d: d["golden_cross"] == True, ["golden_cross"]),  # noqa: E712
    "ma_full_alignment": (lambda d: d["ma_full_alignment"] == True, ["ma_full_alignment"]),  # noqa: E712
    "above_ma200": (lambda d: d["price_vs_ma200"] > 0, ["price_vs_ma200"]),
    "rsi<30 (oversold)": (lambda d: d["rsi14"] < 30, ["rsi14"]),
    "rsi 30-50": (lambda d: (d["rsi14"] >= 30) & (d["rsi14"] < 50), ["rsi14"]),
    "rsi 50-70": (lambda d: (d["rsi14"] >= 50) & (d["rsi14"] < 70), ["rsi14"]),
    "rsi>=70 (overbought)": (lambda d: d["rsi14"] >= 70, ["rsi14"]),
    "macd_bullish (hist>0)": (lambda d: d["macd_histogram"] > 0, ["macd_histogram"]),
    "roc5>0": (lambda d: d["roc5"] > 0, ["roc5"]),
    "roc20>0": (lambda d: d["roc20"] > 0, ["roc20"]),
    "adx>25 (strong trend)": (lambda d: d["adx"] > 25, ["adx"]),
    "adx<20 (weak/no trend)": (lambda d: d["adx"] < 20, ["adx"]),
    "squeeze_on (compressed)": (lambda d: d["squeeze_on"] == True, ["squeeze_on"]),  # noqa: E712
    "squeeze_release": (lambda d: d["squeeze_release"] == True, ["squeeze_release"]),  # noqa: E712
    "supertrend_bullish": (lambda d: d["supertrend_bullish"] == True, ["supertrend_bullish"]),  # noqa: E712
    "atr_breakout (pre-signal)": (lambda d: d["atr_breakout"] == True, ["atr_breakout"]),  # noqa: E712
    "vol_spike (pre-signal day)": (lambda d: d["vol_spike"] == True, ["vol_spike"]),  # noqa: E712
    "vol_ratio_20d (pre-signal) >1.5": (lambda d: d["vol_ratio_20d"] > 1.5, ["vol_ratio_20d"]),
    "deep_below_52w (<-40%)": (lambda d: d["pct_from_52w_high"] < -40, ["pct_from_52w_high"]),
    "near_52w_high (>-10%)": (lambda d: d["pct_from_52w_high"] > -10, ["pct_from_52w_high"]),
    "price_vs_ma200>20": (lambda d: d["price_vs_ma200"] > 20, ["price_vs_ma200"]),
    "roc20>10": (lambda d: d["roc20"] > 10, ["roc20"]),
    "signal==BREAKOUT": (lambda d: d["signal"] == "BREAKOUT", ["signal"]),
}

_EVAL_DAY_FLAGS = {
    "gap_up (>1%)": (lambda d: d["gap_pct"] > 1, ["gap_pct"]),
    "gap_up_big (>3%)": (lambda d: d["gap_pct"] > 3, ["gap_pct"]),
    "gap_down_or_flat (<=0%)": (lambda d: d["gap_pct"] <= 0, ["gap_pct"]),
    "vol_spike_eval_2x": (lambda d: d["vol_ratio_eval"] > 2, ["vol_ratio_eval"]),
    "vol_spike_eval_3x": (lambda d: d["vol_ratio_eval"] > 3, ["vol_ratio_eval"]),
    "breakout_20d_high": (lambda d: d["breakout_20d_high"] == True, ["breakout_20d_high"]),  # noqa: E712
    "close_near_high (>0.7)": (lambda d: d["close_position"] > 0.7, ["close_position"]),
    "open_near_low (<0.3)": (lambda d: d["open_position"] < 0.3, ["open_position"]),
    "long_body (>0.6)": (lambda d: d["body_ratio"] > 0.6, ["body_ratio"]),
    "market_risk_on (IHSG up on eval day)": (lambda d: d["ihsg_pct_change_eval"] > 0, ["ihsg_pct_change_eval"]),
}

# Pre-signal-only composite score — every component must be knowable BEFORE
# the eval-date move, so this is the part that's actually usable as a
# prospective screener filter (unlike the eval-day flags above).
_SCORE_WEIGHTS = [
    ("atr_breakout", lambda d: (d["atr_breakout"] == True).fillna(False), 1),  # noqa: E712
    ("vol_spike", lambda d: (d["vol_spike"] == True).fillna(False), 1),  # noqa: E712
    ("vol_ratio_20d>1.5", lambda d: (d["vol_ratio_20d"] > 1.5).fillna(False), 1),
    ("rsi14>=70", lambda d: (d["rsi14"] >= 70).fillna(False), 1),
    ("price_vs_ma200>20", lambda d: (d["price_vs_ma200"] > 20).fillna(False), 1),
    ("roc20>10", lambda d: (d["roc20"] > 10).fillna(False), 1),
    ("signal==BREAKOUT", lambda d: (d["signal"] == "BREAKOUT"), 1),
    ("squeeze_on", lambda d: (d["squeeze_on"] == True).fillna(False), -2),  # noqa: E712
]


def compute_pre_signal_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0, index=df.index)
    for _, mask_fn, weight in _SCORE_WEIGHTS:
        score = score + weight * mask_fn(df).astype(int)
    return score


def _lift_table(df: pd.DataFrame, flags: dict, min_n: int = 10) -> pd.DataFrame:
    rows = []
    for name, (fn, source_cols) in flags.items():
        mask = fn(df)
        valid = df[source_cols].notna().all(axis=1)
        mask_f = mask.fillna(False)
        n_true = int(mask_f[valid].sum())
        if n_true < min_n:
            continue
        rate_true = df.loc[valid & mask_f, "is_success"].mean()
        rate_false = df.loc[valid & ~mask_f, "is_success"].mean()
        lift = rate_true / rate_false if rate_false else float("inf")
        rows.append({"condition": name, "n_true": n_true,
                     "success_rate_true_pct": round(rate_true * 100, 2),
                     "n_false": int((~mask_f[valid]).sum()),
                     "success_rate_false_pct": round(rate_false * 100, 2),
                     "lift": round(lift, 2)})
    return pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)


def run_analysis(df: pd.DataFrame) -> None:
    for c in ["golden_cross", "ma_full_alignment", "squeeze_on", "squeeze_release",
              "supertrend_bullish", "atr_breakout", "vol_spike", "breakout_20d_high"]:
        df[c] = _tobool(df[c])
    df["is_success"] = df["pct_close"] > _SUCCESS_THRESH

    pre_tbl = _lift_table(df, _PRE_SIGNAL_FLAGS)
    pre_tbl.insert(0, "category", "pre_signal")
    eval_tbl = _lift_table(df, _EVAL_DAY_FLAGS)
    eval_tbl.insert(0, "category", "eval_day")
    lift_tbl = pd.concat([pre_tbl, eval_tbl], ignore_index=True)
    lift_tbl.to_csv(_OUT_LIFT_CSV, index=False)
    logger.info("Wrote {} ({} conditions)", _OUT_LIFT_CSV, len(lift_tbl))

    df["pre_signal_score"] = compute_pre_signal_score(df)
    bands = pd.cut(df["pre_signal_score"], bins=[-3, -1, 1, 3, 5, 8],
                   labels=["<=-1 (squeeze veto)", "0-1 (avoid)", "2-3 (neutral)",
                           "4-5 (good)", "6+ (strong)"])
    score_tbl = df.groupby(bands, observed=True)["is_success"].agg(["mean", "count"])
    score_tbl["mean"] = (score_tbl["mean"] * 100).round(2)
    score_tbl.columns = ["success_rate_pct", "n"]
    score_tbl.to_csv(_OUT_SCORE_CSV)
    logger.info("Wrote {} \n{}", _OUT_SCORE_CSV, score_tbl)


if __name__ == "__main__":
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    dataset.to_csv(_OUT_CSV, index=False)
    logger.info("Wrote {} ({} rows)", _OUT_CSV, len(dataset))
    run_analysis(dataset)

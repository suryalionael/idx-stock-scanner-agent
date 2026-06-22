"""Rule-based signal engine — v2 (audit-fixes-v2).

CHANGES vs v1:
  - Hard liquidity gate: turnover < min_turnover_idr → AVOID (before any scoring)
  - penalty_score weight increased: 0.10 → 0.20
  - squeeze_release: reads pre-computed column from feature_builder (no shift(1) here)
  - _classify() receives full config for gate access
  - Thresholds now match tighter values in scanner_config.yaml

Menerima DataFrame fitur dan menghasilkan:
- 5 komponen score original (0–10) + 2 komponen baru (news, foreign)
- total_score (weighted, clipped 0–10)
- enhanced_total_score — total_score + koreksi dari news dan foreign
- signal label: BREAKOUT | PRE_MARKUP | WATCH | AVOID | NONE
"""
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

# Default thresholds — sync dengan scanner_config.yaml defaults
_DEFAULTS = {
    "signal_thresholds": {
        "breakout":   {"total": 7.5, "breakout": 7.0, "volume": 6.0},
        "pre_markup": {"total": 6.0, "trend": 5.0},
        "watch":      {"total": 5.0},
    },
    "liquidity": {
        "min_turnover_idr": 500_000_000,
    },
}

SIGNAL_BREAKOUT   = "BREAKOUT"
SIGNAL_PRE_MARKUP = "PRE_MARKUP"
SIGNAL_WATCH      = "WATCH"
SIGNAL_AVOID      = "AVOID"
SIGNAL_NONE       = "NONE"


def compute_signal(
    df_features: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Hitung score dan label signal dari DataFrame fitur.

    Kolom yang ditambahkan:
        trend_score, momentum_score, breakout_score, volume_score, penalty_score,
        news_score, foreign_score,
        total_score, enhanced_total_score, signal

    Hard gates (diterapkan sebelum scoring):
        - turnover (close × volume) < liquidity.min_turnover_idr → AVOID
        - volume == 0 (no trades) → AVOID

    Args:
        df_features: DataFrame dengan kolom dari feature_builder.FEATURE_COLS
        config: dict dari scanner_config.yaml (opsional)
    """
    if config is None:
        config = _DEFAULTS

    thresholds = config.get("signal_thresholds", _DEFAULTS["signal_thresholds"])
    df = df_features.copy()

    # --- 5 komponen original ---
    df["trend_score"]    = _trend_score(df)
    df["momentum_score"] = _momentum_score(df)
    df["breakout_score"] = _breakout_score(df)
    df["volume_score"]   = _volume_score(df)
    df["penalty_score"]  = _penalty_score(df)

    # --- Komponen baru (optional, neutral=5.0 jika tidak ada data) ---
    df["news_score"]    = _news_score(df)
    df["foreign_score"] = _foreign_score(df)

    # --- total_score ---
    # penalty weight dinaikkan dari 0.10 → 0.20 untuk memberi efek nyata
    df["total_score"] = (
        df["trend_score"]    * 0.25
        + df["momentum_score"] * 0.25
        + df["breakout_score"] * 0.25
        + df["volume_score"]   * 0.15
        - df["penalty_score"]  * 0.20    # was 0.10
    ).clip(0, 10)

    # --- enhanced_total_score ---
    news_adj    = (df["news_score"]    - 5.0) * 0.08
    foreign_adj = (df["foreign_score"] - 5.0) * 0.08
    df["enhanced_total_score"] = (df["total_score"] + news_adj + foreign_adj).clip(0, 10)

    # --- Signal classification ---
    use_enhanced = config.get("use_enhanced_score", False)
    score_col = "enhanced_total_score" if use_enhanced else "total_score"
    df["signal"] = df.apply(
        lambda row: _classify(row, thresholds, score_col, config),
        axis=1,
    )

    n_signals = df["signal"].value_counts().to_dict()
    n_avoid_liq = (df.get("_avoid_reason", pd.Series(dtype=str)) == "LIQUIDITY").sum() \
                  if "_avoid_reason" in df.columns else "?"
    logger.info(f"Signal distribution: {n_signals} | liquidity-AVOID: {n_avoid_liq}")
    return df


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_signals(df: pd.DataFrame, signals_dir: Path, scan_date: str | None = None) -> None:
    label = scan_date or date.today().strftime("%Y-%m-%d")
    signals_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = signals_dir / f"{label}.parquet"
    csv_path     = signals_dir / f"{label}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    logger.info(f"Signals saved → {parquet_path}")


def load_signals(signals_dir: Path, scan_date: str) -> pd.DataFrame:
    path = signals_dir / f"{scan_date}.parquet"
    if not path.exists():
        path_csv = signals_dir / f"{scan_date}.csv"
        if path_csv.exists():
            return pd.read_csv(path_csv)
        logger.warning(f"Signal file not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Internal scorers
# ---------------------------------------------------------------------------

def _trend_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)

    if "ma_full_alignment" in df.columns:
        full_align = df["ma_full_alignment"].fillna(False).astype(bool)
        score += full_align.astype(float) * 10
        if "ma_partial_alignment" in df.columns:
            partial_only = df["ma_partial_alignment"].fillna(False).astype(bool) & ~full_align
            score += partial_only.astype(float) * 5

    if "slope_ma20" in df.columns:
        score += (df["slope_ma20"].fillna(0) > 0).astype(float) * 2
    if "golden_cross" in df.columns:
        score += df["golden_cross"].fillna(False).astype(float) * 3

    if "adx" in df.columns:
        adx = df["adx"].fillna(0)
        score += (adx >= 25).astype(float) * 1
        score += (adx >= 40).astype(float) * 1

    if "supertrend_bullish" in df.columns:
        score += df["supertrend_bullish"].fillna(False).astype(float) * 2

    return score.clip(0, 10)


def _momentum_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    if "rsi14" in df.columns:
        rsi = df["rsi14"].fillna(50)
        ideal = (rsi >= 40) & (rsi <= 70)
        score += ideal.astype(float) * 5
    if "macd_histogram" in df.columns:
        score += (df["macd_histogram"].fillna(0) > 0).astype(float) * 3
    if "roc5" in df.columns:
        score += (df["roc5"].fillna(0) > 0).astype(float) * 1
    if "roc20" in df.columns:
        score += (df["roc20"].fillna(0) > 0).astype(float) * 1

    if "stoch_rsi_k" in df.columns and "stoch_rsi_d" in df.columns:
        k = df["stoch_rsi_k"].fillna(50)
        d = df["stoch_rsi_d"].fillna(50)
        score += ((k > d) & (k < 80)).astype(float) * 1

    return score.clip(0, 10)


def _breakout_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    if "pct_from_52w_high" in df.columns:
        pct = df["pct_from_52w_high"].fillna(-100)
        score += (pct >= -5).astype(float) * 5
        score += ((pct >= -15) & (pct < -5)).astype(float) * 2
    if "atr_breakout" in df.columns:
        score += df["atr_breakout"].fillna(False).astype(float) * 5

    # squeeze_release: baca dari kolom pre-computed (feature_builder._add_tv_indicators)
    # JANGAN pakai shift(1) di sini — signal_engine hanya lihat single row per ticker
    if "squeeze_release" in df.columns:
        score += df["squeeze_release"].fillna(False).astype(float) * 3

    return score.clip(0, 10)


def _volume_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    if "vol_ratio_20d" in df.columns:
        vr = df["vol_ratio_20d"].fillna(0)
        score += (vr >= 2.0).astype(float) * 5
        score += ((vr >= 1.3) & (vr < 2.0)).astype(float) * 3
    if "obv_trend" in df.columns:
        score += df["obv_trend"].fillna(False).astype(float) * 5
    return score.clip(0, 10)


def _penalty_score(df: pd.DataFrame) -> pd.Series:
    """Nilai positif = besar penalti. Bobot di total_score = ×0.20 (was 0.10)."""
    penalty = pd.Series(0.0, index=df.index)

    if "rsi14" in df.columns:
        penalty += (df["rsi14"].fillna(50) > 80).astype(float) * 8

    if "volume" in df.columns and "close" in df.columns:
        value_idr = df["volume"].fillna(0) * df["close"].fillna(0)
        penalty += (value_idr < 500_000_000).astype(float) * 5    # < Rp 500 juta
    elif "volume" in df.columns:
        penalty += (df["volume"].fillna(0) < 500_000).astype(float) * 5

    if "adx" in df.columns:
        penalty += (df["adx"].fillna(25) < 15).astype(float) * 2

    # squeeze_on: Bollinger Band masih terkompresi (belum release). Validated
    # via backtest audit: 0% dari 67 sinyal historis dengan squeeze_on=True
    # pernah mencapai pct_close>10%, konsisten di setiap time-split,
    # market-regime-split, dan ticker-exclusion check yang diuji.
    if "squeeze_on" in df.columns:
        penalty += df["squeeze_on"].fillna(False).astype(float) * 8

    return penalty.clip(0, 10)


# ---------------------------------------------------------------------------
# News & Foreign scores
# ---------------------------------------------------------------------------

def _news_score(df: pd.DataFrame) -> pd.Series:
    if "news_sentiment_score" not in df.columns:
        return pd.Series(5.0, index=df.index)
    return df["news_sentiment_score"].fillna(5.0).clip(0, 10)


def _foreign_score(df: pd.DataFrame) -> pd.Series:
    if "foreign_flow_score" not in df.columns:
        return pd.Series(5.0, index=df.index)
    return df["foreign_flow_score"].fillna(5.0).clip(0, 10)


# ---------------------------------------------------------------------------
# Hard gates
# ---------------------------------------------------------------------------

def _hard_gate(row: pd.Series, config: dict) -> str | None:
    """Return AVOID reason string if hard gate is triggered, else None.

    Gates (checked in order):
    1. Zero volume — no trades today
    2. Turnover < min_turnover_idr (default Rp 500 juta)

    Returns:
        None          → passed all gates (proceed to scoring)
        "ZERO_VOLUME" → no trades
        "LIQUIDITY"   → too illiquid
    """
    liq_cfg = config.get("liquidity", _DEFAULTS["liquidity"])
    min_turnover = liq_cfg.get("min_turnover_idr", 500_000_000)

    volume = float(row.get("volume", 0) or 0)
    close  = float(row.get("close", 0) or 0)

    if volume == 0:
        return "ZERO_VOLUME"

    turnover = volume * close
    if turnover < min_turnover:
        return "LIQUIDITY"

    return None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify(
    row: pd.Series,
    thresholds: dict,
    score_col: str = "total_score",
    config: dict | None = None,
) -> str:
    # ── Hard gates (checked first, before any score-based logic) ──────────
    gate = _hard_gate(row, config or _DEFAULTS)
    if gate:
        return SIGNAL_AVOID

    ts = row.get(score_col, 0) or 0
    bs = row.get("breakout_score", 0) or 0
    vs = row.get("volume_score", 0) or 0
    tr = row.get("trend_score", 0) or 0
    ps = row.get("penalty_score", 0) or 0

    # Heavy penalty → AVOID regardless of score
    if ps >= 8:
        return SIGNAL_AVOID

    # BREAKOUT
    t_breakout = thresholds.get("breakout", {})
    if (ts >= t_breakout.get("total", 7.5)
            and bs >= t_breakout.get("breakout", 7.0)
            and vs >= t_breakout.get("volume", 6.0)):
        return SIGNAL_BREAKOUT

    # PRE_MARKUP
    t_pre = thresholds.get("pre_markup", {})
    if ts >= t_pre.get("total", 6.0) and tr >= t_pre.get("trend", 5.0):
        return SIGNAL_PRE_MARKUP

    # WATCH
    t_watch = thresholds.get("watch", {})
    if ts >= t_watch.get("total", 5.0):
        return SIGNAL_WATCH

    # Low score → AVOID
    if ts < 2.0:
        return SIGNAL_AVOID

    return SIGNAL_NONE

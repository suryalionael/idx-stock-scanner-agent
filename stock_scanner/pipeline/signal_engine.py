"""Rule-based signal engine.

Menerima DataFrame fitur dan menghasilkan:
- 5 komponen score (0–10)
- total_score (weighted, clipped 0–10)
- signal label: BREAKOUT | PRE_MARKUP | WATCH | AVOID | NONE

Semua threshold baca dari scanner_config.yaml → signal_thresholds.
"""
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

# Default thresholds jika config tidak tersedia
_DEFAULTS = {
    "signal_thresholds": {
        "breakout": {"total": 7.5, "breakout": 7.0, "volume": 6.0},
        "pre_markup": {"total": 5.5, "trend": 5.0},
        "watch": {"total": 3.5},
    }
}

# Label signal (urutan penting — lebih kuat di atas)
SIGNAL_BREAKOUT = "BREAKOUT"
SIGNAL_PRE_MARKUP = "PRE_MARKUP"
SIGNAL_WATCH = "WATCH"
SIGNAL_AVOID = "AVOID"
SIGNAL_NONE = "NONE"


def compute_signal(
    df_features: pd.DataFrame,
    config: dict | None = None,
) -> pd.DataFrame:
    """Hitung score dan label signal dari DataFrame fitur.

    Args:
        df_features: DataFrame dengan kolom dari feature_builder.FEATURE_COLS
        config: dict dari scanner_config.yaml (opsional; gunakan default jika None)

    Returns:
        DataFrame dengan kolom tambahan:
            trend_score, momentum_score, breakout_score, volume_score,
            penalty_score, total_score, signal
    """
    if config is None:
        config = _DEFAULTS

    thresholds = config.get("signal_thresholds", _DEFAULTS["signal_thresholds"])
    df = df_features.copy()

    df["trend_score"] = _trend_score(df)
    df["momentum_score"] = _momentum_score(df)
    df["breakout_score"] = _breakout_score(df)
    df["volume_score"] = _volume_score(df)
    df["penalty_score"] = _penalty_score(df)

    # Weighted total: trend 25%, momentum 25%, breakout 25%, volume 15%, penalty -10%
    df["total_score"] = (
        df["trend_score"] * 0.25
        + df["momentum_score"] * 0.25
        + df["breakout_score"] * 0.25
        + df["volume_score"] * 0.15
        - df["penalty_score"] * 0.10
    ).clip(0, 10)

    df["signal"] = df.apply(lambda row: _classify(row, thresholds), axis=1)

    n_signals = df["signal"].value_counts().to_dict()
    logger.info(f"Signal distribution: {n_signals}")
    return df


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_signals(df: pd.DataFrame, signals_dir: Path, scan_date: str | None = None) -> None:
    label = scan_date or date.today().strftime("%Y-%m-%d")
    signals_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = signals_dir / f"{label}.parquet"
    csv_path = signals_dir / f"{label}.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    logger.info(f"Signals saved → {parquet_path}")


def load_signals(signals_dir: Path, scan_date: str) -> pd.DataFrame:
    path = signals_dir / f"{scan_date}.parquet"
    if not path.exists():
        logger.warning(f"Signal file not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


# --- Internal scorers ---

def _trend_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    if "ma_full_alignment" in df.columns:
        score += df["ma_full_alignment"].fillna(False).astype(float) * 10
    elif "ma_partial_alignment" in df.columns:
        score += df["ma_partial_alignment"].fillna(False).astype(float) * 5
    if "slope_ma20" in df.columns:
        score += (df["slope_ma20"].fillna(0) > 0).astype(float) * 2
    if "golden_cross" in df.columns:
        score += df["golden_cross"].fillna(False).astype(float) * 3
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
    return score.clip(0, 10)


def _breakout_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    if "pct_from_52w_high" in df.columns:
        pct = df["pct_from_52w_high"].fillna(-100)
        # -5% dari 52w high → strong breakout setup
        score += (pct >= -5).astype(float) * 5
        # -15% → approaching
        score += ((pct >= -15) & (pct < -5)).astype(float) * 2
    if "atr_breakout" in df.columns:
        score += df["atr_breakout"].fillna(False).astype(float) * 5
    return score.clip(0, 10)


def _volume_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    if "vol_ratio_20d" in df.columns:
        vr = df["vol_ratio_20d"].fillna(0)
        score += (vr >= 2.0).astype(float) * 5    # strong volume surge
        score += ((vr >= 1.3) & (vr < 2.0)).astype(float) * 3  # moderate
    if "obv_trend" in df.columns:
        score += df["obv_trend"].fillna(False).astype(float) * 5
    return score.clip(0, 10)


def _penalty_score(df: pd.DataFrame) -> pd.Series:
    """Nilai positif = besar penalti (dikurangkan dari total_score)."""
    penalty = pd.Series(0.0, index=df.index)
    if "rsi14" in df.columns:
        penalty += (df["rsi14"].fillna(50) > 80).astype(float) * 8  # overbought ekstrem
    if "volume" in df.columns:
        # Volume sangat rendah: < 100 juta IDR (perkiraan konservatif)
        penalty += (df["volume"].fillna(0) < 100_000).astype(float) * 5
    return penalty.clip(0, 10)


def _classify(row: pd.Series, thresholds: dict) -> str:
    ts = row.get("total_score", 0)
    bs = row.get("breakout_score", 0)
    vs = row.get("volume_score", 0)
    tr = row.get("trend_score", 0)
    ps = row.get("penalty_score", 0)

    # Hard penalty override
    if ps >= 8:
        return SIGNAL_AVOID

    t_breakout = thresholds.get("breakout", {})
    if (ts >= t_breakout.get("total", 7.5)
            and bs >= t_breakout.get("breakout", 7.0)
            and vs >= t_breakout.get("volume", 6.0)):
        return SIGNAL_BREAKOUT

    t_pre = thresholds.get("pre_markup", {})
    if ts >= t_pre.get("total", 5.5) and tr >= t_pre.get("trend", 5.0):
        return SIGNAL_PRE_MARKUP

    t_watch = thresholds.get("watch", {})
    if ts >= t_watch.get("total", 3.5):
        return SIGNAL_WATCH

    if ts < 2.0:
        return SIGNAL_AVOID

    return SIGNAL_NONE

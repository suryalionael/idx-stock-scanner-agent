"""Build technical indicator features dari OHLCV DataFrame.

Semua fitur dihitung inline di sini — tidak ada sub-modul terpisah.
Tambahkan indikator baru ke `build_features()` dan daftarkan di `FEATURE_COLS`.

Dependencies: ta (pip install ta)
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import ta
from loguru import logger

FEATURE_COLS = [
    "date", "ticker",
    # Trend
    "ma20", "ma50", "ma200",
    "ma_full_alignment", "ma_partial_alignment",
    "slope_ma20", "golden_cross", "price_vs_ma200",
    # Momentum
    "rsi14", "macd", "macd_signal", "macd_histogram",
    "roc5", "roc20",
    # Breakout
    "high_52w", "pct_from_52w_high",
    "atr14", "atr_breakout",
    # Volume
    "vol_ratio_20d", "vol_spike", "obv_trend",
    # Volatility
    "atr_pct", "bb_width", "hist_vol_20d",
    # Raw (untuk scoring + ML)
    "close", "volume",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hitung semua fitur teknikal untuk satu ticker DataFrame.

    Input: DataFrame dengan kolom OHLCV (minimal: date, open, high, low, close, volume).
    Output: DataFrame dengan kolom di FEATURE_COLS (kolom yang tidak bisa dihitung di-skip).
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df = _add_trend(df)
    df = _add_momentum(df)
    df = _add_breakout(df)
    df = _add_volume(df)
    df = _add_volatility(df)

    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available]


def build_features_batch(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build features untuk semua ticker dan gabungkan ke satu DataFrame."""
    frames = []
    for ticker, df in data.items():
        try:
            features = build_features(df)
            frames.append(features)
            logger.info(f"{ticker}: features computed ({len(features)} rows)")
        except Exception as e:
            logger.error(f"{ticker}: feature computation failed — {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def save_features(df: pd.DataFrame, features_dir: Path, scan_date: str | None = None) -> None:
    label = scan_date or date.today().strftime("%Y-%m-%d")
    features_dir.mkdir(parents=True, exist_ok=True)
    path = features_dir / f"{label}.parquet"
    df.to_parquet(path, index=False)
    logger.info(f"Feature store saved → {path}")


def load_features(features_dir: Path, scan_date: str) -> pd.DataFrame:
    path = features_dir / f"{scan_date}.parquet"
    if not path.exists():
        logger.warning(f"Feature file not found: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


# --- Internal builders ---

def _add_trend(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ma20"] = c.rolling(20).mean()
    df["ma50"] = c.rolling(50).mean()
    df["ma200"] = c.rolling(200).mean()

    # MA alignment
    has_all = df["ma20"].notna() & df["ma50"].notna() & df["ma200"].notna()
    df["ma_full_alignment"] = has_all & (df["ma20"] > df["ma50"]) & (df["ma50"] > df["ma200"])
    df["ma_partial_alignment"] = has_all & (df["ma20"] > df["ma50"])

    # Slope MA20: positif jika MA20 sekarang > MA20 5 hari lalu
    df["slope_ma20"] = df["ma20"] - df["ma20"].shift(5)

    # Golden cross: MA50 baru saja melewati MA200 ke atas
    ma50_prev = df["ma50"].shift(1)
    ma200_prev = df["ma200"].shift(1)
    df["golden_cross"] = (
        (df["ma50"] > df["ma200"]) & (ma50_prev <= ma200_prev)
    ).fillna(False)

    # Price vs MA200: persentase di atas/bawah MA200
    df["price_vs_ma200"] = ((c - df["ma200"]) / df["ma200"].replace(0, np.nan)) * 100

    return df


def _add_momentum(df: pd.DataFrame) -> pd.DataFrame:
    # RSI 14
    df["rsi14"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

    # MACD
    macd_ind = ta.trend.MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_histogram"] = macd_ind.macd_diff()

    # Rate of Change
    df["roc5"] = df["close"].pct_change(5) * 100
    df["roc20"] = df["close"].pct_change(20) * 100

    return df


def _add_breakout(df: pd.DataFrame) -> pd.DataFrame:
    # 52-week high (252 trading days)
    df["high_52w"] = df["high"].rolling(252, min_periods=50).max()
    df["pct_from_52w_high"] = (
        (df["close"] - df["high_52w"]) / df["high_52w"].replace(0, np.nan) * 100
    )

    # ATR 14
    atr_ind = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14)
    df["atr14"] = atr_ind.average_true_range()

    # ATR breakout: close > (close kemarin + 1.5 * ATR)
    df["atr_breakout"] = df["close"] > (df["close"].shift(1) + 1.5 * df["atr14"].shift(1))
    df["atr_breakout"] = df["atr_breakout"].fillna(False)

    return df


def _add_volume(df: pd.DataFrame) -> pd.DataFrame:
    vol = df["volume"]

    # Volume ratio: volume hari ini / rata-rata 20 hari
    vol_ma20 = vol.rolling(20).mean()
    df["vol_ratio_20d"] = vol / vol_ma20.replace(0, np.nan)

    # Volume spike: ratio > 2.5x
    df["vol_spike"] = df["vol_ratio_20d"] > 2.5

    # OBV trend: OBV sekarang > OBV 10 hari lalu
    obv = ta.volume.OnBalanceVolumeIndicator(df["close"], vol).on_balance_volume()
    df["obv_trend"] = obv > obv.shift(10)
    df["obv_trend"] = df["obv_trend"].fillna(False)

    return df


def _add_volatility(df: pd.DataFrame) -> pd.DataFrame:
    # ATR %: ATR relatif terhadap harga
    if "atr14" not in df.columns:
        df = _add_breakout(df)
    df["atr_pct"] = df["atr14"] / df["close"].replace(0, np.nan) * 100

    # Bollinger Band width
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_mid = bb.bollinger_mavg()
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan) * 100

    # Historical volatility: std return 20 hari (annualized)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["hist_vol_20d"] = log_ret.rolling(20).std() * np.sqrt(252) * 100

    return df

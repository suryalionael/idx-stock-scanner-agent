"""ML Ranker — XGBoost probability ranking untuk kandidat saham IDX.

Tujuan: setelah SignalEngine memfilter kandidat dengan rules,
MLRanker memberi probabilitas (ml_prob) bahwa saham akan naik > X%
dalam horizon N hari ke depan.

Target: return_N_days > threshold_pct (binary classification)
- Default: return_5d > 3% (swing trading horizon)

ALUR:
  1. Kumpulkan data historis dengan fitur + target (perlu minimal ~1 tahun data)
  2. Jalankan train_ranker() → simpan model ke models/ranker.pkl
  3. Saat scan harian: score_candidates(df_today, model, config) → tambah kolom ml_prob

CATATAN:
  - Model ini memerlukan data historis yang cukup untuk dilatih.
  - TODO: buat script terpisah untuk generate training dataset dari parquet historis.
  - Model disimpan sebagai pickle — retrain minimal 1x per bulan atau saat market regime berubah.
"""
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

# Coba import XGBoost, fallback ke sklearn GradientBoosting
try:
    from xgboost import XGBClassifier as _BaseModel

    _MODEL_NAME = "XGBoostClassifier"
except ImportError:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingClassifier as _BaseModel  # type: ignore

    _MODEL_NAME = "GradientBoostingClassifier (XGBoost not installed)"

DEFAULT_FEATURES = [
    "rsi14",
    "macd_histogram",
    "vol_ratio_20d",
    "atr_pct",
    "roc5",
    "roc20",
    "pct_from_52w_high",
    "bb_width",
    "hist_vol_20d",
    "trend_score",
    "momentum_score",
    "breakout_score",
    "volume_score",
]


def train_ranker(
    df_features: pd.DataFrame,
    config: dict,
    model_dir: Path | None = None,
) -> Any:
    """Latih ranking model dari DataFrame historis dengan fitur + harga penutupan.

    Args:
        df_features: DataFrame historis yang sudah berisi kolom fitur + kolom 'close'.
                     Harus terurut by date per ticker.
                     Kolom 'date' dan 'ticker' wajib ada.
        config: dict dari model_config.yaml
        model_dir: direktori untuk simpan model (default: ./models/)

    Returns:
        model yang sudah dilatih

    Notes:
        - Target dibuat otomatis: apakah return N hari ke depan > threshold_pct
        - TODO: pastikan df_features berisi data historis yang cukup (min. 1 tahun)
        - TODO: tambahkan cross-validation dan evaluation metrics sebelum deploy ke prod
    """
    target_pct = config.get("target_return_pct", 3.0)
    horizon = config.get("target_horizon_days", 5)
    feature_cols = config.get("features", DEFAULT_FEATURES)

    logger.info(f"Training {_MODEL_NAME}: target={target_pct}% in {horizon} days, features={len(feature_cols)}")

    # --- Build target variable ---
    # TODO: untuk data multi-ticker, hitung forward return per ticker
    # Saat ini asumsi df sudah per-ticker atau sudah di-sort
    df = df_features.copy().sort_values(["ticker", "date"]).reset_index(drop=True)

    # Forward return: (close N hari ke depan - close sekarang) / close sekarang
    df["_future_close"] = df.groupby("ticker")["close"].shift(-horizon)
    df["_return_fwd"] = (df["_future_close"] - df["close"]) / df["close"].replace(0, np.nan) * 100
    df["_target"] = (df["_return_fwd"] > target_pct).astype(int)

    # Drop baris dengan target NaN (N hari terakhir per ticker tidak punya future close)
    df = df.dropna(subset=["_target"] + [c for c in feature_cols if c in df.columns])

    available_features = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available_features)
    if missing:
        logger.warning(f"Feature columns not found, skipping: {missing}")

    if len(available_features) == 0:
        raise ValueError("Tidak ada feature column yang tersedia untuk training")

    X = df[available_features].fillna(0)
    y = df["_target"]

    pos_rate = y.mean()
    logger.info(f"Training set: {len(X)} samples, positive rate={pos_rate:.1%}")

    # TODO: tambahkan train/test split dan evaluate sebelum simpan ke prod
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = _BaseModel(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X, y)
    logger.info(f"Model trained ({_MODEL_NAME})")

    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "ranker.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "features": available_features, "config": config}, f)
        logger.info(f"Model saved → {model_path}")

    return model


def score_candidates(
    df_today: pd.DataFrame,
    model: Any,
    config: dict,
) -> pd.DataFrame:
    """Tambahkan kolom ml_prob ke DataFrame kandidat hari ini.

    Args:
        df_today: DataFrame baris terkini per ticker (output signal_engine)
        model: model yang sudah dilatih (dari train_ranker atau load_ranker)
        config: dict dari model_config.yaml (harus memiliki key 'features')

    Returns:
        df_today dengan kolom tambahan 'ml_prob' (float 0–1)
    """
    feature_cols = config.get("features", DEFAULT_FEATURES)
    available = [c for c in feature_cols if c in df_today.columns]

    if not available:
        logger.warning("Tidak ada feature column tersedia untuk scoring — ml_prob diset 0.0")
        df_today = df_today.copy()
        df_today["ml_prob"] = 0.0
        return df_today

    X = df_today[available].fillna(0)

    try:
        probs = model.predict_proba(X)[:, 1]
    except AttributeError:
        # Model tidak support predict_proba — fallback ke binary predict
        probs = model.predict(X).astype(float)

    df_today = df_today.copy()
    df_today["ml_prob"] = probs
    logger.info(f"ML scoring done: {len(df_today)} tickers, avg ml_prob={probs.mean():.3f}")
    return df_today


def load_ranker(model_path: Path) -> tuple[Any, list[str], dict]:
    """Load model dari pickle.

    Returns:
        (model, feature_list, config)
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    logger.info(f"Model loaded from {model_path}")
    return bundle["model"], bundle["features"], bundle.get("config", {})

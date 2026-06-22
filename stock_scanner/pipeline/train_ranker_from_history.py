"""Train ML ranker from historical raw OHLCV data.

Workflow:
  1. Scan data/raw/*.parquet (3 years of OHLCV per ticker)
  2. Build technical features via feature_builder.build_features()
  3. Compute binary forward-return label (close N days ahead > threshold_pct)
  4. Concatenate all tickers into a single training dataset
  5. Chronological train/test split (last 20% of calendar time = test)
  6. Train XGBoost classifier
  7. Evaluate: AUC-ROC, precision, recall, F1
  8. Save model → models/ranker.pkl
  9. Save metrics → models/ranker_metrics.yaml

Usage:
    python -m stock_scanner.pipeline.train_ranker_from_history
    python -m stock_scanner.pipeline.train_ranker_from_history --horizon 10 --target-pct 5.0
    python -m stock_scanner.pipeline.train_ranker_from_history --save-dataset  # also dump CSV
    python -m stock_scanner.pipeline.train_ranker_from_history --dry-run       # skip training, just inspect

Notes:
    - Requires ≥ 250 trading-day rows per ticker (< 1 year → skipped)
    - Liquidity gate: turnover (close × volume) < Rp 500M rows are excluded from training
    - Features are computed from RAW OHLCV only (no signal_engine scores needed)
    - Chronological split avoids look-ahead bias
    - If AUC < 0.55: script warns and does NOT save the model (prevents deploying noise)
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from loguru import logger

# Ensure repo root on path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from stock_scanner.pipeline.feature_builder import build_features  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_HORIZON       = 5        # trading days forward
_DEFAULT_TARGET_PCT    = 3.0      # % gain to call it a "positive" trade
_DEFAULT_MIN_ROWS      = 250      # minimum rows per ticker
_DEFAULT_MIN_TURNOVER  = 500_000_000  # Rp 500 juta liquidity gate
_DEFAULT_TEST_FRAC     = 0.20     # last 20% of time → test set
_MIN_AUC_TO_SAVE       = 0.52     # below this → refuse to save model

# Features produced by feature_builder (no signal_engine scores)
_FEATURE_COLS = [
    "rsi14",
    "macd_histogram",
    "macd",
    "vol_ratio_20d",
    "atr_pct",
    "roc5",
    "roc20",
    "pct_from_52w_high",
    "bb_width",
    "hist_vol_20d",
    "ma_full_alignment",
    "ma_partial_alignment",
    "slope_ma20",
    "price_vs_ma200",
    "obv_trend",
    "squeeze_release",
]


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_training_dataset(
    raw_dir: Path,
    horizon_days: int = _DEFAULT_HORIZON,
    target_pct: float = _DEFAULT_TARGET_PCT,
    min_rows: int = _DEFAULT_MIN_ROWS,
    min_turnover_idr: float = _DEFAULT_MIN_TURNOVER,
    max_tickers: int | None = None,
) -> pd.DataFrame:
    """Build a labelled training DataFrame from all raw OHLCV parquets.

    Returns:
        DataFrame with feature columns + '_target' (0/1) + '_date' + '_ticker'.
        Empty if no tickers qualify.
    """
    raw_files = sorted(raw_dir.glob("*.parquet"))
    if max_tickers:
        raw_files = raw_files[:max_tickers]

    logger.info(f"Building training dataset from {len(raw_files)} raw files...")
    logger.info(f"  horizon={horizon_days}d, target={target_pct}%, min_rows={min_rows}")

    all_frames: list[pd.DataFrame] = []
    skipped_short = skipped_error = 0

    for rp in raw_files:
        ticker = rp.stem  # e.g. "BBCA.JK"
        try:
            raw_df = pd.read_parquet(rp)
            raw_df["date"] = pd.to_datetime(raw_df["date"])
            raw_df = raw_df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            logger.debug(f"{ticker}: cannot load raw — {e}")
            skipped_error += 1
            continue

        if len(raw_df) < min_rows:
            logger.debug(f"{ticker}: too short ({len(raw_df)} rows < {min_rows}) — skip")
            skipped_short += 1
            continue

        # Build features
        try:
            feat_df = build_features(raw_df)
        except Exception as e:
            logger.debug(f"{ticker}: feature build failed — {e}")
            skipped_error += 1
            continue

        if feat_df.empty or len(feat_df) < horizon_days + 10:
            skipped_short += 1
            continue

        # Ensure we have date + close + volume
        if "close" not in feat_df.columns or "volume" not in feat_df.columns:
            skipped_error += 1
            continue

        # Compute forward return label
        feat_df = feat_df.copy().reset_index(drop=True)
        feat_df["_future_close"] = feat_df["close"].shift(-horizon_days)
        feat_df["_return_fwd"] = (
            (feat_df["_future_close"] - feat_df["close"]) / feat_df["close"].replace(0, np.nan) * 100
        )
        feat_df["_target"] = (feat_df["_return_fwd"] > target_pct).astype(float)

        # Drop last horizon_days rows (no future close)
        feat_df = feat_df.iloc[:-horizon_days].copy()

        # Liquidity gate: exclude low-turnover rows from training
        turnover = feat_df["close"] * feat_df["volume"]
        feat_df = feat_df[turnover >= min_turnover_idr].copy()

        # Drop rows with NaN target or key indicators
        feat_df = feat_df.dropna(subset=["_target", "close"])
        if feat_df.empty:
            skipped_short += 1
            continue

        feat_df["_ticker"] = ticker
        feat_df["_date"] = feat_df["date"]
        all_frames.append(feat_df)

    logger.info(
        f"Dataset built: {len(all_frames)} tickers OK | "
        f"skipped_short={skipped_short}, skipped_error={skipped_error}"
    )

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info(f"Total rows: {len(combined)} | positive rate: {combined['_target'].mean():.1%}")
    return combined


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_from_dataset(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    test_frac: float = _DEFAULT_TEST_FRAC,
    xgb_params: dict | None = None,
) -> tuple[Any, dict]:
    """Train XGBoost on dataset, chronological split.

    Returns:
        (model, metrics_dict)
    """
    try:
        from xgboost import XGBClassifier
        _model_name = "XGBoostClassifier"
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier  # type: ignore
        _model_name = "GradientBoostingClassifier (XGBoost not installed)"

    try:
        from sklearn.metrics import classification_report, roc_auc_score
    except ImportError as e:
        raise ImportError("scikit-learn required for evaluation: pip install scikit-learn") from e

    # Available features
    available = [c for c in feature_cols if c in dataset.columns]
    missing = set(feature_cols) - set(available)
    if missing:
        logger.warning(f"Missing feature columns (will skip): {sorted(missing)}")
    if not available:
        raise ValueError("No feature columns available for training")

    logger.info(f"Training features ({len(available)}): {available}")

    # Chronological split by date
    dataset = dataset.sort_values("_date").reset_index(drop=True)
    split_idx = int(len(dataset) * (1 - test_frac))
    train_df = dataset.iloc[:split_idx]
    test_df  = dataset.iloc[split_idx:]

    X_train = train_df[available].fillna(0)
    y_train = train_df["_target"].astype(int)
    X_test  = test_df[available].fillna(0)
    y_test  = test_df["_target"].astype(int)

    # Date ranges for logging
    train_dates = (train_df["_date"].min().strftime("%Y-%m-%d"),
                   train_df["_date"].max().strftime("%Y-%m-%d"))
    test_dates  = (test_df["_date"].min().strftime("%Y-%m-%d"),
                   test_df["_date"].max().strftime("%Y-%m-%d"))

    logger.info(f"Train: {len(X_train)} rows ({train_dates[0]} → {train_dates[1]}), "
                f"pos={y_train.mean():.1%}")
    logger.info(f"Test:  {len(X_test)} rows ({test_dates[0]} → {test_dates[1]}), "
                f"pos={y_test.mean():.1%}")

    # Model hyperparams
    default_params = {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,   # prevent overfitting on small groups
        "random_state": 42,
        "eval_metric": "auc",
    }
    if xgb_params:
        default_params.update(xgb_params)

    model = XGBClassifier(**default_params)
    model.fit(X_train, y_train, verbose=False)

    # Evaluate
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "model":          _model_name,
        "auc_roc":        round(float(auc), 4),
        "precision_1":    round(float(report.get("1", {}).get("precision", 0)), 4),
        "recall_1":       round(float(report.get("1", {}).get("recall", 0)), 4),
        "f1_1":           round(float(report.get("1", {}).get("f1-score", 0)), 4),
        "support_1_test": int(report.get("1", {}).get("support", 0)),
        "train_rows":     len(X_train),
        "test_rows":      len(X_test),
        "train_period":   f"{train_dates[0]} → {train_dates[1]}",
        "test_period":    f"{test_dates[0]} → {test_dates[1]}",
        "positive_rate_train": round(float(y_train.mean()), 4),
        "positive_rate_test":  round(float(y_test.mean()), 4),
        "features":       available,
        "n_features":     len(available),
    }

    logger.info(f"{'='*50}")
    logger.info(f"AUC-ROC : {auc:.4f}")
    logger.info(f"Precision (class=1): {metrics['precision_1']:.4f}")
    logger.info(f"Recall    (class=1): {metrics['recall_1']:.4f}")
    logger.info(f"F1        (class=1): {metrics['f1_1']:.4f}")
    logger.info(f"{'='*50}")

    return model, metrics


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    metrics: dict,
    feature_cols: list[str],
    model_config: dict,
    model_dir: Path,
) -> tuple[Path, Path]:
    """Save model + metrics to disk.

    Returns:
        (model_path, metrics_path)
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path   = model_dir / "ranker.pkl"
    metrics_path = model_dir / "ranker_metrics.yaml"

    bundle = {
        "model":    model,
        "features": feature_cols,
        "config":   model_config,
        "metrics":  metrics,
    }
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f, protocol=5)
    logger.info(f"Model saved → {model_path}")

    with open(metrics_path, "w") as f:
        yaml.dump(metrics, f, default_flow_style=False, allow_unicode=True)
    logger.info(f"Metrics saved → {metrics_path}")

    return model_path, metrics_path


# ---------------------------------------------------------------------------
# Feature importance report
# ---------------------------------------------------------------------------

def _print_feature_importance(model: Any, feature_cols: list[str]) -> None:
    try:
        importances = model.feature_importances_
        pairs = sorted(zip(feature_cols, importances), key=lambda x: -x[1])
        print(f"\n{'─'*50}")
        print("FEATURE IMPORTANCE (gain)")
        print(f"{'─'*50}")
        for feat, imp in pairs:
            bar = "█" * int(imp * 200)
            print(f"  {feat:30s}: {imp:.4f}  {bar}")
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    raw_dir: Path,
    model_dir: Path,
    config_path: Path,
    horizon_days: int,
    target_pct: float,
    min_rows: int,
    test_frac: float,
    save_dataset: bool,
    dry_run: bool,
    max_tickers: int | None,
) -> None:
    logger.info("=" * 60)
    logger.info("IDX Scanner — ML Ranker Training from Historical Data")
    logger.info(f"  raw_dir      : {raw_dir}")
    logger.info(f"  model_dir    : {model_dir}")
    logger.info(f"  horizon_days : {horizon_days}")
    logger.info(f"  target_pct   : {target_pct}%")
    logger.info(f"  min_rows     : {min_rows}")
    logger.info(f"  test_frac    : {test_frac:.0%}")
    logger.info(f"  dry_run      : {dry_run}")
    logger.info("=" * 60)

    # Load model config (for XGBoost params + feature list override)
    model_config: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            model_config = yaml.safe_load(f) or {}
        logger.info(f"Loaded model config: {config_path}")

    xgb_params = model_config.get("xgboost", {})
    feature_cols = model_config.get("features", _FEATURE_COLS)
    # Remove signal_engine-only columns (not available in raw history)
    _signal_only = {"trend_score", "momentum_score", "breakout_score", "volume_score"}
    feature_cols = [c for c in feature_cols if c not in _signal_only]
    logger.info(f"Feature columns to use: {feature_cols}")

    # --- Step 1: Build dataset ---
    dataset = build_training_dataset(
        raw_dir=raw_dir,
        horizon_days=horizon_days,
        target_pct=target_pct,
        min_rows=min_rows,
        min_turnover_idr=_DEFAULT_MIN_TURNOVER,
        max_tickers=max_tickers,
    )

    if dataset.empty:
        logger.error("Dataset is empty — cannot train. Check raw_dir and min_rows settings.")
        return

    if save_dataset:
        ds_path = model_dir / "training_dataset.csv"
        model_dir.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(ds_path, index=False)
        logger.info(f"Training dataset saved → {ds_path} ({len(dataset)} rows)")

    if dry_run:
        print(f"\n{'─'*60}")
        print("DRY RUN — dataset summary (training not started):")
        print(f"  Total rows  : {len(dataset):,}")
        print(f"  Tickers     : {dataset['_ticker'].nunique()}")
        date_col = dataset["_date"] if "_date" in dataset.columns else dataset.get("date")
        if date_col is not None:
            print(f"  Date range  : {date_col.min().strftime('%Y-%m-%d')} → {date_col.max().strftime('%Y-%m-%d')}")
        print(f"  Pos rate    : {dataset['_target'].mean():.1%}")
        print(f"  Feature cols: {[c for c in feature_cols if c in dataset.columns]}")
        print(f"{'─'*60}")
        return

    # --- Step 2: Train ---
    model, metrics = train_from_dataset(
        dataset=dataset,
        feature_cols=feature_cols,
        test_frac=test_frac,
        xgb_params=xgb_params,
    )

    # Feature importance
    trained_features = metrics["features"]
    _print_feature_importance(model, trained_features)

    # --- Step 3: Quality gate ---
    auc = metrics["auc_roc"]
    print(f"\n{'─'*60}")
    if auc < _MIN_AUC_TO_SAVE:
        logger.warning(
            f"AUC {auc:.4f} < threshold {_MIN_AUC_TO_SAVE} — model NOT saved.\n"
            "Suggestions:\n"
            "  • Try --horizon 10 (longer horizon may be more predictable)\n"
            "  • Try --target-pct 5.0 (higher bar → cleaner signal)\n"
            "  • Add more tickers or longer history\n"
            "  • Consider regime-specific models (bull/bear)"
        )
        return

    # --- Step 4: Save ---
    # Update model_config with actual training params
    model_config.update({
        "target_return_pct":   target_pct,
        "target_horizon_days": horizon_days,
        "features":            trained_features,
    })

    mp, metp = save_model(
        model=model,
        metrics=metrics,
        feature_cols=trained_features,
        model_config=model_config,
        model_dir=model_dir,
    )

    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE")
    print(f"  Model   : {mp}")
    print(f"  Metrics : {metp}")
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  Period  : {metrics['train_period']} [train]")
    print(f"          : {metrics['test_period']} [test]")
    print(f"  Tickers in training: {dataset['_ticker'].nunique()}")
    print("\nNext: run the daily scan — ml_prob column will now be active.")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ML ranker for IDX Stock Scanner from historical OHLCV"
    )
    parser.add_argument(
        "--raw-dir", type=Path,
        default=repo_root / "data" / "raw",
        help="Directory of *.parquet raw OHLCV files",
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=repo_root / "models",
        help="Directory to save ranker.pkl and ranker_metrics.yaml",
    )
    parser.add_argument(
        "--config", type=Path,
        default=repo_root / "stock_scanner" / "configs" / "model_config.yaml",
        help="Path to model_config.yaml",
    )
    parser.add_argument(
        "--horizon", type=int, default=_DEFAULT_HORIZON,
        help=f"Forward-looking trading days for return label (default {_DEFAULT_HORIZON})",
    )
    parser.add_argument(
        "--target-pct", type=float, default=_DEFAULT_TARGET_PCT,
        help=f"Minimum return %% to label as positive (default {_DEFAULT_TARGET_PCT})",
    )
    parser.add_argument(
        "--min-rows", type=int, default=_DEFAULT_MIN_ROWS,
        help=f"Minimum ticker history rows required (default {_DEFAULT_MIN_ROWS})",
    )
    parser.add_argument(
        "--test-frac", type=float, default=_DEFAULT_TEST_FRAC,
        help=f"Fraction of chronological data for test set (default {_DEFAULT_TEST_FRAC})",
    )
    parser.add_argument(
        "--save-dataset", action="store_true",
        help="Also save full training DataFrame to models/training_dataset.csv",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build dataset only — do not train or save model",
    )
    parser.add_argument(
        "--max-tickers", type=int, default=None,
        help="Limit number of tickers processed (useful for quick tests)",
    )
    args = parser.parse_args()

    main(
        raw_dir=args.raw_dir,
        model_dir=args.model_dir,
        config_path=args.config,
        horizon_days=args.horizon,
        target_pct=args.target_pct,
        min_rows=args.min_rows,
        test_frac=args.test_frac,
        save_dataset=args.save_dataset,
        dry_run=args.dry_run,
        max_tickers=args.max_tickers,
    )

"""Entry point scan harian IDX Stock Scanner Agent.

Jalankan:
    python -m stock_scanner.pipeline.run_daily_scan
    python -m stock_scanner.pipeline.run_daily_scan --config stock_scanner/configs/scanner_config.yaml
    idx-scan  (jika sudah pip install -e .)

Pipeline:
    load config → load tickers → incremental update → validate
    → features → signal (rules) → ML ranking (opsional)
    → explain (opsional) → simpan output → log ringkasan
"""
import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from stock_scanner.pipeline.explain_agent import explain_batch
from stock_scanner.pipeline.feature_builder import build_features, save_features
from stock_scanner.pipeline.fetch_yfinance import YFinanceFetcher, incremental_update
from stock_scanner.pipeline.ml_ranker import load_ranker, score_candidates
from stock_scanner.pipeline.signal_engine import compute_signal, save_signals
from stock_scanner.pipeline.validator import validate

_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "scanner_config.yaml"


def main(config_path: Path = _DEFAULT_CONFIG) -> None:
    config = _load_config(config_path)
    scan_date = date.today().strftime("%Y-%m-%d")
    logger.info(f"=== IDX Daily Scan: {scan_date} ===")

    # --- Paths dari config ---
    base_dir = Path(config.get("base_dir", "."))
    raw_dir = base_dir / config.get("data_dir", "data/raw")
    features_dir = base_dir / config.get("features_dir", "data/features")
    signals_dir = base_dir / config.get("signals_dir", "data/signals")
    ranked_dir = base_dir / config.get("ranked_dir", "data/ranked")
    model_path = base_dir / config.get("model_path", "models/ranker.pkl")
    universe_path = base_dir / config.get("universe_path", "stock_scanner/configs/idx_universe.csv")

    ranked_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load ticker universe ---
    tickers = _load_universe(universe_path)
    if not tickers:
        logger.error(f"Universe kosong atau tidak ditemukan: {universe_path}")
        return
    logger.info(f"Universe: {len(tickers)} tickers")

    # --- Step 2 & 3: Incremental update + validate ---
    fetcher = YFinanceFetcher(batch_size=config.get("batch_size", 20))
    lookback = config.get("lookback_years", 3)
    max_gap = config.get("max_gap_fill_days", 2)

    all_latest_features: list[pd.DataFrame] = []
    skipped = 0

    for ticker in tickers:
        try:
            df_raw = incremental_update(ticker, fetcher, raw_dir, lookback)
            if df_raw.empty:
                logger.warning(f"{ticker}: data kosong setelah update, skip")
                skipped += 1
                continue

            clean, report = validate(df_raw, ticker, max_gap)
            if clean.empty:
                logger.warning(f"{ticker}: gagal validasi, skip")
                skipped += 1
                continue

            # Step 4: Features — ambil baris terakhir saja untuk scanning
            features = build_features(clean)
            if features.empty:
                skipped += 1
                continue
            latest = features.iloc[[-1]].copy()
            all_latest_features.append(latest)

        except Exception as e:
            logger.error(f"{ticker}: error saat processing — {e}")
            skipped += 1

    logger.info(f"Processing done: {len(all_latest_features)} OK, {skipped} skipped")

    if not all_latest_features:
        logger.error("Tidak ada fitur yang berhasil dihitung. Scan dihentikan.")
        return

    feature_df = pd.concat(all_latest_features, ignore_index=True)
    save_features(feature_df, features_dir, scan_date)

    # --- Step 5: Signal engine (rules) ---
    signals_df = compute_signal(feature_df, config)

    # --- Step 6: ML ranking (opsional — skip jika model belum ada) ---
    model_config = _load_model_config(config_path.parent / "model_config.yaml")
    if model_path.exists():
        try:
            model, _, _ = load_ranker(model_path)
            signals_df = score_candidates(signals_df, model, model_config)
            logger.info("ML ranking applied")
        except Exception as e:
            logger.warning(f"ML ranking gagal (skip): {e}")
    else:
        logger.info(f"Model belum ada ({model_path}) — skip ML ranking")

    # --- Step 7: Explain agent (opsional — hanya top candidates) ---
    explain_cfg = config.get("explain", {})
    if explain_cfg.get("enabled", False):
        max_explain = explain_cfg.get("max_tickers", 10)
        signals_df = explain_batch(signals_df, model_info=model_config, config=config, max_explain=max_explain)
    else:
        logger.info("Explain agent dinonaktifkan (set explain.enabled: true di config untuk aktifkan)")

    # --- Step 8: Simpan output ---
    save_signals(signals_df, signals_dir, scan_date)
    _save_ranked(signals_df, ranked_dir, scan_date)

    # --- Ringkasan ---
    _print_summary(signals_df, scan_date)
    logger.info(f"=== Scan selesai: {scan_date} ===")


def _load_config(path: Path) -> dict:
    if not path.exists():
        logger.warning(f"Config tidak ditemukan: {path} — menggunakan defaults")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_model_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_universe(universe_path: Path) -> list[str]:
    if not universe_path.exists():
        logger.error(f"Universe file tidak ditemukan: {universe_path}")
        return []
    df = pd.read_csv(universe_path)
    if "is_active" in df.columns:
        df = df[df["is_active"].astype(str).str.lower().isin(["true", "1", "yes"])]
    return df["ticker"].tolist()


def _save_ranked(df: pd.DataFrame, ranked_dir: Path, scan_date: str) -> None:
    priority_signals = ["BREAKOUT", "PRE_MARKUP", "WATCH"]
    ranked = df[df["signal"].isin(priority_signals)].copy()
    if "ml_prob" in ranked.columns:
        ranked = ranked.sort_values(["signal", "ml_prob", "total_score"], ascending=[True, False, False])
    else:
        ranked = ranked.sort_values(["signal", "total_score"], ascending=[True, False])

    path = ranked_dir / f"ranked_{scan_date}.csv"
    ranked.to_csv(path, index=False)
    logger.info(f"Ranked output → {path} ({len(ranked)} tickers)")


def _print_summary(df: pd.DataFrame, scan_date: str) -> None:
    total = len(df)
    dist = df["signal"].value_counts().to_dict()
    logger.info(f"--- Ringkasan {scan_date} ---")
    logger.info(f"Total tickers: {total}")
    for signal in ["BREAKOUT", "PRE_MARKUP", "WATCH", "AVOID", "NONE"]:
        count = dist.get(signal, 0)
        logger.info(f"  {signal}: {count}")

    # Print top 5 ke console
    top = df[df["signal"].isin(["BREAKOUT", "PRE_MARKUP"])].head(5)
    if not top.empty:
        logger.info("Top candidates:")
        for _, row in top.iterrows():
            prob_str = f" ml_prob={row['ml_prob']:.3f}" if "ml_prob" in row and pd.notna(row.get("ml_prob")) else ""
            logger.info(f"  {row.get('ticker')} | {row.get('signal')} | score={row.get('total_score', 0):.1f}{prob_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDX Stock Scanner — daily scan")
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path ke scanner_config.yaml",
    )
    args = parser.parse_args()
    main(args.config)

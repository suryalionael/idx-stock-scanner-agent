"""Entry point scan harian IDX Stock Scanner Agent.

Jalankan:
    python -m stock_scanner.pipeline.run_daily_scan
    python -m stock_scanner.pipeline.run_daily_scan --config stock_scanner/configs/scanner_config.yaml
    idx-scan  (jika sudah pip install -e .)

Pipeline:
    load config → load tickers → incremental update → validate
    → features (OHLCV + TV indicators) → enrich news → enrich foreign
    → enrich shareholder → signal (rules) → ML ranking (opsional)
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
from stock_scanner.pipeline.foreign_flow import PlaceholderForeignFetcher, enrich_with_foreign
from stock_scanner.pipeline.fundamental import enrich_with_fundamentals
from stock_scanner.pipeline.ml_ranker import load_ranker, score_candidates
from stock_scanner.pipeline.news_sentiment import enrich_with_news
from stock_scanner.pipeline.quality_filters import (
    enrich_df_with_quality_filters,
    load_risk_overrides,
    EXCLUDED_STATUSES,
)
from stock_scanner.pipeline.shareholder import PlaceholderShareholderFetcher, enrich_with_shareholders
from stock_scanner.alerts.level_calculator import enrich_df_with_levels
from stock_scanner.pipeline.scalping import enrich_df_with_scalping
from stock_scanner.pipeline.signal_engine import compute_signal, save_signals
from stock_scanner.pipeline.validator import validate

_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "scanner_config.yaml"


def main(config_path: Path = _DEFAULT_CONFIG) -> None:
    config = _load_config(config_path)
    scan_date = date.today().strftime("%Y-%m-%d")
    logger.info(f"=== IDX Daily Scan: {scan_date} ===")

    # --- Paths dari config ---
    base_dir = Path(config.get("base_dir", "."))
    raw_dir       = base_dir / config.get("data_dir",       "data/raw")
    features_dir  = base_dir / config.get("features_dir",   "data/features")
    signals_dir   = base_dir / config.get("signals_dir",    "data/signals")
    ranked_dir    = base_dir / config.get("ranked_dir",     "data/ranked")
    news_dir         = base_dir / config.get("news_dir",          "data/news")
    foreign_dir      = base_dir / config.get("foreign_dir",       "data/foreign")
    broker_dir       = base_dir / config.get("broker_dir",        "data/broker")
    shareholder_dir  = base_dir / config.get("shareholder_dir",   "data/shareholders")
    fundamentals_dir = base_dir / config.get("fundamentals_dir",  "data/fundamentals")
    model_path       = base_dir / config.get("model_path",        "models/ranker.pkl")
    universe_path    = base_dir / config.get("universe_path",     "stock_scanner/configs/idx_universe.csv")

    risk_dir = base_dir / config.get("risk_dir", "data/risk")

    for d in [ranked_dir, news_dir, foreign_dir, broker_dir, shareholder_dir, fundamentals_dir, risk_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load ticker universe ---
    tickers = _load_universe(universe_path)
    if not tickers:
        logger.error(f"Universe kosong atau tidak ditemukan: {universe_path}")
        return
    logger.info(f"Universe: {len(tickers)} tickers")

    # --- Step 2 & 3: Incremental update + validate ---
    fetcher = YFinanceFetcher(batch_size=config.get("batch_size", 20))
    lookback = config.get("lookback_years", 3)
    max_gap  = config.get("max_gap_fill_days", 2)

    all_latest_features: list[pd.DataFrame] = []
    skipped = 0

    for ticker in tickers:
        try:
            df_raw = incremental_update(ticker, fetcher, raw_dir, lookback)
            if df_raw.empty:
                skipped += 1
                continue

            clean, _ = validate(df_raw, ticker, max_gap)
            if clean.empty:
                skipped += 1
                continue

            # Step 4: Features (ambil baris terakhir untuk scanning)
            features = build_features(clean)
            if features.empty:
                skipped += 1
                continue
            all_latest_features.append(features.iloc[[-1]].copy())

        except Exception as e:
            logger.error(f"{ticker}: error saat processing — {e}")
            skipped += 1

    logger.info(f"Processing done: {len(all_latest_features)} OK, {skipped} skipped")
    if not all_latest_features:
        logger.error("Tidak ada fitur yang berhasil dihitung. Scan dihentikan.")
        return

    feature_df = pd.concat(all_latest_features, ignore_index=True)
    save_features(feature_df, features_dir, scan_date)

    # --- Step 5: Enrichment (news, foreign, shareholder) ---
    enrich_cfg = config.get("enrichment", {})

    if enrich_cfg.get("news", {}).get("enabled", True):
        news_days = enrich_cfg.get("news", {}).get("days", 3)
        feature_df = enrich_with_news(
            feature_df, news_dir=news_dir, news_days=news_days,
            save=True, scan_date=scan_date,
        )

    if enrich_cfg.get("foreign", {}).get("enabled", True):
        feature_df = enrich_with_foreign(
            feature_df,
            fetcher=PlaceholderForeignFetcher(),
            foreign_dir=foreign_dir,
        )

    if enrich_cfg.get("shareholder", {}).get("enabled", True):
        feature_df = enrich_with_shareholders(
            feature_df,
            fetcher=PlaceholderShareholderFetcher(),
            shareholder_dir=shareholder_dir,
        )

    if enrich_cfg.get("fundamental", {}).get("enabled", True):
        delay = enrich_cfg.get("fundamental", {}).get("delay_between_tickers", 0.3)
        feature_df = enrich_with_fundamentals(
            feature_df,
            fundamentals_dir=fundamentals_dir,
            save=True,
            scan_date=scan_date,
            delay_between_tickers=delay,
        )

    # --- Step 5d: Load UMA / special monitoring overrides ---
    risk_overrides = load_risk_overrides(risk_dir)
    if risk_overrides:
        logger.info(f"Risk overrides loaded: {len(risk_overrides)} tickers flagged")
    else:
        logger.info("No UMA/special monitoring overrides active")

    # --- Step 6: Signal engine (rules) ---
    signals_df = compute_signal(feature_df, config)

    # --- Step 6b: Quality filters (hard exclusion + risk flags) ---
    signals_df = enrich_df_with_quality_filters(signals_df, config, risk_overrides)
    eligible_count   = (signals_df["final_status"] == "eligible").sum()
    watch_count      = (signals_df["final_status"] == "watch_with_risk").sum()
    excluded_count   = signals_df["final_status"].isin(EXCLUDED_STATUSES).sum()
    insuff_count     = (signals_df["final_status"] == "insufficient_data").sum()
    logger.info(
        f"Quality filter: {eligible_count} eligible, {watch_count} watch_with_risk, "
        f"{excluded_count} excluded, {insuff_count} insufficient_data"
    )

    # --- Step 7: ML ranking (opsional) ---
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

    # --- Step 8: Explain agent (opsional) ---
    explain_cfg = config.get("explain", {})
    if explain_cfg.get("enabled", False):
        max_explain = explain_cfg.get("max_tickers", 10)
        signals_df = explain_batch(signals_df, model_info=model_config,
                                   config=config, max_explain=max_explain)
    else:
        logger.info("Explain agent dinonaktifkan")

    # --- Step 8b: Scalping scores ---
    signals_df = enrich_df_with_scalping(signals_df)
    scalp_high = (signals_df.get("scalping_label") == "SCALPING_HIGH").sum() \
                 if "scalping_label" in signals_df.columns else 0
    logger.info(f"Scalping: {scalp_high} SCALPING_HIGH candidates")

    # --- Step 8c: Trading levels (entry / TP / cutloss) with R:R validation ---
    min_rr = config.get("min_rr", 1.5)
    signals_df = enrich_df_with_levels(signals_df, min_rr=min_rr)
    active_count = (signals_df.get("trade_setup_status") == "active").sum() \
                   if "trade_setup_status" in signals_df.columns else 0
    low_rr_count = (signals_df.get("trade_setup_status") == "low_rr").sum() \
                   if "trade_setup_status" in signals_df.columns else 0
    logger.info(f"Trading levels: {active_count} active, {low_rr_count} low_rr (R:R < {min_rr})")

    # --- Step 9: Simpan output ---
    save_signals(signals_df, signals_dir, scan_date)
    _save_ranked(signals_df, ranked_dir, scan_date, config=config)
    _print_summary(signals_df, scan_date)
    logger.info(f"=== Scan selesai: {scan_date} ===")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _save_ranked(
    df: pd.DataFrame,
    ranked_dir: Path,
    scan_date: str,
    config: dict | None = None,
) -> None:
    """Save ranked signals with per-tier caps and quality filter applied.

    Excludes tickers with final_status in EXCLUDED_STATUSES
    (excluded_fundamental, excluded_float_structure, excluded_regulatory).
    'insufficient_data' tickers are kept but sorted last within each tier.

    Caps are read from config['signal_caps'] (default: BREAKOUT=15, PRE_MARKUP=30, WATCH=50).
    Sort order: signal tier → quality_adjusted_score (if present) → ml_prob → total_score.
    """
    caps_cfg = (config or {}).get("signal_caps", {})
    caps = {
        "BREAKOUT":   caps_cfg.get("breakout",   15),
        "PRE_MARKUP": caps_cfg.get("pre_markup",  30),
        "WATCH":      caps_cfg.get("watch",       50),
    }

    priority = ["BREAKOUT", "PRE_MARKUP", "WATCH"]
    ranked = df[df["signal"].isin(priority)].copy()

    # Exclude hard-excluded tickers (leave insufficient_data tickers in, flagged)
    if "final_status" in ranked.columns:
        from stock_scanner.pipeline.quality_filters import EXCLUDED_STATUSES
        before = len(ranked)
        ranked = ranked[~ranked["final_status"].isin(EXCLUDED_STATUSES)]
        excluded_n = before - len(ranked)
        if excluded_n:
            logger.info(f"Ranked: removed {excluded_n} tickers with excluded_* status")

    # Sort within each tier: quality_adjusted_score > ml_prob > total_score
    sort_cols = ["signal"]
    asc = [True]
    if "quality_adjusted_score" in ranked.columns:
        sort_cols.append("quality_adjusted_score")
        asc.append(False)
    elif "ml_prob" in ranked.columns:
        sort_cols.append("ml_prob")
        asc.append(False)
    sort_cols.append("total_score")
    asc.append(False)
    ranked = ranked.sort_values(sort_cols, ascending=asc)

    # Apply per-tier caps
    capped_frames = []
    for sig in priority:
        tier = ranked[ranked["signal"] == sig]
        cap = caps.get(sig, 999)
        if len(tier) > cap:
            logger.info(f"Cap applied: {sig} {len(tier)} → {cap} tickers")
            tier = tier.head(cap)
        capped_frames.append(tier)

    if capped_frames:
        ranked = pd.concat(capped_frames, ignore_index=True)
    else:
        ranked = ranked.head(0)

    ranked_dir.mkdir(parents=True, exist_ok=True)
    path = ranked_dir / f"ranked_{scan_date}.csv"
    ranked.to_csv(path, index=False)

    sig_counts = ranked["signal"].value_counts().to_dict()
    status_counts = ranked["final_status"].value_counts().to_dict() if "final_status" in ranked.columns else {}
    logger.info(f"Ranked output → {path} ({len(ranked)} tickers) | signals={sig_counts} | status={status_counts}")


def _print_summary(df: pd.DataFrame, scan_date: str) -> None:
    dist = df["signal"].value_counts().to_dict()
    logger.info(f"--- Ringkasan {scan_date} ({len(df)} tickers) ---")
    for sig in ["BREAKOUT", "PRE_MARKUP", "WATCH", "AVOID", "NONE"]:
        logger.info(f"  {sig}: {dist.get(sig, 0)}")

    top = df[df["signal"].isin(["BREAKOUT", "PRE_MARKUP"])].head(5)
    for _, row in top.iterrows():
        enh = f" enh={row.get('enhanced_total_score', 0):.1f}" if "enhanced_total_score" in row else ""
        prob = f" ml={row.get('ml_prob', float('nan')):.3f}" if "ml_prob" in row and pd.notna(row.get("ml_prob")) else ""
        logger.info(f"  {row.get('ticker')} | {row.get('signal')} | score={row.get('total_score', 0):.1f}{enh}{prob}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDX Stock Scanner — daily scan")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    args = parser.parse_args()
    main(args.config)

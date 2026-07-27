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
import json
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from stock_scanner.alerts.level_calculator import enrich_df_with_levels
from stock_scanner.pipeline.explain_agent import explain_batch
from stock_scanner.pipeline.feature_builder import build_features, save_features
from stock_scanner.pipeline.fetch_yfinance import YFinanceFetcher, incremental_update
from stock_scanner.pipeline.foreign_flow import PlaceholderForeignFetcher, enrich_with_foreign
from stock_scanner.pipeline.fundamental import enrich_with_fundamentals
from stock_scanner.pipeline.knowledge_application import apply_knowledge_ranking
from stock_scanner.pipeline.ml_ranker import load_ranker, score_candidates
from stock_scanner.pipeline.news_sentiment import enrich_with_news
from stock_scanner.pipeline.quality_filters import (
    EXCLUDED_STATUSES,
    enrich_df_with_quality_filters,
    load_risk_overrides,
)
from stock_scanner.pipeline.scalping import enrich_df_with_scalping
from stock_scanner.pipeline.shareholder import (
    PlaceholderShareholderFetcher,
    enrich_with_shareholders,
)
from stock_scanner.pipeline.signal_engine import compute_signal, save_signals
from stock_scanner.pipeline.validator import validate

_DEFAULT_CONFIG = Path(__file__).parent.parent / "configs" / "scanner_config.yaml"


def main(config_path: Path = _DEFAULT_CONFIG, force_holiday: bool = False) -> None:
    config = _load_config(config_path)

    # ── Trading calendar guard ───────────────────────────────────────────────
    # Bedakan execution_date (kapan script jalan) vs scan_date/market_date
    # (tanggal data market yang sebenarnya dipakai).
    # scan_date akan diperbarui dari data aktual setelah feature build.
    from stock_scanner.utils.trading_calendar import is_trading_day, last_trading_day

    execution_date = date.today()
    execution_date_str = execution_date.strftime("%Y-%m-%d")

    if not is_trading_day(execution_date):
        last_td = last_trading_day(execution_date)
        if not force_holiday:
            logger.info(
                f"=== {execution_date_str} bukan hari bursa IDX "
                f"(last trading day: {last_td}). "
                f"Pipeline tidak dijalankan. Gunakan --force-holiday untuk override. ==="
            )
            return
        logger.warning(
            f"--force-holiday aktif: {execution_date_str} bukan hari bursa, "
            f"pipeline tetap dijalankan. Last trading day: {last_td}."
        )

    # Tentative scan_date = execution date; akan di-override dari data aktual
    # setelah feature build (lihat "Determine actual market_date" di bawah).
    scan_date = execution_date_str
    logger.info(f"=== IDX Daily Scan: execution={execution_date_str} ===")

    # --- Paths dari config ---
    base_dir = Path(config.get("base_dir", "."))
    raw_dir = base_dir / config.get("data_dir", "data/raw")
    features_dir = base_dir / config.get("features_dir", "data/features")
    signals_dir = base_dir / config.get("signals_dir", "data/signals")
    ranked_dir = base_dir / config.get("ranked_dir", "data/ranked")
    news_dir = base_dir / config.get("news_dir", "data/news")
    foreign_dir = base_dir / config.get("foreign_dir", "data/foreign")
    broker_dir = base_dir / config.get("broker_dir", "data/broker")
    shareholder_dir = base_dir / config.get("shareholder_dir", "data/shareholders")
    fundamentals_dir = base_dir / config.get("fundamentals_dir", "data/fundamentals")
    model_path = base_dir / config.get("model_path", "models/ranker.pkl")
    universe_path = base_dir / config.get("universe_path", "stock_scanner/configs/idx_universe.csv")

    risk_dir = base_dir / config.get("risk_dir", "data/risk")

    for d in [
        ranked_dir,
        news_dir,
        foreign_dir,
        broker_dir,
        shareholder_dir,
        fundamentals_dir,
        risk_dir,
    ]:
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
    max_gap = config.get("max_gap_fill_days", 2)

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

            # ── Strip trailing zero-volume rows (yfinance holiday artefact) ──────
            # yfinance kadang mengembalikan baris dengan volume=0 dan OHLC di-ffill
            # untuk hari libur (misal: cuti bersama, Kenaikan Isa Almasih).
            # Jika baris terakhir volume=0, _hard_gate() akan langsung AVOID semua.
            # Solusi: potong baris trailing zero-volume agar feature_builder
            # selalu menggunakan hari perdagangan terakhir yang nyata.
            non_zero_vol = clean["volume"].fillna(0) > 0
            if non_zero_vol.any():
                last_real_idx = clean[non_zero_vol].index[-1]
                n_trailing = (clean.index > last_real_idx).sum()
                if n_trailing > 0:
                    logger.debug(
                        f"{ticker}: stripped {n_trailing} trailing zero-volume row(s) "
                        f"(holiday artefact from yfinance)"
                    )
                    clean = clean.loc[:last_real_idx]
            else:
                # Seluruh series tidak punya volume — biasanya suspended/delisted
                logger.debug(f"{ticker}: entire series has zero volume — skipped")
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

    # ── Determine actual market_date from data ───────────────────────────────
    # scan_date = tanggal market data yang benar-benar dipakai (bukan execution date).
    # Setelah zero-volume strip, setiap ticker sudah punya baris dari last real
    # trading day. Kita ambil tanggal terbaru dari feature_df sebagai scan_date.
    if "date" in feature_df.columns:
        market_date = pd.to_datetime(feature_df["date"]).max().date()
        market_date_str = market_date.strftime("%Y-%m-%d")
        if market_date_str != scan_date:
            logger.info(
                f"Market data date: {market_date_str} "
                f"(execution: {execution_date_str}) — scan_date diperbarui dari data."
            )
        scan_date = market_date_str

    is_live_scan: bool = scan_date == execution_date_str
    if not is_live_scan:
        logger.info(
            f"⚠ Non-live scan: execution={execution_date_str}, "
            f"market_data={scan_date}. "
            f"Dashboard akan menampilkan staleness warning."
        )

    save_features(feature_df, features_dir, scan_date)

    # --- Step 5: Enrichment (news, foreign, shareholder) ---
    enrich_cfg = config.get("enrichment", {})

    if enrich_cfg.get("news", {}).get("enabled", True):
        news_days = enrich_cfg.get("news", {}).get("days", 3)
        feature_df = enrich_with_news(
            feature_df,
            news_dir=news_dir,
            news_days=news_days,
            save=True,
            scan_date=scan_date,
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
    eligible_count = (signals_df["final_status"] == "eligible").sum()
    watch_count = (signals_df["final_status"] == "watch_with_risk").sum()
    excluded_count = signals_df["final_status"].isin(EXCLUDED_STATUSES).sum()
    insuff_count = (signals_df["final_status"] == "insufficient_data").sum()
    logger.info(
        f"Quality filter: {eligible_count} eligible, {watch_count} watch_with_risk, "
        f"{excluded_count} excluded, {insuff_count} insufficient_data"
    )

    # --- Step 6c: Promoted challenger score (DB-driven closed loop, optional) ---
    # Ranking-only enrichment — see docs/SELF_IMPROVING_ARCHITECTURE.md. Never
    # touches signal classification (already fixed above in compute_signal /
    # quality filters) and never touches SQLite (reads the committed JSON
    # registry mirror only). No-op when no model is currently promoted.
    signals_df = _apply_promoted_challenger_score(signals_df)

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

    # --- Step 7b: Knowledge Application (opsional, ranking-only, deterministic) ---
    # Rules First (Step 6/6b) → ML Second (Step 7) → Knowledge Third (here).
    # Never touches signal/final_status — see stock_scanner/pipeline/knowledge_application.py.
    signals_df = _apply_knowledge_ranking(signals_df, config)

    # --- Step 8: Explain agent (opsional) ---
    explain_cfg = config.get("explain", {})
    if explain_cfg.get("enabled", False):
        max_explain = explain_cfg.get("max_tickers", 10)
        signals_df = explain_batch(
            signals_df, model_info=model_config, config=config, max_explain=max_explain
        )
    else:
        logger.info("Explain agent dinonaktifkan")

    # --- Step 8b: Scalping scores ---
    signals_df = enrich_df_with_scalping(signals_df)
    scalp_high = (
        (signals_df.get("scalping_label") == "SCALPING_HIGH").sum()
        if "scalping_label" in signals_df.columns
        else 0
    )
    logger.info(f"Scalping: {scalp_high} SCALPING_HIGH candidates")

    # --- Step 8c: Trading levels (entry / TP / cutloss) with R:R validation ---
    min_rr = config.get("min_rr", 1.5)
    signals_df = enrich_df_with_levels(signals_df, min_rr=min_rr)
    active_count = (
        (signals_df.get("trade_setup_status") == "active").sum()
        if "trade_setup_status" in signals_df.columns
        else 0
    )
    low_rr_count = (
        (signals_df.get("trade_setup_status") == "low_rr").sum()
        if "trade_setup_status" in signals_df.columns
        else 0
    )
    logger.info(f"Trading levels: {active_count} active, {low_rr_count} low_rr (R:R < {min_rr})")

    # --- Step 9: Simpan output ---
    save_signals(signals_df, signals_dir, scan_date)
    _save_ranked(signals_df, ranked_dir, scan_date, config=config)
    _print_summary(signals_df, scan_date)

    # --- Step 9b: Pre-warm multi-period financials store for ranked candidates ---
    # so the dashboard reads committed data instead of a live yfinance call at
    # view time (unreliable on Streamlit Cloud). Non-fatal.
    _prewarm_financials(signals_df)

    # --- Step 10: Publish payload untuk dashboard online (non-fatal) ---
    _publish_dashboard_data(
        signals_df,
        scan_date,
        base_dir,
        execution_date=execution_date_str,
        is_live_scan=is_live_scan,
    )

    # --- Step 10b: Recent-OHLC bundle for dashboard charts (non-fatal) ---
    # data/raw/ is gitignored (not deployed), so commit a compact recent-OHLC
    # bundle the deployed dashboard can read for charts without live yfinance.
    try:
        from stock_scanner.pipeline.publisher import export_recent_ohlc
        from stock_scanner.pipeline.suspension import filter_active

        _pub_tickers = filter_active(signals_df)["ticker"].astype(str).tolist()
        export_recent_ohlc(_pub_tickers, raw_dir, sessions=250)
    except Exception as exc:  # noqa: BLE001
        logger.warning("OHLC bundle export skipped: {}", exc)

    # --- Step 10c: IHSG (composite index) bundle for the benchmark panel ---
    try:
        from stock_scanner.pipeline.publisher import export_ihsg_bundle

        export_ihsg_bundle(sessions=250)
    except Exception as exc:  # noqa: BLE001
        logger.warning("IHSG bundle export skipped: {}", exc)

    # --- Step 11: Signal List performance tracking (non-fatal) ---
    # Archives today's Swing/Scalping signals and evaluates any pending prior
    # signals against this freshly-scanned session's OHLC. Writes CSV + Excel.
    try:
        from stock_scanner.pipeline.performance import run_performance

        run_performance()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Performance tracking skipped: {}", exc)

    # --- Step 12: AI Lab automation chain (non-fatal) ---
    # generation -> resolution -> reflection -> hypothesis + statistical
    # validation -> knowledge base. Runs strictly after every production
    # scoring/ranking/filtering/publishing step above has already completed —
    # AI Lab never feeds back into today's production output. Each of the 5
    # stages is independently isolated inside run_ai_pipeline() itself; this
    # outer try/except is a last-resort safety net on top of that. Gated by
    # ai_lab.enabled (scanner_config.yaml) so the whole chain can be disabled
    # without a code revert. See docs/ADR_AI_AUTOMATION_AND_STOCK_DICTIONARY.md.
    if config.get("ai_lab", {}).get("enabled", True):
        try:
            import asyncio

            from stock_scanner.ai_lab.pipeline import run_ai_pipeline

            ai_lab_summary = asyncio.run(run_ai_pipeline(scan_date, config))
            logger.info(f"AI Lab automation summary: {ai_lab_summary}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("AI Lab automation chain skipped entirely: {}", exc)
    else:
        logger.info("AI Lab automation disabled via config (ai_lab.enabled: false)")

    logger.info(f"=== Scan selesai: {scan_date} ===")


def _apply_promoted_challenger_score(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the currently-promoted 'rule_score' challenger (if any) as an
    additional ranking signal — see docs/SELF_IMPROVING_ARCHITECTURE.md.

    Reads only the committed data/published/model_registry.json (no SQLite
    dependency in the morning scan). Purely additive: adds
    promoted_rule_score (used as a ranking tie-breaker by _save_ranked) plus
    audit-trail columns (promoted_model_id, promoted_model_type,
    promoted_threshold_used, promoted_at, ranking_source) so a ranked file
    can be traced back to the exact promotion decision that produced it,
    months later. Never touches signal classification or hard gates.
    Best-effort: any failure here is logged and the scan continues with
    today's (pre-existing) behavior — a promoted-model lookup must never
    break the daily scan.
    """
    from stock_scanner.db.model_lookup import get_promoted_model
    from stock_scanner.pipeline.challenger_score import compute_rule_score

    model_row = get_promoted_model("rule_score")
    if model_row is None:
        logger.info("No promoted rule_score challenger yet — skip")
        return df

    try:
        metrics_json = (
            model_row.get("test_metrics_json") or model_row.get("train_metrics_json") or "{}"
        )
        metrics = json.loads(metrics_json)
        thresh = metrics.get("vol_ratio_20d_threshold", 2.0)

        df = df.copy()
        df["promoted_rule_score"] = compute_rule_score(df, thresh)
        df["promoted_model_id"] = model_row.get("model_version_id")
        df["promoted_model_type"] = model_row.get("model_type")
        df["promoted_threshold_used"] = thresh
        df["promoted_at"] = model_row.get("promoted_at")
        df["ranking_source"] = "promoted_challenger"
        logger.info(
            f"Promoted challenger applied: model={model_row.get('model_version_id')} "
            f"threshold={thresh}"
        )
    except Exception as e:  # noqa: BLE001 — a lookup/scoring failure must never break the scan
        logger.warning(f"Promoted challenger score failed (skip, non-fatal): {e}")
        return df

    return df


def _apply_knowledge_ranking(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply curated AI Lab knowledge (data/published/knowledge_report.json)
    as a bounded ranking adjustment — see stock_scanner/pipeline/
    knowledge_application.py. Reads only the committed JSON mirror (no
    SQLite dependency in the morning scan, same rule as the promoted
    challenger step above). Purely additive: adds knowledge_bonus/
    knowledge_adjusted_score/knowledge_matched_ids/knowledge_applied_rules
    audit columns. Never touches signal classification or hard gates.
    Best-effort: any failure here is logged and the scan continues with
    today's (pre-existing) behavior — a knowledge lookup must never break
    the daily scan."""
    try:
        return apply_knowledge_ranking(df, config=config.get("knowledge_application", {}))
    except Exception as e:  # noqa: BLE001 — a knowledge lookup/scoring failure must never break the scan
        logger.warning(f"Knowledge application failed (skip, non-fatal): {e}")
        return df


def _prewarm_financials(signals_df: pd.DataFrame, max_tickers: int = 60) -> None:
    """Fetch + persist multi-period financials for ranked candidates.

    Populates data/published/financials/{ticker}.json so the dashboard (incl.
    Streamlit Cloud) serves committed data instead of a live yfinance call at
    view time. Best-effort: failures are logged, never fatal.
    """
    try:
        if signals_df is None or signals_df.empty or "signal" not in signals_df.columns:
            return
        from stock_scanner.pipeline.long_term import compare_financial_statements

        cand = signals_df[signals_df["signal"].isin(["BREAKOUT", "PRE_MARKUP", "WATCH"])]
        tickers = cand["ticker"].astype(str).head(max_tickers).tolist()
        ok = 0
        for t in tickers:
            try:
                r = compare_financial_statements(t)  # writes the cache on success
                if r.get("status") == "ok":
                    ok += 1
            except Exception:  # noqa: BLE001
                continue
        logger.info("Financials pre-warm: {}/{} ranked candidates cached.", ok, len(tickers))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Financials pre-warm skipped: {}", exc)


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
        "BREAKOUT": caps_cfg.get("breakout", 15),
        "PRE_MARKUP": caps_cfg.get("pre_markup", 30),
        "WATCH": caps_cfg.get("watch", 50),
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

    # Sort within each tier: knowledge_adjusted_score > quality_adjusted_score
    # > promoted_rule_score > ml_prob > total_score. knowledge_adjusted_score
    # (Rules First, ML Second, Knowledge Third — see
    # stock_scanner/pipeline/knowledge_application.py) is `<whichever of the
    # other three would otherwise be used> + knowledge_bonus`, so when no
    # applicable knowledge exists (knowledge_bonus == 0 everywhere, or the
    # column is entirely absent) this cascade sorts identically to before —
    # byte-identical output is a property of the column's definition, not a
    # separate check here. promoted_rule_score ranks ahead of ml_prob because
    # it's validated against the exact live signal population/label —
    # ml_prob's XGBoost track is trained on a broader, mismatched population
    # (see docs/SELF_IMPROVING_ARCHITECTURE.md §0). Absent columns fall
    # through unchanged.
    sort_cols = ["signal"]
    asc = [True]
    if "knowledge_adjusted_score" in ranked.columns:
        sort_cols.append("knowledge_adjusted_score")
        asc.append(False)
    elif "quality_adjusted_score" in ranked.columns:
        sort_cols.append("quality_adjusted_score")
        asc.append(False)
    elif "promoted_rule_score" in ranked.columns:
        sort_cols.append("promoted_rule_score")
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
    status_counts = (
        ranked["final_status"].value_counts().to_dict() if "final_status" in ranked.columns else {}
    )
    logger.info(
        f"Ranked output → {path} ({len(ranked)} tickers) | signals={sig_counts} | status={status_counts}"
    )


def _publish_dashboard_data(
    signals_df: pd.DataFrame,
    scan_date: str,
    base_dir: Path,
    execution_date: str | None = None,
    is_live_scan: bool = True,
) -> None:
    """Generate data/published/latest_scan.json — non-fatal wrapper.

    Jika publish gagal (disk full, permission error, dll) pipeline tetap
    dianggap berhasil. Error di-log tapi tidak di-raise.
    """
    try:
        from stock_scanner.pipeline.publisher import export_latest_dashboard_data

        output_path = base_dir / "data" / "published" / "latest_scan.json"
        export_latest_dashboard_data(
            signals_df=signals_df,
            scan_date=scan_date,
            ai_summary=None,  # daily_report summary akan di-attach oleh runner.py
            output_path=output_path,
            execution_date=execution_date,
            is_live_scan=is_live_scan,
        )
    except Exception as exc:
        logger.warning(f"Publish dashboard data gagal (non-fatal): {exc}")


def _print_summary(df: pd.DataFrame, scan_date: str) -> None:
    dist = df["signal"].value_counts().to_dict()
    logger.info(f"--- Ringkasan {scan_date} ({len(df)} tickers) ---")
    for sig in ["BREAKOUT", "PRE_MARKUP", "WATCH", "AVOID", "NONE"]:
        logger.info(f"  {sig}: {dist.get(sig, 0)}")

    top = df[df["signal"].isin(["BREAKOUT", "PRE_MARKUP"])].head(5)
    for _, row in top.iterrows():
        enh = (
            f" enh={row.get('enhanced_total_score', 0):.1f}"
            if "enhanced_total_score" in row
            else ""
        )
        prob = (
            f" ml={row.get('ml_prob', float('nan')):.3f}"
            if "ml_prob" in row and pd.notna(row.get("ml_prob"))
            else ""
        )
        logger.info(
            f"  {row.get('ticker')} | {row.get('signal')} | score={row.get('total_score', 0):.1f}{enh}{prob}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDX Stock Scanner — daily scan")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    parser.add_argument(
        "--force-holiday",
        action="store_true",
        default=False,
        help=(
            "Jalankan pipeline meskipun hari ini adalah non-trading day IDX. "
            "Berguna untuk backfill manual atau testing pada hari libur."
        ),
    )
    args = parser.parse_args()
    main(args.config, force_holiday=args.force_holiday)

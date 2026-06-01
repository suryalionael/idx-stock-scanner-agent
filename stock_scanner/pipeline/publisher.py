"""Publisher — mengubah hasil scan harian menjadi artefak ringkas untuk dashboard online.

Menghasilkan: data/published/latest_scan.json

Format payload:
  {
    "scan_date":    "2026-05-27",
    "generated_at": "2026-05-27T06:45:12+07:00",
    "meta": {
      "total_tickers":   int,
      "breakout_count":  int,
      "pre_markup_count":int,
      "watch_count":     int,
      "avoid_count":     int,
    },
    "breakout":   [ { ticker fields... }, ... ],
    "pre_markup": [ { ticker fields... }, ... ],
    "scalping":   [ { ticker fields... }, ... ],   # SCALPING_HIGH saja
    "watch":      [ { ticker fields... }, ... ],
    "ai_summary": str | null,                      # Claude AI summary jika ada
  }

Usage:
    from stock_scanner.pipeline.publisher import build_published_payload, export_latest_dashboard_data

    payload = build_published_payload(signals_df, scan_date)
    export_latest_dashboard_data(signals_df, scan_date)   # tulis ke data/published/
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

_ROOT           = Path(__file__).parent.parent.parent
_PUBLISHED_DIR  = _ROOT / "data" / "published"
_PUBLISHED_PATH = _PUBLISHED_DIR / "latest_scan.json"

# Kolom ticker yang diikutsertakan dalam payload (tier lists + all_tickers).
# Harus mencakup semua kolom yang dibutuhkan Scalping tab, Long Term tab,
# dan Swing tab di dashboard agar parity local == deployed terjaga.
_TICKER_COLS = [
    # Identity
    "ticker",
    "signal",
    # Composite scores
    "total_score",
    "quality_adjusted_score",
    "enhanced_total_score",
    "trend_score", "momentum_score", "breakout_score", "volume_score",
    "penalty_score", "news_score",
    # Scalping outputs (computed in pipeline step 8b)
    "scalping_label",
    "scalping_score",
    "scalping_reason",
    # Price levels (from level_calculator)
    "close",
    "entry_low", "entry_high",
    "tp_low",    "tp_high",
    "cutloss",
    "trade_setup_status",
    # Technical indicators — core
    "rsi14", "adx", "atr_pct", "atr14",
    "vol_ratio_20d", "pct_from_52w_high",
    "supertrend_bullish", "squeeze_on",
    # Technical indicators — scalping signals
    "atr_breakout", "vol_spike", "hist_vol_20d",
    "roc5", "roc20",
    # Technical indicators — long-term / trend
    "ma20", "ma50", "ma200",
    "macd_histogram", "obv_trend",
    # Fundamental — valuation
    "pe_ratio", "pbv", "roe_pct", "der", "public_float_pct",
    "ebitda", "fundamental_status", "eps",
    # Fundamental — growth (needed by Long Term tab)
    "revenue_growth_pct", "profit_growth_pct", "div_yield_pct",
    # News
    "news_sentiment_score", "news_count_3d", "news_data_status",
    # Quality filters
    "final_status", "risk_flags",
    # ML
    "ml_prob",
]

# Kolom yang harus dinormalisasi agar JSON-serializable
_FLOAT_COLS = {
    "total_score", "quality_adjusted_score", "enhanced_total_score",
    "scalping_score",
    "close", "entry_low", "entry_high", "tp_low", "tp_high", "cutloss",
    "trend_score", "momentum_score", "breakout_score", "volume_score",
    "penalty_score", "news_score",
    "rsi14", "adx", "atr_pct", "atr14",
    "vol_ratio_20d", "pct_from_52w_high",
    "roc5", "roc20", "hist_vol_20d",
    "ma20", "ma50", "ma200", "macd_histogram",
    "pe_ratio", "pbv", "roe_pct", "der", "public_float_pct",
    "ebitda", "eps", "revenue_growth_pct", "profit_growth_pct", "div_yield_pct",
    "news_sentiment_score", "ml_prob",
}

_WIB = timezone(timedelta(hours=7))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_val(v: Any) -> Any:
    """Convert a value to JSON-serializable form."""
    if v is None:
        return None
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 4)
    if isinstance(v, (bool, int, str)):
        return v
    # numpy scalar
    try:
        import numpy as np
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            f = float(v)
            return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
        if isinstance(v, np.bool_):
            return bool(v)
    except ImportError:
        pass
    return str(v)


def _row_to_dict(row: pd.Series) -> dict:
    """Convert a DataFrame row to a clean JSON-serializable dict."""
    out: dict = {}
    for col in _TICKER_COLS:
        if col in row.index:
            out[col] = _safe_val(row[col])
    return out


def _filter_tier(df: pd.DataFrame, signal: str) -> list[dict]:
    """Return list of row dicts for given signal tier, sorted by score desc."""
    tier = df[df["signal"] == signal].copy()
    # Sort by quality_adjusted_score if present, else total_score
    sort_col = "quality_adjusted_score" if "quality_adjusted_score" in tier.columns else "total_score"
    tier = tier.sort_values(sort_col, ascending=False)
    return [_row_to_dict(row) for _, row in tier.iterrows()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_published_payload(
    signals_df: pd.DataFrame,
    scan_date: str,
    ai_summary: str | None = None,
    execution_date: str | None = None,
    is_live_scan: bool = True,
) -> dict:
    """Bangun payload JSON dari signals DataFrame.

    Args:
        signals_df     : DataFrame hasil scan harian (sudah di-enrich semua step).
        scan_date      : Tanggal data market (YYYY-MM-DD) — bisa berbeda dari
                         execution_date pada hari libur bursa.
        ai_summary     : Ringkasan teks Claude (opsional).
        execution_date : Tanggal script dijalankan (YYYY-MM-DD). Sama dengan
                         scan_date pada hari bursa normal. Berbeda saat libur.
        is_live_scan   : True jika execution_date == scan_date (hari bursa aktif).
                         False = data berasal dari last trading day sebelumnya.

    Returns:
        dict yang bisa di-json.dumps().
    """
    df = signals_df.copy()
    # Simpan referensi ke semua ticker (sebelum filter EXCLUDED_STATUSES)
    # agar all_tickers payload mencakup SEMUA 957 ticker persis seperti local.
    signals_df_original = df.copy()

    # -- Hitung meta --
    sig_col = df["signal"] if "signal" in df.columns else pd.Series(dtype=str)
    total    = len(df)
    bo_cnt   = int((sig_col == "BREAKOUT").sum())
    pm_cnt   = int((sig_col == "PRE_MARKUP").sum())
    wa_cnt   = int((sig_col == "WATCH").sum())
    av_cnt   = int((sig_col == "AVOID").sum())

    # Exclude hard-excluded tickers dari tier lists (keep dalam meta count)
    try:
        from stock_scanner.pipeline.quality_filters import EXCLUDED_STATUSES
        if "final_status" in df.columns:
            df = df[~df["final_status"].isin(EXCLUDED_STATUSES)].copy()
    except ImportError:
        pass

    # -- Tier lists --
    breakout_list   = _filter_tier(df, "BREAKOUT")
    pre_markup_list = _filter_tier(df, "PRE_MARKUP")
    watch_list      = _filter_tier(df, "WATCH")

    # Scalping: SCALPING_HIGH, sorted by scalping_score
    scalp_list: list[dict] = []
    if "scalping_label" in df.columns:
        scalp_df = df[df["scalping_label"] == "SCALPING_HIGH"].copy()
        if "scalping_score" in scalp_df.columns:
            scalp_df = scalp_df.sort_values("scalping_score", ascending=False)
        scalp_list = [_row_to_dict(row) for _, row in scalp_df.iterrows()]

    # ALL tickers — untuk parity local == deployed di Scalping & Long Term tabs.
    # Menggunakan signals_df asli (sebelum exclude EXCLUDED_STATUSES di atas)
    # agar semua 957 ticker tersedia di dashboard cloud persis seperti local.
    # signals_df_original sudah di-copy sebelum filter EXCLUDED_STATUSES.
    all_ticker_list = [_row_to_dict(row) for _, row in signals_df_original.iterrows()]

    now_wib = datetime.now(_WIB).strftime("%Y-%m-%dT%H:%M:%S+07:00")

    payload = {
        # scan_date = tanggal data market yang dipakai (last real trading day)
        # execution_date = kapan script dijalankan (bisa sama atau lebih baru)
        # is_live_scan = True jika pipeline jalan pada hari bursa aktif
        "scan_date":      scan_date,
        "execution_date": execution_date or scan_date,
        "is_live_scan":   is_live_scan,
        "generated_at":   now_wib,
        "meta": {
            "total_tickers":    total,
            "breakout_count":   bo_cnt,
            "pre_markup_count": pm_cnt,
            "watch_count":      wa_cnt,
            "avoid_count":      av_cnt,
            "scalping_high_count": len(scalp_list),
        },
        "breakout":    breakout_list,
        "pre_markup":  pre_markup_list,
        "scalping":    scalp_list,
        "watch":       watch_list,
        "all_tickers": all_ticker_list,   # semua ticker — dashboard parity
        "ai_summary":  ai_summary,
    }

    logger.debug(
        "Payload built: %d breakout, %d pre_markup, %d scalping, %d watch, %d all_tickers",
        len(breakout_list), len(pre_markup_list), len(scalp_list), len(watch_list), len(all_ticker_list),
    )
    return payload


def export_latest_dashboard_data(
    signals_df: pd.DataFrame,
    scan_date: str,
    ai_summary: str | None = None,
    output_path: Path | None = None,
    execution_date: str | None = None,
    is_live_scan: bool = True,
) -> Path:
    """Build payload dan tulis ke data/published/latest_scan.json.

    Args:
        signals_df  : DataFrame hasil scan harian.
        scan_date   : Format YYYY-MM-DD.
        ai_summary  : Ringkasan Claude (opsional).
        output_path : Override lokasi file (default: data/published/latest_scan.json).

    Returns:
        Path ke file yang ditulis.

    Raises:
        IOError jika tidak bisa menulis file.
    """
    out_path = output_path or _PUBLISHED_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_published_payload(
        signals_df, scan_date, ai_summary,
        execution_date=execution_date,
        is_live_scan=is_live_scan,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    size_kb = out_path.stat().st_size / 1024
    logger.info(
        "Published payload written: %s (%.1f KB, breakout=%d, pre_markup=%d, scalping=%d, all=%d)",
        out_path.name,
        size_kb,
        len(payload["breakout"]),
        len(payload["pre_markup"]),
        len(payload["scalping"]),
        len(payload.get("all_tickers", [])),
    )
    return out_path

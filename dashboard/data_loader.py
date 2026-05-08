"""Data loading utilities untuk dashboard.

Semua I/O terpusat di sini agar app.py tetap bersih dari path logic.
"""
from pathlib import Path
from datetime import date

import pandas as pd

# --- Path roots (relatif dari root repo) ---
_ROOT = Path(__file__).parent.parent
_RANKED_DIR    = _ROOT / "data" / "ranked"
_RAW_DIR       = _ROOT / "data" / "raw"
_SIGNALS_DIR   = _ROOT / "data" / "signals"
_NEWS_DIR      = _ROOT / "data" / "news"
_FOREIGN_DIR   = _ROOT / "data" / "foreign"
_BROKER_DIR    = _ROOT / "data" / "broker"

# Kolom tabel utama (urutan display) — termasuk kolom baru
TABLE_COLS = [
    "ticker", "signal", "total_score", "enhanced_total_score",
    "trend_score", "momentum_score", "breakout_score", "volume_score", "penalty_score",
    "news_score", "foreign_score",
    "close", "rsi14", "vol_ratio_20d", "pct_from_52w_high",
    "adx", "supertrend_bullish", "squeeze_on",
    "atr_breakout", "vol_spike",
    "news_sentiment_score", "news_count_3d",
]

HISTORY_COLS = [
    "date", "ticker", "signal", "total_score",
    "close", "rsi14", "vol_ratio_20d", "pct_from_52w_high",
    "news_sentiment_score", "foreign_flow_score",
]


# ---------------------------------------------------------------------------
# Date discovery
# ---------------------------------------------------------------------------

def list_ranked_dates() -> list[str]:
    """Kembalikan daftar tanggal yang punya file ranked, urutan descending."""
    files = sorted(_RANKED_DIR.glob("ranked_*.csv"), reverse=True)
    dates = []
    for f in files:
        stem = f.stem
        parts = stem.split("_", 1)
        if len(parts) == 2:
            dates.append(parts[1])
    return dates


def list_signals_dates() -> list[str]:
    """Tanggal yang punya file signals (lebih lengkap dari ranked)."""
    files = sorted(_SIGNALS_DIR.glob("*.parquet"), reverse=True)
    return [f.stem for f in files]


def latest_ranked_date() -> str | None:
    dates = list_ranked_dates()
    return dates[0] if dates else None


def available_dates() -> list[str]:
    """Gabungan tanggal dari ranked dan signals, deduplicated, descending."""
    ranked = set(list_ranked_dates())
    signals = set(list_signals_dates())
    all_dates = sorted(ranked | signals, reverse=True)
    return all_dates


# ---------------------------------------------------------------------------
# Load ranked / signals
# ---------------------------------------------------------------------------

def _normalize_bool_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Normalisasi tipe kolom boolean dan numeric."""
    bool_cols = ["atr_breakout", "vol_spike", "ma_full_alignment", "ma_partial_alignment",
                 "golden_cross", "obv_trend", "supertrend_bullish", "squeeze_on"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1"])

    score_cols = ["trend_score", "momentum_score", "breakout_score", "volume_score",
                  "penalty_score", "total_score", "enhanced_total_score",
                  "news_score", "foreign_score"]
    for col in score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    return df


def load_ranked(scan_date: str) -> pd.DataFrame:
    """Load ranked_{scan_date}.csv. Return DataFrame kosong jika tidak ada."""
    path = _RANKED_DIR / f"ranked_{scan_date}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return _normalize_bool_cols(df)


def load_signals_for_date(scan_date: str) -> pd.DataFrame:
    """Load signals/{scan_date}.parquet atau .csv — berisi SEMUA ticker."""
    parquet = _SIGNALS_DIR / f"{scan_date}.parquet"
    csv = _SIGNALS_DIR / f"{scan_date}.csv"
    if parquet.exists():
        df = pd.read_parquet(parquet)
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        return pd.DataFrame()
    return _normalize_bool_cols(df)


def load_all_tickers_for_date(scan_date: str) -> pd.DataFrame:
    """Load semua ticker untuk tanggal tertentu.

    Prioritas: signals (semua ticker) → ranked (hanya WATCH+ke atas).
    Fallback ke ranked jika signals tidak ada.
    """
    df = load_signals_for_date(scan_date)
    if not df.empty:
        return df
    return load_ranked(scan_date)


# ---------------------------------------------------------------------------
# Load raw OHLCV
# ---------------------------------------------------------------------------

def load_raw(ticker: str) -> pd.DataFrame:
    """Load OHLCV parquet untuk satu ticker."""
    path = _RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Load history
# ---------------------------------------------------------------------------

def load_all_ranked(
    min_signal: list[str] | None = None,
    ticker_filter: str | None = None,
    limit_rows: int = 500,
) -> pd.DataFrame:
    """Concat semua ranked CSV jadi satu DataFrame untuk halaman History."""
    files = sorted(_RANKED_DIR.glob("ranked_*.csv"), reverse=True)
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            stem = f.stem.split("_", 1)
            if "date" not in df.columns and len(stem) == 2:
                df["date"] = stem[1]
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")

    if min_signal:
        combined = combined[combined["signal"].isin(min_signal)]
    if ticker_filter:
        combined = combined[combined["ticker"].str.contains(ticker_filter, case=False, na=False)]

    combined = combined.sort_values(
        ["date", "total_score"], ascending=[False, False]
    ).reset_index(drop=True)

    available = [c for c in HISTORY_COLS if c in combined.columns]
    return combined[available].head(limit_rows)


# ---------------------------------------------------------------------------
# Table display helper
# ---------------------------------------------------------------------------

def get_table_df(df: pd.DataFrame) -> pd.DataFrame:
    """Pilih dan urutkan kolom untuk tabel sinyal utama."""
    available = [c for c in TABLE_COLS if c in df.columns]
    result = df[available].copy()
    if "total_score" in result.columns:
        result = result.sort_values("total_score", ascending=False)
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Broker data
# ---------------------------------------------------------------------------

def load_broker_for_ticker(ticker: str, selected_date: str, use_mock: bool = True) -> pd.DataFrame:
    """Load broker summary untuk ticker + tanggal.

    Jika tidak ada file nyata dan use_mock=True, kembalikan mock data.
    """
    from stock_scanner.pipeline.broker_summary import get_broker_summary, PlaceholderBrokerFetcher
    return get_broker_summary(
        ticker=ticker,
        date=selected_date,
        broker_dir=_BROKER_DIR,
        fetcher=PlaceholderBrokerFetcher() if use_mock else None,
        top_n=10,
        use_mock_if_empty=use_mock,
    )


# ---------------------------------------------------------------------------
# News data
# ---------------------------------------------------------------------------

def load_news_for_date(scan_date: str) -> pd.DataFrame:
    """Load news sentiment summary untuk semua ticker pada tanggal tertentu."""
    path = _NEWS_DIR / f"{scan_date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

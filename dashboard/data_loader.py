"""Data loading utilities untuk dashboard.

Semua I/O terpusat di sini agar app.py tetap bersih dari path logic.

Mode operasi dikontrol oleh environment variable DATA_SOURCE:
  DATA_SOURCE=local   (default) — baca file lokal di data/
  DATA_SOURCE=remote            — baca published JSON dari GitHub raw URL

Untuk deploy ke Streamlit Community Cloud, set:
  DATA_SOURCE=remote
  REMOTE_DATA_URL=https://raw.githubusercontent.com/<user>/<repo>/main/data/published/latest_scan.json
"""
import os
from pathlib import Path
from datetime import date
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Data source config
# ---------------------------------------------------------------------------

# Dua mode: "local" (default, baca file lokal) atau "remote" (baca GitHub JSON)
_DATA_SOURCE: str = os.environ.get("DATA_SOURCE", "local").strip().lower()

# URL published JSON untuk mode remote
# Ganti dengan URL GitHub raw repo kamu setelah push ke GitHub.
# Format: https://raw.githubusercontent.com/<user>/<repo>/<branch>/data/published/latest_scan.json
_REMOTE_DATA_URL: str = os.environ.get(
    "REMOTE_DATA_URL",
    "https://raw.githubusercontent.com/PLACEHOLDER_USER/PLACEHOLDER_REPO/main/data/published/latest_scan.json",
)

# --- Path roots (relatif dari root repo) ---
_ROOT = Path(__file__).parent.parent
_RANKED_DIR    = _ROOT / "data" / "ranked"
_RAW_DIR       = _ROOT / "data" / "raw"
_SIGNALS_DIR   = _ROOT / "data" / "signals"
_NEWS_DIR          = _ROOT / "data" / "news"
_NEWS_ARTICLES_DIR = _ROOT / "data" / "news" / "articles"
_FOREIGN_DIR       = _ROOT / "data" / "foreign"
_BROKER_DIR        = _ROOT / "data" / "broker"
_FUNDAMENTALS_DIR  = _ROOT / "data" / "fundamentals"

# Kolom tabel utama (urutan display) — termasuk kolom baru
TABLE_COLS = [
    "ticker", "signal", "total_score", "enhanced_total_score",
    "trend_score", "momentum_score", "breakout_score", "volume_score", "penalty_score",
    "news_score", "foreign_score",
    "close", "rsi14", "vol_ratio_20d", "pct_from_52w_high",
    "adx", "supertrend_bullish", "squeeze_on",
    "atr_breakout", "vol_spike",
    "news_sentiment_score", "news_count_3d", "news_data_status",
    "pe_ratio", "pbv", "roe_pct", "der", "div_yield_pct", "fundamental_status",
    "entry_low", "entry_high", "tp_low", "tp_high", "cutloss", "trade_setup_status",
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


def load_broker_history(ticker: str, n_days: int = 20) -> pd.DataFrame:
    """Load last n_days of broker transaction data for a single ticker.

    Scans data/broker/ for files matching ``{ticker_clean}_{date}.parquet``
    and combines them into one DataFrame with a ``date`` column.

    This is used by ``compute_broker_intelligence()`` for multi-day accumulation
    analysis. Returns an empty DataFrame when no files are found (e.g., broker
    fetcher not yet set up).

    Args:
        ticker : IDX ticker (with or without .JK suffix).
        n_days : Maximum number of daily files to load (most-recent first).

    Returns:
        DataFrame with columns:
            date, broker_code, broker_name, buy_lot, sell_lot, net_lot
        Empty DataFrame if no files exist.
    """
    # Try both clean (BBCA) and raw (BBCA.JK) filename patterns
    ticker_clean = ticker.replace(".JK", "")
    patterns = [f"{ticker_clean}_*.parquet", f"{ticker}_*.parquet"]

    files: list[Path] = []
    for pattern in patterns:
        found = sorted(_BROKER_DIR.glob(pattern), reverse=True)
        if found:
            files = found
            break

    if not files:
        return pd.DataFrame()

    # Load up to n_days files
    files = files[:n_days]
    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # Extract date from filename stem: {ticker}_{YYYY-MM-DD}
            stem = f.stem
            parts = stem.split("_")
            date_str = parts[-1] if len(parts) >= 2 else ""
            if date_str and "date" not in df.columns:
                df["date"] = date_str
            frames.append(df)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Normalise net_lot
    if "net_lot" not in combined.columns and "buy_lot" in combined.columns and "sell_lot" in combined.columns:
        combined["net_lot"] = (
            pd.to_numeric(combined["buy_lot"], errors="coerce").fillna(0)
            - pd.to_numeric(combined["sell_lot"], errors="coerce").fillna(0)
        )

    return combined


# ---------------------------------------------------------------------------
# News data
# ---------------------------------------------------------------------------

def load_news_for_date(scan_date: str) -> pd.DataFrame:
    """Load news sentiment summary untuk semua ticker pada tanggal tertentu."""
    path = _NEWS_DIR / f"{scan_date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_news_articles_for_ticker(ticker: str, scan_date: str) -> list[dict]:
    """Load labeled articles for one ticker from the articles parquet.

    Returns:
        List of dicts with: ticker, title, published, publisher,
        sentiment_score, sentiment_label.
        Empty list if file not found or ticker not in file.
    """
    path = _NEWS_ARTICLES_DIR / f"{scan_date}.parquet"
    if not path.exists():
        return []
    try:
        df = pd.read_parquet(path)
        if "ticker" not in df.columns:
            return []
        rows = df[df["ticker"] == ticker]
        if rows.empty:
            return []
        return rows.to_dict(orient="records")
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Fundamental data
# ---------------------------------------------------------------------------

def load_fundamentals_for_date(scan_date: str) -> pd.DataFrame:
    """Load fundamental snapshot untuk semua ticker pada tanggal tertentu."""
    path = _FUNDAMENTALS_DIR / f"{scan_date}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def get_fundamental_row(ticker: str, scan_date: str) -> dict:
    """Return fundamental dict for one ticker from cached parquet, or empty dict."""
    df = load_fundamentals_for_date(scan_date)
    if df.empty or "ticker" not in df.columns:
        return {}
    row = df[df["ticker"] == ticker]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Remote mode — baca published JSON dari GitHub
# ---------------------------------------------------------------------------

def is_remote_mode() -> bool:
    """Return True jika DATA_SOURCE=remote."""
    return _DATA_SOURCE == "remote"


def load_published_payload(url: str | None = None) -> dict:
    """Fetch latest_scan.json dari GitHub raw URL.

    Args:
        url: Override URL (default: _REMOTE_DATA_URL dari env var).

    Returns:
        Parsed dict payload, atau {} jika gagal.
    """
    import json
    import urllib.request

    target_url = url or _REMOTE_DATA_URL
    try:
        with urllib.request.urlopen(target_url, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
        payload = json.loads(raw)
        return payload
    except Exception as exc:
        # Log ke stderr — tidak crash dashboard
        import sys
        print(f"[data_loader] WARNING: Gagal load remote payload dari {target_url}: {exc}", file=sys.stderr)
        return {}


def df_from_published_payload(payload: dict) -> pd.DataFrame:
    """Ubah payload JSON menjadi DataFrame mirip signals_df.

    Menggabungkan semua tier (breakout + pre_markup + watch + scalping-only)
    menjadi satu DataFrame, dengan kolom yang sama seperti file lokal.

    Scalping tickers yang bukan BREAKOUT/PRE_MARKUP/WATCH tidak dimasukkan
    ke DataFrame utama (sudah tercovering dari signal tier mereka).
    """
    rows: list[dict] = []
    seen_tickers: set = set()

    # Urutkan tier agar ranking konsisten: breakout > pre_markup > watch
    for tier_key in ("breakout", "pre_markup", "watch"):
        for row in payload.get(tier_key, []):
            ticker = row.get("ticker", "")
            if ticker and ticker not in seen_tickers:
                rows.append(row)
                seen_tickers.add(ticker)

    # Tambah scalping yang belum masuk tier above (edge case: AVOID ticker dengan scalping_label)
    for row in payload.get("scalping", []):
        ticker = row.get("ticker", "")
        if ticker and ticker not in seen_tickers:
            rows.append(row)
            seen_tickers.add(ticker)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return _normalize_bool_cols(df)


def available_dates_remote(payload: dict) -> list[str]:
    """Kembalikan list tanggal dari payload remote.

    Untuk sementara hanya satu tanggal (latest_scan.json hanya berisi 1 hari).
    """
    scan_date = payload.get("scan_date", "")
    if scan_date:
        return [scan_date]
    return []


# ---------------------------------------------------------------------------
# Unified API — pakai local atau remote tergantung DATA_SOURCE
# ---------------------------------------------------------------------------

def available_dates_unified(payload: dict | None = None) -> list[str]:
    """Kembalikan daftar tanggal yang tersedia.

    - Mode local : baca dari file ranked + signals.
    - Mode remote: kembalikan [scan_date] dari payload.
    """
    if is_remote_mode():
        if payload is None:
            payload = load_published_payload()
        return available_dates_remote(payload)
    return available_dates()


def load_all_tickers_unified(
    scan_date: str,
    payload: dict | None = None,
) -> pd.DataFrame:
    """Load semua ticker — local file atau remote payload.

    - Mode local : load_all_tickers_for_date(scan_date)
    - Mode remote: df_from_published_payload(payload)
    """
    if is_remote_mode():
        if payload is None:
            payload = load_published_payload()
        return df_from_published_payload(payload)
    return load_all_tickers_for_date(scan_date)

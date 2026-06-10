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
import yaml

# ---------------------------------------------------------------------------
# Data source config
# ---------------------------------------------------------------------------

# Dua mode: "local" (default, baca file lokal) atau "remote" (baca GitHub JSON)
_DATA_SOURCE: str = os.environ.get("DATA_SOURCE", "local").strip().lower()

# URL published JSON untuk mode remote.
# Default sudah diisi dengan URL repo yang benar — tidak perlu set env var
# kecuali repo dipindah atau branch berubah.
# Override via Streamlit secrets: REMOTE_DATA_URL = "https://raw.githubusercontent.com/..."
_REMOTE_DATA_URL: str = os.environ.get(
    "REMOTE_DATA_URL",
    "https://raw.githubusercontent.com/suryalionael/idx-stock-scanner-agent/main/data/published/latest_scan.json",
)

# File lokal fallback — tersedia di Streamlit Cloud karena repo di-clone ke /mount/src/
_LOCAL_PUBLISHED_PATH = Path(__file__).parent.parent / "data" / "published" / "latest_scan.json"

# Marker string yang menandakan URL belum dikonfigurasi dengan benar
_URL_PLACEHOLDERS = ("PLACEHOLDER_USER", "PLACEHOLDER_REPO", "<username>", "<user>", "<repo>")

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
_BROKER_CONFIG     = _ROOT / "stock_scanner" / "configs" / "broker_config.yaml"

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
# Broker Config
# ---------------------------------------------------------------------------

def load_broker_config() -> dict:
    """Load broker_config.yaml untuk Broker Analytics modul.

    Returns:
        dict dengan structure:
        {
          'broker_groups': {
            'foreign': {'codes': [...], 'description': ...},
            'institution': {...},
            'retail': {...},
            'big_local': {...},
            'local': {...}
          },
          'metrics': {
            'far': {'thresholds': {...}, 'description': ...},
            'retail_ratio': {...},
            'ridr': {...},
            'smart_money_score': {...}
          },
          'display': {...}
        }

    Returns empty dict jika file tidak ditemukan atau gagal parse.
    """
    if not _BROKER_CONFIG.exists():
        import warnings
        warnings.warn(
            f"broker_config.yaml tidak ditemukan di {_BROKER_CONFIG}. "
            f"Broker Analytics tidak tersedia. Jalankan setup untuk membuat file ini.",
            UserWarning,
            stacklevel=2,
        )
        return {}

    try:
        with open(_BROKER_CONFIG, "r") as f:
            config = yaml.safe_load(f) or {}
        return config
    except Exception as e:
        import warnings
        warnings.warn(
            f"Gagal load broker_config.yaml: {e}",
            UserWarning,
            stacklevel=2,
        )
        return {}


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

# Process-level cache for live-fetched OHLCV (deployed env has no data/raw/).
_RAW_LIVE_CACHE: dict[str, pd.DataFrame] = {}


def load_raw(ticker: str) -> pd.DataFrame:
    """Load OHLCV for one ticker.

    1) Local parquet `data/raw/{ticker}.parquet` (fast path — present locally).
    2) Fallback: live yfinance fetch — `data/raw/` is gitignored and therefore
       NOT deployed to Streamlit Cloud, so the parquet is absent there. Without
       this fallback every chart on the deployed app shows "Tidak ada data raw".
       Cached per-process to avoid refetching on each rerun.
    Returns an empty DataFrame only when the ticker is truly unavailable.
    """
    path = _RAW_DIR / f"{ticker}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        return df.sort_values("date").reset_index(drop=True)
    return _load_raw_live(ticker)


def _load_raw_live(ticker: str) -> pd.DataFrame:
    """Live OHLCV fetch fallback (deployed env). ~430 days so MA200 renders."""
    if ticker in _RAW_LIVE_CACHE:
        return _RAW_LIVE_CACHE[ticker]
    df = pd.DataFrame()
    try:
        from datetime import datetime, timedelta
        from stock_scanner.pipeline.fetch_yfinance import YFinanceFetcher
        start = (datetime.today() - timedelta(days=430)).strftime("%Y-%m-%d")
        end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
        raw = YFinanceFetcher(batch_size=1).fetch_single(ticker, start, end)
        if raw is not None and not raw.empty and "date" in raw.columns:
            raw = raw.copy()
            raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None).dt.normalize()
            df = raw.sort_values("date").reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        import sys
        print(f"[data_loader] load_raw live fallback failed for {ticker}: {exc}", file=sys.stderr)
    _RAW_LIVE_CACHE[ticker] = df
    return df


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
# Broker data — REAL Index Alpha API (cache-first). No mock data.
# ---------------------------------------------------------------------------
#
# Source of truth: Index Alpha API
#   GET https://api.indexalpha.id/stocks/broker-summary
#   params: ticker, from, to, investor (all|f|or|d), market (RG|NG|ALL)
# Service layer: stock_scanner/pipeline/fetch_indexalpha.py (IndexAlphaFetcher)
# Auth: env var INDEX_ALPHA_API_KEY (free plan = 5 requests/day → cache-first).
#
# Cache: data/broker/{TICKER}.JK_{YYYY-MM-DD}.parquet
# Normalized columns:
#   broker_code, broker_name, buy_lot, sell_lot, net_lot,
#   buy_value, sell_value, net_value, buy_avg_price, sell_avg_price,
#   buy_freq, sell_freq


def _broker_cache_file(ticker: str, date: str) -> Path:
    """Canonical broker cache path (consistent with fetch_indexalpha)."""
    clean = ticker.upper().replace(".JK", "").strip()
    return _BROKER_DIR / f"{clean}.JK_{date}.parquet"


def available_broker_dates_for_ticker(ticker: str) -> list[str]:
    """Cached broker dates for ONE ticker, newest first. No API call."""
    if not _BROKER_DIR.exists():
        return []
    clean = ticker.upper().replace(".JK", "").strip()
    dates: set[str] = set()
    from datetime import datetime as _dt
    for pattern in (f"{clean}.JK_*.parquet", f"{clean}_*.parquet"):
        for f in _BROKER_DIR.glob(pattern):
            parts = f.stem.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    _dt.strptime(parts[1], "%Y-%m-%d")
                    dates.add(parts[1])
                except ValueError:
                    continue
    return sorted(dates, reverse=True)


def fetch_broker_summary(
    ticker: str,
    date: str,
    investor: str = "all",
    market: str = "RG",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, str | None]:
    """Real Index Alpha broker summary for ticker+date, CACHE-FIRST.

    Returns (df, error). `error` is None on success, else a human-readable
    message for the UI (never raises). Quota-safe: only calls the API when no
    cache exists or force_refresh=True.

    Args:
        ticker        : IDX ticker (with/without .JK).
        date          : "YYYY-MM-DD".
        investor      : "all" | "f" (foreign) | "d" (domestic) | "or".
        market        : "RG" | "NG" | "ALL".
        force_refresh : True = hit the API even if cache exists (uses 1 quota).
    """
    cache_path = _broker_cache_file(ticker, date)

    # 1) Cache hit (and not forcing a refresh) → real cached Index Alpha data.
    if cache_path.exists() and not force_refresh:
        try:
            return pd.read_parquet(cache_path), None
        except Exception as exc:  # noqa: BLE001
            return pd.DataFrame(), f"Gagal membaca cache broker: {exc}"

    # 2) Need the API. Require a key — otherwise fall back to cache or warn.
    if not os.environ.get("INDEX_ALPHA_API_KEY", "").strip():
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path), None
            except Exception:  # noqa: BLE001
                pass
        return pd.DataFrame(), (
            "INDEX_ALPHA_API_KEY belum diset. Set environment variable "
            "INDEX_ALPHA_API_KEY untuk mengambil data broker dari Index Alpha."
        )

    # 3) Call the Index Alpha service layer (caches the parquet on success).
    try:
        from stock_scanner.pipeline.fetch_indexalpha import fetch_with_cache
        df = fetch_with_cache(
            ticker, date, _BROKER_DIR,
            investor=investor, market=market, force_refresh=force_refresh,
        )
    except Exception as exc:  # noqa: BLE001
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path), None
            except Exception:  # noqa: BLE001
                pass
        return pd.DataFrame(), f"Gagal mengambil data dari Index Alpha: {exc}"

    if df is None or df.empty:
        return pd.DataFrame(), (
            f"Tidak ada data broker dari Index Alpha untuk {ticker} @ {date}. "
            "Kemungkinan bukan hari bursa, di luar cakupan data, atau kuota harian habis."
        )
    return df, None


def load_broker_for_ticker(ticker: str, selected_date: str, use_mock: bool = False) -> pd.DataFrame:
    """[DEPRECATED] Cache-first broker load. Mock removed — use fetch_broker_summary().

    Kept for backward compatibility. `use_mock` is ignored (always real data).
    """
    df, _ = fetch_broker_summary(ticker, selected_date)
    return df


# Index Alpha Regular-market broker data availability starts June 2025.
_INDEXALPHA_RG_START = "2025-06-01"


def _broker_range_cache_file(
    ticker: str, from_date: str, to_date: str, investor: str, market: str,
) -> Path:
    """Cache path for a HISTORICAL range aggregate.

    Cache key = ticker + from + to + investor + market (kept in a `range/`
    subdir so the single-day date scanner never picks these up). Net/Gross is a
    display-only toggle on the same data, so it is NOT part of the key.
    """
    clean = ticker.upper().replace(".JK", "").strip()
    return _BROKER_DIR / "range" / f"{clean}_{from_date}_{to_date}_{investor}_{market}.parquet"


def fetch_broker_range(
    ticker: str,
    from_date: str,
    to_date: str,
    investor: str = "all",
    market: str = "RG",
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, str | None]:
    """REAL Index Alpha broker summary AGGREGATED over [from_date, to_date].

    Uses GET /stocks/broker-summary with from/to (one request per range — the
    API returns per-broker totals across the whole period, no per-day breakdown).
    Cache-first; returns (df, error). Never raises.
    """
    cache_path = _broker_range_cache_file(ticker, from_date, to_date, investor, market)

    if cache_path.exists() and not force_refresh:
        try:
            return pd.read_parquet(cache_path), None
        except Exception as exc:  # noqa: BLE001
            return pd.DataFrame(), f"Gagal membaca cache range: {exc}"

    if not os.environ.get("INDEX_ALPHA_API_KEY", "").strip():
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path), None
            except Exception:  # noqa: BLE001
                pass
        return pd.DataFrame(), (
            "INDEX_ALPHA_API_KEY belum diset. Set environment variable untuk "
            "mengambil broker summary historical dari Index Alpha."
        )

    try:
        from stock_scanner.pipeline.fetch_indexalpha import IndexAlphaFetcher
        df = IndexAlphaFetcher().fetch_range(
            ticker, from_date, to_date, investor=investor, market=market,
        )
    except Exception as exc:  # noqa: BLE001
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path), None
            except Exception:  # noqa: BLE001
                pass
        return pd.DataFrame(), f"Gagal mengambil historical dari Index Alpha: {exc}"

    if df is None or df.empty:
        return pd.DataFrame(), (
            f"Tidak ada data broker untuk {ticker} pada {from_date}…{to_date}. "
            "Kemungkinan di luar cakupan data (RG mulai Jun 2025) atau kuota habis."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(cache_path, index=False)
    except Exception:  # noqa: BLE001
        pass  # caching is best-effort; still return the data
    return df, None


def fetch_broker_latest(
    ticker: str,
    investor: str = "all",
    market: str = "RG",
) -> tuple[pd.DataFrame, str | None, dict]:
    """Broker summary for the LAST COMPLETED trading session — fresh-first.

    Order: (1) if that exact session is already cached → use it (0 API calls);
    (2) else, with an API key, fetch it (1 quota); (3) else / on failure, fall
    back to the most recent cached session and flag it clearly.

    Returns (df, note, info) where info = {"date": str, "source": fresh|cache|fallback}.
    `note` is a non-blocking warning (e.g. fallback reason); may be set even when
    df is non-empty.
    """
    from datetime import datetime, timezone, timedelta
    from stock_scanner.utils.trading_calendar import expected_market_date

    now_wib = datetime.now(timezone(timedelta(hours=7)))
    target = expected_market_date(now_wib).strftime("%Y-%m-%d")
    info: dict = {"date": target, "source": None}

    # 1) Right session already cached → fresh enough, no API call.
    cache_path = _broker_cache_file(ticker, target)
    if cache_path.exists():
        try:
            info["source"] = "cache"
            return pd.read_parquet(cache_path), None, info
        except Exception:  # noqa: BLE001
            pass

    # 2) Not cached. With a key, fetch the target session (1 quota).
    if os.environ.get("INDEX_ALPHA_API_KEY", "").strip():
        df, err = fetch_broker_summary(
            ticker, target, investor=investor, market=market, force_refresh=True,
        )
        if df is not None and not df.empty:
            info["source"] = "fresh"
            return df, None, info
        fallback_reason = err or "fetch terbaru gagal"
    else:
        fallback_reason = "INDEX_ALPHA_API_KEY belum diset"

    # 3) Fallback: most recent cached session, clearly flagged.
    dates = available_broker_dates_for_ticker(ticker)
    if dates:
        fb = dates[0]
        try:
            df = pd.read_parquet(_broker_cache_file(ticker, fb))
            info["date"] = fb
            info["source"] = "fallback"
            note = (f"Menampilkan cache {fb} — sesi terbaru {target} belum tersedia "
                    f"({fallback_reason}).")
            return df, note, info
        except Exception:  # noqa: BLE001
            pass

    return pd.DataFrame(), f"Tidak ada data broker. {fallback_reason}.", info


def broker_range_bounds(quick: str) -> tuple[str, str]:
    """Compute (from_date, to_date) for a quick historical range.

    to_date = last completed trading session (WIB). from_date = to_date minus the
    range, clamped to Index Alpha RG availability (Jun 2025). `quick` ∈
    {"1W","1M","3M","6M","1Y"}.
    """
    from datetime import datetime, timezone, timedelta, date as _date
    from stock_scanner.utils.trading_calendar import expected_market_date

    now_wib = datetime.now(timezone(timedelta(hours=7)))
    to_d = expected_market_date(now_wib)
    days = {"1W": 7, "1M": 30, "3M": 91, "6M": 182, "1Y": 365}.get(quick, 30)
    from_d = to_d - timedelta(days=days)
    rg_start = _date.fromisoformat(_INDEXALPHA_RG_START)
    if from_d < rg_start:
        from_d = rg_start
    return from_d.strftime("%Y-%m-%d"), to_d.strftime("%Y-%m-%d")


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


def available_broker_dates() -> list[str]:
    """Scan broker directory dan return daftar tanggal unik tersedia.

    Tujuan: Discover tanggal apa saja yang ada broker data di cache.

    Returns:
        List of date strings (YYYY-MM-DD), sorted descending (newest first).
        Empty list jika tidak ada broker files.

    Logic:
        1. Scan data/broker/*.parquet
        2. Extract date dari filename {TICKER}.JK_{YYYY-MM-DD}.parquet
        3. Deduplicate dan sort descending.
    """
    if not _BROKER_DIR.exists():
        return []

    files = list(_BROKER_DIR.glob("*.parquet"))
    dates_set = set()

    for f in files:
        stem = f.stem
        # Format: {TICKER}.JK_{YYYY-MM-DD}
        parts = stem.rsplit("_", 1)  # split from right to isolate date
        if len(parts) == 2:
            date_str = parts[1]
            # Validate date format
            try:
                from datetime import datetime
                datetime.strptime(date_str, "%Y-%m-%d")
                dates_set.add(date_str)
            except ValueError:
                continue

    return sorted(dates_set, reverse=True)


def load_broker_history_for_ticker(
    ticker: str,
    date_start: str | None = None,
    date_end: str | None = None,
    n_days: int = 20,
) -> pd.DataFrame:
    """Load broker history untuk satu ticker dengan opsi fleksibel.

    Tujuan: Wrapper sederhana untuk ambil broker history dengan dua modus:
    1. Recent mode: last n_days (default)
    2. Range mode: specific [date_start, date_end]

    Args:
        ticker      : IDX ticker (dengan atau tanpa .JK)
        date_start  : YYYY-MM-DD (optional, untuk range mode)
        date_end    : YYYY-MM-DD (optional, untuk range mode)
        n_days      : default 20, dipakai jika date_start/end tidak ada

    Returns:
        DataFrame dengan kolom: date, broker_code, broker_name, buy_lot, sell_lot, net_lot
        Sorted ascending by date (oldest first).
        Empty DataFrame jika tidak ada data.

    Logic:
        - Jika date_start dan date_end ada: gunakan range mode
        - Otherwise: gunakan recent n_days mode
    """
    # Range mode
    if date_start and date_end:
        from stock_scanner.pipeline.broker_analytics import load_broker_history_multi_day
        return load_broker_history_multi_day(ticker, date_start, date_end)

    # Recent mode (existing behavior)
    return load_broker_history(ticker, n_days)


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
    """Fetch latest_scan.json dari GitHub raw URL, dengan fallback ke file lokal.

    Urutan prioritas:
      1. Remote HTTP fetch dari `url` (atau _REMOTE_DATA_URL)
      2. File lokal data/published/latest_scan.json (tersedia di Streamlit Cloud
         karena repo di-clone ke /mount/src/)

    Args:
        url: Override URL. Default: _REMOTE_DATA_URL dari env var.

    Returns:
        Parsed dict payload, atau {} jika semua sumber gagal.
    """
    import json
    import sys
    import urllib.request

    target_url = url or _REMOTE_DATA_URL

    # Deteksi URL yang masih placeholder — langsung skip ke local fallback
    if any(p in target_url for p in _URL_PLACEHOLDERS):
        print(
            f"[data_loader] INFO: REMOTE_DATA_URL masih placeholder ({target_url!r}) "
            f"— langsung pakai file lokal.",
            file=sys.stderr,
        )
        return _load_local_published()

    # ── Coba remote ──────────────────────────────────────────────────────────
    try:
        with urllib.request.urlopen(target_url, timeout=10) as resp:
            raw = resp.read().decode("utf-8")
        payload = json.loads(raw)
        print(
            f"[data_loader] INFO: Remote payload loaded dari {target_url} "
            f"(scan_date={payload.get('scan_date', '?')})",
            file=sys.stderr,
        )
        return payload
    except Exception as exc:
        print(
            f"[data_loader] WARNING: Gagal load remote payload dari {target_url}: {exc} "
            f"— mencoba file lokal sebagai fallback.",
            file=sys.stderr,
        )

    # ── Fallback ke file lokal ────────────────────────────────────────────────
    return _load_local_published()


def _load_local_published() -> dict:
    """Baca data/published/latest_scan.json dari disk.

    File ini ada di repo (di-commit ke git) sehingga tersedia di Streamlit
    Cloud pada path /mount/src/<repo>/data/published/latest_scan.json.
    """
    import json
    import sys

    if not _LOCAL_PUBLISHED_PATH.exists():
        print(
            f"[data_loader] WARNING: File lokal tidak ditemukan: {_LOCAL_PUBLISHED_PATH}",
            file=sys.stderr,
        )
        return {}

    try:
        with open(_LOCAL_PUBLISHED_PATH, encoding="utf-8") as f:
            payload = json.load(f)
        print(
            f"[data_loader] INFO: Local payload loaded dari {_LOCAL_PUBLISHED_PATH.name} "
            f"(scan_date={payload.get('scan_date', '?')})",
            file=sys.stderr,
        )
        return payload
    except Exception as exc:
        print(f"[data_loader] ERROR: Gagal baca file lokal: {exc}", file=sys.stderr)
        return {}


def df_from_published_payload(payload: dict) -> pd.DataFrame:
    """Ubah payload JSON menjadi DataFrame mirip signals_df.

    Strategi (v2 payload dengan all_tickers):
      - Jika payload berisi kunci "all_tickers", gunakan langsung sebagai
        sumber data utama — berisi SEMUA ticker (957 baris) persis seperti
        file lokal, termasuk AVOID/NONE/scalping candidates.
      - Fallback ke rekonstruksi dari tier lists (breakout + pre_markup +
        watch + scalping) untuk payload lama yang belum punya all_tickers.

    Ini memastikan parity local == deployed di semua tab dashboard.
    """
    # ── v2 payload: all_tickers tersedia (parity mode) ───────────────────────
    all_tickers = payload.get("all_tickers")
    if all_tickers:
        df = pd.DataFrame(all_tickers)
        return _normalize_bool_cols(df)

    # ── v1 fallback: rekonstruksi dari tier lists (25 baris) ─────────────────
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
    # LOCAL: also surface the committed published scan_date when it is FRESHER than
    # the newest local signals file, so `git pull` brings the automation's latest
    # session into a normal `streamlit run` without a local rescan. Additive:
    # if it is not newer, the local signal dates are returned unchanged.
    dates = available_dates()
    pub = str(_load_local_published().get("scan_date") or "")
    if pub and (not dates or pub > dates[0]):
        dates = [pub] + [d for d in dates if d != pub]
    return dates


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
    # LOCAL: prefer local signal files; if this date exists only in the committed
    # published JSON (fresher, from automation + git pull), load from that.
    df = load_all_tickers_for_date(scan_date)
    if df is None or df.empty:
        pub = _load_local_published()
        if pub and str(pub.get("scan_date")) == str(scan_date):
            return df_from_published_payload(pub)
    return df

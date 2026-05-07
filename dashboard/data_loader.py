"""Data loading utilities untuk dashboard.

Semua I/O terpusat di sini agar app.py tetap bersih dari path logic.
"""
from pathlib import Path
from datetime import date

import pandas as pd

# --- Path roots (relatif dari root repo) ---
_ROOT = Path(__file__).parent.parent
_RANKED_DIR = _ROOT / "data" / "ranked"
_RAW_DIR = _ROOT / "data" / "raw"
_SIGNALS_DIR = _ROOT / "data" / "signals"

# Kolom yang ditampilkan di tabel utama (urutan display)
TABLE_COLS = [
    "ticker", "signal", "total_score",
    "trend_score", "momentum_score", "breakout_score", "volume_score", "penalty_score",
    "close", "rsi14", "vol_ratio_20d", "pct_from_52w_high",
    "atr_breakout", "vol_spike",
]

HISTORY_COLS = [
    "date", "ticker", "signal", "total_score",
    "close", "rsi14", "vol_ratio_20d", "pct_from_52w_high",
]


def list_ranked_dates() -> list[str]:
    """Kembalikan daftar tanggal yang punya file ranked, urutan descending."""
    files = sorted(_RANKED_DIR.glob("ranked_*.csv"), reverse=True)
    dates = []
    for f in files:
        stem = f.stem  # "ranked_2026-05-07"
        parts = stem.split("_", 1)
        if len(parts) == 2:
            dates.append(parts[1])
    return dates


def latest_ranked_date() -> str | None:
    """Tanggal terbaru yang punya file ranked, atau None jika belum ada."""
    dates = list_ranked_dates()
    return dates[0] if dates else None


def load_ranked(scan_date: str) -> pd.DataFrame:
    """Load ranked_{scan_date}.csv. Return DataFrame kosong jika tidak ada."""
    path = _RANKED_DIR / f"ranked_{scan_date}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Normalisasi tipe
    for col in ["atr_breakout", "vol_spike", "ma_full_alignment", "ma_partial_alignment",
                 "golden_cross", "obv_trend"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1"])
    for col in ["trend_score", "momentum_score", "breakout_score",
                "volume_score", "penalty_score", "total_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    return df


def load_raw(ticker: str) -> pd.DataFrame:
    """Load OHLCV parquet untuk satu ticker. Return DataFrame kosong jika tidak ada."""
    path = _RAW_DIR / f"{ticker}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def load_all_ranked(
    min_signal: list[str] | None = None,
    ticker_filter: str | None = None,
    limit_rows: int = 500,
) -> pd.DataFrame:
    """Concat semua ranked CSV jadi satu DataFrame untuk halaman History.

    Args:
        min_signal: filter sinyal, mis. ["WATCH", "PRE_MARKUP", "BREAKOUT"]
        ticker_filter: filter satu ticker (opsional)
        limit_rows: batasi total baris untuk performa

    Returns:
        DataFrame gabungan, diurutkan date desc → total_score desc
    """
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
        combined = combined[combined["ticker"] == ticker_filter]

    combined = combined.sort_values(
        ["date", "total_score"], ascending=[False, False]
    ).reset_index(drop=True)

    # Kembalikan kolom yang tersedia dari HISTORY_COLS
    available = [c for c in HISTORY_COLS if c in combined.columns]
    return combined[available].head(limit_rows)


def get_table_df(df: pd.DataFrame) -> pd.DataFrame:
    """Pilih dan urutkan kolom untuk ditampilkan di tabel sinyal utama."""
    available = [c for c in TABLE_COLS if c in df.columns]
    result = df[available].copy()
    if "total_score" in result.columns:
        result = result.sort_values("total_score", ascending=False)
    return result.reset_index(drop=True)

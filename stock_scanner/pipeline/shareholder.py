"""Shareholder Metric Pipeline.

Tracking pertumbuhan jumlah pemegang saham per ticker IDX.
Data ini biasanya tidak tersedia harian — umumnya mingguan atau bulanan
dari C-BEST (Central Depository & Book Entry Settlement System) KSEI.

Kolom output:
    num_shareholders       — jumlah pemegang saham terakhir
    holder_growth_1m       — pertumbuhan 1 bulan (%)
    holder_growth_3m       — pertumbuhan 3 bulan (%)
    shareholder_score      — skor 0–10 (5.0 = neutral)

Source potensial:
    - KSEI C-BEST: https://www.ksei.co.id/
    - Stockbit / Ajaib API (pemegang saham ritail)
    - IDX disclosure (laporan pemegang saham 5% keatas)
    - yfinance.Ticker.major_holders / institutional_holders (sangat terbatas untuk IDX)

Catatan:
    Karena data tidak harian, disimpan sebagai time series per ticker:
    data/shareholders/{ticker}_shareholders.parquet
    Schema: date (Timestamp), ticker (str), num_shareholders (int),
            holder_growth_1m (float %), holder_growth_3m (float %)
"""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaseShareholderFetcher(ABC):
    """Interface untuk data provider shareholder."""

    @abstractmethod
    def fetch(self, ticker: str) -> pd.DataFrame:
        """Ambil history jumlah pemegang saham untuk satu ticker.

        Returns:
            DataFrame dengan kolom: date (Timestamp), ticker (str),
            num_shareholders (int). Diurutkan by date ascending.
        """
        ...


# ---------------------------------------------------------------------------
# Placeholder
# ---------------------------------------------------------------------------

class PlaceholderShareholderFetcher(BaseShareholderFetcher):
    """Mengembalikan DataFrame kosong.

    TODO: Implementasi nyata via KSEI API atau scraping publik KSEI
          Report: https://www.ksei.co.id/services/securities-administration
    """

    def fetch(self, ticker: str) -> pd.DataFrame:
        logger.debug(f"{ticker}: PlaceholderShareholderFetcher — no real source yet")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _path(ticker: str, shareholder_dir: Path) -> Path:
    return shareholder_dir / f"{ticker}_shareholders.parquet"


def save_shareholders(ticker: str, df: pd.DataFrame, shareholder_dir: Path) -> None:
    shareholder_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_path(ticker, shareholder_dir), index=False)
    logger.debug(f"{ticker}: shareholders saved")


def load_shareholders(ticker: str, shareholder_dir: Path) -> pd.DataFrame:
    p = _path(ticker, shareholder_dir)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_shareholder_features(df_holders: pd.DataFrame) -> dict:
    """Hitung fitur pertumbuhan dari history shareholder.

    Returns:
        dict: num_shareholders, holder_growth_1m, holder_growth_3m, shareholder_score
    """
    default = {
        "num_shareholders": np.nan,
        "holder_growth_1m": np.nan,
        "holder_growth_3m": np.nan,
        "shareholder_score": 5.0,
    }
    if df_holders.empty or "num_shareholders" not in df_holders.columns:
        return default

    df = df_holders.sort_values("date")
    latest = float(df["num_shareholders"].iloc[-1])

    def _growth_pct(n_rows_back: int) -> float:
        if len(df) <= n_rows_back:
            return np.nan
        old = float(df["num_shareholders"].iloc[-(n_rows_back + 1)])
        if old == 0:
            return np.nan
        return round((latest - old) / old * 100, 2)

    # Asumsikan data mingguan: 4 row ≈ 1 bulan, 12 row ≈ 3 bulan
    g1m = _growth_pct(4)
    g3m = _growth_pct(12)

    # Skor: pertumbuhan positif 3m → skor > 5
    score = 5.0
    if not np.isnan(g3m):
        score = float(np.clip(5.0 + g3m * 0.2, 0, 10))  # 10% growth → +2 skor

    return {
        "num_shareholders": int(latest),
        "holder_growth_1m": g1m,
        "holder_growth_3m": g3m,
        "shareholder_score": round(score, 2),
    }


# ---------------------------------------------------------------------------
# Enrichment entry point
# ---------------------------------------------------------------------------

def enrich_with_shareholders(
    df_features: pd.DataFrame,
    fetcher: BaseShareholderFetcher | None = None,
    shareholder_dir: Path | None = None,
) -> pd.DataFrame:
    """Tambahkan kolom shareholder ke feature DataFrame.

    Jika data tidak tersedia, kolom akan NaN (kecuali shareholder_score = 5.0).
    """
    SH_COLS = ["num_shareholders", "holder_growth_1m", "holder_growth_3m", "shareholder_score"]

    if fetcher is None:
        fetcher = PlaceholderShareholderFetcher()

    df = df_features.copy()
    rows = []

    for ticker in df["ticker"].unique():
        features = {"ticker": ticker}
        try:
            if shareholder_dir:
                df_sh = load_shareholders(ticker, shareholder_dir)
            else:
                df_sh = pd.DataFrame()

            if df_sh.empty:
                df_sh = fetcher.fetch(ticker)
                if not df_sh.empty and shareholder_dir:
                    save_shareholders(ticker, df_sh, shareholder_dir)

            features.update(compute_shareholder_features(df_sh))
        except Exception as e:
            logger.warning(f"{ticker}: shareholder error — {e}")
            features.update({c: np.nan for c in SH_COLS})
            features["shareholder_score"] = 5.0

        rows.append(features)

    sh_df = pd.DataFrame(rows)
    df = df.merge(sh_df[["ticker"] + SH_COLS], on="ticker", how="left")
    df["shareholder_score"] = df["shareholder_score"].fillna(5.0)
    return df

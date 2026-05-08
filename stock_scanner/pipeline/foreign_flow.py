"""Foreign Flow Pipeline — Net buy/sell asing per ticker IDX.

Abstraksi data foreign flow. Source data asli belum tersedia secara bebas;
gunakan IDX API, RTI Business, atau CNBC Indonesia untuk implementasi nyata.

Kolom output:
    foreign_buy        — volume beli asing (lot atau IDR, tergantung source)
    foreign_sell       — volume jual asing
    foreign_net        — foreign_buy - foreign_sell (hari ini)
    foreign_net_5d     — total foreign_net 5 hari terakhir
    foreign_net_20d    — total foreign_net 20 hari terakhir
    foreign_flow_score — skor 0–10 (5.0 = neutral)

Cara pakai:
    1. Extend BaseForeignFetcher dengan implementasi nyata
    2. Ganti PlaceholderForeignFetcher di run_daily_scan.py
    3. Data disimpan sebagai Parquet per ticker: data/foreign/{ticker}_foreign.parquet

Skema CSV/Parquet:
    date (Timestamp), ticker (str), foreign_buy (float), foreign_sell (float), foreign_net (float)
"""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaseForeignFetcher(ABC):
    """Interface untuk data provider foreign flow IDX."""

    @abstractmethod
    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Ambil data foreign flow untuk satu ticker.

        Returns:
            DataFrame dengan kolom minimal:
            date (Timestamp), ticker (str), foreign_buy (float),
            foreign_sell (float), foreign_net (float)
        """
        ...


# ---------------------------------------------------------------------------
# Placeholder implementation
# ---------------------------------------------------------------------------

class PlaceholderForeignFetcher(BaseForeignFetcher):
    """Placeholder: tidak mengambil data nyata.

    Mengembalikan DataFrame kosong. Ganti dengan implementasi nyata
    saat data source tersedia (mis. IDX API, RTI, Stockbit).

    TODO:
        - IDX: https://www.idx.co.id/api/... (perlu reverse engineer)
        - RTI Business API: berbayar tapi reliable
        - IHSG.com: scraping publik (risiko TOS)
        - Stockbit API: butuh akun premium
    """

    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        logger.debug(f"{ticker}: PlaceholderForeignFetcher — returning empty (no real data source)")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def save_foreign(ticker: str, df: pd.DataFrame, foreign_dir: Path) -> None:
    foreign_dir.mkdir(parents=True, exist_ok=True)
    path = foreign_dir / f"{ticker}_foreign.parquet"
    df.to_parquet(path, index=False)
    logger.debug(f"{ticker}: foreign flow saved → {path}")


def load_foreign(ticker: str, foreign_dir: Path) -> pd.DataFrame:
    path = foreign_dir / f"{ticker}_foreign.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_foreign_features(df_foreign: pd.DataFrame) -> dict:
    """Hitung kolom turunan dari data foreign flow historis satu ticker.

    Args:
        df_foreign: DataFrame dengan kolom date, foreign_buy, foreign_sell, foreign_net
                    (diurutkan ascending by date)

    Returns:
        dict dengan: foreign_net, foreign_net_5d, foreign_net_20d, foreign_flow_score
    """
    default = {
        "foreign_net": np.nan,
        "foreign_net_5d": np.nan,
        "foreign_net_20d": np.nan,
        "foreign_flow_score": 5.0,  # neutral
    }
    if df_foreign.empty or "foreign_net" not in df_foreign.columns:
        return default

    df = df_foreign.sort_values("date").reset_index(drop=True)

    latest_net = float(df["foreign_net"].iloc[-1]) if len(df) >= 1 else np.nan
    net_5d = float(df["foreign_net"].tail(5).sum()) if len(df) >= 5 else np.nan
    net_20d = float(df["foreign_net"].tail(20).sum()) if len(df) >= 20 else np.nan

    # Normalisasi ke score 0–10 berdasarkan z-score dari net_20d
    # Jika net_20d positif (net buy) → skor > 5, negatif → < 5
    score = 5.0
    if not np.isnan(net_20d):
        # Standar deviasi 20d sebagai referensi
        std_20d = float(df["foreign_net"].tail(20).std())
        if std_20d > 0:
            z = net_20d / std_20d
            score = float(np.clip(5.0 + z * 1.5, 0, 10))

    return {
        "foreign_net": latest_net,
        "foreign_net_5d": net_5d,
        "foreign_net_20d": net_20d,
        "foreign_flow_score": round(score, 2),
    }


# ---------------------------------------------------------------------------
# Enrichment entry point
# ---------------------------------------------------------------------------

def enrich_with_foreign(
    df_features: pd.DataFrame,
    fetcher: BaseForeignFetcher | None = None,
    foreign_dir: Path | None = None,
) -> pd.DataFrame:
    """Tambahkan kolom foreign flow ke feature DataFrame.

    Jika fetcher mengembalikan data kosong (PlaceholderForeignFetcher),
    kolom foreign_* akan NaN kecuali foreign_flow_score = 5.0 (neutral).

    Args:
        df_features: DataFrame satu baris per ticker, harus punya kolom 'ticker'
        fetcher: implementasi BaseForeignFetcher (default: PlaceholderForeignFetcher)
        foreign_dir: direktori Parquet foreign flow (opsional)
    """
    FOREIGN_COLS = ["foreign_net", "foreign_net_5d", "foreign_net_20d", "foreign_flow_score"]

    if fetcher is None:
        fetcher = PlaceholderForeignFetcher()

    df = df_features.copy()
    rows = []

    for ticker in df["ticker"].unique():
        features = {"ticker": ticker}
        try:
            # Coba load dari cache lokal dulu
            if foreign_dir:
                df_foreign = load_foreign(ticker, foreign_dir)
            else:
                df_foreign = pd.DataFrame()

            # Jika cache kosong, fetch dari source
            if df_foreign.empty:
                df_foreign = fetcher.fetch(ticker, start="2020-01-01", end="2099-12-31")
                if not df_foreign.empty and foreign_dir:
                    save_foreign(ticker, df_foreign, foreign_dir)

            features.update(compute_foreign_features(df_foreign))
        except Exception as e:
            logger.warning(f"{ticker}: foreign flow error — {e}")
            features.update({c: np.nan for c in FOREIGN_COLS})
            features["foreign_flow_score"] = 5.0

        rows.append(features)

    foreign_df = pd.DataFrame(rows)
    df = df.merge(foreign_df[["ticker"] + FOREIGN_COLS], on="ticker", how="left")
    df["foreign_flow_score"] = df["foreign_flow_score"].fillna(5.0)

    has_real = df["foreign_net"].notna().any()
    logger.info(f"Foreign flow enrichment: has_real_data={has_real}")
    return df

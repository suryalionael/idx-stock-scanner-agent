"""Broker Summary — ringkasan aktivitas broker/sekuritas per ticker per tanggal.

Data model: siapa yang paling banyak beli/jual pada tanggal tertentu.
Source asli: IDX JATS (Jakarta Automated Trading System) broker transaction data.

Format data yang diharapkan:
    data/broker/{ticker}_{YYYY-MM-DD}.parquet
    Kolom: broker_code, broker_name, buy_lot, sell_lot, net_lot

Cara pakai:
    1. Extend BaseBrokerFetcher dengan implementasi nyata
    2. Dashboard memanggil load_broker_summary(ticker, date) untuk chart/tooltip

Source potensial:
    - RTI Business: menyediakan data broker per saham (berbayar)
    - Stockbit Premium API
    - IDX JATS data (via member sekuritas)
    - Scraping ChartNexus / ajaib.co.id (risiko TOS)
"""
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Data schema (dokumentasi untuk implementasi nyata)
# ---------------------------------------------------------------------------

BROKER_SCHEMA = {
    "broker_code":  "str   — kode broker, mis. 'YP', 'CC', 'ZP'",
    "broker_name":  "str   — nama sekuritas, mis. 'Indo Premier Sekuritas'",
    "buy_lot":      "float — total lot beli pada tanggal tersebut",
    "sell_lot":     "float — total lot jual pada tanggal tersebut",
    "net_lot":      "float — buy_lot - sell_lot (positif = net buyer)",
}

# Contoh broker IDX populer (untuk mock data)
_SAMPLE_BROKERS = [
    ("YP", "Indo Premier Sekuritas"),
    ("CC", "Mandiri Sekuritas"),
    ("ZP", "Kim Eng Sekuritas"),
    ("AK", "UBS Sekuritas Indonesia"),
    ("BQ", "BNI Sekuritas"),
    ("PD", "Indo Premier Online"),
    ("ML", "Merrill Lynch"),
    ("RX", "Macquarie Sekuritas Indonesia"),
    ("CP", "Valbury Asia Securities"),
    ("FZ", "Waterfront Sekuritas"),
]


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BaseBrokerFetcher(ABC):
    """Interface untuk data provider broker activity."""

    @abstractmethod
    def fetch(self, ticker: str, date: str) -> pd.DataFrame:
        """Ambil summary broker untuk satu ticker pada satu tanggal.

        Returns:
            DataFrame dengan kolom: broker_code, broker_name,
            buy_lot, sell_lot, net_lot
            Diurutkan descending by abs(net_lot).
        """
        ...


# ---------------------------------------------------------------------------
# Placeholder implementation (dengan mock data untuk demo)
# ---------------------------------------------------------------------------

class PlaceholderBrokerFetcher(BaseBrokerFetcher):
    """Mengembalikan data mock untuk demo UI.

    TODO: Ganti dengan implementasi nyata saat source tersedia.
          Mock data TIDAK mencerminkan transaksi nyata.
    """

    def fetch(self, ticker: str, date: str) -> pd.DataFrame:
        import numpy as np
        rng = sum(ord(c) for c in ticker + date)  # deterministik per ticker+date
        random = __import__("random")
        random.seed(rng)
        np.random.seed(rng % 2**32)

        rows = []
        for code, name in _SAMPLE_BROKERS:
            buy = abs(np.random.normal(5000, 3000))
            sell = abs(np.random.normal(5000, 3000))
            rows.append({
                "broker_code": code,
                "broker_name": name,
                "buy_lot": round(buy),
                "sell_lot": round(sell),
                "net_lot": round(buy - sell),
            })

        df = pd.DataFrame(rows)
        df = df.sort_values("net_lot", key=abs, ascending=False).reset_index(drop=True)
        logger.debug(f"{ticker} @ {date}: PlaceholderBrokerFetcher — mock data returned")
        return df


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _broker_path(ticker: str, date: str, broker_dir: Path) -> Path:
    return broker_dir / f"{ticker}_{date}.parquet"


def save_broker_summary(ticker: str, date: str, df: pd.DataFrame, broker_dir: Path) -> None:
    broker_dir.mkdir(parents=True, exist_ok=True)
    path = _broker_path(ticker, date, broker_dir)
    df.to_parquet(path, index=False)
    logger.debug(f"Broker summary saved → {path}")


def load_broker_summary(ticker: str, date: str, broker_dir: Path) -> pd.DataFrame:
    """Load broker summary untuk ticker + tanggal. Return kosong jika tidak ada."""
    path = _broker_path(ticker, date, broker_dir)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def get_broker_summary(
    ticker: str,
    date: str,
    broker_dir: Path,
    fetcher: BaseBrokerFetcher | None = None,
    top_n: int = 10,
    use_mock_if_empty: bool = True,
) -> pd.DataFrame:
    """Load dari cache, atau fetch + simpan jika belum ada.

    Args:
        use_mock_if_empty: jika True dan fetcher adalah Placeholder,
                           tetap kembalikan mock data untuk demo UI.
    Returns:
        DataFrame top N broker, atau kosong jika tidak tersedia.
    """
    df = load_broker_summary(ticker, date, broker_dir)
    if not df.empty:
        return df.head(top_n)

    if fetcher is None:
        fetcher = PlaceholderBrokerFetcher()

    try:
        df = fetcher.fetch(ticker, date)
        if not df.empty:
            save_broker_summary(ticker, date, df, broker_dir)
    except Exception as e:
        logger.warning(f"{ticker} @ {date}: broker fetch gagal — {e}")
        return pd.DataFrame()

    return df.head(top_n) if not df.empty else pd.DataFrame()

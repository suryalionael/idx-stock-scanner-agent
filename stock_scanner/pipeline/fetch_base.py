from abc import ABC, abstractmethod

import pandas as pd

OHLCV_COLS = ["date", "ticker", "open", "high", "low", "close", "volume", "adj_close"]


class BaseFetcher(ABC):
    """Interface contract untuk semua data provider.

    Implementasi baru (mis. Bloomberg, Refinitiv) cukup extend class ini
    tanpa menyentuh layer validate/features/signals.
    """

    @abstractmethod
    def fetch(self, tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
        """Download OHLCV untuk list ticker.

        Returns:
            dict { ticker: DataFrame dengan kolom OHLCV_COLS }
        """
        ...

    @abstractmethod
    def fetch_single(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Download OHLCV untuk satu ticker.

        Returns:
            DataFrame dengan kolom OHLCV_COLS, atau DataFrame kosong jika gagal.
        """
        ...

import pandas as pd
from loguru import logger

REQUIRED_COLS = {"date", "ticker", "open", "high", "low", "close", "volume"}

_FFILL_GAP_LIMIT = 2       # hari kerja — ffill tanpa log (weekend panjang, cuti bersama)
_HOLIDAY_GAP_LIMIT = 30    # hari kerja — lebih dari ini → WARNING suspend/delisted


def validate(df: pd.DataFrame, ticker: str, max_gap_fill_days: int = _FFILL_GAP_LIMIT) -> tuple[pd.DataFrame, dict]:
    """Validate dan clean raw OHLCV DataFrame untuk satu ticker.

    Returns:
        (cleaned_df, report) — report berisi info anomali yang ditemukan.
        cleaned_df kosong jika kolom wajib tidak ada.
    """
    report: dict = {"ticker": ticker, "issues": [], "rows_dropped": 0, "gaps_filled": 0}

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        msg = f"Missing columns: {missing}"
        report["issues"].append(msg)
        logger.warning(f"{ticker}: {msg}")
        return pd.DataFrame(), report

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    # Drop NaN OHLC
    original_len = len(df)
    df = df.dropna(subset=["close", "open", "high", "low"])
    dropped = original_len - len(df)
    if dropped:
        report["rows_dropped"] += dropped
        report["issues"].append(f"Dropped {dropped} rows with NaN OHLC")

    # Drop harga negatif
    neg = (df[["open", "high", "low", "close"]] < 0).any(axis=1)
    if neg.any():
        count = int(neg.sum())
        df = df[~neg]
        report["issues"].append(f"Dropped {count} rows with negative prices")

    # Drop high < low
    hl_bad = df["high"] < df["low"]
    if hl_bad.any():
        count = int(hl_bad.sum())
        df = df[~hl_bad]
        report["issues"].append(f"Dropped {count} rows where high < low")

    if df.empty:
        return df, report

    # Gap detection
    df = df.set_index("date")
    all_biz_days = pd.date_range(df.index.min(), df.index.max(), freq="B")
    missing_dates = all_biz_days.difference(df.index)
    max_consec = _max_consecutive_gap(missing_dates)

    if max_consec > 0:
        if max_consec <= max_gap_fill_days:
            df = df.reindex(all_biz_days)
            _ffill_ohlc(df, ticker, max_gap_fill_days)
            report["gaps_filled"] = len(missing_dates)
        elif max_consec <= _HOLIDAY_GAP_LIMIT:
            logger.info(
                f"{ticker}: {len(missing_dates)} missing business days "
                f"(max consecutive: {max_consec}) — likely IDX holidays or provider gaps"
            )
            report["issues"].append(f"holiday/provider gaps ({len(missing_dates)} days)")
        else:
            logger.warning(
                f"{ticker}: {len(missing_dates)} missing business days "
                f"(max consecutive: {max_consec}) — possible suspension or delisting"
            )
            report["issues"].append(f"large gap: {max_consec} consecutive business days")

    df = df.reset_index().rename(columns={"index": "date"})
    df = df.dropna(subset=["close"])

    real_issues = [i for i in report["issues"] if "holiday" not in i]
    if real_issues:
        logger.warning(f"{ticker}: {real_issues}")
    else:
        logger.info(f"{ticker}: validation OK ({len(df)} rows)")

    return df, report


def validate_batch(
    data: dict[str, pd.DataFrame],
    max_gap_fill_days: int = _FFILL_GAP_LIMIT,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Validate semua ticker sekaligus. Returns (clean_data, all_reports)."""
    clean: dict[str, pd.DataFrame] = {}
    reports: list[dict] = []
    for ticker, df in data.items():
        cleaned, report = validate(df, ticker, max_gap_fill_days)
        if not cleaned.empty:
            clean[ticker] = cleaned
        reports.append(report)
    return clean, reports


def _max_consecutive_gap(missing_dates: pd.DatetimeIndex) -> int:
    """Hitung panjang gap berturut-turut terbesar dari tanggal yang hilang."""
    if len(missing_dates) == 0:
        return 0
    sorted_dates = missing_dates.sort_values()
    max_run = run = 1
    for i in range(1, len(sorted_dates)):
        diff = (sorted_dates[i] - sorted_dates[i - 1]).days
        if diff <= 3:  # toleransi weekend di antara hari libur
            run += 1
            max_run = max(max_run, run)
        else:
            run = 1
    return max_run


def _ffill_ohlc(df: pd.DataFrame, ticker: str, limit: int) -> None:
    price_cols = [c for c in ["open", "high", "low", "close", "adj_close"] if c in df.columns]
    df[price_cols] = df[price_cols].ffill(limit=limit)
    df["volume"] = df["volume"].fillna(0)
    df["ticker"] = ticker

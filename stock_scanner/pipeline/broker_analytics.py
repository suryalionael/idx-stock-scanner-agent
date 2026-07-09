"""Broker Analytics — metrik dan analisis aktivitas broker berbasis net_lot.

Modul ini menyediakan fungsi-fungsi untuk menganalisis aktivitas broker
menggunakan data broker summary (buy_lot, sell_lot, net_lot) dari parquet cache.

Schema yang diharapkan:
    broker_code (str)      — 2-letter IDX code, mis. 'YP', 'ML', 'ZP'
    broker_name (str)      — full broker name
    buy_lot (int|float)    — total buy volume (lot)
    sell_lot (int|float)   — total sell volume (lot)
    net_lot (int|float)    — buy_lot - sell_lot (can be negative)

Metrics dalam MVP:
    - classify_brokers_in_df()      — assign broker_type ke setiap broker
    - calculate_far()               — Foreign Accumulation Ratio (%)
    - calculate_retail_ratio()      — Retail participation (%)
    - calculate_smart_money_score() — Composite weighted score (0-100)

Extended metrics (ditunda):
    - RIDR (function stub tersedia, tapi jangan dipakai di MVP)
    - Value-based metrics (memerlukan buy_value, sell_value dari Index Alpha)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

# Kategori broker type yang valid
VALID_BROKER_TYPES = {"foreign", "institution", "retail", "big_local", "local"}


# ---------------------------------------------------------------------------
# MAIN FUNCTIONS
# ---------------------------------------------------------------------------

def classify_brokers_in_df(
    df_broker: pd.DataFrame,
    broker_config: dict,
) -> pd.DataFrame:
    """Assign broker_type ke setiap row berdasarkan broker_code.

    Input DataFrame harus punya kolom: broker_code

    Output DataFrame berisi kolom baru: broker_type
    Nilai: "foreign" | "institution" | "retail" | "big_local" | "local"

    Args:
        df_broker       : DataFrame dengan minimal kolom 'broker_code'
        broker_config   : Dict dari load_broker_config()

    Returns:
        DataFrame original + kolom 'broker_type'

    Logic:
        1. Extract broker_groups dari config
        2. Build lookup: broker_code → broker_type
        3. Assign type ke setiap row (default "local" jika tidak match)
    """
    if df_broker is None or df_broker.empty:
        return df_broker.copy() if df_broker is not None else pd.DataFrame()

    df = df_broker.copy()

    # Jika broker_code tidak ada, return tanpa type
    if "broker_code" not in df.columns:
        logger.warning("broker_code column not found in DataFrame")
        df["broker_type"] = "unknown"
        return df

    # Build lookup dict: broker_code → broker_type
    broker_groups: dict = broker_config.get("broker_groups", {})
    code_to_type: dict[str, str] = {}

    for broker_type, group_config in broker_groups.items():
        codes = group_config.get("codes", [])
        for code in codes:
            code_to_type[str(code).upper()] = broker_type

    # Assign type ke setiap row
    def _classify(code: str) -> str:
        code_upper = str(code).upper().strip() if pd.notna(code) else ""
        return code_to_type.get(code_upper, "local")

    df["broker_type"] = df["broker_code"].apply(_classify)

    return df


def calculate_far(
    df_broker: pd.DataFrame,
    broker_config: dict,
) -> Optional[float]:
    """Calculate FAR — Foreign Accumulation Ratio.

    FAR = Foreign Net Buy / Total Positive Net Buy × 100

    Interpretasi (dari config):
        < 30%   : Retail Dominated (retail ambil, asing sedikit)
        30-60%  : Mixed
        > 60%   : Foreign Dominated
        > 80%   : Strong Foreign Accumulation

    Args:
        df_broker       : DataFrame dengan kolom: broker_code, net_lot
                         dan broker_type (hasil classify_brokers_in_df)
        broker_config   : Dict dari load_broker_config()

    Returns:
        float 0-100, atau None jika tidak ada positive net_lot

    Logic:
        1. Classify brokers jika belum ada broker_type
        2. Sum net_lot untuk foreign brokers dengan net_lot > 0
        3. Sum semua net_lot positif (semua broker)
        4. Ratio = foreign positive / total positive × 100
    """
    if df_broker is None or df_broker.empty:
        return None

    df = df_broker.copy()

    # Pastikan ada broker_type
    if "broker_type" not in df.columns:
        df = classify_brokers_in_df(df, broker_config)

    # Pastikan net_lot numeric
    if "net_lot" not in df.columns:
        return None

    df["net_lot"] = pd.to_numeric(df["net_lot"], errors="coerce")

    # Foreign net buy (only positive)
    foreign_mask = df["broker_type"] == "foreign"
    foreign_positive = df[foreign_mask & (df["net_lot"] > 0)]["net_lot"].sum()

    # Total positive net
    total_positive = df[df["net_lot"] > 0]["net_lot"].sum()

    # Avoid division by zero
    if total_positive <= 0:
        return None

    far = (foreign_positive / total_positive) * 100
    return float(far)


def calculate_retail_ratio(
    df_broker: pd.DataFrame,
    broker_config: dict,
) -> Optional[float]:
    """Calculate Retail Ratio — retail participation dari total positive net.

    Retail Ratio = Retail Net Buy / Total Positive Net Buy × 100

    Interpretasi:
        > 50%  : Strong retail participation
        20-50% : Moderate
        < 20%  : Weak retail participation

    Args:
        df_broker       : DataFrame dengan kolom: broker_code, net_lot
        broker_config   : Dict dari load_broker_config()

    Returns:
        float 0-100, atau None jika tidak ada positive net_lot

    Logic:
        Similar dengan FAR, tapi menggunakan retail brokers sebagai filter.
    """
    if df_broker is None or df_broker.empty:
        return None

    df = df_broker.copy()

    # Pastikan ada broker_type
    if "broker_type" not in df.columns:
        df = classify_brokers_in_df(df, broker_config)

    # Pastikan net_lot numeric
    if "net_lot" not in df.columns:
        return None

    df["net_lot"] = pd.to_numeric(df["net_lot"], errors="coerce")

    # Retail net buy (only positive)
    retail_mask = df["broker_type"] == "retail"
    retail_positive = df[retail_mask & (df["net_lot"] > 0)]["net_lot"].sum()

    # Total positive net
    total_positive = df[df["net_lot"] > 0]["net_lot"].sum()

    # Avoid division by zero
    if total_positive <= 0:
        return None

    ratio = (retail_positive / total_positive) * 100
    return float(ratio)


def calculate_smart_money_score(
    df_broker: pd.DataFrame,
    broker_config: dict,
) -> Optional[float]:
    """Calculate Smart Money Score — weighted composite score (0-100).

    SMS = (Foreign Net × weight_foreign)
        + (Institution Net × weight_institution)
        - (Retail Net × weight_retail)

    Kemudian normalize ke range 0-100.

    Interpretasi (dari config):
        0-25   : Weak (retail dominasi)
        25-50  : Moderate
        50-75  : Strong (smart money ambil)
        75-100 : Very Strong (smart money accumulating)

    Args:
        df_broker       : DataFrame dengan kolom: broker_code, net_lot
        broker_config   : Dict dari load_broker_config()

    Returns:
        float 0-100, atau None jika insufficient data

    Logic:
        1. Classify brokers
        2. Aggregate net_lot per broker_type
        3. Apply weights dari config
        4. Normalize using min-max scaling
    """
    if df_broker is None or df_broker.empty:
        return None

    df = df_broker.copy()

    # Pastikan ada broker_type
    if "broker_type" not in df.columns:
        df = classify_brokers_in_df(df, broker_config)

    # Pastikan net_lot numeric
    if "net_lot" not in df.columns:
        return None

    df["net_lot"] = pd.to_numeric(df["net_lot"], errors="coerce")

    # Get weights dari config
    sms_config = broker_config.get("metrics", {}).get("smart_money_score", {})
    weights = sms_config.get("weights", {})
    weight_foreign = float(weights.get("foreign", 1.0))
    weight_institution = float(weights.get("institution", 0.7))
    weight_retail = float(weights.get("retail", -0.3))

    # Aggregate per type
    foreign_net = df[df["broker_type"] == "foreign"]["net_lot"].sum()
    institution_net = df[df["broker_type"] == "institution"]["net_lot"].sum()
    retail_net = df[df["broker_type"] == "retail"]["net_lot"].sum()

    # Raw score
    raw_score = (
        (foreign_net * weight_foreign)
        + (institution_net * weight_institution)
        + (retail_net * weight_retail)
    )

    # Normalize to 0-100
    # Use min-max scaling: assume raw_score range approximately [-max_lot, +max_lot]
    # For stability, use configurable or estimated bounds
    max_lot = df["net_lot"].abs().max() if not df.empty else 100000
    min_bound = -max_lot * 2
    max_bound = max_lot * 2

    if max_bound <= min_bound:
        # Edge case: all net_lot is zero
        sms = 50.0  # neutral
    else:
        sms_raw = ((raw_score - min_bound) / (max_bound - min_bound)) * 100
        sms = max(0, min(100, sms_raw))  # clip to 0-100

    return float(sms)


# ---------------------------------------------------------------------------
# STUB FUNCTIONS (EXTENDED PHASE — JANGAN DIPAKAI MVP)
# ---------------------------------------------------------------------------

def calculate_ridr(
    df_broker: pd.DataFrame,
    broker_config: dict,
) -> Optional[float]:
    """[EXTENDED PHASE — STUB ONLY]

    Calculate RIDR — Retail-to-Institution Distribution Ratio.

    RIDR = Retail Net Buy / abs(Institution Net Sell)

    Interpretasi:
        < 1.0    : Institutional Accumulation (mereka ambil, retail tidak beli)
        1.0-1.5  : Neutral
        > 1.5    : Retail Distribution Risk (retail ambil alih)

    Args:
        df_broker       : DataFrame dengan broker data
        broker_config   : Dict dari load_broker_config()

    Returns:
        float (ratio), atau None jika insufficient data

    TODO: Implement setelah MVP FAR/Retail Ratio/SMS validated.
    """
    # Stub: return None untuk MVP
    logger.debug("RIDR calculation not yet implemented (MVP phase)")
    return None


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def get_broker_classification_label(
    broker_type: str,
    metric_value: Optional[float],
    metric_name: str,
    broker_config: dict,
) -> str:
    """Map metric score ke human-readable label berdasarkan config thresholds.

    Args:
        broker_type  : "foreign" | "institution" | "retail" | "big_local" | "local"
        metric_value : float score (0-100 untuk FAR/Retail, 0-100 untuk SMS)
        metric_name  : "far" | "retail_ratio" | "smart_money_score"
        broker_config: Dict dari load_broker_config()

    Returns:
        str label, mis. "Strong Foreign Accumulation", "Weak Retail"
    """
    if metric_value is None:
        return "No Data"

    thresholds = broker_config.get("metrics", {}).get(metric_name, {}).get("thresholds", {})

    if metric_name == "far":
        if metric_value > thresholds.get("strong_accumulation", 80):
            return "🟢 Strong Foreign"
        elif metric_value > thresholds.get("foreign_dominated", 60):
            return "🟠 Foreign Dominated"
        elif metric_value >= thresholds.get("retail_dominated", 30):
            return "⚪ Mixed"
        else:
            return "🔴 Retail Dominated"

    elif metric_name == "retail_ratio":
        if metric_value > thresholds.get("significant", 50):
            return "🟢 Strong Retail"
        elif metric_value > thresholds.get("moderate", 20):
            return "⚪ Moderate Retail"
        else:
            return "🔴 Weak Retail"

    elif metric_name == "smart_money_score":
        categories = broker_config.get("metrics", {}).get("smart_money_score", {}).get("categories", {})
        if metric_value >= categories.get("very_strong", [75, 100])[0]:
            return "🟢 Very Strong"
        elif metric_value >= categories.get("strong", [50, 75])[0]:
            return "🟠 Strong"
        elif metric_value >= categories.get("moderate", [25, 50])[0]:
            return "⚪ Moderate"
        else:
            return "🔴 Weak"

    return "Unknown"


def get_top_brokers_by_metric(
    df_broker: pd.DataFrame,
    metric: str = "net_lot",
    direction: str = "positive",
    n: int = 5,
) -> list[dict]:
    """Get top N brokers berdasarkan metric tertentu.

    Args:
        df_broker : DataFrame dengan kolom: broker_code, broker_name, net_lot
        metric    : "net_lot" saja untuk MVP
        direction : "positive" (net_lot > 0) atau "negative" (net_lot < 0)
        n         : Jumlah top broker

    Returns:
        List of dicts:
        [
            {
                "broker_code": str,
                "broker_name": str,
                "net_lot": float,
                "type": str  (jika ada broker_type column)
            },
            ...
        ]
    """
    if df_broker is None or df_broker.empty or metric != "net_lot":
        return []

    df = df_broker.copy()

    # Pastikan net_lot numeric
    if "net_lot" not in df.columns:
        return []

    df["net_lot"] = pd.to_numeric(df["net_lot"], errors="coerce")

    # Filter by direction
    if direction == "positive":
        df = df[df["net_lot"] > 0]
    elif direction == "negative":
        df = df[df["net_lot"] < 0]
    else:
        pass  # no filter

    # Sort dan ambil top n
    df = df.sort_values("net_lot", ascending=(direction == "negative"), key=abs).head(n)

    # Format output
    result = []
    for _, row in df.iterrows():
        item = {
            "broker_code": str(row.get("broker_code", "?")),
            "broker_name": str(row.get("broker_name", "?")),
            "net_lot": float(row.get("net_lot", 0)),
        }
        if "broker_type" in row:
            item["type"] = str(row.get("broker_type", ""))
        result.append(item)

    return result


# ---------------------------------------------------------------------------
# AGGREGATION FUNCTIONS (STEP 2)
# ---------------------------------------------------------------------------

def aggregate_broker_activity(
    df_broker: pd.DataFrame,
    broker_config: dict,
    date: str | None = None,
) -> dict:
    """Ringkas aktivitas broker single-day untuk satu ticker.

    Tujuan: Mengagregasi broker summary satu saham satu hari menjadi
    metrik ringkas untuk display atau leaderboard.

    Args:
        df_broker       : DataFrame hasil load broker untuk satu ticker satu hari
                         (schema: broker_code, broker_name, buy_lot, sell_lot, net_lot)
        broker_config   : Dict dari load_broker_config()
        date            : Optional YYYY-MM-DD, untuk dokumentasi output

    Returns:
        Dict dengan struktur:
        {
            "date": str | None,
            "broker_count": int,
            "total_buy_lot": float,
            "total_sell_lot": float,
            "total_net_lot": float,
            "far": float | None,
            "retail_ratio": float | None,
            "smart_money_score": float | None,
            "foreign_net": float,
            "institution_net": float,
            "retail_net": float,
            "top_buyers": list[dict],  — top 3 buyers
            "top_sellers": list[dict], — top 3 sellers (abs value)
            "broker_type_breakdown": dict — count per type
        }

    Graceful: Return sensible defaults jika df_broker kosong atau ada error.
    """
    if df_broker is None or df_broker.empty:
        return {
            "date": date,
            "broker_count": 0,
            "total_buy_lot": 0,
            "total_sell_lot": 0,
            "total_net_lot": 0,
            "far": None,
            "retail_ratio": None,
            "smart_money_score": None,
            "foreign_net": 0,
            "institution_net": 0,
            "retail_net": 0,
            "top_buyers": [],
            "top_sellers": [],
            "broker_type_breakdown": {},
        }

    df = df_broker.copy()

    # Classify brokers jika belum ada type
    if "broker_type" not in df.columns:
        df = classify_brokers_in_df(df, broker_config)

    # Numeric conversion
    df["buy_lot"] = pd.to_numeric(df["buy_lot"], errors="coerce")
    df["sell_lot"] = pd.to_numeric(df["sell_lot"], errors="coerce")
    df["net_lot"] = pd.to_numeric(df["net_lot"], errors="coerce")

    # Aggregate lots
    total_buy = df["buy_lot"].sum()
    total_sell = df["sell_lot"].sum()
    total_net = df["net_lot"].sum()

    # Net lot per broker type
    foreign_net = df[df["broker_type"] == "foreign"]["net_lot"].sum()
    institution_net = df[df["broker_type"] == "institution"]["net_lot"].sum()
    retail_net = df[df["broker_type"] == "retail"]["net_lot"].sum()

    # Metrics
    far = calculate_far(df, broker_config)
    retail_ratio = calculate_retail_ratio(df, broker_config)
    sms = calculate_smart_money_score(df, broker_config)

    # Top buyers/sellers
    top_buyers = get_top_brokers_by_metric(df, "net_lot", "positive", 3)
    top_sellers = get_top_brokers_by_metric(df, "net_lot", "negative", 3)

    # Broker type breakdown
    type_breakdown = df["broker_type"].value_counts().to_dict()

    return {
        "date": date,
        "broker_count": len(df),
        "total_buy_lot": float(total_buy),
        "total_sell_lot": float(total_sell),
        "total_net_lot": float(total_net),
        "far": far,
        "retail_ratio": retail_ratio,
        "smart_money_score": sms,
        "foreign_net": float(foreign_net),
        "institution_net": float(institution_net),
        "retail_net": float(retail_net),
        "top_buyers": top_buyers,
        "top_sellers": top_sellers,
        "broker_type_breakdown": type_breakdown,
    }


def load_broker_history_multi_day(
    ticker: str,
    date_start: str,
    date_end: str,
) -> pd.DataFrame:
    """Load broker activity untuk satu ticker dalam range tanggal.

    Tujuan: Kumpulkan data broker multi-hari untuk satu ticker.

    Args:
        ticker      : IDX ticker (dengan atau tanpa .JK)
        date_start  : YYYY-MM-DD
        date_end    : YYYY-MM-DD (inclusive)

    Returns:
        DataFrame gabungan dengan kolom:
        - date (str, YYYY-MM-DD)
        - broker_code
        - broker_name
        - buy_lot
        - sell_lot
        - net_lot
        Sorted ascending by date (oldest first).

    Logic:
        1. Scan folder data/broker untuk files matching {TICKER}.JK_{DATE}.parquet
        2. Load semua files dalam range
        3. Add kolom date
        4. Gabung dan sort by date
    """
    from pathlib import Path

    ticker_clean = str(ticker).upper().replace(".JK", "")
    broker_dir = Path.cwd() / "data" / "broker"

    if not broker_dir.exists():
        logger.warning(f"Broker directory tidak ada: {broker_dir}")
        return pd.DataFrame()

    # Parse date range
    try:
        from datetime import datetime, timedelta

        start = datetime.strptime(date_start, "%Y-%m-%d")
        end = datetime.strptime(date_end, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        return pd.DataFrame()

    # Scan files
    dfs = []
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        file_path = broker_dir / f"{ticker_clean}.JK_{date_str}.parquet"

        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                df["date"] = date_str
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        current += timedelta(days=1)

    if not dfs:
        logger.debug(f"No broker data found for {ticker_clean} [{date_start} to {date_end}]")
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    result = result.sort_values("date").reset_index(drop=True)

    return result


def aggregate_broker_history(
    ticker: str,
    date_start: str,
    date_end: str,
    broker_code: str,
) -> pd.DataFrame:
    """Get history satu broker untuk satu ticker pada date range.

    Tujuan: Track aktivitas satu broker sepanjang periode.

    Args:
        ticker      : IDX ticker
        date_start  : YYYY-MM-DD
        date_end    : YYYY-MM-DD
        broker_code : 2-letter code (mis. "YP", "ML")

    Returns:
        DataFrame per broker dengan kolom:
        - date
        - broker_code
        - broker_name
        - buy_lot
        - sell_lot
        - net_lot
        - rolling_5d (rolling sum net_lot, 5-day window)
        - rolling_20d (rolling sum net_lot, 20-day window)
        - cumulative_net_lot (cumsum dari net_lot)
        Sorted ascending by date.

    Graceful: Return empty DataFrame jika broker tidak ada dalam range.
    """
    df_history = load_broker_history_multi_day(ticker, date_start, date_end)

    if df_history.empty:
        return pd.DataFrame()

    # Filter by broker code
    broker_code_upper = str(broker_code).upper()
    df = df_history[df_history["broker_code"] == broker_code_upper].copy()

    if df.empty:
        logger.debug(f"No data for {broker_code} on {ticker} [{date_start} to {date_end}]")
        return pd.DataFrame()

    # Ensure sorted by date
    df = df.sort_values("date").reset_index(drop=True)

    # Convert net_lot to numeric
    df["net_lot"] = pd.to_numeric(df["net_lot"], errors="coerce")

    # Rolling windows
    df["rolling_5d"] = df["net_lot"].rolling(window=5, min_periods=1).sum()
    df["rolling_20d"] = df["net_lot"].rolling(window=20, min_periods=1).sum()

    # Cumulative
    df["cumulative_net_lot"] = df["net_lot"].cumsum()

    return df


def aggregate_all_stocks_single_date(
    date: str,
    broker_config: dict,
) -> pd.DataFrame:
    """Leaderboard: aggregate metrics untuk semua ticker pada satu tanggal.

    Tujuan: Scan semua file broker untuk satu date dan compute metrics
    per ticker, kemudian ranking.

    Args:
        date            : YYYY-MM-DD
        broker_config   : Dict dari load_broker_config()

    Returns:
        DataFrame dengan kolom:
        - ticker (str, includes .JK)
        - date (str)
        - broker_count (int)
        - total_net_lot (float)
        - far (float, %)
        - retail_ratio (float, %)
        - smart_money_score (float, 0-100)
        - foreign_net (float)
        - institution_net (float)
        - retail_net (float)
        Sorted descending by smart_money_score (strongest accumulation first).

    Graceful: Return empty DataFrame jika tidak ada broker data untuk date tersebut.
    """
    from pathlib import Path

    broker_dir = Path.cwd() / "data" / "broker"

    if not broker_dir.exists():
        logger.warning(f"Broker directory tidak ada: {broker_dir}")
        return pd.DataFrame()

    # Find all files matching {TICKER}.JK_{DATE}.parquet
    pattern = f"*_{date}.parquet"
    files = list(broker_dir.glob(pattern))

    if not files:
        logger.debug(f"No broker files found for date {date}")
        return pd.DataFrame()

    results = []

    for file_path in files:
        # Parse ticker dari filename
        # Format: {TICKER}.JK_{DATE}.parquet
        stem = file_path.stem
        parts = stem.rsplit("_", 1)  # split from right, only date
        if len(parts) != 2:
            continue

        ticker_with_jk = parts[0]  # e.g., "BBCA.JK"

        try:
            df = pd.read_parquet(file_path)
            if df.empty:
                continue

            # Aggregate
            agg = aggregate_broker_activity(df, broker_config, date)

            result_row = {
                "ticker": ticker_with_jk,
                "date": date,
                "broker_count": agg["broker_count"],
                "total_net_lot": agg["total_net_lot"],
                "far": agg["far"],
                "retail_ratio": agg["retail_ratio"],
                "smart_money_score": agg["smart_money_score"],
                "foreign_net": agg["foreign_net"],
                "institution_net": agg["institution_net"],
                "retail_net": agg["retail_net"],
            }
            results.append(result_row)

        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            continue

    if not results:
        return pd.DataFrame()

    df_result = pd.DataFrame(results)

    # Sort by smart_money_score descending (strongest accumulation first)
    df_result = df_result.sort_values(
        "smart_money_score",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return df_result

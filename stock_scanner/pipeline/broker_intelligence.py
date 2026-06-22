"""Broker Intelligence module for IDX stocks.

Classifies brokers into foreign / big-local / local tiers and computes
accumulation scores from multi-day broker transaction data.

⚠️ Requires real broker transaction data in data/broker/{ticker}_{date}.parquet.
When no data is available, all outputs are zero / NO_SIGNAL — the module will
never raise an exception.

Broker data schema (per-day parquet):
    broker_code : str   — 2-letter IDX code
    broker_name : str
    buy_lot     : float — total buy (lot)
    sell_lot    : float — total sell (lot)
    net_lot     : float — buy_lot - sell_lot
    date        : str | pd.Timestamp  (added by load_broker_history)

Public API
----------
classify_broker(broker_code) -> str
    "foreign" | "big_local" | "local" | "unknown"

compute_broker_intelligence(ticker, broker_df, metadata_df=None) -> dict
    Aggregate multi-day broker data into accumulation signals.

compute_longterm_broker_alert(stock_data) -> dict
    3-layer alert: broker accumulation + fundamental quality + valuation.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Broker classification tables (IDX 2-letter codes)
# ---------------------------------------------------------------------------

#: Foreign institutional brokers active on IDX
FOREIGN_BROKER_CODES: frozenset[str] = frozenset({
    "CS",   # Credit Suisse / UBS Securities (post-merger)
    "ML",   # Merrill Lynch / BofA Securities
    "MS",   # Morgan Stanley
    "JP",   # JPMorgan Securities
    "DB",   # Deutsche Bank
    "YU",   # CLSA Securities Indonesia
    "ZP",   # Macquarie Sekuritas
    "RB",   # CGS-CIMB Securities
    "GS",   # Goldman Sachs
    "UX",   # UOB Kay Hian Securities
    "AK",   # UBS Securities Indonesia
    "DP",   # DBS Vickers Securities
    "SB",   # Nomura Securities
    "BK",   # JP Securities / Barclays
    "QA",   # Citigroup Securities
    "LS",   # Credit Suisse International
    "MK",   # Daiwa Capital Markets
    "KK",   # Maybank Securities
    "OX",   # CIMB Securities
    "SA",   # Societe Generale
})

#: Large domestic (local) brokers with significant market presence
BIG_LOCAL_BROKER_CODES: frozenset[str] = frozenset({
    "YP",   # Indo Premier Sekuritas / Mirae Asset
    "OD",   # Mandiri Sekuritas
    "DH",   # Bahana Sekuritas
    "YJ",   # Trimegah Sekuritas
    "ZH",   # CIMB Niaga Sekuritas
    "BW",   # BNI Sekuritas
    "DR",   # OSO Sekuritas
    "CC",   # Mirae Asset Sekuritas
    "ID",   # Sinarmas Sekuritas
    "YB",   # Panin Sekuritas
    "GI",   # BRI Danareksa Sekuritas
    "LG",   # Phillip Sekuritas
    "ZU",   # Danareksa Sekuritas
    "HD",   # HD Capital Sekuritas
    "PD",   # Henan Putihrai
    "EP",   # MNC Sekuritas
    "FZ",   # Manulife Sekuritas
    "KI",   # Ciptadana Sekuritas
    "AD",   # Phintraco Sekuritas
    "NI",   # BCA Sekuritas
})

_LABEL_STRONG = "ACCUMULATION_STRONG"
_LABEL_WATCH  = "ACCUMULATION_WATCH"
_LABEL_NONE   = "NO_SIGNAL"

_SCORE_STRONG = 6.0
_SCORE_WATCH  = 3.5

_INACTIVE_RESULT: dict[str, Any] = {
    "foreign_net_buy_1d":          0.0,
    "foreign_net_buy_5d":          0.0,
    "foreign_net_buy_20d":         0.0,
    "big_broker_net_buy_5d":       0.0,
    "big_broker_net_buy_20d":      0.0,
    "active_broker_count_1d":      0,
    "active_broker_count_5d":      0,
    "active_broker_count_20d":     0,
    "top_buyer_brokers":           [],
    "top_seller_brokers":          [],
    "broker_accumulation_score":   0.0,
    "broker_accumulation_label":   _LABEL_NONE,
    "broker_summary":              "Data broker tidak tersedia",
    "strengths":                   [],
    "red_flags":                   [],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_broker(broker_code: str) -> str:
    """Classify a broker by its 2-letter IDX code.

    Returns:
        "foreign"   — foreign institutional broker
        "big_local" — large domestic broker
        "local"     — other domestic broker
        "unknown"   — unrecognised code
    """
    code = str(broker_code).strip().upper()
    if not code:
        return "unknown"
    if code in FOREIGN_BROKER_CODES:
        return "foreign"
    if code in BIG_LOCAL_BROKER_CODES:
        return "big_local"
    if len(code) == 2 and code.isalpha():
        return "local"
    return "unknown"


def compute_broker_intelligence(
    ticker: str,
    broker_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None = None,
) -> dict:
    """Compute broker accumulation intelligence from multi-day transaction data.

    Scoring rubric (max 10 pts):
    ┌──────────────────────────────────┬────────┐
    │ Factor                           │ Max pts│
    ├──────────────────────────────────┼────────┤
    │ Foreign net buy 1d               │ 2.0    │
    │ Foreign net buy 5d               │ 1.5    │
    │ Foreign net buy 20d              │ 1.0    │
    │ Big local net buy 5d             │ 1.5    │
    │ Big local net buy 20d            │ 1.0    │
    │ Active net-buying broker count   │ 0.5    │
    │ Accumulation consistency (trend) │ 1.0    │
    │ Divergence: foreign ≠ retail     │ 0.5    │
    │ Heavy buy concentration (top 3)  │ 1.0    │
    └──────────────────────────────────┴────────┘

    Args:
        ticker     : IDX ticker (used for logging only)
        broker_df  : Multi-day broker transaction DataFrame.
                     Must have: broker_code, net_lot (or buy_lot/sell_lot).
                     Optional: date (for time-slicing).
        metadata_df: Optional metadata (not used currently, reserved for future).

    Returns:
        dict with aggregated broker intelligence. See _INACTIVE_RESULT for keys.
    """
    if broker_df is None or broker_df.empty:
        return dict(_INACTIVE_RESULT)

    df = broker_df.copy()

    # ── Normalise required columns ────────────────────────────────────────
    if "broker_code" not in df.columns:
        return dict(_INACTIVE_RESULT)

    # Compute net_lot if missing
    if "net_lot" not in df.columns:
        if "buy_lot" in df.columns and "sell_lot" in df.columns:
            df["net_lot"] = (
                pd.to_numeric(df["buy_lot"], errors="coerce").fillna(0)
                - pd.to_numeric(df["sell_lot"], errors="coerce").fillna(0)
            )
        else:
            return dict(_INACTIVE_RESULT)
    else:
        df["net_lot"] = pd.to_numeric(df["net_lot"], errors="coerce").fillna(0)

    # Classify brokers
    df["broker_type"] = df["broker_code"].apply(classify_broker)

    # ── Date partitioning ─────────────────────────────────────────────────
    has_date = "date" in df.columns
    if has_date:
        df["_date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["_date"])
        df = df.sort_values("_date")
        latest_date = df["_date"].max()
        df_1d  = df[df["_date"] == latest_date]
        df_5d  = df[df["_date"] >= (latest_date - pd.Timedelta(days=7))]   # 5 trading ≈ 7 cal days
        df_20d = df  # already sliced upstream by load_broker_history
    else:
        # No date info — treat all rows as "latest" period
        df_1d = df_5d = df_20d = df

    # ── Aggregate by broker type ──────────────────────────────────────────
    def _net(frame: pd.DataFrame, btype: str) -> float:
        sub = frame[frame["broker_type"] == btype]
        return float(sub["net_lot"].sum()) if not sub.empty else 0.0

    foreign_1d   = _net(df_1d,  "foreign")
    foreign_5d   = _net(df_5d,  "foreign")
    foreign_20d  = _net(df_20d, "foreign")
    big_5d       = _net(df_5d,  "big_local")
    big_20d      = _net(df_20d, "big_local")

    # Active net-buying broker counts
    def _active_count(frame: pd.DataFrame) -> int:
        return int((frame.groupby("broker_code")["net_lot"].sum() > 0).sum())

    active_1d  = _active_count(df_1d)
    active_5d  = _active_count(df_5d)
    active_20d = _active_count(df_20d)

    # Top buyers/sellers (by net_lot over 5d)
    broker_agg = (
        df_5d.groupby(["broker_code", "broker_type"])["net_lot"]
        .sum()
        .reset_index()
        .sort_values("net_lot", ascending=False)
    )
    top_buyers  = _format_top_brokers(broker_agg.head(5),  direction="buy")
    top_sellers = _format_top_brokers(broker_agg.tail(5).iloc[::-1], direction="sell")

    # ── Scoring ───────────────────────────────────────────────────────────
    score = 0.0
    strengths: list[str] = []
    red_flags: list[str] = []
    reasons:   list[str] = []

    # 1. Foreign net buy 1d (2.0)
    if foreign_1d > 0:
        pts = min(2.0, 0.5 + (foreign_1d / 100_000) * 1.5)  # scale to 100K lot
        score += pts
        if pts >= 1.5:
            reasons.append(f"Foreign beli kuat 1d (+{foreign_1d:,.0f} lot)")
            strengths.append(f"Foreign net buy 1d: +{foreign_1d:,.0f} lot")
        elif pts >= 0.8:
            reasons.append(f"Foreign beli 1d (+{foreign_1d:,.0f} lot)")
    elif foreign_1d < -50_000:
        score -= 0.5
        red_flags.append(f"Foreign jual 1d: {foreign_1d:,.0f} lot")

    # 2. Foreign net buy 5d (1.5)
    if foreign_5d > 0:
        pts = min(1.5, 0.3 + (foreign_5d / 200_000) * 1.2)
        score += pts
        if pts >= 1.0:
            strengths.append(f"Foreign net buy 5d: +{foreign_5d:,.0f} lot")
    elif foreign_5d < -100_000:
        score -= 0.3
        red_flags.append(f"Foreign distribution 5d: {foreign_5d:,.0f} lot")

    # 3. Foreign net buy 20d (1.0)
    if foreign_20d > 0:
        pts = min(1.0, 0.2 + (foreign_20d / 500_000) * 0.8)
        score += pts
        if pts >= 0.6:
            strengths.append(f"Foreign kumulatif 20d: +{foreign_20d:,.0f} lot")
    elif foreign_20d < -200_000:
        score -= 0.2

    # 4. Big local net buy 5d (1.5)
    if big_5d > 0:
        pts = min(1.5, 0.3 + (big_5d / 150_000) * 1.2)
        score += pts
        if pts >= 0.8:
            reasons.append(f"Big broker beli 5d (+{big_5d:,.0f} lot)")
            strengths.append(f"Big broker net buy 5d: +{big_5d:,.0f} lot")
    elif big_5d < -80_000:
        score -= 0.2
        red_flags.append(f"Big broker jual 5d: {big_5d:,.0f} lot")

    # 5. Big local net buy 20d (1.0)
    if big_20d > 0:
        pts = min(1.0, 0.2 + (big_20d / 300_000) * 0.8)
        score += pts

    # 6. Active broker count (0.5)
    if active_5d >= 10:
        score += 0.5
        strengths.append(f"{active_5d} broker aktif beli 5d")
    elif active_5d >= 5:
        score += 0.25

    # 7. Consistency (1.0) — foreign and big both positive = convergence signal
    if foreign_5d > 0 and big_5d > 0:
        score += 1.0
        reasons.append("Foreign + big broker convergence")
        strengths.append("Foreign + big broker beli bersamaan (5d)")
    elif foreign_5d > 0 and big_5d < 0:
        # Divergence: foreign buying while big local selling → accumulation with distribution
        score += 0.3
    elif foreign_5d < 0 and big_5d < 0:
        score -= 0.5
        red_flags.append("Foreign + big broker keduanya jual")

    # 8. Divergence bonus (0.5) — foreign accumulating vs retail selling = strong smart-money signal
    if has_date and len(df_5d) > 0:
        local_5d = _net(df_5d, "local")
        if foreign_5d > 0 and local_5d < 0:
            score += 0.5
            strengths.append("Foreign beli saat retail jual (smart-money divergence)")

    # 9. Heavy concentration (1.0) — top 3 buyers are foreign = conviction
    if len(broker_agg) >= 3:
        top3_types = broker_agg.head(3)["broker_type"].tolist()
        foreign_in_top3 = top3_types.count("foreign")
        if foreign_in_top3 >= 2:
            score += 1.0
            strengths.append(f"{foreign_in_top3}/3 top buyer adalah broker asing")
        elif foreign_in_top3 == 1:
            score += 0.4

    # ── Clip + label ──────────────────────────────────────────────────────
    score = round(max(0.0, min(10.0, score)), 2)

    if score >= _SCORE_STRONG:
        label = _LABEL_STRONG
    elif score >= _SCORE_WATCH:
        label = _LABEL_WATCH
    else:
        label = _LABEL_NONE

    summary = ", ".join(reasons[:3]) if reasons else "Tidak ada sinyal akumulasi signifikan"

    return {
        "foreign_net_buy_1d":          round(foreign_1d, 0),
        "foreign_net_buy_5d":          round(foreign_5d, 0),
        "foreign_net_buy_20d":         round(foreign_20d, 0),
        "big_broker_net_buy_5d":       round(big_5d, 0),
        "big_broker_net_buy_20d":      round(big_20d, 0),
        "active_broker_count_1d":      active_1d,
        "active_broker_count_5d":      active_5d,
        "active_broker_count_20d":     active_20d,
        "top_buyer_brokers":           top_buyers,
        "top_seller_brokers":          top_sellers,
        "broker_accumulation_score":   score,
        "broker_accumulation_label":   label,
        "broker_summary":              summary,
        "strengths":                   strengths,
        "red_flags":                   red_flags,
    }


def compute_longterm_broker_alert(stock_data: dict) -> dict:
    """3-layer long-term alert: broker accumulation + fundamental + valuation.

    Trigger conditions (ALL must be true for STRONG):
        Layer 1 — Broker: broker_accumulation_score >= 3.5 (foreign/big buying)
        Layer 2 — Fundamental: ROE >= 10% AND DER <= 2.5
        Layer 3 — Valuation: valuation_status in (UNDERVALUED, FAIR)

    Alert levels:
        LONGTERM_BROKER_ALERT_STRONG — all 3 layers met + broker_score >= 6.0
        LONGTERM_BROKER_ALERT_WATCH  — broker layer + at least 1 of the other 2
        NO_ALERT                      — broker layer not met, or fundamental very weak

    Args:
        stock_data: dict containing keys from compute_broker_intelligence output
                    plus fundamental columns (roe_pct, der, valuation_status, etc.)

    Returns:
        {
            "alert_status"   : str,
            "alert_reason"   : str,
            "confidence_score": float  (0–10)
        }
    """
    def _get(k, default=None):
        return stock_data.get(k, default)

    broker_score    = _safe_float(_get("broker_accumulation_score", 0))
    broker_label    = str(_get("broker_accumulation_label", "NO_SIGNAL"))
    roe             = _safe_float(_get("roe_pct"))
    der             = _safe_float(_get("der"))
    val_status      = str(_get("valuation_status", "UNKNOWN")).upper()
    lt_score        = _safe_float(_get("long_term_score", 0))

    reasons: list[str] = []
    confidence = 0.0

    # ── Layer 1: Broker accumulation ─────────────────────────────────────
    broker_ok = broker_score >= _SCORE_WATCH  # >= 3.5

    if not broker_ok:
        return {
            "alert_status":    "NO_ALERT",
            "alert_reason":    "Tidak ada sinyal akumulasi broker yang signifikan",
            "confidence_score": 0.0,
        }

    confidence += min(3.5, broker_score * 0.5)
    reasons.append(f"Broker score {broker_score:.1f} ({broker_label})")

    # ── Layer 2: Fundamental quality ─────────────────────────────────────
    fund_ok = False
    if roe is not None and roe >= 10.0:
        fund_ok = True
        reasons.append(f"ROE {roe:.1f}%")
        confidence += 1.5
        if roe >= 15:
            confidence += 0.5
    if der is not None and der <= 2.5:
        if not fund_ok:
            fund_ok = True
        reasons.append(f"DER {der:.2f}×")
        confidence += 1.0
    elif der is not None and der > 3.0:
        reasons.append(f"⚠️ DER tinggi {der:.2f}×")
        confidence -= 0.5

    # Long-term score as additional quality signal
    if lt_score >= 6.0:
        confidence += 1.0
        reasons.append(f"LT Score {lt_score:.1f}/10")

    # ── Layer 3: Valuation ────────────────────────────────────────────────
    val_ok = val_status in ("UNDERVALUED", "FAIR")

    if val_status == "UNDERVALUED":
        confidence += 2.0
        reasons.append("Valuasi UNDERVALUED (margin of safety positif)")
    elif val_status == "FAIR":
        confidence += 1.0
        reasons.append("Valuasi FAIR")
    elif val_status == "OVERVALUED":
        confidence -= 1.0
        reasons.append("⚠️ Valuasi OVERVALUED")

    # ── Determine alert level ─────────────────────────────────────────────
    confidence = round(max(0.0, min(10.0, confidence)), 2)
    reason_str = " · ".join(reasons)

    if broker_score >= _SCORE_STRONG and fund_ok and val_ok:
        return {
            "alert_status":    "LONGTERM_BROKER_ALERT_STRONG",
            "alert_reason":    reason_str,
            "confidence_score": confidence,
        }

    if broker_ok and (fund_ok or val_ok):
        return {
            "alert_status":    "LONGTERM_BROKER_ALERT_WATCH",
            "alert_reason":    reason_str,
            "confidence_score": confidence,
        }

    return {
        "alert_status":    "NO_ALERT",
        "alert_reason":    reason_str or "Kondisi tidak memenuhi kriteria long-term alert",
        "confidence_score": confidence,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_top_brokers(df: pd.DataFrame, direction: str) -> list[dict]:
    """Return top broker list as list-of-dicts for easy display."""
    if df.empty or "broker_code" not in df.columns:
        return []
    result = []
    for _, row in df.iterrows():
        code = str(row.get("broker_code", "?"))
        btype = str(row.get("broker_type", classify_broker(code)))
        net = float(row.get("net_lot", 0))
        if direction == "buy" and net <= 0:
            continue
        if direction == "sell" and net >= 0:
            continue
        result.append({
            "broker_code":  code,
            "broker_type":  btype,
            "net_lot":      net,
            "type_label":   _type_label(btype),
        })
    return result[:5]


def _type_label(btype: str) -> str:
    return {"foreign": "🌍 Asing", "big_local": "🏦 Big Local", "local": "🏠 Lokal"}.get(
        btype, "❓ Unknown"
    )


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        v = float(val)
        return None if v != v else v  # NaN guard
    except (TypeError, ValueError):
        return None

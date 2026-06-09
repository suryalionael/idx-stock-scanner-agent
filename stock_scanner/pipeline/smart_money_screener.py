"""Smart Money Screener — detect accumulation footprints before price markup.

Five detectors, all config-driven (configs/smart_money_config.yaml) and tolerant
of missing data. Inputs come from the daily-scan signals DataFrame (94 enriched
columns: technical + foreign flow + shareholder + fundamental) plus, where
available, real Index Alpha broker-summary parquet files for per-broker analysis.

Detectors
---------
1. detect_volume_accumulation(row, cfg)
     Volume rising while price is still sideways / below breakout.
2. detect_ownership_change(row, cfg)            -> Accumulation | Neutral | Distribution
     Foreign flow (5d/20d) + shareholder base trend.
3. detect_single_broker_absorption(broker_df, cfg) -> Strong | Moderate | No ... Accumulation
     One broker's net buy dominating supply from many different sellers.
4. detect_hidden_accumulation(row, ma5, broker_label, cfg) -> Hidden | Early | Strong | None
     Price depressed (MA200 > MA5 > close) yet being accumulated.
5. score_fundamentals(row, cfg)                 -> score 0-100 + grade A/B/C
     Revenue/profit growth, ROE, DER, positive EBITDA (OCF proxy).

screen_smart_money(...) combines them into a ranked candidate table.

Column dependencies (from signals parquet): vol_ratio_20d, roc20,
pct_from_52w_high, foreign_net_5d, foreign_net_20d, foreign_flow_score,
shareholder_score, holder_growth_1m, ma200, close, roe_pct, der,
revenue_growth_pct, profit_growth_pct, ebitda, fundamental_status.
ma5 is supplied separately (compute_ma5_map) since older scans predate it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml
from loguru import logger

_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_CFG = _ROOT / "stock_scanner" / "configs" / "smart_money_config.yaml"
_RAW_DIR = _ROOT / "data" / "raw"
_BROKER_DIR = _ROOT / "data" / "broker"


# ---------------------------------------------------------------------------
# Config + small numeric helpers
# ---------------------------------------------------------------------------

def load_smart_money_config(path: Path | None = None) -> dict:
    """Load smart_money_config.yaml. Returns {} (with a warning) if missing."""
    path = path or _DEFAULT_CFG
    if not path.exists():
        logger.warning("smart_money_config.yaml not found at {} — using defaults.", path)
        return {}
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse smart_money_config.yaml: {}", exc)
        return {}


def _num(val: Any) -> Optional[float]:
    """Coerce to float; None on NaN/invalid."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if pd.isna(f):
        return None
    return f


def _broker_ownership_net(broker_df: pd.DataFrame) -> Optional[dict]:
    """REAL per-tier net lot from Index Alpha broker data (classify each broker).

    Returns {foreign_net, big_local_net, local_net, total_net} or None when no
    broker data — so ownership is reported honestly instead of a misleading
    'Neutral' from placeholder foreign-flow columns.
    """
    if broker_df is None or broker_df.empty or "net_lot" not in broker_df.columns:
        return None
    from stock_scanner.pipeline.broker_intelligence import classify_broker

    df = broker_df.copy()
    df["net_lot"] = pd.to_numeric(df["net_lot"], errors="coerce").fillna(0)
    if "broker_code" not in df.columns:
        return None
    df["_cls"] = df["broker_code"].apply(lambda c: classify_broker(str(c)))
    return {
        "foreign_net": float(df.loc[df["_cls"] == "foreign", "net_lot"].sum()),
        "big_local_net": float(df.loc[df["_cls"] == "big_local", "net_lot"].sum()),
        "local_net": float(df.loc[df["_cls"] == "local", "net_lot"].sum()),
        "total_net": float(df["net_lot"].sum()),
    }


# ---------------------------------------------------------------------------
# 1) Volume rises before price
# ---------------------------------------------------------------------------

def detect_volume_accumulation(row: pd.Series, cfg: dict) -> dict:
    """Volume spike while price is still sideways / below breakout.

    Returns {detected, strength, vol_ratio, roc20, reason}.
    """
    c = cfg.get("volume_accumulation", {})
    min_vr = float(c.get("min_vol_ratio", 1.5))
    strong_vr = float(c.get("strong_vol_ratio", 2.5))
    max_roc20 = float(c.get("max_roc20_pct", 8.0))
    max_pct52 = float(c.get("max_pct_from_52w_high", -7.0))

    vr = _num(row.get("vol_ratio_20d"))
    roc20 = _num(row.get("roc20"))
    pct52 = _num(row.get("pct_from_52w_high"))

    if vr is None:
        return {"detected": False, "strength": "—", "vol_ratio": None,
                "roc20": roc20, "reason": "volume data tidak tersedia"}

    vol_ok = vr >= min_vr
    price_sideways = (roc20 is None) or (roc20 <= max_roc20)
    below_breakout = (pct52 is None) or (pct52 <= max_pct52)

    detected = bool(vol_ok and price_sideways and below_breakout)
    if not detected:
        reason = "tidak memenuhi (" + ", ".join(filter(None, [
            None if vol_ok else f"vol {vr:.1f}x < {min_vr}x",
            None if price_sideways else f"roc20 {roc20:.1f}% > {max_roc20}%",
            None if below_breakout else "sudah dekat 52w high",
        ])) + ")"
        return {"detected": False, "strength": "—", "vol_ratio": vr,
                "roc20": roc20, "reason": reason}

    strength = "Strong" if vr >= strong_vr else "Moderate"
    return {"detected": True, "strength": strength, "vol_ratio": vr, "roc20": roc20,
            "reason": f"vol {vr:.1f}x avg, harga sideways (roc20 {roc20 if roc20 is not None else 0:.1f}%)"}


# ---------------------------------------------------------------------------
# 2) Ownership change — Accumulation / Neutral / Distribution
# ---------------------------------------------------------------------------

def detect_ownership_change(row: pd.Series, broker_df: pd.DataFrame, cfg: dict) -> dict:
    """Classify ownership trend → Accumulation | Neutral | Distribution | No Data.

    Primary (real) source: Index Alpha broker data, summed by broker tier
    (foreign / big-local). Falls back to real foreign_net_* columns only if those
    are populated (they are placeholder/NaN in the current pipeline, so the
    detector returns 'No Data' rather than a misleading 'Neutral').

    Returns {status, points, source, foreign_net, reason}.
    """
    c = cfg.get("ownership_change", {})
    accum_min = int(c.get("accum_min_points", 2))
    distrib_max = int(c.get("distrib_max_points", -1))

    own = _broker_ownership_net(broker_df)
    if own is not None:
        points = 0
        notes: list[str] = []
        if own["foreign_net"] > 0:
            points += 1
            notes.append("foreign net buy")
        elif own["foreign_net"] < 0:
            points -= 1
            notes.append("foreign net sell")
        if own["big_local_net"] > 0:
            points += 1
            notes.append("big-local net buy")
        elif own["big_local_net"] < 0:
            points -= 1
            notes.append("big-local net sell")
        status = ("Accumulation" if points >= accum_min
                  else "Distribution" if points <= distrib_max else "Neutral")
        return {"status": status, "points": points, "source": "broker",
                "foreign_net": own["foreign_net"], "reason": ", ".join(notes) or "netral"}

    # Fallback: real foreign-flow columns (skip when placeholder/NaN).
    f5 = _num(row.get("foreign_net_5d"))
    f20 = _num(row.get("foreign_net_20d"))
    if f5 is None and f20 is None:
        return {"status": "No Data", "points": 0, "source": None, "foreign_net": None,
                "reason": "data kepemilikan real tidak tersedia (perlu broker/foreign feed)"}

    points = 0
    notes = []
    for val, lbl in [(f5, "5d"), (f20, "20d")]:
        if val is None:
            continue
        if val > 0:
            points += 1
            notes.append(f"foreign {lbl} inflow")
        elif val < 0:
            points -= 1
            notes.append(f"foreign {lbl} outflow")
    status = ("Accumulation" if points >= accum_min
              else "Distribution" if points <= distrib_max else "Neutral")
    return {"status": status, "points": points, "source": "foreign_cols",
            "foreign_net": f5, "reason": ", ".join(notes) or "netral"}


# ---------------------------------------------------------------------------
# 3) Single broker absorbing supply from many sellers
# ---------------------------------------------------------------------------

def detect_single_broker_absorption(broker_df: pd.DataFrame, cfg: dict) -> dict:
    """One broker's net buy dominating supply from several distinct sellers.

    Returns {label, top_broker, top_name, dominant_share, seller_count, top_net}.
    label ∈ {Strong Accumulation, Moderate Accumulation, No Significant Accumulation, No Data}.
    """
    c = cfg.get("single_broker_absorption", {})
    min_sellers = int(c.get("min_seller_count", 3))
    moderate = float(c.get("moderate_share", 0.25))
    strong = float(c.get("strong_share", 0.40))

    empty = {"label": "No Data", "top_broker": None, "top_name": None,
             "dominant_share": None, "seller_count": 0, "top_net": None}
    if broker_df is None or broker_df.empty or "net_lot" not in broker_df.columns:
        return empty

    df = broker_df.copy()
    df["net_lot"] = pd.to_numeric(df["net_lot"], errors="coerce")
    df = df.dropna(subset=["net_lot"])
    if df.empty:
        return empty

    total_pos = df.loc[df["net_lot"] > 0, "net_lot"].sum()
    seller_count = int((df["net_lot"] < 0).sum())
    if total_pos <= 0:
        return {**empty, "label": "No Significant Accumulation", "seller_count": seller_count}

    top = df.sort_values("net_lot", ascending=False).iloc[0]
    top_net = float(top["net_lot"])
    share = top_net / total_pos if total_pos > 0 else 0.0

    if share >= strong and seller_count >= min_sellers:
        label = "Strong Accumulation"
    elif share >= moderate and seller_count >= min_sellers:
        label = "Moderate Accumulation"
    else:
        label = "No Significant Accumulation"

    return {
        "label": label,
        "top_broker": str(top.get("broker_code", "?")),
        "top_name": str(top.get("broker_name", "")),
        "dominant_share": round(share, 4),
        "seller_count": seller_count,
        "top_net": top_net,
    }


# ---------------------------------------------------------------------------
# 4) Not-yet-risen but accumulated (price depressed under MAs)
# ---------------------------------------------------------------------------

def detect_hidden_accumulation(
    row: pd.Series,
    ma5: Optional[float],
    broker_absorption_label: Optional[str],
    cfg: dict,
    own_net: Optional[dict] = None,
) -> dict:
    """Price below short & long MA (MA200 > MA5 > close) yet being accumulated.

    Accumulation signals use REAL inputs: volume ratio (full coverage) plus, where
    broker data exists, foreign / big-local net buy and single-broker absorption.

    Returns {label: Hidden|Early|Strong|None, signals, reason}.
    """
    c = cfg.get("hidden_accumulation", {})
    require_stack = bool(c.get("require_ma_stack", True))
    min_vr = float(c.get("min_vol_ratio", 1.2))
    early_n = int(c.get("early_min_signals", 1))
    hidden_n = int(c.get("hidden_min_signals", 2))
    strong_n = int(c.get("strong_min_signals", 3))

    close = _num(row.get("close"))
    ma200 = _num(row.get("ma200"))
    ma5v = _num(ma5)

    # Gate: depressed price under the MA stack.
    stacked = (
        ma200 is not None and ma5v is not None and close is not None
        and ma200 > ma5v > close
    )
    if require_stack and not stacked:
        return {"label": "None", "signals": 0, "reason": "tidak memenuhi MA200>MA5>close"}

    # Count aligned (real) accumulation signals.
    signals = 0
    notes: list[str] = []
    vr = _num(row.get("vol_ratio_20d"))
    if vr is not None and vr >= min_vr:
        signals += 1
        notes.append(f"volume {vr:.1f}x")
    if own_net is not None:
        if own_net.get("foreign_net", 0) > 0:
            signals += 1
            notes.append("foreign net buy")
        if own_net.get("big_local_net", 0) > 0:
            signals += 1
            notes.append("big-local net buy")
    if broker_absorption_label in ("Strong Accumulation", "Moderate Accumulation"):
        signals += 1
        notes.append(f"broker {broker_absorption_label.split()[0].lower()} absorb")

    if signals >= strong_n:
        label = "Strong Accumulation"
    elif signals >= hidden_n:
        label = "Hidden Accumulation"
    elif signals >= early_n:
        label = "Early Accumulation"
    else:
        label = "None"

    return {"label": label, "signals": signals,
            "reason": ("harga di bawah MA200/MA5; " + ", ".join(notes)) if label != "None"
                       else "belum ada sinyal akumulasi"}


# ---------------------------------------------------------------------------
# 5) Fundamentals — score 0-100 + grade A/B/C
# ---------------------------------------------------------------------------

def score_fundamentals(row: pd.Series, cfg: dict) -> dict:
    """Healthy-fundamentals score from growth, ROE, DER, positive EBITDA.

    Returns {score, grade, reason}. grade ∈ {A,B,C,N/A}.
    """
    c = cfg.get("fundamentals", {})
    min_roe = float(c.get("min_roe_pct", 10.0))
    max_der = float(c.get("max_der", 1.5))
    min_rev = float(c.get("min_revenue_growth_pct", 0.0))
    min_prof = float(c.get("min_profit_growth_pct", 0.0))
    need_ebitda = bool(c.get("require_positive_ebitda", True))
    grade_a = float(c.get("grade_a_min", 80))
    grade_b = float(c.get("grade_b_min", 60))

    status = str(row.get("fundamental_status", "")).lower()
    rev = _num(row.get("revenue_growth_pct"))
    prof = _num(row.get("profit_growth_pct"))
    roe = _num(row.get("roe_pct"))
    der = _num(row.get("der"))
    ebitda = _num(row.get("ebitda"))

    if status not in ("ok", "partial") and all(v is None for v in (rev, prof, roe, der)):
        return {"score": None, "grade": "N/A", "reason": "data fundamental tidak tersedia"}

    score = 0
    checks: list[str] = []
    if rev is not None and rev > min_rev:
        score += 20
        checks.append("revenue growth +")
    if prof is not None and prof > min_prof:
        score += 20
        checks.append("profit growth +")
    if roe is not None and roe >= min_roe:
        score += 20
        checks.append(f"ROE {roe:.0f}%")
    if der is not None and der <= max_der:
        score += 20
        checks.append(f"DER {der:.2f}")
    if not need_ebitda or (ebitda is not None and ebitda > 0):
        score += 20
        if ebitda is not None and ebitda > 0:
            checks.append("EBITDA +")

    grade = "A" if score >= grade_a else ("B" if score >= grade_b else "C")
    return {"score": score, "grade": grade, "reason": ", ".join(checks) or "lemah"}


# ---------------------------------------------------------------------------
# Helpers: ma5 from raw OHLCV + per-ticker broker load
# ---------------------------------------------------------------------------

def compute_ma5_map(tickers: list[str], raw_dir: Path | None = None) -> dict[str, float]:
    """Latest MA5 (5-day SMA of close) per ticker, read from raw OHLCV parquet."""
    raw_dir = raw_dir or _RAW_DIR
    out: dict[str, float] = {}
    for t in tickers:
        path = raw_dir / f"{t}.parquet"
        if not path.exists():
            continue
        try:
            close = pd.read_parquet(path, columns=["close"])["close"].dropna()
            if len(close) >= 5:
                out[t] = float(close.tail(5).mean())
        except Exception:  # noqa: BLE001
            continue
    return out


def _load_broker_for(ticker: str, date: str, broker_dir: Path) -> pd.DataFrame:
    """Load a single-day broker parquet for ticker+date, or empty."""
    clean = ticker.upper().replace(".JK", "")
    for name in (f"{clean}.JK_{date}.parquet", f"{clean}_{date}.parquet"):
        p = broker_dir / name
        if p.exists():
            try:
                return pd.read_parquet(p)
            except Exception:  # noqa: BLE001
                return pd.DataFrame()
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Combined screener
# ---------------------------------------------------------------------------

def screen_smart_money(
    df_signals: pd.DataFrame,
    scan_date: str,
    cfg: dict | None = None,
    broker_dir: Path | None = None,
    raw_dir: Path | None = None,
    ma5_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Run all detectors over the signals DataFrame; return a ranked candidate table.

    Each row gets: volume_accum, ownership, broker_absorption, hidden_accum,
    fundamental_grade/score, plus a combined smart_money_score (# of aligned
    pillars) and smart_money_label (Strong Candidate / Watch / —).
    """
    if df_signals is None or df_signals.empty:
        return pd.DataFrame()

    # Drop suspended / recently-unsuspended names so they never become candidates.
    try:
        from stock_scanner.pipeline.suspension import filter_active
        df_signals = filter_active(df_signals)
    except Exception as exc:  # noqa: BLE001
        logger.warning("suspension filter skipped in screener: {}", exc)

    cfg = cfg or load_smart_money_config()
    broker_dir = broker_dir or _BROKER_DIR
    comb = cfg.get("combined", {})
    pillars_strong = int(comb.get("pillars_strong", 4))
    pillars_watch = int(comb.get("pillars_watch", 2))
    fund_ok_grades = set(comb.get("require_fundamental_grade", ["A", "B"]))

    tickers = df_signals["ticker"].astype(str).tolist()
    if ma5_map is None:
        ma5_map = compute_ma5_map(tickers, raw_dir)

    rows: list[dict] = []
    for _, row in df_signals.iterrows():
        ticker = str(row.get("ticker", "?"))

        vol = detect_volume_accumulation(row, cfg)
        fund = score_fundamentals(row, cfg)

        broker_df = _load_broker_for(ticker, scan_date, broker_dir)
        own = detect_ownership_change(row, broker_df, cfg)
        own_net = _broker_ownership_net(broker_df)
        absorb = detect_single_broker_absorption(broker_df, cfg)

        hidden = detect_hidden_accumulation(
            row, ma5_map.get(ticker), absorb.get("label"), cfg, own_net=own_net,
        )

        # Combined pillars (smart-money confluence)
        pillars = 0
        if vol["detected"]:
            pillars += 1
        if own["status"] == "Accumulation":
            pillars += 1
        if absorb["label"] in ("Strong Accumulation", "Moderate Accumulation"):
            pillars += 1
        if hidden["label"] in ("Early Accumulation", "Hidden Accumulation", "Strong Accumulation"):
            pillars += 1
        if fund["grade"] in fund_ok_grades:
            pillars += 1

        fundamentally_ok = fund["grade"] in fund_ok_grades
        if pillars >= pillars_strong and fundamentally_ok:
            label = "🔥 Strong Candidate"
        elif pillars >= pillars_watch and fundamentally_ok:
            label = "👀 Watch"
        else:
            label = "—"

        reasons = []
        if vol["detected"]:
            reasons.append(f"Vol↑ ({vol['strength']})")
        if own["status"] != "Neutral":
            reasons.append(f"Ownership: {own['status']}")
        if absorb["label"] not in ("No Data", "No Significant Accumulation"):
            reasons.append(f"Broker: {absorb['label']} ({absorb['top_broker']})")
        if hidden["label"] != "None":
            reasons.append(f"Hidden: {hidden['label']}")

        rows.append({
            "ticker": ticker,
            "close": _num(row.get("close")),
            "signal": row.get("signal", ""),
            "smart_money_label": label,
            "smart_money_pillars": pillars,
            "volume_accum": vol["strength"] if vol["detected"] else "—",
            "vol_ratio_20d": vol["vol_ratio"],
            "roc20": vol["roc20"],
            "ownership": own["status"],
            "broker_absorption": absorb["label"],
            "absorb_broker": absorb["top_broker"],
            "absorb_share": absorb["dominant_share"],
            "hidden_accum": hidden["label"],
            "fundamental_grade": fund["grade"],
            "fundamental_score": fund["score"],
            "reasons": " · ".join(reasons) if reasons else "—",
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    # Rank: pillars desc, then fundamental score desc, then volume ratio desc.
    result = result.sort_values(
        ["smart_money_pillars", "fundamental_score", "vol_ratio_20d"],
        ascending=[False, False, False], na_position="last",
    ).reset_index(drop=True)
    return result

"""Suspension / halt exclusion — keep suspended & just-unsuspended names out of
screener, dashboard, and Telegram candidate lists.

No real-time IDX suspension feed exists here (see configs/suspension.yaml). The
safest available signals are used:
  1) manual authoritative lists (suspended / recently_unsuspended + cooldown),
  2) automatic staleness: a ticker whose latest TRADED bar is more than
     `max_staleness_trading_days` sessions behind the current market session has
     effectively stopped trading — derived for free from the per-ticker scan
     date already present in the signals data (no extra fetch).

Statuses surfaced in `suspension_status`:
  active | excluded_suspended | excluded_recent_unsuspend
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from loguru import logger

_ROOT = Path(__file__).parent.parent.parent
_CFG = _ROOT / "stock_scanner" / "configs" / "suspension.yaml"

STATUS_ACTIVE = "active"
STATUS_SUSPENDED = "excluded_suspended"
STATUS_RECENT = "excluded_recent_unsuspend"


def load_suspension_config(path: Path | None = None) -> dict:
    path = path or _CFG
    if not path.exists():
        logger.warning("suspension.yaml not found at {} — suspension filter OFF.", path)
        return {"enabled": False}
    try:
        return yaml.safe_load(open(path)) or {"enabled": False}
    except Exception as exc:  # noqa: BLE001
        logger.warning("suspension.yaml parse failed: {} — filter OFF.", exc)
        return {"enabled": False}


def _clean(t) -> str:
    return str(t).upper().replace(".JK", "").strip()


def _to_date(v) -> Optional[date]:
    try:
        d = pd.to_datetime(v)
        return None if pd.isna(d) else d.date()
    except Exception:  # noqa: BLE001
        return None


def _trading_days_between(d0: date, d1: date, cap: int | None = None) -> int:
    """Number of IDX trading sessions in the half-open interval (d0, d1].

    `cap` early-exits once the count reaches it (we only ever need to know
    whether a small threshold is exceeded — this avoids iterating years for
    long-delisted names and silences missing-holiday-year warnings).
    """
    if not d0 or not d1 or d1 <= d0:
        return 0
    import warnings
    from stock_scanner.utils.trading_calendar import is_trading_day
    n, cur = 0, d0 + timedelta(days=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        while cur <= d1:
            if is_trading_day(cur):
                n += 1
                if cap is not None and n >= cap:
                    return n
            cur += timedelta(days=1)
    return n


def classify_suspension(
    ticker: str,
    last_bar_date: Optional[date],
    market_date: Optional[date],
    cfg: dict,
) -> tuple[str, Optional[str]]:
    """Return (status, reason)."""
    if not cfg.get("enabled", False):
        return STATUS_ACTIVE, None

    code = _clean(ticker)
    manual = cfg.get("manual", {}) or {}

    # 1) Manual: currently suspended
    if code in {_clean(x) for x in (manual.get("suspended") or [])}:
        return STATUS_SUSPENDED, "manual suspended list"

    # 2) Manual: recently unsuspended within cooldown
    cooldown = int(cfg.get("unsuspend_cooldown_days", 5))
    for k, v in (manual.get("recently_unsuspended") or {}).items():
        if _clean(k) == code:
            resume = _to_date(v)
            if resume and market_date and _trading_days_between(resume, market_date, cap=cooldown + 1) < cooldown:
                return STATUS_RECENT, f"baru buka {v} (cooldown {cooldown} sesi)"

    # 3) Automatic staleness (no extra feed)
    max_stale = int(cfg.get("max_staleness_trading_days", 2))
    if last_bar_date and market_date:
        stale = _trading_days_between(last_bar_date, market_date, cap=max_stale + 1)
        if stale > max_stale:
            return STATUS_SUSPENDED, (
                f"tidak ada transaksi >{max_stale} sesi "
                f"(data terakhir {last_bar_date.isoformat()})"
            )

    return STATUS_ACTIVE, None


def annotate_suspension(
    df: pd.DataFrame,
    cfg: dict | None = None,
    market_date: Optional[date] = None,
) -> pd.DataFrame:
    """Add `suspension_status` + `suspension_reason` columns (cheap; uses df['date'])."""
    if df is None or df.empty or "ticker" not in df.columns:
        return df
    cfg = cfg or load_suspension_config()
    out = df.copy()
    if not cfg.get("enabled", False):
        out["suspension_status"] = STATUS_ACTIVE
        out["suspension_reason"] = None
        return out

    mkt = market_date
    if mkt is None and "date" in out.columns:
        mkt = _to_date(out["date"].max())

    has_date = "date" in out.columns
    statuses, reasons = [], []
    for _, r in out.iterrows():
        lbd = _to_date(r.get("date")) if has_date else None
        s, why = classify_suspension(r.get("ticker"), lbd, mkt, cfg)
        statuses.append(s)
        reasons.append(why)
    out["suspension_status"] = statuses
    out["suspension_reason"] = reasons
    return out


def filter_active(
    df: pd.DataFrame,
    cfg: dict | None = None,
    market_date: Optional[date] = None,
) -> pd.DataFrame:
    """Return only rows that are NOT suspended / recently-unsuspended."""
    if df is None or df.empty or "ticker" not in df.columns:
        return df
    ann = annotate_suspension(df, cfg, market_date)
    return ann[ann["suspension_status"] == STATUS_ACTIVE].reset_index(drop=True)


def suspended_tickers(
    df: pd.DataFrame,
    cfg: dict | None = None,
    market_date: Optional[date] = None,
) -> set[str]:
    """Set of tickers in df that are excluded (suspended/recent-unsuspend)."""
    if df is None or df.empty or "ticker" not in df.columns:
        return set()
    ann = annotate_suspension(df, cfg, market_date)
    return set(ann.loc[ann["suspension_status"] != STATUS_ACTIVE, "ticker"].astype(str))

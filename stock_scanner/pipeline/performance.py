"""Signal List performance tracking — daily archive + next-session evaluation.

A reporting/tracking layer on top of the screener (the screener logic is NOT
changed). For each trading day it:

  1. Archives the Swing and Scalping signal lists (permanent snapshot).
  2. Evaluates each signal against the NEXT valid market session's OHLC.
  3. Writes daily CSV + a ready-to-read Excel workbook, and updates the
     permanent archive / results CSVs.

Convention (per product spec)
-----------------------------
  Signal date     = the session the ticker appeared in the Swing/Scalping list
  Evaluation date = next valid trading session after the signal date
  Entry  (Open)   = evaluation-date open
  High            = evaluation-date high
  Close           = evaluation-date close
  Percentage High = (High  - Open) / Open * 100
  Percentage Close= (Close - Open) / Open * 100
  W/L             = "W" if Close > Open else "L"
  Win Rate        = #W / #evaluated * 100

If the next session's OHLC is not available yet, the record is kept as
status="pending" (NOT faked) and filled on the next run.

Strategy membership
  Swing    : signal ∈ {BREAKOUT, PRE_MARKUP}
  Scalping : scalping_label ∈ {SCALPING_HIGH}

Outputs (all under data/performance/)
  signals_archive.csv                 — permanent signal snapshots (append/dedup)
  signal_results.csv                  — permanent evaluated results (upsert)
  daily/swing_YYYY-MM-DD.csv           — Swing signals for that signal date
  daily/scalping_YYYY-MM-DD.csv        — Scalping signals for that signal date
  daily/signal_list_YYYY-MM-DD.xlsx    — Swing + Scalping sheets + win-rate
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

_ROOT = Path(__file__).parent.parent.parent
_SIGNALS_DIR = _ROOT / "data" / "signals"
_RAW_DIR = _ROOT / "data" / "raw"
_PERF_DIR = _ROOT / "data" / "performance"
_DAILY_DIR = _PERF_DIR / "daily"

_ARCHIVE_CSV = _PERF_DIR / "signals_archive.csv"
_RESULTS_CSV = _PERF_DIR / "signal_results.csv"

SWING_SIGNALS = {"BREAKOUT", "PRE_MARKUP"}
SCALPING_LABELS = {"SCALPING_HIGH"}

# Column order for the daily table. Reference price = PREVIOUS CLOSE (the close
# of the signal session); High/Close are the next session's, measured vs Prev.
_RESULT_COLS = [
    "signal_date", "eval_date", "strategy", "ticker", "signal",
    "prev", "close", "high", "pct_high", "pct_close", "wl", "status",
]


# ---------------------------------------------------------------------------
# Signal-list extraction
# ---------------------------------------------------------------------------

def _signal_list(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Return [ticker, signal] for one strategy from a signals frame."""
    if df is None or df.empty or "ticker" not in df.columns:
        return pd.DataFrame(columns=["ticker", "signal"])
    if strategy == "swing":
        if "signal" not in df.columns:
            return pd.DataFrame(columns=["ticker", "signal"])
        sub = df[df["signal"].astype(str).isin(SWING_SIGNALS)]
        out = sub[["ticker", "signal"]].copy()
    else:  # scalping
        if "scalping_label" not in df.columns:
            return pd.DataFrame(columns=["ticker", "signal"])
        sub = df[df["scalping_label"].astype(str).isin(SCALPING_LABELS)]
        out = sub[["ticker"]].copy()
        out["signal"] = sub["scalping_label"].astype(str).values
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Next-session OHLC lookup (never fabricates)
# ---------------------------------------------------------------------------

def _next_session_ohlc(ticker: str, signal_date: str, raw_dir: Path) -> Optional[dict]:
    """Previous-close reference + next valid session's high/close for `ticker`.

    Returns {prev_close, eval_date, high, close} or None when not available yet
    (caller marks the record pending). `prev_close` is the close of the session
    immediately BEFORE the evaluation session (i.e. the signal-session close).
    Skips zero-volume / NaN bars.
    """
    path = raw_dir / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path, columns=["date", "open", "high", "low", "close", "volume"])
    except Exception:  # noqa: BLE001
        try:
            df = pd.read_parquet(path)
        except Exception:  # noqa: BLE001
            return None
    if df.empty or "date" not in df.columns:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)
    if "volume" in df.columns:
        df = df[pd.to_numeric(df["volume"], errors="coerce").fillna(0) > 0].reset_index(drop=True)
    sig_ts = pd.Timestamp(signal_date)
    fwd_idx = df.index[df["date"] > sig_ts]
    for i in fwd_idx:
        if i == 0:  # no prior session → no previous close
            continue
        ev = df.loc[i]
        prev_close = df.loc[i - 1, "close"]
        h, c = ev.get("high"), ev.get("close")
        if (pd.notna(prev_close) and float(prev_close) > 0
                and pd.notna(h) and pd.notna(c)):
            return {
                "prev_close": float(prev_close),
                "eval_date": ev["date"].strftime("%Y-%m-%d"),
                "high": float(h), "close": float(c),
            }
    return None


def _evaluate_row(signal_date: str, strategy: str, ticker: str, signal: str,
                  raw_dir: Path) -> dict:
    """Build one result row — evaluated or pending. Reference = previous close."""
    base = {"signal_date": signal_date, "strategy": strategy, "ticker": ticker,
            "signal": signal, "eval_date": None, "prev": None, "close": None,
            "high": None, "pct_high": None, "pct_close": None, "wl": None,
            "status": "pending"}
    ohlc = _next_session_ohlc(ticker, signal_date, raw_dir)
    if ohlc is None:
        return base
    p, h, c = ohlc["prev_close"], ohlc["high"], ohlc["close"]
    base.update({
        "eval_date": ohlc["eval_date"],
        "prev": round(p, 2), "high": round(h, 2), "close": round(c, 2),
        "pct_high": round((h - p) / p * 100, 2),
        "pct_close": round((c - p) / p * 100, 2),
        "wl": "W" if c > p else "L",
        "status": "evaluated",
    })
    return base


# ---------------------------------------------------------------------------
# Persistent archive / results helpers (idempotent upserts)
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:  # noqa: BLE001
            return pd.DataFrame()
    return pd.DataFrame()


def _upsert(existing: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new.reset_index(drop=True)
    if new is None or new.empty:
        return existing.reset_index(drop=True)
    for k in keys:
        if k not in existing.columns:
            existing[k] = None
    merged = pd.concat([existing, new], ignore_index=True)
    merged = merged.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    return merged


# ---------------------------------------------------------------------------
# Excel writer
# ---------------------------------------------------------------------------

def _write_excel(path: Path, swing: pd.DataFrame, scalping: pd.DataFrame, signal_date: str) -> None:
    """Ready-to-read workbook: a sheet per strategy, % formatting, W/L colour,
    win-rate summary at the top. Best-effort (skipped if openpyxl missing)."""
    try:
        from openpyxl.styles import Font, PatternFill, Alignment
    except Exception as exc:  # noqa: BLE001
        logger.warning("openpyxl not available — skipping Excel ({}).", exc)
        return

    disp_cols = ["ticker", "signal", "prev", "close", "high",
                 "pct_high", "pct_close", "wl", "eval_date", "status"]
    headers = ["Signal", "Type", "Prev", "Close", "High",
               "Percentage High", "Percentage Close", "W/L", "Eval Date", "Status"]

    win_fill = PatternFill("solid", fgColor="C6EFCE")
    loss_fill = PatternFill("solid", fgColor="FFC7CE")
    hdr_fill = PatternFill("solid", fgColor="1F2937")
    hdr_font = Font(color="FFFFFF", bold=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for sheet, df in (("Swing", swing), ("Scalping", scalping)):
            ev = df[df["status"] == "evaluated"] if not df.empty else df
            n = len(ev)
            wins = int((ev["wl"] == "W").sum()) if n else 0
            wr = round(wins / n * 100, 1) if n else 0.0
            pend = int((df["status"] == "pending").sum()) if not df.empty else 0

            ws = xw.book.create_sheet(sheet)
            ws["A1"] = f"{sheet} Signal List — {signal_date}"
            ws["A1"].font = Font(bold=True, size=13)
            ws["A2"] = (f"Win Rate: {wr}%  ({wins}W / {n - wins}L of {n} evaluated"
                        + (f", {pend} pending)" if pend else ")"))
            ws["A2"].font = Font(bold=True, color="2563EB")

            start = 4
            for j, h in enumerate(headers, 1):
                cell = ws.cell(row=start, column=j, value=h)
                cell.fill = hdr_fill
                cell.font = hdr_font
                cell.alignment = Alignment(horizontal="center")

            if df.empty:
                ws.cell(row=start + 1, column=1, value="(tidak ada sinyal)")
            else:
                for i, (_, r) in enumerate(df.iterrows(), start=start + 1):
                    for j, col in enumerate(disp_cols, 1):
                        ws.cell(row=i, column=j, value=r.get(col))
                    # % formatting
                    for jc in (6, 7):  # pct_high, pct_close columns
                        ws.cell(row=i, column=jc).number_format = '0.00"%"'
                    # W/L colour
                    wl = r.get("wl")
                    if wl == "W":
                        ws.cell(row=i, column=8).fill = win_fill
                    elif wl == "L":
                        ws.cell(row=i, column=8).fill = loss_fill
            # widths
            for j, w in enumerate([12, 13, 10, 10, 10, 15, 16, 6, 12, 11], 1):
                ws.column_dimensions[chr(64 + j)].width = w

        # openpyxl always creates a default "Sheet" — remove it.
        if "Sheet" in xw.book.sheetnames:
            del xw.book["Sheet"]


# ---------------------------------------------------------------------------
# Per-date processing + orchestrator
# ---------------------------------------------------------------------------

def process_date(signal_date: str, signals_dir: Path | None = None,
                 raw_dir: Path | None = None, perf_dir: Path | None = None,
                 exclude_suspended: bool = True) -> dict:
    """Archive + evaluate one signal date. Returns per-strategy summary."""
    signals_dir = signals_dir or _SIGNALS_DIR
    raw_dir = raw_dir or _RAW_DIR
    perf_dir = perf_dir or _PERF_DIR
    daily_dir = perf_dir / "daily"

    f = signals_dir / f"{signal_date}.parquet"
    if not f.exists():
        f = signals_dir / f"{signal_date}.csv"
    if not f.exists():
        return {}
    df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)

    if exclude_suspended:
        try:
            from stock_scanner.pipeline.suspension import filter_active
            df = filter_active(df)
        except Exception:  # noqa: BLE001
            pass

    archive_rows, result_rows, per_strategy = [], [], {}
    daily_dir.mkdir(parents=True, exist_ok=True)

    for strategy in ("swing", "scalping"):
        lst = _signal_list(df, strategy)
        rows = [
            _evaluate_row(signal_date, strategy, str(r["ticker"]), str(r["signal"]), raw_dir)
            for _, r in lst.iterrows()
        ]
        sdf = pd.DataFrame(rows, columns=_RESULT_COLS) if rows else pd.DataFrame(columns=_RESULT_COLS)

        # daily per-strategy CSV (signal-date keyed)
        sdf.to_csv(daily_dir / f"{strategy}_{signal_date}.csv", index=False)

        for _, r in lst.iterrows():
            archive_rows.append({"signal_date": signal_date, "strategy": strategy,
                                 "ticker": str(r["ticker"]), "signal": str(r["signal"])})
        result_rows.extend(rows)

        ev = sdf[sdf["status"] == "evaluated"]
        n, wins = len(ev), int((ev["wl"] == "W").sum()) if len(ev) else 0
        per_strategy[strategy] = {
            "signals": len(sdf), "evaluated": n, "wins": wins, "losses": n - wins,
            "pending": int((sdf["status"] == "pending").sum()),
            "win_rate": round(wins / n * 100, 1) if n else None,
            "df": sdf,
        }

    # Permanent stores (upsert)
    _ARCHIVE_CSV.parent.mkdir(parents=True, exist_ok=True)
    _upsert(_read_csv(_ARCHIVE_CSV), pd.DataFrame(archive_rows),
            ["signal_date", "strategy", "ticker"]).to_csv(_ARCHIVE_CSV, index=False)
    _upsert(_read_csv(_RESULTS_CSV), pd.DataFrame(result_rows, columns=_RESULT_COLS),
            ["signal_date", "strategy", "ticker"]).to_csv(_RESULTS_CSV, index=False)

    # Daily Excel (both sheets)
    _write_excel(daily_dir / f"signal_list_{signal_date}.xlsx",
                 per_strategy["swing"]["df"], per_strategy["scalping"]["df"], signal_date)

    return per_strategy


def run_performance(signals_dir: Path | None = None, raw_dir: Path | None = None,
                    perf_dir: Path | None = None, dates: list[str] | None = None) -> dict:
    """Archive + (re)evaluate. Default: every available signal date (so pending
    rows get filled once their next session exists). Returns {date: summary}."""
    signals_dir = signals_dir or _SIGNALS_DIR
    if dates is None:
        dates = sorted(p.stem for p in signals_dir.glob("*.parquet"))
        dates += [p.stem for p in signals_dir.glob("*.csv") if p.stem not in dates]
        dates = sorted(set(dates))
    out = {}
    for d in dates:
        try:
            res = process_date(d, signals_dir, raw_dir, perf_dir)
            if res:
                out[d] = res
        except Exception as exc:  # noqa: BLE001
            logger.warning("performance: failed on {}: {}", d, exc)
    if out:
        last = sorted(out)[-1]
        sw, sc = out[last].get("swing", {}), out[last].get("scalping", {})
        logger.info("Performance updated thru {} | swing {}sig/{}eval, scalping {}sig/{}eval",
                    last, sw.get("signals"), sw.get("evaluated"),
                    sc.get("signals"), sc.get("evaluated"))
    return out


# ---------------------------------------------------------------------------
# Read helpers for dashboard / Telegram
# ---------------------------------------------------------------------------

def load_results(perf_dir: Path | None = None) -> pd.DataFrame:
    perf_dir = perf_dir or _PERF_DIR
    return _read_csv(perf_dir / "signal_results.csv")


# ---------------------------------------------------------------------------
# Evening (18:00 WIB) entrypoint — fetch same-day OHLC, then evaluate
# ---------------------------------------------------------------------------

def _pending_tickers(perf_dir: Path | None = None) -> set[str]:
    df = load_results(perf_dir)
    if df.empty or "status" not in df.columns:
        return set()
    return set(df.loc[df["status"] == "pending", "ticker"].astype(str))


def _refresh_raw_for(tickers: list[str], raw_dir: Path, lookback_years: int = 1) -> int:
    """Fetch the latest OHLC for specific tickers, INCLUDING today's just-closed
    session (end = today+1, since yfinance's end is exclusive and the standard
    incremental_update uses end=today which omits the current day). Run after
    market close so today's bar is final. Reuses fetch_yfinance helpers; does not
    touch the screener. Returns count attempted. Best-effort per ticker.
    """
    if not tickers:
        return 0
    from datetime import datetime, timedelta
    from stock_scanner.pipeline.fetch_yfinance import (
        YFinanceFetcher, load_raw, save_raw, default_start_date,
    )
    fetcher = YFinanceFetcher(batch_size=20)
    end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")  # inclusive of today
    n = 0
    for t in tickers:
        try:
            existing = load_raw(t, raw_dir)
            if existing.empty:
                start = default_start_date(lookback_years)
            else:
                last = pd.to_datetime(existing["date"]).max()
                start = (last + timedelta(days=1)).strftime("%Y-%m-%d")
            if start >= end:
                n += 1
                continue
            new = fetcher.fetch_single(t, start, end)
            if new is None or new.empty:
                n += 1
                continue
            combined = pd.concat([existing, new], ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None).dt.normalize()
            combined = (combined.drop_duplicates(subset=["date", "ticker"])
                        .sort_values("date").reset_index(drop=True))
            save_raw(t, combined, raw_dir)
            n += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("refresh_raw {}: {}", t, exc)
    return n


def run_performance_evening(signals_dir: Path | None = None, raw_dir: Path | None = None,
                            perf_dir: Path | None = None, lookback_years: int = 1) -> dict:
    """18:00 WIB entrypoint. Refreshes OHLC for the tickers awaiting evaluation
    (pending results + the latest signal date's lists), so that if the market
    data source has already published today's close, pending rows are filled on
    THIS run instead of waiting for the next morning. Then runs the standard
    performance pass. Does NOT touch the screener.
    """
    signals_dir = signals_dir or _SIGNALS_DIR
    raw_dir = raw_dir or _RAW_DIR
    perf_dir = perf_dir or _PERF_DIR

    need = _pending_tickers(perf_dir)
    files = sorted(signals_dir.glob("*.parquet"))
    if files:
        try:
            df = pd.read_parquet(files[-1])
            for strat in ("swing", "scalping"):
                need |= set(_signal_list(df, strat)["ticker"].astype(str))
        except Exception:  # noqa: BLE001
            pass

    refreshed = _refresh_raw_for(sorted(need), raw_dir, lookback_years)
    logger.info("Evening performance: refreshed OHLC for {} ticker(s) awaiting eval.", refreshed)
    return run_performance(signals_dir, raw_dir, perf_dir)


def win_rate_recap(signal_date: str, perf_dir: Path | None = None) -> dict:
    """Per-strategy win-rate summary for one signal date (for Telegram/dashboard)."""
    df = load_results(perf_dir)
    out = {}
    if df.empty:
        return out
    for strat in ("swing", "scalping"):
        sub = df[(df["signal_date"] == signal_date) & (df["strategy"] == strat)
                 & (df["status"] == "evaluated")]
        n = len(sub)
        wins = int((sub["wl"] == "W").sum()) if n else 0
        out[strat] = {"signals": n, "wins": wins, "losses": n - wins,
                      "win_rate": round(wins / n * 100, 1) if n else None}
    return out


def latest_evaluated_date(perf_dir: Path | None = None) -> Optional[str]:
    df = load_results(perf_dir)
    if df.empty:
        return None
    ev = df[df["status"] == "evaluated"]
    return None if ev.empty else str(ev["signal_date"].max())


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Signal List performance tracker")
    p.add_argument("--date", default=None, help="Only this signal date (YYYY-MM-DD).")
    p.add_argument("--evening", action="store_true",
                   help="18:00 WIB mode: fetch same-day OHLC for pending tickers, then evaluate.")
    a = p.parse_args()
    if a.evening:
        run_performance_evening()
    else:
        run_performance(dates=[a.date] if a.date else None)

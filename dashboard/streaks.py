"""Consecutive up/down streak screener with a trend-state filter.

Finds stocks that closed higher ("naik") or lower ("turun") for several
consecutive trading days, restricted to names whose broader structure is
bullish or sideways — bearish names are excluded. OHLC comes from the committed
recent bundle (data/published/ohlc_recent.parquet) so the scan behaves
identically locally and on Streamlit Cloud.

────────────────────────────────────────────────────────────────────────────
Streak logic (close-to-close direction)
────────────────────────────────────────────────────────────────────────────
Per ticker, take the daily closes sorted oldest→newest and form the sign of each
close-to-close move:
    up   move : close[t] >  close[t-1]
    down move : close[t] <  close[t-1]
    flat      : close[t] == close[t-1]   (breaks any streak)
The *current* streak is the run of identical-sign moves ending at the latest
session. ``streak_len`` = how many consecutive same-direction days; ``streak_dir``
= "up" / "down". So "naik 3 hari berturut-turut" == an up streak of length ≥ 3,
"turun 5 hari" == a down streak of length ≥ 5. (A 6-day up streak therefore also
satisfies the ≥3 and ≥5 'naik' buckets — it genuinely has risen 3 and 5 days.)

────────────────────────────────────────────────────────────────────────────
Trend-state ruleset (explicit; documented)
────────────────────────────────────────────────────────────────────────────
From the close series we derive SMA20, SMA50, SMA200 and the 10-session percent
change of SMA50 (``ma50_slope``, in %). With c=last close:

  BEARISH (excluded) if ANY of — strict enough to drop even weak downtrends:
    • c < SMA50  AND  SMA20 < SMA50            # price below the medium MA and
                                               #   short-term MA below it (downtrend
                                               #   structure)
    • SMA50 < SMA200  AND  c < SMA200          # long-term down (death-cross side)
    • ma50_slope < -0.3                        # SMA50 falling over ~2 weeks — this
                                               #   catches WEAK downtrends too

  BULLISH if NOT bearish AND:
    • c ≥ SMA50  AND  SMA20 ≥ SMA50            # price holds above a rising stack
    • ma50_slope > -0.1                        # SMA50 not meaningfully falling
    • c ≥ SMA200  OR  SMA50 ≥ SMA200           # long-term not broken

  SIDEWAYS otherwise — NOT bearish and NOT bullish: a flat/consolidating range
  that is explicitly not in a confirmed (or weak) downtrend.

Only BULLISH and SIDEWAYS rows are returned; BEARISH is dropped.
"""
from __future__ import annotations

import pandas as pd

# Tunable thresholds for the trend classifier (percent change of SMA50 over the
# last 10 sessions). Kept here so the ruleset is easy to find and adjust.
_SLOPE_BEARISH = -0.3   # SMA50 falling at least this much (%/10 sessions) → bearish
_SLOPE_BULLISH = -0.1   # bullish requires SMA50 not falling beyond this
_MIN_SESSIONS = 60      # need at least this much history to classify a trend


def current_streak(closes: list[float]) -> tuple[str, int]:
    """Return (direction, length) of the close-to-close streak ending at the last
    close. direction ∈ {"up","down","flat"}; flat (or <2 closes) → length 0."""
    if len(closes) < 2:
        return "flat", 0
    moves = []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        moves.append(1 if d > 0 else (-1 if d < 0 else 0))
    last = moves[-1]
    if last == 0:
        return "flat", 0
    n = 0
    for m in reversed(moves):
        if m == last:
            n += 1
        else:
            break
    return ("up" if last > 0 else "down"), n


def trend_state(closes: pd.Series) -> tuple[str, str]:
    """Classify the broader structure. Returns (state, reason) where state ∈
    {"bullish","sideways","bearish"} per the documented ruleset above."""
    c = pd.to_numeric(closes, errors="coerce").dropna()
    if len(c) < _MIN_SESSIONS:
        return "sideways", "histori terbatas — diperlakukan sideways"

    close = float(c.iloc[-1])
    ma20 = float(c.rolling(20).mean().iloc[-1])
    ma50 = float(c.rolling(50).mean().iloc[-1])
    ma200_win = 200 if len(c) >= 200 else len(c)
    ma200 = float(c.rolling(ma200_win).mean().iloc[-1])

    ma50_s = c.rolling(50).mean().dropna()
    if len(ma50_s) >= 11 and ma50_s.iloc[-11] > 0:
        slope = (ma50_s.iloc[-1] - ma50_s.iloc[-11]) / ma50_s.iloc[-11] * 100.0
    else:
        slope = 0.0

    bearish = (
        (close < ma50 and ma20 < ma50)
        or (ma50 < ma200 and close < ma200)
        or (slope < _SLOPE_BEARISH)
    )
    if bearish:
        if slope < _SLOPE_BEARISH and not (close < ma50 and ma20 < ma50):
            why = f"MA50 menurun ({slope:+.1f}% / 10 sesi)"
        elif ma50 < ma200 and close < ma200:
            why = "harga & MA50 di bawah MA200 (tren turun jangka panjang)"
        else:
            why = "harga di bawah MA50 dengan MA20<MA50 (struktur turun)"
        return "bearish", why

    bullish = (
        close >= ma50 and ma20 >= ma50
        and slope > _SLOPE_BULLISH
        and (close >= ma200 or ma50 >= ma200)
    )
    if bullish:
        return "bullish", (f"harga di atas MA50 & MA20≥MA50, MA50 naik "
                           f"({slope:+.1f}% / 10 sesi)")
    return "sideways", (f"konsolidasi — bukan tren turun (MA50 {slope:+.1f}% / 10 sesi, "
                        f"harga {'≥' if close >= ma50 else '<'} MA50)")


def screen_streaks(min_len: int = 3) -> pd.DataFrame:
    """Scan the committed OHLC bundle for current up/down streaks of length
    ≥ ``min_len`` whose trend state is bullish or sideways (bearish excluded).

    Returns a DataFrame: ticker, last_close, last_date, streak_dir, streak_len,
    trend_state, reason — sorted by streak_len descending.
    """
    from dashboard.data_loader import _get_ohlc_bundle

    bundle = _get_ohlc_bundle()
    if bundle is None or bundle.empty or "ticker" not in bundle.columns:
        return pd.DataFrame(columns=["ticker", "last_close", "last_date",
                                     "streak_dir", "streak_len", "trend_state", "reason"])

    rows = []
    for tk, g in bundle.groupby("ticker"):
        g = g.sort_values("date")
        closes = pd.to_numeric(g["close"], errors="coerce").dropna().tolist()
        if len(closes) < 2:
            continue
        sdir, slen = current_streak(closes)
        if sdir == "flat" or slen < min_len:
            continue
        state, reason = trend_state(g["close"])
        if state == "bearish":
            continue  # exclude bearish — strict downtrend filter
        rows.append({
            "ticker": str(tk),
            "last_close": float(closes[-1]),
            "last_date": str(pd.to_datetime(g["date"]).max().date()),
            "streak_dir": sdir,
            "streak_len": int(slen),
            "trend_state": state,
            "reason": reason,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["streak_len", "ticker"], ascending=[False, True]).reset_index(drop=True)
    return df

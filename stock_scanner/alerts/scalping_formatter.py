"""Scalping alert formatter — levels + Telegram messages for SCALPING_HIGH.

Design principles:
- Levels are deterministic (ATR-based), not AI-generated.
- Only SCALPING_HIGH rows get active levels; others get trade_setup_status=inactive.
- Message format: compact, actionable, one ticker per block.

Public API
----------
compute_scalping_levels(row) -> dict
    Compute entry/TP/cutloss for one row. Returns inactive dict for non-SCALPING_HIGH.

format_scalping_alert(rows, scan_date) -> list[str]
    Format a list of signal dicts (SCALPING_HIGH filtered) into Telegram messages.
    Returns list of strings, each ≤ 4000 chars.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from stock_scanner.alerts.level_calculator import get_tick_size, round_to_tick

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SAFE_LIMIT = 4000       # chars per Telegram message
_SCALPING_HIGH = "SCALPING_HIGH"

_INACTIVE = {
    "entry_low":          0,
    "entry_high":         0,
    "tp_low":             0,
    "tp_high":            0,
    "cutloss":            0,
    "trade_setup_status": "inactive",
}

_STOCKBIT_URL = "https://stockbit.com/symbol/{code}"


# ---------------------------------------------------------------------------
# Scalping level calculator
# ---------------------------------------------------------------------------

def compute_scalping_levels(row: dict | pd.Series) -> dict:
    """Compute entry / TP / cutloss for scalping (short horizon).

    Logic (for SCALPING_HIGH only):
        Entry range : close ± 0.2×ATR (tight, immediate action)
        TP          : entry_low + 1.5×ATR  (short scalping target, ~1-2 day)
        Cutloss     : close - 0.5×ATR      (fast invalidation)

    If ATR is unavailable, fall back to % of close:
        ATR fallback : 1.5% of close

    For sub-Rp100 stocks (very illiquid/volatile):
        Widen entry range to ±0.3×ATR.

    Returns:
        dict with keys: entry_low, entry_high, tp_low, tp_high, cutloss,
                        trade_setup_status, rr_ratio
    """
    label = str(row.get("scalping_label", "")).upper()
    if label != _SCALPING_HIGH:
        return dict(_INACTIVE)

    close = _sf(row.get("close"))
    if not close or close <= 0:
        return dict(_INACTIVE)

    tick = get_tick_size(close)

    # ATR: prefer atr14, fallback to atr_pct*close, then 1.5% of close
    atr = _sf(row.get("atr14")) or 0.0
    if atr <= 0:
        atr_pct_val = _sf(row.get("atr_pct")) or 0.0
        atr = close * atr_pct_val / 100 if atr_pct_val > 0 else close * 0.015

    # For very cheap stocks, widen range slightly
    entry_buffer = 0.3 if close < 100 else 0.2

    # Entry range: close ± entry_buffer × ATR
    entry_low_raw  = close - atr * entry_buffer
    entry_high_raw = close + atr * entry_buffer

    # Cutloss: close - 0.5×ATR, floor at -3% from close
    cutloss_raw = max(close - atr * 0.5, close * 0.97)

    # TP: entry_low + 1.5×ATR (scalping horizon — short, realistic)
    risk = entry_low_raw - cutloss_raw
    if risk <= 0:
        risk = close * 0.012   # fallback 1.2% risk
    tp_low_raw  = entry_high_raw + risk * 1.5
    tp_high_raw = entry_high_raw + risk * 2.5

    # Round to tick
    el = round_to_tick(entry_low_raw,  tick)
    eh = round_to_tick(entry_high_raw, tick)
    cl = round_to_tick(cutloss_raw,    tick, down=True)
    tl = round_to_tick(tp_low_raw,     tick)
    th = round_to_tick(tp_high_raw,    tick)

    # R:R check
    status = "active"
    if el > cl > 0 and tl > el:
        rr = (tl - el) / (el - cl)
        if rr < 1.2:
            status = "low_rr"
    else:
        rr = 0.0

    return {
        "entry_low":          el,
        "entry_high":         eh,
        "tp_low":             tl,
        "tp_high":            th,
        "cutloss":            cl,
        "trade_setup_status": status,
        "rr_ratio":           round(rr, 2),
    }


# ---------------------------------------------------------------------------
# Alert formatter
# ---------------------------------------------------------------------------

def format_scalping_alert(
    rows: list[dict],
    scan_date: str = "",
) -> list[str]:
    """Format SCALPING_HIGH rows into Telegram-ready HTML message(s).

    Filters rows to SCALPING_HIGH only.
    Splits into multiple messages if total length > _SAFE_LIMIT.

    Args:
        rows      : List of signal dicts (output of signals_df_to_list).
                    Each dict must have: ticker, scalping_label, scalping_score,
                    scalping_reason, close, atr14/atr_pct, rsi14, vol_ratio_20d.
        scan_date : YYYY-MM-DD string for the header.

    Returns:
        List of HTML strings, each ≤ 4000 chars. Empty list if no SCALPING_HIGH.
    """
    # Filter to SCALPING_HIGH only
    high_rows = [r for r in rows if str(r.get("scalping_label", "")).upper() == _SCALPING_HIGH]
    if not high_rows:
        logger.info("Scalping: no SCALPING_HIGH candidates — alert skipped.")
        return []

    # Sort by scalping_score desc
    high_rows.sort(key=lambda r: _sf(r.get("scalping_score")) or 0.0, reverse=True)

    date_label = _format_date_id(scan_date)
    now_str    = datetime.now().strftime("%H:%M WIB")

    header_lines = [
        f"⚡ <b>SCALPING HIGH</b>  —  {date_label}",
        "─────────────────────",
        f"<i>{len(high_rows)} kandidat momentum kuat hari ini</i>",
        "",
    ]
    footer_lines = [
        "",
        "─────────────────────",
        "<i>⚠️ Scalping = risiko tinggi. Selalu pasang cutloss.</i>",
        f"<i>📡 {now_str}  ·  🤖 IDX Scanner Agent</i>",
    ]

    # Build per-ticker blocks
    ticker_blocks: list[str] = []
    for i, row in enumerate(high_rows, 1):
        block = _build_scalping_block(i, row)
        ticker_blocks.append(block)

    # Assemble and split
    messages = _pack_into_messages(
        header_lines=header_lines,
        footer_lines=footer_lines,
        blocks=ticker_blocks,
        limit=_SAFE_LIMIT,
    )

    logger.info("Scalping alert: %d SCALPING_HIGH → %d message(s)", len(high_rows), len(messages))
    return messages


# ---------------------------------------------------------------------------
# Internal builders
# ---------------------------------------------------------------------------

def _build_scalping_block(idx: int, row: dict) -> str:
    """Build HTML block for one SCALPING_HIGH ticker."""
    ticker  = _clean(row.get("ticker", "?"))
    score   = _sf(row.get("scalping_score")) or 0.0
    reason  = str(row.get("scalping_reason", "")) or "—"
    close   = _sf(row.get("close"))
    rsi     = _sf(row.get("rsi14"))
    vol     = _sf(row.get("vol_ratio_20d"))
    signal  = str(row.get("signal", "")).upper()

    url = _STOCKBIT_URL.format(code=ticker)

    # Compute levels
    levels = compute_scalping_levels(row)
    el  = levels["entry_low"]
    eh  = levels["entry_high"]
    tl  = levels["tp_low"]
    th  = levels["tp_high"]
    cl  = levels["cutloss"]
    status = levels["trade_setup_status"]
    rr  = levels.get("rr_ratio", 0.0)

    sig_note = f" [{signal}]" if signal and signal not in ("NONE", "") else ""

    lines: list[str] = []
    lines.append(f'{idx}. <a href="{url}"><b>${ticker}</b></a>  Score <code>{score:.1f}</code>{sig_note}')

    if close:
        lines.append(f"   Harga     : <code>Rp{int(close):,}</code>")

    if status == "active":
        if el != eh:
            lines.append(f"   Entry     : <code>{el:,} – {eh:,}</code>")
        else:
            lines.append(f"   Entry     : <code>{el:,}</code>")

        if tl != th:
            lines.append(f"   TP        : <code>{tl:,} – {th:,}</code>")
        else:
            lines.append(f"   TP        : <code>{tl:,}</code>")

        lines.append(f"   Cutloss   : <code>{cl:,}</code>")
        if rr >= 1.0:
            lines.append(f"   <i>R:R ≈ 1:{rr:.1f}</i>")
    elif status == "low_rr":
        lines.append(f"   <i>⚠️ Level ada tapi R:R kurang ideal — sizing kecil</i>")
        if el:
            lines.append(f"   Entry~CL  : <code>{el:,} / {cl:,}</code>  TP~<code>{tl:,}</code>")
    else:
        lines.append(f"   <i>Level tidak dihitung (data kurang)</i>")

    # Reason line — shorten if too long
    short_reason = reason[:80] + "…" if len(reason) > 80 else reason
    lines.append(f"   Alasan    : <i>{short_reason}</i>")

    # Quick stats
    stats: list[str] = []
    if rsi:
        stats.append(f"RSI {rsi:.0f}")
    if vol and vol > 0:
        stats.append(f"Vol×{vol:.1f}")
    if stats:
        lines.append(f"   <i>{'  ·  '.join(stats)}</i>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Message packer (smart split)
# ---------------------------------------------------------------------------

def _pack_into_messages(
    header_lines: list[str],
    footer_lines: list[str],
    blocks: list[str],
    limit: int = _SAFE_LIMIT,
) -> list[str]:
    """Pack header + blocks + footer into ≤limit-char Telegram messages.

    Each message starts with the header (abbreviated for continuations)
    and ends with the footer.
    """
    header = "\n".join(header_lines)
    footer = "\n".join(footer_lines)
    separator = "\n\n"

    messages: list[str] = []
    current_parts: list[str] = [header]
    current_len = len(header) + len(footer) + len(separator)

    for block in blocks:
        block_len = len(block) + len(separator)
        if current_len + block_len > limit and len(current_parts) > 1:
            # Flush current
            msg = separator.join(current_parts) + "\n" + footer
            messages.append(msg)
            # Start new chunk with continuation header
            cont_hdr = f"⚡ <b>SCALPING HIGH</b> (sambungan)"
            current_parts = [cont_hdr, block]
            current_len = len(cont_hdr) + block_len + len(footer) + len(separator)
        else:
            current_parts.append(block)
            current_len += block_len

    # Flush last chunk
    if len(current_parts) > 1:
        msg = separator.join(current_parts) + "\n" + footer
        messages.append(msg)
    elif not messages:
        # Nothing was written (shouldn't happen if blocks non-empty)
        msg = header + "\n<i>Tidak ada kandidat.</i>" + "\n" + footer
        messages.append(msg)

    return messages


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sf(val: Any) -> float | None:
    try:
        v = float(val)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _clean(ticker: str) -> str:
    return str(ticker).replace(".JK", "").replace(".jk", "").strip()


def _format_date_id(date_str: str) -> str:
    months = ["", "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
              "Jul", "Ags", "Sep", "Okt", "Nov", "Des"]
    days   = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    try:
        from datetime import datetime as _dt
        dt = _dt.strptime(date_str, "%Y-%m-%d")
        return f"{days[dt.weekday()]}, {dt.day} {months[dt.month]} {dt.year}"
    except (ValueError, IndexError):
        return date_str

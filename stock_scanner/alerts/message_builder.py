"""Telegram message builder for IDX daily alerts.

Two alert formats:
  1. Deep Dive  — full analysis for 2 top BREAKOUT candidates (DEWA-style).
  2. Top Picks  — compact trading levels for 7 best candidates (DPUM-style).

Public API
----------
signals_df_to_list(df, scan_date) -> list[dict]
    Convert signals DataFrame rows to list of dicts (adapter layer).

select_breakout_deep_dive_candidates(signals, top_n=2) -> list[dict]
    Pick the best BREAKOUT / PRE_MARKUP tickers for deep dive.

select_top_picks(signals, top_n=7) -> list[dict]
    Pick the 7 best non-AVOID tickers for compact level listing.

build_telegram_deep_dive_message(signal, articles=None) -> str
    Full DEWA-style analysis message in HTML.

build_all_breakout_deep_dive_messages(signals, articles_by_ticker=None) -> list[str]
    One deep-dive message per candidate.

build_telegram_top_picks_message(signals, date) -> str
    Compact DPUM/DFAM-style list with entry/TP/CL levels.

split_long_message(text, limit=4000) -> list[str]
    Split a message at paragraph boundaries to respect Telegram's 4096-char cap.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from stock_scanner.alerts.level_calculator import compute_trading_levels

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SIGNAL_PRIORITY = {"BREAKOUT": 5, "PRE_MARKUP": 4, "WATCH": 3, "NONE": 1, "AVOID": 0}
_SIG_EMOJI = {
    "BREAKOUT":   "🟢",
    "PRE_MARKUP": "🔵",
    "WATCH":      "🟠",
    "AVOID":      "🔴",
    "NONE":       "⚪",
}
_STOCKBIT_URL = "https://stockbit.com/symbol/{code}"
_MAX_TELEGRAM = 4096
_SAFE_LIMIT = 4000  # leave headroom for Telegram's counting differences


# ---------------------------------------------------------------------------
# Adapter: DataFrame → list[dict]
# ---------------------------------------------------------------------------

def signals_df_to_list(df: pd.DataFrame, scan_date: str = "") -> list[dict]:
    """Convert a signals DataFrame to a list of plain dicts.

    Each dict contains all columns as native Python scalars (NaN→None).
    Adds 'scan_date' key for downstream use.

    Args:
        df        : Signals / ranked DataFrame.
        scan_date : Date string injected into every row dict.

    Returns:
        List of dicts, one per row.
    """
    if df.empty:
        return []
    records = df.where(df.notna(), other=None).to_dict(orient="records")
    for r in records:
        r.setdefault("scan_date", scan_date)
    return records


# ---------------------------------------------------------------------------
# Candidate selectors
# ---------------------------------------------------------------------------

def select_breakout_deep_dive_candidates(
    signals: list[dict],
    top_n: int = 2,
) -> list[dict]:
    """Select the best candidates for deep-dive analysis.

    Priority:
    1. BREAKOUT first, then PRE_MARKUP.
    2. Within same signal: sort by enhanced_total_score → total_score (desc).
    3. Prefer tickers with real news data (news_data_status == 'ok').
    4. Prefer tickers with real fundamental data (fundamental_status == 'ok').

    Args:
        signals : Output of signals_df_to_list().
        top_n   : How many candidates to return (default 2).

    Returns:
        List of dicts (≤ top_n), sorted best-first.
    """
    eligible = [
        s for s in signals
        if str(s.get("signal", "")).upper() in ("BREAKOUT", "PRE_MARKUP")
    ]
    if not eligible:
        # Fallback to WATCH if no breakout/pre_markup
        eligible = [s for s in signals if str(s.get("signal", "")).upper() == "WATCH"]

    def _sort_key(s: dict) -> tuple:
        sig_rank = _SIGNAL_PRIORITY.get(str(s.get("signal", "")).upper(), 0)
        score    = _safe_num(s.get("enhanced_total_score")) or _safe_num(s.get("total_score")) or 0.0
        news_ok  = 1 if s.get("news_data_status") == "ok" else 0
        fund_ok  = 1 if s.get("fundamental_status") == "ok" else 0
        return (sig_rank, news_ok, fund_ok, score)

    eligible.sort(key=_sort_key, reverse=True)
    return eligible[:top_n]


def select_top_picks(
    signals: list[dict],
    top_n: int = 7,
) -> list[dict]:
    """Select the top N picks for the compact trading level list.

    Excludes AVOID and NONE signals.
    Sorted by enhanced_total_score (or total_score) descending.

    Args:
        signals : Output of signals_df_to_list().
        top_n   : Number of picks (default 7).

    Returns:
        List of dicts (≤ top_n).
    """
    picks = [
        s for s in signals
        if str(s.get("signal", "")).upper() not in ("AVOID", "NONE", "")
    ]

    def _sort_key(s: dict) -> float:
        return _safe_num(s.get("enhanced_total_score")) or _safe_num(s.get("total_score")) or 0.0

    picks.sort(key=_sort_key, reverse=True)
    return picks[:top_n]


# ---------------------------------------------------------------------------
# Deep Dive message builder  (DEWA-style)
# ---------------------------------------------------------------------------

def build_telegram_deep_dive_message(
    signal: dict,
    articles: list[dict] | None = None,
) -> str:
    """Build a full deep-dive analysis message for one ticker.

    Format (HTML, Telegram parse_mode=HTML):
    ─────────────────────────────────────────
    🔍 DEEP DIVE: $BBCA — Senin, 12 Mei 2026
    ─────────────────────────────────────────
    📌 SINYAL  : BREAKOUT 🟢 | Skor 8.4
    💰 Harga   : Rp9,250

    📊 TEKNIKAL
    ...

    📰 NEWS INTEL
    ...

    🏦 FUNDAMENTAL
    ...

    🎯 LEVEL TRADING
    ...

    💡 KESIMPULAN
    ...

    Args:
        signal  : Dict from signals_df_to_list().
        articles: Labeled article dicts for this ticker (from news articles parquet).

    Returns:
        HTML-formatted string, guaranteed ≤ 4000 chars.
        May be split downstream with split_long_message() if needed.
    """
    ticker = _ticker_clean(signal.get("ticker", "?"))
    signal_type = str(signal.get("signal", "")).upper()
    emoji = _SIG_EMOJI.get(signal_type, "⚪")

    scan_date_str = signal.get("scan_date", "")
    date_label = _format_date_id(scan_date_str)

    close = _safe_num(signal.get("close"))

    # Score
    score_enh = _safe_num(signal.get("enhanced_total_score"))
    score_base = _safe_num(signal.get("total_score"))
    score_display = score_enh if score_enh else score_base
    score_str = f"{score_display:.1f}" if score_display else "—"

    url = _STOCKBIT_URL.format(code=ticker)

    parts: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    parts.append(
        f'🔍 <b>DEEP DIVE: <a href="{url}">${ticker}</a></b>'
        f"  —  {date_label}"
    )
    parts.append("─────────────────────")
    parts.append(
        f"📌 <b>Sinyal</b>  : <b>{signal_type}</b> {emoji}  |  Skor <code>{score_str}</code>"
    )
    if close:
        parts.append(f"💰 <b>Harga</b>   : <code>Rp{int(close):,}</code>")
    parts.append("")

    # ── Technical ─────────────────────────────────────────────────────────────
    parts.append("📊 <b>TEKNIKAL</b>")
    tech_lines = _build_technical_lines(signal)
    parts.extend(tech_lines)
    parts.append("")

    # ── News ──────────────────────────────────────────────────────────────────
    parts.append("📰 <b>NEWS INTEL</b>")
    news_lines = _build_news_lines(signal, articles)
    parts.extend(news_lines)
    parts.append("")

    # ── Fundamental ───────────────────────────────────────────────────────────
    parts.append("🏦 <b>FUNDAMENTAL</b>")
    fund_lines = _build_fundamental_lines(signal)
    parts.extend(fund_lines)
    parts.append("")

    # ── Trading levels ────────────────────────────────────────────────────────
    parts.append("🎯 <b>LEVEL TRADING</b>")
    level_lines = _build_level_lines(signal)
    parts.extend(level_lines)
    parts.append("")

    # ── Conclusion ────────────────────────────────────────────────────────────
    parts.append("💡 <b>KESIMPULAN</b>")
    conclusion = _build_conclusion(signal, articles)
    parts.append(conclusion)
    parts.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    now_str = datetime.now().strftime("%H:%M WIB")
    parts.append("─────────────────────")
    parts.append(f"<i>📡 {now_str}  ·  🤖 IDX Scanner Agent</i>")

    text = "\n".join(parts)

    # Safety trim — keep under 4000 chars
    if len(text) > _SAFE_LIMIT:
        text = _trim_to_limit(text, _SAFE_LIMIT)

    return text


def build_all_breakout_deep_dive_messages(
    signals: list[dict],
    articles_by_ticker: dict[str, list[dict]] | None = None,
    top_n: int = 2,
) -> list[str]:
    """Build deep-dive messages for the top N BREAKOUT candidates.

    Args:
        signals           : Full list of signal dicts.
        articles_by_ticker: {ticker: [article_dicts, ...]} mapping.
        top_n             : How many deep dives to produce.

    Returns:
        List of HTML strings, one per candidate.
    """
    candidates = select_breakout_deep_dive_candidates(signals, top_n=top_n)
    messages: list[str] = []
    for s in candidates:
        ticker = _ticker_clean(s.get("ticker", ""))
        arts = (articles_by_ticker or {}).get(ticker, [])
        msg = build_telegram_deep_dive_message(s, articles=arts)
        messages.append(msg)
        logger.info("Built deep-dive message for %s (%d chars)", ticker, len(msg))
    return messages


# ---------------------------------------------------------------------------
# Top Picks message builder  (DPUM-style compact list)
# ---------------------------------------------------------------------------

def build_telegram_top_picks_message(
    signals: list[dict],
    date: str = "",
    top_n: int = 7,
) -> str:
    """Build a compact trading level list for the top N picks.

    Format (HTML):
    ─────────────────────────────────────
    🏆 TOP 7 PICK HARI INI — 12 Mei 2026
    ─────────────────────────────────────
    1. $DPUM 🔵
       Area Entry      : 200 – 207
       Target Profit   : 255 – 262
       Cutloss         : 198

    2. $BBCA 🟢
       ...

    Args:
        signals : List of signal dicts.
        date    : Scan date string (YYYY-MM-DD).
        top_n   : Number of picks to show.

    Returns:
        HTML-formatted string ≤ 4000 chars.
    """
    picks = select_top_picks(signals, top_n=top_n)
    date_label = _format_date_id(date)

    parts: list[str] = [
        f"🏆 <b>TOP {top_n} PICK HARI INI</b>  —  {date_label}",
        "─────────────────────",
        "",
    ]

    for i, s in enumerate(picks, 1):
        ticker = _ticker_clean(s.get("ticker", "?"))
        signal_type = str(s.get("signal", "")).upper()
        emoji = _SIG_EMOJI.get(signal_type, "⚪")
        url = _STOCKBIT_URL.format(code=ticker)

        # Compute levels
        try:
            row_series = pd.Series(s)
            levels = compute_trading_levels(row_series)
        except Exception as exc:
            logger.warning("Level calc failed for %s: %s", ticker, exc)
            levels = {"entry_low": 0, "entry_high": 0, "tp_low": 0, "tp_high": 0, "cutloss": 0}

        entry_low  = levels["entry_low"]
        entry_high = levels["entry_high"]
        tp_low     = levels["tp_low"]
        tp_high    = levels["tp_high"]
        cutloss    = levels["cutloss"]

        # Format line
        parts.append(
            f'{i}. <a href="{url}"><b>${ticker}</b></a>  {emoji} {signal_type}'
        )

        if entry_low and entry_high and entry_low != entry_high:
            parts.append(f"   Area Entry    : <code>{entry_low:,} – {entry_high:,}</code>")
        elif entry_low:
            parts.append(f"   Area Entry    : <code>{entry_low:,}</code>")

        if tp_low and tp_high and tp_low != tp_high:
            parts.append(f"   Target Profit : <code>{tp_low:,} – {tp_high:,}</code>")
        elif tp_low:
            parts.append(f"   Target Profit : <code>{tp_low:,}</code>")

        if cutloss:
            parts.append(f"   Cutloss       : <code>{cutloss:,}</code>")

        # Optional score/RSI mini-info
        score = _safe_num(s.get("enhanced_total_score")) or _safe_num(s.get("total_score"))
        rsi = _safe_num(s.get("rsi14"))
        vol = _safe_num(s.get("vol_ratio_20d"))
        info_parts: list[str] = []
        if score:
            info_parts.append(f"Skor {score:.1f}")
        if rsi:
            info_parts.append(f"RSI {rsi:.0f}")
        if vol:
            info_parts.append(f"Vol×{vol:.1f}")
        if info_parts:
            parts.append(f"   <i>{'  ·  '.join(info_parts)}</i>")

        parts.append("")

    # Footer
    now_str = datetime.now().strftime("%H:%M WIB")
    parts.append("─────────────────────")
    parts.append(f"<i>⚠️ Bukan rekomendasi investasi. DYOR.</i>")
    parts.append(f"<i>📡 {now_str}  ·  🤖 IDX Scanner Agent</i>")

    text = "\n".join(parts)

    if len(text) > _SAFE_LIMIT:
        text = _trim_to_limit(text, _SAFE_LIMIT)

    return text


# ---------------------------------------------------------------------------
# Message splitting
# ---------------------------------------------------------------------------

def split_long_message(text: str, limit: int = _SAFE_LIMIT) -> list[str]:
    """Split a long message at paragraph boundaries to respect Telegram's limit.

    Strategy:
    1. Try to split at double-newline (paragraph break).
    2. Fall back to single-newline if no paragraph break found.
    3. Hard-split if even that fails.

    Args:
        text  : The message text.
        limit : Max chars per chunk (default: 4000).

    Returns:
        List of chunks, each ≤ limit chars.
    """
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text

    while len(remaining) > limit:
        # Find split point: prefer paragraph break (\n\n) within limit
        window = remaining[:limit]
        split_idx = window.rfind("\n\n")

        if split_idx == -1:
            # Fall back to single newline
            split_idx = window.rfind("\n")

        if split_idx == -1 or split_idx < limit // 2:
            # Hard split as last resort
            split_idx = limit - 1

        chunk = remaining[:split_idx].rstrip()
        chunks.append(chunk)
        remaining = remaining[split_idx:].lstrip("\n")

    if remaining.strip():
        chunks.append(remaining.strip())

    return chunks


# ---------------------------------------------------------------------------
# Section builders (internal helpers)
# ---------------------------------------------------------------------------

def _build_technical_lines(s: dict) -> list[str]:
    """Build technical analysis bullet lines."""
    lines: list[str] = []
    close = _safe_num(s.get("close"))
    rsi = _safe_num(s.get("rsi14"))
    adx = _safe_num(s.get("adx"))
    vol = _safe_num(s.get("vol_ratio_20d"))
    pct52 = _safe_num(s.get("pct_from_52w_high"))
    ma20 = _safe_num(s.get("ma20"))
    ma50 = _safe_num(s.get("ma50"))
    atr14 = _safe_num(s.get("atr14"))
    macd_hist = _safe_num(s.get("macd_histogram"))
    supertrend = s.get("supertrend_bullish")
    squeeze = s.get("squeeze_on")
    atr_brk = s.get("atr_breakout")
    vol_spike = s.get("vol_spike")

    # Price vs MAs
    if close and ma20:
        rel = ((close - ma20) / ma20 * 100)
        pos = "di atas" if close > ma20 else "di bawah"
        lines.append(f"• Harga <code>Rp{int(close):,}</code>  {pos} MA20 ({rel:+.1f}%)")
    if close and ma50:
        rel50 = ((close - ma50) / ma50 * 100)
        pos50 = "di atas" if close > ma50 else "di bawah"
        lines.append(f"• vs MA50: {pos50} ({rel50:+.1f}%)")

    # RSI
    if rsi is not None:
        rsi_label = ""
        if rsi >= 70:
            rsi_label = "⚠️ overbought"
        elif rsi <= 30:
            rsi_label = "⚡ oversold"
        elif 50 < rsi < 70:
            rsi_label = "momentum naik"
        elif 30 < rsi <= 50:
            rsi_label = "momentum turun"
        lines.append(f"• RSI14: <code>{rsi:.1f}</code>  {rsi_label}")

    # ADX
    if adx is not None:
        trend_str = "tren kuat" if adx >= 25 else "tren lemah/sideways"
        lines.append(f"• ADX: <code>{adx:.1f}</code>  ({trend_str})")

    # Volume
    if vol is not None:
        vol_label = "spike signifikan 🔥" if vol >= 3 else ("volume tinggi" if vol >= 1.5 else "normal")
        lines.append(f"• Volume Ratio: <code>{vol:.1f}×</code>  {vol_label}")

    # 52w high
    if pct52 is not None:
        lines.append(f"• Jarak dari 52w High: <code>{pct52:+.1f}%</code>")

    # MACD
    if macd_hist is not None:
        macd_dir = "positif (bullish)" if macd_hist > 0 else "negatif (bearish)"
        lines.append(f"• MACD Histogram: <code>{macd_hist:+.3f}</code>  ({macd_dir})")

    # ATR
    if atr14 is not None and close:
        atr_pct = atr14 / close * 100
        lines.append(f"• ATR14: <code>Rp{int(atr14):,}</code>  ({atr_pct:.1f}% dari harga)")

    # Special flags
    flags: list[str] = []
    if supertrend is True or supertrend == 1:
        flags.append("Supertrend Bullish ✅")
    if squeeze is True or squeeze == 1:
        flags.append("Squeeze ON 🔋")
    if atr_brk is True or atr_brk == 1:
        flags.append("ATR Breakout 🚀")
    if vol_spike is True or vol_spike == 1:
        flags.append("Volume Spike 🔥")
    if flags:
        lines.append("• Konfirmasi: " + "  |  ".join(flags))

    if not lines:
        lines.append("• Data teknikal tidak tersedia.")

    return lines


def _build_news_lines(s: dict, articles: list[dict] | None = None) -> list[str]:
    """Build news intel lines with article bullet summaries."""
    status = str(s.get("news_data_status", "")).lower()
    sentiment_score = _safe_num(s.get("news_sentiment_score"))
    news_count = _safe_num(s.get("news_count_3d"))

    if status == "failed":
        return [
            "⚠️ <i>Gagal mengambil data berita. Cek koneksi atau coba lagi.</i>"
        ]

    if status == "none" or not articles:
        if status == "none":
            return ["• <i>Tidak ada berita relevan dalam 3 hari terakhir.</i>"]
        if sentiment_score is not None:
            bar = _score_bar(sentiment_score)
            return [
                f"• Sentimen: <code>{sentiment_score:.1f}/10</code>  {bar}",
                f"• Artikel (3h): {int(news_count) if news_count else 0}",
                "• <i>Detail artikel tidak tersedia.</i>",
            ]
        return ["• <i>Data berita tidak tersedia.</i>"]

    # Rich mode: summarize articles
    try:
        from stock_scanner.pipeline.news_summarizer import (
            summarize_news_articles,
            format_news_bullets,
        )
        summary = summarize_news_articles(articles)
        count = int(news_count) if news_count else len(articles)
        bar = _score_bar(sentiment_score) if sentiment_score is not None else ""
        bullets_text = format_news_bullets(
            summary,
            total_article_count=count,
            sentiment_score=sentiment_score,
            score_bar=bar,
            max_chars=700,
        )
        # Convert markdown bold (**) to HTML <b>
        bullets_html = _md_to_html(bullets_text)
        return [bullets_html]
    except Exception as exc:
        logger.warning("News summarizer error: %s", exc)
        lines: list[str] = []
        if sentiment_score is not None:
            lines.append(f"• Sentimen: <code>{sentiment_score:.1f}/10</code>")
        if news_count:
            lines.append(f"• Artikel (3h): {int(news_count)}")
        return lines


def _build_fundamental_lines(s: dict) -> list[str]:
    """Build fundamental snapshot lines."""
    status = str(s.get("fundamental_status", "")).lower()

    if status == "missing":
        ticker = _ticker_clean(s.get("ticker", ""))
        url = _STOCKBIT_URL.format(code=ticker)
        return [
            "• <i>Data fundamental belum tersedia via API.</i>",
            f'• Cek manual: <a href="{url}">Stockbit ${ticker}</a>',
        ]

    lines: list[str] = []
    pe  = _safe_num(s.get("pe_ratio"))
    pbv = _safe_num(s.get("pbv"))
    roe = _safe_num(s.get("roe_pct"))
    der = _safe_num(s.get("der"))
    div = _safe_num(s.get("div_yield_pct"))

    # PE
    if pe is not None:
        pe_label = ""
        if pe < 10:
            pe_label = "murah"
        elif pe < 20:
            pe_label = "wajar"
        elif pe < 30:
            pe_label = "premium"
        else:
            pe_label = "mahal"
        lines.append(f"• P/E: <code>{pe:.1f}×</code>  ({pe_label})")

    # PBV
    if pbv is not None:
        pbv_label = "undervalue" if pbv < 1.5 else ("premium" if pbv > 4.0 else "wajar")
        lines.append(f"• P/BV: <code>{pbv:.2f}×</code>  ({pbv_label})")

    # ROE
    if roe is not None:
        roe_label = "sangat profitable" if roe >= 25 else ("profitable" if roe >= 15 else "lemah")
        lines.append(f"• ROE: <code>{roe:.1f}%</code>  ({roe_label})")

    # DER
    if der is not None:
        der_label = "aman" if der <= 1.0 else ("moderat" if der <= 2.5 else "leverage tinggi ⚠️")
        lines.append(f"• DER: <code>{der:.2f}×</code>  ({der_label})")

    # Dividend
    if div is not None and div > 0:
        lines.append(f"• Dividend Yield: <code>{div:.2f}%</code>")

    if not lines:
        if status == "partial":
            lines.append("• <i>Data fundamental parsial — beberapa metrik tidak tersedia.</i>")
        else:
            lines.append("• <i>Data fundamental tidak tersedia.</i>")

    return lines


def _build_level_lines(s: dict) -> list[str]:
    """Build trading level lines using level_calculator."""
    try:
        row_series = pd.Series(s)
        levels = compute_trading_levels(row_series)
    except Exception as exc:
        logger.warning("Level calc error: %s", exc)
        return ["• <i>Level trading tidak dapat dihitung.</i>"]

    el  = levels["entry_low"]
    eh  = levels["entry_high"]
    tpl = levels["tp_low"]
    tph = levels["tp_high"]
    cl  = levels["cutloss"]

    lines: list[str] = []

    if el and eh and el != eh:
        lines.append(f"• <b>Area Entry</b>    : <code>Rp{el:,} – Rp{eh:,}</code>")
    elif el:
        lines.append(f"• <b>Area Entry</b>    : <code>Rp{el:,}</code>")

    if tpl and tph and tpl != tph:
        lines.append(f"• <b>Target Profit</b> : <code>Rp{tpl:,} – Rp{tph:,}</code>")
    elif tpl:
        lines.append(f"• <b>Target Profit</b> : <code>Rp{tpl:,}</code>")

    if cl:
        lines.append(f"• <b>Cutloss</b>       : <code>Rp{cl:,}</code>")

    # Show R:R ratio
    if el and tpl and cl and el > cl:
        risk = el - cl
        reward = tpl - el
        rr = reward / risk
        lines.append(f"• <i>Risk/Reward ≈ 1:{rr:.1f}</i>")

    if not lines:
        lines.append("• <i>Level tidak dapat dihitung — data ATR/MA tidak tersedia.</i>")

    return lines


def _build_conclusion(s: dict, articles: list[dict] | None = None) -> str:
    """Build a 2–3 sentence conclusion string."""
    ticker = _ticker_clean(s.get("ticker", ""))
    signal_type = str(s.get("signal", "")).upper()
    close = _safe_num(s.get("close"))
    rsi = _safe_num(s.get("rsi14"))
    sentiment = _safe_num(s.get("news_sentiment_score"))
    pe = _safe_num(s.get("pe_ratio"))
    roe = _safe_num(s.get("roe_pct"))
    news_status = str(s.get("news_data_status", "")).lower()

    sentences: list[str] = []

    # Technical stance
    if signal_type == "BREAKOUT":
        sentences.append(
            f"${ticker} menunjukkan sinyal <b>breakout</b> dengan konfirmasi teknikal."
        )
    elif signal_type == "PRE_MARKUP":
        sentences.append(
            f"${ticker} sedang dalam fase <b>akumulasi / pre-markup</b> yang menarik."
        )
    elif signal_type == "WATCH":
        sentences.append(
            f"${ticker} masuk watchlist — belum ada konfirmasi breakout."
        )
    else:
        sentences.append(f"${ticker} dalam fase konsolidasi.")

    # RSI condition
    if rsi is not None:
        if rsi < 35:
            sentences.append("Momentum RSI mendekati oversold — potensi reversal.")
        elif rsi > 65:
            sentences.append("RSI overbought — waspadai tekanan jual jangka pendek.")
        elif 45 <= rsi <= 65:
            sentences.append("RSI sehat — ruang rally masih terbuka.")

    # News sentiment
    if news_status == "ok" and sentiment is not None:
        if sentiment >= 7:
            sentences.append("Katalis berita positif mendukung momentum.")
        elif sentiment <= 4:
            sentences.append("⚠️ Sentimen berita negatif — perhatikan risiko downside.")

    # Fundamental
    if pe is not None and roe is not None:
        if pe < 15 and roe >= 15:
            sentences.append(
                f"Valuasi menarik (PE {pe:.1f}×, ROE {roe:.0f}%) — margin of safety cukup baik."
            )
        elif pe > 25:
            sentences.append(
                f"Valuasi premium (PE {pe:.1f}×) — pastikan pertumbuhan mendukung harga."
            )

    # Disclaimer
    sentences.append("<i>Selalu pasang cutloss dan kelola risiko dengan bijak.</i>")

    return "  ".join(sentences)


# ---------------------------------------------------------------------------
# Formatting utilities
# ---------------------------------------------------------------------------

def _ticker_clean(ticker: str) -> str:
    """Remove .JK suffix if present."""
    return str(ticker).replace(".JK", "").replace(".jk", "").strip()


def _safe_num(val: Any) -> float | None:
    """Convert value to float, return None on failure or NaN."""
    try:
        if val is None:
            return None
        v = float(val)
        return None if (v != v) else v  # NaN check
    except (TypeError, ValueError):
        return None


def _score_bar(score: float | None, width: int = 8) -> str:
    """Build a visual █░ bar from a 0–10 score."""
    if score is None:
        return ""
    filled = round((score / 10) * width)
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _format_date_id(date_str: str) -> str:
    """Format YYYY-MM-DD to Indonesian date string."""
    months_id = [
        "", "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
        "Jul", "Ags", "Sep", "Okt", "Nov", "Des",
    ]
    days_id = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return f"{days_id[dt.weekday()]}, {dt.day} {months_id[dt.month]} {dt.year}"
    except (ValueError, IndexError):
        return date_str


def _md_to_html(text: str) -> str:
    """Convert simple **bold** markdown to HTML <b> tags."""
    import re
    # Replace **text** → <b>text</b>
    result = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return result


def _trim_to_limit(text: str, limit: int) -> str:
    """Trim text to limit chars, cutting at the last paragraph break."""
    if len(text) <= limit:
        return text
    window = text[:limit - 30]  # leave room for trim marker
    cut = window.rfind("\n\n")
    if cut == -1:
        cut = window.rfind("\n")
    if cut == -1:
        cut = limit - 30
    return text[:cut].rstrip() + "\n\n<i>… (pesan dipotong)</i>"

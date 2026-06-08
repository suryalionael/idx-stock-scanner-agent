"""Daily report generator — one Markdown document per scan day.

Generates a comprehensive daily analysis document containing:
  A. Header / market overview
  B. Scalping High candidates
  C. Swing — Breakout + Pre-Markup
  D. Long Term picks
  E. AI Summary (Claude API if ANTHROPIC_API_KEY is set, else rule-based)
  F. Risk notes

The document is saved to data/reports/report_YYYY-MM-DD.md
and can be sent to Telegram via TelegramSender.send_document().

Public API
----------
generate_daily_report(context) -> dict
    Build the full report. Returns:
        {
          "report_markdown": str,       # full .md content
          "report_summary":  str,       # short Telegram text message (HTML)
          "output_path":     Path,      # where the .md was saved
        }

build_report_context(signals_df, scan_date) -> dict
    Helper to assemble the context dict from a signals DataFrame.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

_ROOT        = Path(__file__).parent.parent.parent
_REPORTS_DIR = _ROOT / "data" / "reports"

_CLAUDE_MODEL = "claude-sonnet-4-6"
_AI_SUMMARY_MAX_TOKENS = 600

_MONTHS_ID = ["", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
               "Juli", "Agustus", "September", "Oktober", "November", "Desember"]
_DAYS_ID   = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_report_context(
    signals_df: pd.DataFrame,
    scan_date: str,
) -> dict:
    """Assemble a context dict for generate_daily_report() from a DataFrame.

    Args:
        signals_df : Full signals DataFrame (output of run_daily_scan with all enrichments).
        scan_date  : 'YYYY-MM-DD'.

    Returns:
        Context dict with all sections pre-separated.
    """
    df = signals_df.copy()
    df = df.where(df.notna(), other=None)

    def _pick(col: str) -> list:
        return df[df["signal"] == col].to_dict(orient="records") if "signal" in df.columns else []

    def _scalp_high() -> list:
        if "scalping_label" not in df.columns:
            return []
        return df[df["scalping_label"] == "SCALPING_HIGH"].to_dict(orient="records")

    def _long_term() -> list:
        if "long_term_label" in df.columns:
            lt = df[df["long_term_label"].isin(["LT_STRONG_BUY", "LT_BUY"])]
            return lt.sort_values("long_term_score", ascending=False).head(10).to_dict(orient="records") \
                   if not lt.empty else []
        # Fallback: pick BREAKOUT/PRE_MARKUP with good fundamentals
        eligible = df[
            df["signal"].isin(["BREAKOUT", "PRE_MARKUP"]) &
            (pd.to_numeric(df.get("roe_pct", pd.Series(dtype=float)), errors="coerce") >= 12)
        ]
        return eligible.head(5).to_dict(orient="records")

    total_scanned = len(df)
    signal_dist   = df["signal"].value_counts().to_dict() if "signal" in df.columns else {}

    return {
        "scan_date":       scan_date,
        "total_scanned":   total_scanned,
        "signal_dist":     signal_dist,
        "scalping_high":   _scalp_high(),
        "breakouts":       _pick("BREAKOUT"),
        "pre_markups":     _pick("PRE_MARKUP"),
        "watches":         _pick("WATCH"),
        "long_term_picks": _long_term(),
    }


def generate_daily_report(context: dict) -> dict:
    """Build the full daily Markdown report.

    Args:
        context: Output of build_report_context() or manually assembled dict with keys:
            scan_date, total_scanned, signal_dist,
            scalping_high, breakouts, pre_markups, watches, long_term_picks

    Returns:
        {
            "report_markdown": str,   # Full .md content
            "report_summary":  str,   # Short HTML summary for Telegram text
            "output_path":     Path,  # Path where .md was saved
        }
    """
    scan_date = context.get("scan_date", datetime.now().strftime("%Y-%m-%d"))
    logger.info("Generating daily report for %s...", scan_date)

    sections: list[str] = []

    # ── A. Header ─────────────────────────────────────────────────────────
    sections.append(_section_header(context))

    # ── B. Scalping High ──────────────────────────────────────────────────
    sections.append(_section_scalping(context))

    # ── C. Swing ──────────────────────────────────────────────────────────
    sections.append(_section_swing(context))

    # ── D. Long Term ──────────────────────────────────────────────────────
    sections.append(_section_long_term(context))

    # ── E. AI Summary ─────────────────────────────────────────────────────
    ai_section, ai_short = _section_ai_summary(context)
    sections.append(ai_section)

    # ── F. Risk Notes ─────────────────────────────────────────────────────
    sections.append(_section_risk_notes(context))

    # ── Footer ────────────────────────────────────────────────────────────
    now_str = datetime.now().strftime("%H:%M WIB")
    sections.append(f"\n---\n*Generated: {now_str} · IDX Scanner Agent*")

    report_md = "\n\n".join(s for s in sections if s.strip())

    # Save to disk
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _REPORTS_DIR / f"report_{scan_date}.md"
    output_path.write_text(report_md, encoding="utf-8")
    logger.info("Daily report saved → %s (%d chars)", output_path, len(report_md))

    # Build short Telegram summary (HTML)
    summary_html = _build_telegram_summary(context, ai_short, scan_date)

    return {
        "report_markdown": report_md,
        "report_summary":  summary_html,
        "output_path":     output_path,
    }


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _section_header(ctx: dict) -> str:
    scan_date = ctx["scan_date"]
    try:
        dt = datetime.strptime(scan_date, "%Y-%m-%d")
        date_full = f"{_DAYS_ID[dt.weekday()]}, {dt.day} {_MONTHS_ID[dt.month]} {dt.year}"
    except (ValueError, IndexError):
        date_full = scan_date

    total = ctx.get("total_scanned", 0)
    dist  = ctx.get("signal_dist", {})

    now = datetime.now()
    try:
        report_full = f"{_DAYS_ID[now.weekday()]}, {now.day} {_MONTHS_ID[now.month]} {now.year}"
    except (ValueError, IndexError):
        report_full = now.strftime("%d %b %Y")

    lines = [
        f"# 📊 IDX Daily Report — {report_full}",
        f"> Data market: {date_full}",
        "",
        "## Market Overview",
        "",
        f"- **Report date**: `{now.strftime('%Y-%m-%d')}` (WIB)",
        f"- **Data market (sesi)**: `{scan_date}`",
        f"- **Total tickers scanned**: {total}",
        "",
        "### Signal Distribution",
        "",
        "| Signal | Count |",
        "|---|---|",
    ]
    for sig in ["BREAKOUT", "PRE_MARKUP", "WATCH", "AVOID", "NONE"]:
        cnt = dist.get(sig, 0)
        lines.append(f"| {sig} | {cnt} |")

    return "\n".join(lines)


def _section_scalping(ctx: dict) -> str:
    from stock_scanner.alerts.scalping_formatter import compute_scalping_levels

    rows = ctx.get("scalping_high", [])
    lines = ["## ⚡ Scalping High", ""]

    if not rows:
        lines.append("*Tidak ada kandidat Scalping High hari ini.*")
        return "\n".join(lines)

    lines.append(
        "| # | Ticker | Harga | Entry | TP | Cutloss | R:R | Alasan |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    for i, row in enumerate(sorted(rows, key=lambda r: _sf(r.get("scalping_score")) or 0, reverse=True), 1):
        ticker = _clean(row.get("ticker", "?"))
        close  = _sf(row.get("close"))
        reason = str(row.get("scalping_reason", ""))[:60]
        score  = _sf(row.get("scalping_score")) or 0

        lvl    = compute_scalping_levels(row)
        el, eh = lvl["entry_low"], lvl["entry_high"]
        tl, th = lvl["tp_low"],   lvl["tp_high"]
        cl     = lvl["cutloss"]
        rr     = lvl.get("rr_ratio", 0.0)
        status = lvl["trade_setup_status"]

        close_str = f"{int(close):,}" if close else "—"
        entry_str = f"{el:,}–{eh:,}" if el and eh and el != eh else (f"{el:,}" if el else "—")
        tp_str    = f"{tl:,}–{th:,}" if tl and th and tl != th else (f"{tl:,}" if tl else "—")
        cl_str    = f"{cl:,}" if cl else "—"
        rr_str    = f"1:{rr:.1f}" if rr >= 1.0 else ("low" if status == "low_rr" else "—")

        lines.append(
            f"| {i} | **{ticker}** (skor {score:.1f}) | {close_str} | {entry_str} | {tp_str} | {cl_str} | {rr_str} | {reason} |"
        )

    lines.append("")
    lines.append("> ⚠️ Scalping = risiko tinggi. Horizon 1 hari. Selalu pasang cutloss.")
    return "\n".join(lines)


def _section_swing(ctx: dict) -> str:
    breakouts   = ctx.get("breakouts", [])
    pre_markups = ctx.get("pre_markups", [])
    lines = ["## 📈 Swing Trading", ""]

    def _render_group(group: list[dict], label: str, emoji: str) -> list[str]:
        from stock_scanner.alerts.level_calculator import compute_trading_levels
        out = [f"### {emoji} {label} ({len(group)})", ""]
        if not group:
            out.append(f"*Tidak ada sinyal {label} hari ini.*")
            return out

        out.append("| # | Ticker | Score | Harga | Entry | TP | Cutloss | RSI |")
        out.append("|---|---|---|---|---|---|---|---|")

        grp_sorted = sorted(group,
                            key=lambda r: _sf(r.get("enhanced_total_score")) or _sf(r.get("total_score")) or 0,
                            reverse=True)
        for i, row in enumerate(grp_sorted, 1):
            ticker = _clean(row.get("ticker", "?"))
            score  = _sf(row.get("enhanced_total_score")) or _sf(row.get("total_score")) or 0
            close  = _sf(row.get("close"))
            rsi    = _sf(row.get("rsi14"))

            try:
                lvl = compute_trading_levels(pd.Series(row))
            except Exception:
                lvl = {}

            el   = lvl.get("entry_low", 0)
            eh   = lvl.get("entry_high", 0)
            tl   = lvl.get("tp_low", 0)
            th   = lvl.get("tp_high", 0)
            cl   = lvl.get("cutloss", 0)
            st   = lvl.get("trade_setup_status", "inactive")

            close_str = f"{int(close):,}" if close else "—"
            entry_str = f"{el:,}–{eh:,}" if el and eh and el != eh else (f"{el:,}" if el else "—")
            tp_str    = f"{tl:,}–{th:,}" if tl and th and tl != th else (f"{tl:,}" if tl else "—")
            cl_str    = f"{cl:,}" if cl else "—"
            rsi_str   = f"{rsi:.0f}" if rsi else "—"

            # Mark inactive levels
            if st != "active":
                entry_str = f"~~{entry_str}~~" if entry_str != "—" else "—"
                tp_str    = "—"

            out.append(
                f"| {i} | **{ticker}** | {score:.1f} | {close_str} | {entry_str} | {tp_str} | {cl_str} | {rsi_str} |"
            )
        return out

    lines.extend(_render_group(breakouts,   "Breakout",   "🟢"))
    lines.append("")
    lines.extend(_render_group(pre_markups, "Pre-Markup", "🔵"))
    lines.append("")
    lines.append("> ⚠️ Level trading bersifat indikatif. Cek chart sebelum entry.")
    return "\n".join(lines)


def _section_long_term(ctx: dict) -> str:
    picks = ctx.get("long_term_picks", [])
    lines = ["## 🏦 Long Term / Fundamental", ""]

    if not picks:
        lines.append("*Data long-term tidak tersedia atau tidak ada kandidat hari ini.*")
        return "\n".join(lines)

    lines.append("| # | Ticker | Harga | ROE | P/E | P/BV | DER | Div% | Valuation |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for i, row in enumerate(picks, 1):
        ticker = _clean(row.get("ticker", "?"))
        close  = _sf(row.get("close"))
        roe    = _sf(row.get("roe_pct"))
        pe     = _sf(row.get("pe_ratio"))
        pbv    = _sf(row.get("pbv"))
        der    = _sf(row.get("der"))
        div    = _sf(row.get("div_yield_pct"))

        # Valuation note
        val = "—"
        if pe and roe:
            if pe < 12 and roe >= 15:
                val = "Murah"
            elif pe < 20:
                val = "Wajar"
            elif pe >= 25:
                val = "Premium"

        def _fmt(v, fmt, suffix="", default="—"):
            return f"{v:{fmt}}{suffix}" if v is not None else default

        lines.append(
            f"| {i} | **{ticker}** | "
            f"{_fmt(close, ',.0f')} | "
            f"{_fmt(roe, '.1f', '%')} | "
            f"{_fmt(pe, '.1f', '×')} | "
            f"{_fmt(pbv, '.2f', '×')} | "
            f"{_fmt(der, '.2f', '×')} | "
            f"{_fmt(div, '.1f', '%')} | "
            f"{val} |"
        )

    lines.append("")
    lines.append("> 🏦 Long term = horizon 6–24 bulan. Riset lebih lanjut sebelum investasi.")
    return "\n".join(lines)


def _section_ai_summary(ctx: dict) -> tuple[str, str]:
    """Return (markdown_section, short_text_for_telegram)."""
    scan_date   = ctx["scan_date"]
    breakouts   = ctx.get("breakouts", [])
    pre_markups = ctx.get("pre_markups", [])
    scalp_high  = ctx.get("scalping_high", [])
    dist        = ctx.get("signal_dist", {})
    total       = ctx.get("total_scanned", 0)

    # Build data summary for AI prompt
    top_b  = [_clean(r.get("ticker", "")) for r in breakouts[:5]]
    top_pm = [_clean(r.get("ticker", "")) for r in pre_markups[:5]]
    top_sc = [_clean(r.get("ticker", "")) for r in scalp_high[:3]]

    # Check sectors from top picks
    sectors: list[str] = []
    for row in (breakouts + pre_markups)[:10]:
        s = str(row.get("sector", "") or "").strip()
        if s and s not in sectors:
            sectors.append(s)

    # Try Claude API
    ai_narrative = _try_claude_summary(ctx, top_b, top_pm, top_sc, sectors)

    if not ai_narrative:
        # Fallback: deterministic rule-based summary
        ai_narrative = _rule_based_summary(dist, total, top_b, top_pm, top_sc, sectors, scan_date)

    section = "\n".join([
        "## 🤖 AI Summary",
        "",
        ai_narrative,
        "",
    ])

    # Short version for Telegram (strip markdown)
    short_lines = []
    for line in ai_narrative.split("\n")[:5]:
        line = line.strip().lstrip("- •").strip()
        if line:
            short_lines.append(line)
    short = " ".join(short_lines[:3])[:400]

    return section, short


def _try_claude_summary(
    ctx: dict,
    top_b: list[str],
    top_pm: list[str],
    top_sc: list[str],
    sectors: list[str],
) -> str | None:
    """Try to generate AI summary via Claude API. Returns None on any failure."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.debug("ANTHROPIC_API_KEY not set — using rule-based summary.")
        return None

    try:
        import anthropic

        dist  = ctx.get("signal_dist", {})
        total = ctx.get("total_scanned", 0)

        prompt = (
            f"Kamu adalah analis pasar saham IDX Indonesia yang berpengalaman. "
            f"Berikan ringkasan analisa harian dalam bahasa Indonesia (5–8 bullet point). "
            f"PENTING: angka dan nama saham sudah diberikan — jangan mengarang. "
            f"Fokus pada: momentum pasar hari ini, sektor dominan, kualitas sinyal, dan rekomendasi taktis.\n\n"
            f"DATA SCAN ({ctx['scan_date']}):\n"
            f"- Total ticker dipindai: {total}\n"
            f"- BREAKOUT: {dist.get('BREAKOUT', 0)} saham → {', '.join(top_b) or 'tidak ada'}\n"
            f"- PRE_MARKUP: {dist.get('PRE_MARKUP', 0)} saham → {', '.join(top_pm) or 'tidak ada'}\n"
            f"- WATCH: {dist.get('WATCH', 0)} saham\n"
            f"- SCALPING_HIGH: {len(ctx.get('scalping_high', []))} saham → {', '.join(top_sc) or 'tidak ada'}\n"
            f"- Sektor terlihat: {', '.join(sectors[:5]) or 'beragam'}\n\n"
            f"Tulis ringkasan singkat dalam format bullet list. Maksimal 400 kata. "
            f"Gunakan bahasa Indonesia yang mudah dipahami oleh trader ritel."
        )

        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=_CLAUDE_MODEL,
            max_tokens=_AI_SUMMARY_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        logger.info("AI summary generated via Claude API (%d chars).", len(text))
        return text

    except ImportError:
        logger.debug("anthropic package not installed — using rule-based summary.")
        return None
    except Exception as exc:
        logger.warning("Claude API failed for summary: %s — using rule-based.", exc)
        return None


def _rule_based_summary(
    dist: dict,
    total: int,
    top_b: list[str],
    top_pm: list[str],
    top_sc: list[str],
    sectors: list[str],
    scan_date: str,
) -> str:
    """Deterministic AI-summary fallback when Claude API is unavailable."""
    n_breakout = dist.get("BREAKOUT", 0)
    n_premark  = dist.get("PRE_MARKUP", 0)
    n_watch    = dist.get("WATCH", 0)
    n_scalp    = len(top_sc)

    lines: list[str] = []

    # Market momentum
    if n_breakout >= 5:
        lines.append(f"- 📈 Momentum pasar **kuat** — {n_breakout} BREAKOUT dari {total} ticker yang dipindai.")
    elif n_breakout >= 1:
        lines.append(f"- 📊 Momentum pasar **moderat** — {n_breakout} BREAKOUT, {n_premark} Pre-Markup.")
    else:
        lines.append(f"- 🔵 Pasar dalam fase **akumulasi** — {n_premark} Pre-Markup, belum ada Breakout.")

    # Top picks
    if top_b:
        lines.append(f"- 🟢 Breakout terkuat: **{', '.join(top_b[:3])}**")
    if top_pm:
        lines.append(f"- 🔵 Pre-Markup menarik: **{', '.join(top_pm[:3])}**")
    if top_sc:
        lines.append(f"- ⚡ Scalping High: **{', '.join(top_sc[:3])}** — volume/momentum kuat hari ini.")

    # Sector
    if sectors:
        lines.append(f"- 🏭 Sektor dominan hari ini: **{', '.join(sectors[:3])}**.")
    else:
        lines.append(f"- 🏭 Sinyal tersebar di berbagai sektor.")

    # Signal quality
    if n_watch > n_premark * 3:
        lines.append(f"- ⚠️ Banyak sinyal WATCH ({n_watch}) vs Pre-Markup ({n_premark}) — pasar belum fully confirmed trend.")
    elif n_breakout + n_premark > 20:
        lines.append(f"- ✅ Sinyal berkualitas tinggi banyak ({n_breakout + n_premark} BREAKOUT+PRE_MARKUP).")

    # Risk note
    lines.append(f"- 💡 Selalu cek chart individual sebelum entry. Level di atas bersifat indikatif.")

    return "\n".join(lines)


def _section_risk_notes(ctx: dict) -> str:
    dist  = ctx.get("signal_dist", {})
    rows  = ctx.get("breakouts", []) + ctx.get("pre_markups", [])
    lines = ["## ⚠️ Risk Notes", ""]

    notes: list[str] = []

    # Low-RR setups
    low_rr = [_clean(r.get("ticker", "")) for r in rows if str(r.get("trade_setup_status", "")) == "low_rr"]
    if low_rr:
        notes.append(f"- Saham dengan **R:R rendah** (di bawah 1.5) setelah tick rounding: **{', '.join(low_rr[:5])}**. Sizing lebih kecil atau skip.")

    # AVOID signal
    n_avoid = dist.get("AVOID", 0)
    if n_avoid > 0:
        notes.append(f"- **{n_avoid} ticker** mendapat sinyal AVOID (likuiditas rendah / kondisi teknikal buruk).")

    # General
    notes.append("- Level entry/TP/cutloss dihitung dari data EOD — harga bisa bergerak signifikan saat open.")
    notes.append("- Sinyal WATCH belum konfirmasi — jangan masuk tanpa trigger / breakout harga aktual.")
    notes.append("- Selalu manage position sizing. Jangan lebih dari 5% portofolio per saham.")
    notes.append("- **Dokumen ini bukan rekomendasi investasi.** DYOR.")

    lines.extend(notes)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Telegram summary (text message sent BEFORE the file)
# ---------------------------------------------------------------------------

def _build_telegram_summary(ctx: dict, ai_short: str, scan_date: str) -> str:
    """Build a short HTML Telegram text message to accompany the document."""
    from datetime import datetime as _dt
    try:
        dt = _dt.strptime(scan_date, "%Y-%m-%d")
        date_label = f"{_DAYS_ID[dt.weekday()]}, {dt.day} {_MONTHS_ID[dt.month]} {dt.year}"
    except (ValueError, IndexError):
        date_label = scan_date

    dist  = ctx.get("signal_dist", {})
    scalp = ctx.get("scalping_high", [])

    now = datetime.now()
    now_str = now.strftime("%H:%M WIB")
    # Title date = report/send date (today, WIB); market session shown separately.
    try:
        report_label = f"{_DAYS_ID[now.weekday()]}, {now.day} {_MONTHS_ID[now.month]} {now.year}"
    except (ValueError, IndexError):
        report_label = now.strftime("%d %b %Y")

    lines = [
        f"📋 <b>Daily Report — {report_label}</b>",
        f"<i>Data market: {date_label}</i>",
        "─────────────────────",
        f"🟢 BREAKOUT   : {dist.get('BREAKOUT', 0)}",
        f"🔵 PRE-MARKUP : {dist.get('PRE_MARKUP', 0)}",
        f"🟠 WATCH      : {dist.get('WATCH', 0)}",
        f"⚡ SCALPING H : {len(scalp)}",
        "",
    ]

    if ai_short:
        lines.append("<b>Highlight:</b>")
        lines.append(f"<i>{ai_short[:350]}</i>")
        lines.append("")

    lines += [
        "📄 <i>Laporan lengkap ada di file terlampir.</i>",
        "─────────────────────",
        f"<i>📡 {now_str}  ·  🤖 IDX Scanner Agent</i>",
    ]

    return "\n".join(lines)


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

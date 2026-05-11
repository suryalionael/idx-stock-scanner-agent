"""Explanation generator untuk sinyal saham IDX.

Dua level:
  1. explain_signal()     — rule-based, selalu jalan tanpa API key
  2. explain_signal_llm() — Claude API (skeleton, uncomment saat siap)

Penjelasan dibagi 3 dimensi:
  🔧 Teknikal    — indikator harga & volume (MA, RSI, OBV, ATR, dll.)
  📰 News/Catalyst — sentimen berita 3 hari terakhir; respects news_data_status
  📊 Fundamental  — PE, PBV, ROE, DER, dividend yield dari yfinance pipeline

Logika rule-based menggunakan kondisi fitur nyata sehingga penjelasan
bisa berbeda tiap ticker, bukan template statis.
"""
import os
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Fundamental interpretation thresholds (IDX context)
# ---------------------------------------------------------------------------

_ROE_GOOD     = 15.0   # % — ROE ≥ 15% dianggap baik
_ROE_GREAT    = 25.0   # % — ROE ≥ 25% dianggap sangat baik
_DER_SAFE     = 1.0    # DER ≤ 1 dianggap aman
_DER_HIGH     = 2.5    # DER > 2.5 dianggap tinggi (bank-sector exception applies)
_PE_CHEAP     = 10.0   # PE ≤ 10 dianggap murah relatif ke IDX
_PE_FAIR      = 20.0   # PE 10–20 dianggap wajar
_PE_EXPENSIVE = 30.0   # PE > 30 dianggap mahal
_DIV_GOOD     = 3.0    # yield ≥ 3% dianggap menarik
_PBV_CHEAP    = 1.5    # PBV ≤ 1.5 dianggap undervalued
_PBV_EXPENSIVE = 4.0   # PBV > 4 dianggap premium


# ---------------------------------------------------------------------------
# Level 1: Rule-based explanation (selalu tersedia)
# ---------------------------------------------------------------------------

def explain_signal(row: pd.Series) -> str:
    """Buat penjelasan berbasis rule dari fitur satu ticker.

    Tidak butuh API key. Output adalah teks Markdown dengan 3 seksi:
      🔧 Teknikal
      📰 News & Catalyst
      📊 Fundamental
    """
    signal = str(row.get("signal", "NONE"))
    ticker = str(row.get("ticker", "?"))
    total  = _fmt_num(row.get("total_score"), 1)
    enh    = _fmt_num(row.get("enhanced_total_score"), 1)

    parts: list[str] = []

    # === Kalimat pembuka per signal =========================================
    opener = {
        "BREAKOUT":   (f"**{ticker}** mendapat sinyal **BREAKOUT** — "
                       f"skor {total}/10 (enhanced: {enh}/10)."),
        "PRE_MARKUP": (f"**{ticker}** mendapat sinyal **PRE_MARKUP** "
                       f"(kandidat menjelang breakout) — skor {total}/10 (enhanced: {enh}/10)."),
        "WATCH":      (f"**{ticker}** masuk watchlist dengan skor {total}/10 "
                       f"— belum cukup kuat untuk aksi, tapi layak dipantau."),
        "AVOID":      (f"**{ticker}** mendapat sinyal **AVOID** (skor {total}/10) "
                       f"— kondisi saat ini tidak mendukung entry."),
        "NONE":       f"**{ticker}** tidak menghasilkan sinyal yang jelas (skor {total}/10).",
    }
    parts.append(opener.get(signal, opener["NONE"]))
    parts.append("")

    # =========================================================================
    # SEKSI 1: TEKNIKAL
    # =========================================================================
    parts.append("### 🔧 Teknikal")
    parts.extend(_build_technical_section(row))

    # =========================================================================
    # SEKSI 2: NEWS / CATALYST
    # =========================================================================
    parts.append("")
    parts.append("### 📰 News & Catalyst (3 hari terakhir)")
    parts.extend(_build_news_section(row))

    # =========================================================================
    # SEKSI 3: FUNDAMENTAL
    # =========================================================================
    parts.append("")
    parts.append("### 📊 Fundamental")
    parts.extend(_build_fundamental_section(row))

    # === Disclaimer ==========================================================
    parts.append("")
    parts.append(
        "---\n"
        "⚠️ *Ini adalah output analisis teknikal otomatis, bukan rekomendasi investasi. "
        "Selalu lakukan riset sendiri (DYOR) sebelum mengambil keputusan.*"
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _build_technical_section(row: pd.Series) -> list[str]:
    """Build lines for the Technical section."""
    tech: list[str] = []

    # --- Tren MA ---
    ma_full    = _truthy(row.get("ma_full_alignment"))
    ma_partial = _truthy(row.get("ma_partial_alignment"))
    ma200      = _num(row.get("ma200"))
    ma50       = _num(row.get("ma50"))
    close_num  = _num(row.get("close"))
    close_str  = _fmt_price(row.get("close"))

    if ma_full:
        tech.append(
            "Tren jangka pendek, menengah, dan panjang selaras (MA20 > MA50 > MA200) "
            "— struktur teknikal bullish."
        )
    elif ma_partial:
        tech.append(
            "MA20 berada di atas MA50, namun MA200 belum dilewati "
            "— tren menengah mulai positif tapi tren panjang belum konfirmasi."
        )
    else:
        if ma200 and close_num and close_num < ma200:
            tech.append(
                f"Harga ({close_str}) masih di bawah MA200 ({_fmt_price(ma200)}) "
                "— tren jangka panjang masih turun."
            )
        elif ma50 and close_num and close_num < ma50:
            tech.append("Harga masih di bawah MA50 — tren menengah belum mendukung.")

    # ADX
    adx = _num(row.get("adx"))
    if adx is not None:
        if adx >= 40:
            tech.append(f"ADX {_fmt_num(adx, 0)} — tren sangat kuat.")
        elif adx >= 25:
            tech.append(f"ADX {_fmt_num(adx, 0)} — pasar sedang trending.")
        else:
            tech.append(f"ADX {_fmt_num(adx, 0)} — tren lemah, pasar cenderung sideways.")

    if _truthy(row.get("supertrend_bullish")):
        tech.append("Supertrend menunjukkan sinyal bullish.")

    # RSI
    rsi = _num(row.get("rsi14"))
    if rsi is not None:
        if rsi > 80:
            tech.append(f"RSI {_fmt_num(rsi, 1)} — sangat overbought, risiko koreksi jangka pendek tinggi.")
        elif rsi > 70:
            tech.append(f"RSI {_fmt_num(rsi, 1)} — mendekati overbought, momentum kuat tapi perlu hati-hati.")
        elif 50 <= rsi <= 70:
            tech.append(f"RSI {_fmt_num(rsi, 1)} — zona momentum sehat, tidak overbought.")
        elif 40 <= rsi < 50:
            tech.append(f"RSI {_fmt_num(rsi, 1)} — momentum netral, belum ada dorongan kuat ke atas.")
        else:
            tech.append(f"RSI {_fmt_num(rsi, 1)} — oversold, potensi rebound tapi konfirmasi volume diperlukan.")

    # MACD
    macd_hist = _num(row.get("macd_histogram"))
    if macd_hist is not None:
        if macd_hist > 0:
            tech.append(f"MACD histogram positif ({_fmt_num(macd_hist, 3)}) — momentum bullish.")
        else:
            tech.append(f"MACD histogram negatif ({_fmt_num(macd_hist, 3)}) — momentum bearish.")

    # Squeeze
    if _truthy(row.get("squeeze_on")):
        tech.append("Bollinger Band Squeeze aktif — potensi breakout directional segera.")

    # Volume
    vol_ratio = _num(row.get("vol_ratio_20d"))
    if vol_ratio is not None:
        if vol_ratio >= 2.0:
            tech.append(
                f"Volume melonjak {_fmt_num(vol_ratio, 1)}x rata-rata 20 hari "
                "— sinyal partisipasi pasar yang kuat (akumulasi institusional mungkin terjadi)."
            )
        elif vol_ratio >= 1.3:
            tech.append(
                f"Volume di atas rata-rata ({_fmt_num(vol_ratio, 1)}x) "
                "— ada peningkatan minat, belum signifikan."
            )
        else:
            tech.append(
                "Volume relatif normal atau di bawah rata-rata "
                "— belum ada konfirmasi partisipasi besar."
            )

    if _truthy(row.get("obv_trend")):
        tech.append("OBV (On-Balance Volume) sedang naik, mengindikasikan akumulasi secara kumulatif.")

    # 52w high
    pct_52w = _num(row.get("pct_from_52w_high"))
    if pct_52w is not None:
        if pct_52w >= -5:
            tech.append(
                f"Harga sangat dekat dengan 52-week high (hanya {_fmt_num(abs(pct_52w), 1)}% di bawah) "
                "— potensi breakout ke all-time high area."
            )
        elif pct_52w >= -15:
            tech.append(
                f"Harga masih {_fmt_num(abs(pct_52w), 1)}% di bawah 52-week high "
                "— sedang mendekati area kritis."
            )
        else:
            tech.append(
                f"Harga masih {_fmt_num(abs(pct_52w), 1)}% di bawah 52-week high "
                "— butuh perjalanan panjang untuk uji level tersebut."
            )

    if _truthy(row.get("atr_breakout")):
        tech.append(
            "Harga bergerak lebih dari 1.5× ATR di atas penutupan kemarin "
            "— breakout harga yang signifikan secara statistik."
        )

    roc5 = _num(row.get("roc5"))
    if roc5 is not None:
        if roc5 > 5:
            tech.append(f"ROC 5 hari sebesar +{_fmt_num(roc5, 1)}% — momentum jangka sangat pendek kuat.")
        elif roc5 < -5:
            tech.append(f"ROC 5 hari sebesar {_fmt_num(roc5, 1)}% — pelemahan jangka sangat pendek.")

    return tech if tech else ["Data teknikal tidak lengkap untuk ticker ini."]


def _build_news_section(row: pd.Series) -> list[str]:
    """Build lines for the News & Catalyst section.

    Respects news_data_status:
      "ok"     → show count, breakdown, sentiment score
      "none"   → explain no news found (valid, not a failure)
      "failed" → honest error message, explain impact on scoring
      (missing) → legacy scan without status column
    """
    news_status  = str(row.get("news_data_status", "")).strip().lower()
    news_count   = _num(row.get("news_count_3d"))
    news_mean    = _num(row.get("news_sentiment_mean"))
    news_pos     = _num(row.get("news_positive_count"))
    news_neg     = _num(row.get("news_negative_count"))
    news_score_v = _num(row.get("news_sentiment_score"))
    news_source  = str(row.get("news_source", "")).strip()
    news_err     = row.get("news_error_message")

    lines: list[str] = []

    if news_status == "failed":
        err_hint = f"\n  _(Detail: {news_err})_" if news_err else ""
        lines.append(
            "⚠️ **Data berita gagal diambil untuk ticker ini hari ini** "
            "— komponen news tidak ikut dipertimbangkan dalam skor.{}\n"
            "\nNilai news_score diset ke netral (5.0) secara internal agar tidak "
            "merugikan sinyal teknikal yang mungkin valid.".format(err_hint)
        )

    elif news_status == "none":
        lines.append(
            "Tidak ada berita relevan yang ditemukan dalam 3 hari terakhir untuk ticker ini "
            f"(sumber: {news_source or 'google_rss'}).\n"
            "\nPergerakan harga kemungkinan lebih didorong oleh teknikal atau aliran dana, "
            "bukan katalis berita spesifik."
        )

    elif news_status == "ok" and news_count is not None:
        n   = int(news_count)
        pos = int(news_pos) if news_pos is not None else 0
        neg = int(news_neg) if news_neg is not None else 0
        neu = max(0, n - pos - neg)

        src_label = news_source or "google_rss"
        lines.append(
            f"Ditemukan **{n} artikel** dalam 3 hari terakhir (sumber: {src_label})."
        )

        if news_score_v is not None:
            bar = _score_bar(news_score_v)
            lines.append(
                f"Breakdown: 🟢 {pos} positif · 🔴 {neg} negatif · ⚪ {neu} netral  "
                f"→ Skor sentimen: **{_fmt_num(news_score_v, 1)}/10** {bar}"
            )

        if news_mean is not None:
            if news_mean > 0.3:
                lines.append(
                    "Mayoritas berita bersifat **positif** — ada potensi katalis fundamental "
                    "yang mendukung momentum harga."
                )
            elif news_mean < -0.3:
                lines.append(
                    "Mayoritas berita bersifat **negatif** — waspadai risiko fundamental "
                    "yang dapat menekan harga."
                )
            else:
                lines.append(
                    "Berita cukup berimbang — tidak ada sentimen kuat ke satu arah."
                )

    else:
        # news_data_status kolom belum ada (scan lama sebelum upgrade pipeline)
        if news_count is not None and news_count > 0 and news_score_v is not None:
            # Bisa tampilkan data meski tanpa status field
            lines.append(
                f"{int(news_count)} artikel, skor sentimen: {_fmt_num(news_score_v, 1)}/10. "
                "(Status field belum tersedia — jalankan ulang scan untuk info lengkap.)"
            )
        else:
            lines.append(
                "_Data news tidak tersedia untuk scan ini (format lama). "
                "Jalankan ulang scan untuk mendapatkan status news yang akurat._"
            )

    return lines


def _build_fundamental_section(row: pd.Series) -> list[str]:
    """Build lines for the Fundamental section.

    Uses real data from fundamental pipeline columns if available.
    Falls back to honest 'data not available' only when genuinely missing.
    Never fabricates numbers.
    """
    fund_status = str(row.get("fundamental_status", "")).strip().lower()
    pe     = _num(row.get("pe_ratio"))
    pbv    = _num(row.get("pbv"))
    roe    = _num(row.get("roe_pct"))
    der    = _num(row.get("der"))
    div    = _num(row.get("div_yield_pct"))
    eps    = _num(row.get("eps"))
    rev_g  = _num(row.get("revenue_growth_pct"))
    prof_g = _num(row.get("profit_growth_pct"))

    # Count actually available fields
    available_fields = sum(1 for v in [pe, pbv, roe, der, div, eps] if v is not None)

    lines: list[str] = []

    if fund_status == "missing" or available_fields == 0:
        lines.append(
            "_Data fundamental terbaru belum tersedia di pipeline untuk emiten ini. "
            "Silakan cek laporan keuangan terakhir di "
            "[IDX](https://www.idx.co.id) atau platform seperti stockbit.com / RTI Business._"
        )
        return lines

    # ── Build narrative from available data ────────────────────────────────
    narratives: list[str] = []

    # Valuation (PE + PBV)
    val_parts: list[str] = []
    if pe is not None:
        if pe <= _PE_CHEAP:
            val_parts.append(f"PE {_fmt_num(pe, 1)}x (murah)")
        elif pe <= _PE_FAIR:
            val_parts.append(f"PE {_fmt_num(pe, 1)}x (wajar)")
        elif pe <= _PE_EXPENSIVE:
            val_parts.append(f"PE {_fmt_num(pe, 1)}x (agak mahal)")
        else:
            val_parts.append(f"PE {_fmt_num(pe, 1)}x (premium tinggi)")

    if pbv is not None:
        if pbv <= _PBV_CHEAP:
            val_parts.append(f"PBV {_fmt_num(pbv, 2)}x (undervalued)")
        elif pbv <= _PBV_EXPENSIVE:
            val_parts.append(f"PBV {_fmt_num(pbv, 2)}x (wajar)")
        else:
            val_parts.append(f"PBV {_fmt_num(pbv, 2)}x (premium)")

    if val_parts:
        narratives.append("Valuasi: " + " · ".join(val_parts) + ".")

    # Profitability (ROE + EPS)
    prof_parts: list[str] = []
    if roe is not None:
        if roe >= _ROE_GREAT:
            prof_parts.append(f"ROE {_fmt_num(roe, 1)}% (sangat baik)")
        elif roe >= _ROE_GOOD:
            prof_parts.append(f"ROE {_fmt_num(roe, 1)}% (baik)")
        elif roe > 0:
            prof_parts.append(f"ROE {_fmt_num(roe, 1)}% (moderat)")
        else:
            prof_parts.append(f"ROE {_fmt_num(roe, 1)}% (merugi / negatif)")

    if eps is not None:
        prof_parts.append(f"EPS Rp{_fmt_num(eps, 0)}")

    if prof_parts:
        narratives.append("Profitabilitas: " + " · ".join(prof_parts) + ".")

    # Leverage (DER)
    if der is not None:
        if der <= _DER_SAFE:
            narratives.append(f"DER {_fmt_num(der, 2)} — leverage rendah, struktur modal aman.")
        elif der <= _DER_HIGH:
            narratives.append(f"DER {_fmt_num(der, 2)} — leverage moderat, masih dalam batas wajar.")
        else:
            narratives.append(f"DER {_fmt_num(der, 2)} — leverage tinggi, perhatikan risiko bunga.")

    # Dividend
    if div is not None and div > 0:
        if div >= _DIV_GOOD:
            narratives.append(
                f"Dividend yield {_fmt_num(div, 2)}% — imbal hasil dividen menarik untuk investor defensif."
            )
        else:
            narratives.append(f"Dividend yield {_fmt_num(div, 2)}%.")

    # Growth
    growth_parts: list[str] = []
    if rev_g is not None:
        sign = "+" if rev_g >= 0 else ""
        growth_parts.append(f"revenue {sign}{_fmt_num(rev_g, 1)}%")
    if prof_g is not None:
        sign = "+" if prof_g >= 0 else ""
        growth_parts.append(f"laba {sign}{_fmt_num(prof_g, 1)}%")
    if growth_parts:
        narratives.append("Pertumbuhan YoY: " + ", ".join(growth_parts) + ".")

    lines.extend(narratives)

    # ── Overall fundamental summary sentence ──────────────────────────────
    if fund_status == "partial":
        lines.append("_(Data fundamental parsial — tidak semua metrik tersedia.)_")

    return lines if lines else [
        "_Data fundamental tersedia tapi tidak ada metrik yang dapat diinterpretasi._"
    ]


# ---------------------------------------------------------------------------
# Level 2: LLM explanation (skeleton — uncomment saat API key tersedia)
# ---------------------------------------------------------------------------

def explain_signal_llm(
    row: pd.Series,
    api_key: str | None = None,
) -> str:
    """Gunakan Claude API untuk narasi yang lebih natural.

    Jika api_key tidak tersedia, fallback otomatis ke explain_signal().
    """
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return explain_signal(row)

    prompt = _build_llm_prompt(row)

    # TODO: Uncomment dan install `anthropic` untuk integrasi nyata
    #
    # import anthropic
    # client = anthropic.Anthropic(api_key=key)
    # response = client.messages.create(
    #     model="claude-sonnet-4-6",
    #     max_tokens=500,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # return response.content[0].text.strip()

    return explain_signal(row)


def _build_llm_prompt(row: pd.Series) -> str:
    """Build prompt terstruktur 3 dimensi untuk Claude."""
    ticker  = row.get("ticker", "?")
    signal  = row.get("signal", "NONE")
    total   = _fmt_num(row.get("total_score"), 1)
    close   = _fmt_price(row.get("close"))
    ml_prob = row.get("ml_prob")

    # Teknikal
    tech_lines = "\n".join([
        f"  RSI 14: {_fmt_num(row.get('rsi14'), 1)}",
        f"  MACD Histogram: {_fmt_num(row.get('macd_histogram'), 3)}",
        f"  Volume Ratio 20d: {_fmt_num(row.get('vol_ratio_20d'), 2)}x",
        f"  % dari 52-week High: {_fmt_num(row.get('pct_from_52w_high'), 1)}%",
        f"  ATR Breakout: {row.get('atr_breakout', False)}",
        f"  MA Full Alignment (bullish): {row.get('ma_full_alignment', False)}",
        f"  Supertrend Bullish: {row.get('supertrend_bullish', False)}",
        f"  OBV Trend Naik: {row.get('obv_trend', False)}",
        f"  ROC 5 hari: {_fmt_num(row.get('roc5'), 1)}%",
        f"  ADX: {_fmt_num(row.get('adx'), 0)}",
    ])

    # News
    news_status = str(row.get("news_data_status", "")).lower()
    news_count  = _num(row.get("news_count_3d"))
    news_mean   = _num(row.get("news_sentiment_mean"))
    if news_status == "failed":
        news_lines = "  Status: GAGAL — provider error"
    elif news_status == "none":
        news_lines = "  Status: KOSONG — tidak ada berita 3 hari terakhir"
    elif news_status == "ok" and news_count is not None:
        news_lines = "\n".join([
            f"  Jumlah artikel: {int(news_count)}",
            f"  Sentimen rata-rata: {_fmt_num(news_mean, 2)} (-1 s/d 1)",
            f"  Positif: {_fmt_num(row.get('news_positive_count'), 0)} | "
            f"Negatif: {_fmt_num(row.get('news_negative_count'), 0)}",
            f"  Skor sentimen (0-10): {_fmt_num(row.get('news_sentiment_score'), 1)}",
        ])
    else:
        news_lines = "  Status: tidak tersedia (scan lama)"

    # Fundamental
    fund_status = str(row.get("fundamental_status", "")).lower()
    pe  = _num(row.get("pe_ratio"))
    pbv = _num(row.get("pbv"))
    roe = _num(row.get("roe_pct"))
    der = _num(row.get("der"))
    div = _num(row.get("div_yield_pct"))

    if fund_status == "missing" or all(v is None for v in [pe, pbv, roe, der, div]):
        fund_lines = "  Data fundamental tidak tersedia."
    else:
        fund_lines = "\n".join(filter(None, [
            f"  PE: {_fmt_num(pe, 1)}x" if pe else None,
            f"  PBV: {_fmt_num(pbv, 2)}x" if pbv else None,
            f"  ROE: {_fmt_num(roe, 1)}%" if roe else None,
            f"  DER: {_fmt_num(der, 2)}" if der else None,
            f"  Dividend Yield: {_fmt_num(div, 2)}%" if div else None,
            f"  Status: {fund_status}",
        ]))

    ml_line = (
        f"\nML Probability (return >3% dalam 5 hari): {round(float(ml_prob), 3)}"
        if ml_prob is not None and not pd.isna(ml_prob)
        else ""
    )

    return f"""Kamu adalah analis saham IDX Indonesia. Berikan penjelasan singkat (maksimal 180 kata, dalam bahasa Indonesia) tentang sinyal berikut dalam 3 bagian ringkas.

Ticker: {ticker}
Harga Penutupan: {close}
Signal: {signal}
Total Score: {total}/10
{ml_line}

=== DIMENSI 1: TEKNIKAL ===
{tech_lines}

=== DIMENSI 2: NEWS & CATALYST ===
{news_lines}

=== DIMENSI 3: FUNDAMENTAL ===
{fund_lines}

Format output:
🔧 **Teknikal:** [2-3 kalimat tentang indikator utama]
📰 **News:** [1-2 kalimat; jika status GAGAL tulis "data tidak tersedia hari ini"; jika KOSONG tulis "tidak ada katalis berita"]
📊 **Fundamental:** [1-2 kalimat dari data yang ada; jika tidak ada data tulis jujur]
⚠️ **Risiko:** [1 kalimat risiko utama atau hal yang perlu dikonfirmasi]
*Bukan rekomendasi investasi.*"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num(val: Any) -> float | None:
    try:
        v = float(val)
        return None if pd.isna(v) else v
    except (TypeError, ValueError):
        return None


def _fmt_num(val: Any, decimals: int = 2) -> str:
    n = _num(val)
    return f"{n:.{decimals}f}" if n is not None else "N/A"


def _fmt_price(val: Any) -> str:
    n = _num(val)
    if n is None:
        return "N/A"
    if n >= 1000:
        return f"Rp {n:,.0f}"
    return f"Rp {n:.2f}"


def _truthy(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    try:
        return bool(val)
    except Exception:
        return False


def _score_bar(score: float, width: int = 10) -> str:
    """Visualisasi mini bar untuk skor 0–10."""
    filled = max(0, min(width, round(score)))
    return "█" * filled + "░" * (width - filled)

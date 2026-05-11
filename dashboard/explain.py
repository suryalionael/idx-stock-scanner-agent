"""Explanation generator untuk sinyal saham IDX.

Dua level:
  1. explain_signal()     — rule-based, selalu jalan tanpa API key
  2. explain_signal_llm() — Claude API (skeleton, uncomment saat siap)

Penjelasan dibagi 3 dimensi:
  🔧 Teknikal  — indikator harga & volume (MA, RSI, OBV, ATR, dll.)
  📰 News/Catalyst — sentimen berita 3 hari terakhir via news_sentiment pipeline
  📊 Fundamental — data emiten (placeholder sampai data tersedia)

Logika rule-based menggunakan kondisi fitur nyata sehingga penjelasan
bisa berbeda tiap ticker, bukan template statis.
"""
import os
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Level 1: Rule-based explanation (selalu tersedia)
# ---------------------------------------------------------------------------

def explain_signal(row: pd.Series) -> str:
    """Buat penjelasan berbasis rule dari fitur satu ticker.

    Tidak butuh API key. Output adalah teks Markdown dengan 3 seksi.
    """
    signal = str(row.get("signal", "NONE"))
    ticker = str(row.get("ticker", "?"))
    total = _fmt_num(row.get("total_score"), 1)
    enh   = _fmt_num(row.get("enhanced_total_score"), 1)

    parts: list[str] = []

    # === Kalimat pembuka per signal =========================================
    opener = {
        "BREAKOUT":   f"**{ticker}** mendapat sinyal **BREAKOUT** — "
                      f"skor {total}/10 (enhanced: {enh}/10).",
        "PRE_MARKUP": f"**{ticker}** mendapat sinyal **PRE_MARKUP** (kandidat menjelang breakout) "
                      f"— skor {total}/10 (enhanced: {enh}/10).",
        "WATCH":      f"**{ticker}** masuk watchlist dengan skor {total}/10 "
                      f"— belum cukup kuat untuk aksi, tapi layak dipantau.",
        "AVOID":      f"**{ticker}** mendapat sinyal **AVOID** (skor {total}/10) "
                      f"— kondisi saat ini tidak mendukung entry.",
        "NONE":       f"**{ticker}** tidak menghasilkan sinyal yang jelas (skor {total}/10).",
    }
    parts.append(opener.get(signal, opener["NONE"]))
    parts.append("")

    # === SEKSI 1: TEKNIKAL ===================================================
    parts.append("### 🔧 Teknikal")

    tech_parts: list[str] = []

    # --- Tren MA ---
    ma_full    = _truthy(row.get("ma_full_alignment"))
    ma_partial = _truthy(row.get("ma_partial_alignment"))
    ma200      = _num(row.get("ma200"))
    ma50       = _num(row.get("ma50"))
    close_num  = _num(row.get("close"))
    close_str  = _fmt_price(row.get("close"))

    if ma_full:
        tech_parts.append(
            "Tren jangka pendek, menengah, dan panjang selaras (MA20 > MA50 > MA200) "
            "— struktur teknikal bullish."
        )
    elif ma_partial:
        tech_parts.append(
            "MA20 berada di atas MA50, namun MA200 belum dilewati "
            "— tren menengah mulai positif tapi tren panjang belum konfirmasi."
        )
    else:
        if ma200 and close_num and close_num < ma200:
            tech_parts.append(
                f"Harga ({close_str}) masih di bawah MA200 ({_fmt_price(ma200)}) "
                "— tren jangka panjang masih turun."
            )
        elif ma50 and close_num and close_num < ma50:
            tech_parts.append("Harga masih di bawah MA50 — tren menengah belum mendukung.")

    # ADX / Supertrend
    adx = _num(row.get("adx"))
    if adx is not None:
        if adx >= 40:
            tech_parts.append(f"ADX {_fmt_num(adx, 0)} — tren sangat kuat.")
        elif adx >= 25:
            tech_parts.append(f"ADX {_fmt_num(adx, 0)} — pasar sedang trending.")
        else:
            tech_parts.append(f"ADX {_fmt_num(adx, 0)} — tren lemah, pasar cenderung sideways.")

    if _truthy(row.get("supertrend_bullish")):
        tech_parts.append("Supertrend menunjukkan sinyal bullish.")

    # --- Momentum RSI ---
    rsi = _num(row.get("rsi14"))
    if rsi is not None:
        if rsi > 80:
            tech_parts.append(
                f"RSI {_fmt_num(rsi, 1)} — sangat overbought, risiko koreksi jangka pendek tinggi."
            )
        elif rsi > 70:
            tech_parts.append(
                f"RSI {_fmt_num(rsi, 1)} — mendekati overbought, momentum kuat tapi perlu hati-hati."
            )
        elif 50 <= rsi <= 70:
            tech_parts.append(f"RSI {_fmt_num(rsi, 1)} — zona momentum sehat, tidak overbought.")
        elif 40 <= rsi < 50:
            tech_parts.append(
                f"RSI {_fmt_num(rsi, 1)} — momentum netral, belum ada dorongan kuat ke atas."
            )
        else:
            tech_parts.append(
                f"RSI {_fmt_num(rsi, 1)} — oversold, potensi rebound tapi konfirmasi volume diperlukan."
            )

    # MACD
    macd_hist = _num(row.get("macd_histogram"))
    if macd_hist is not None:
        if macd_hist > 0:
            tech_parts.append(f"MACD histogram positif ({_fmt_num(macd_hist, 3)}) — momentum bullish.")
        else:
            tech_parts.append(f"MACD histogram negatif ({_fmt_num(macd_hist, 3)}) — momentum bearish.")

    # Squeeze
    if _truthy(row.get("squeeze_on")):
        tech_parts.append("Bollinger Band Squeeze aktif — potensi breakout directional segera.")

    # --- Volume ---
    vol_ratio = _num(row.get("vol_ratio_20d"))
    obv_trend = _truthy(row.get("obv_trend"))

    if vol_ratio is not None:
        if vol_ratio >= 2.0:
            tech_parts.append(
                f"Volume melonjak {_fmt_num(vol_ratio, 1)}x rata-rata 20 hari "
                "— sinyal partisipasi pasar yang kuat (akumulasi institusional mungkin terjadi)."
            )
        elif vol_ratio >= 1.3:
            tech_parts.append(
                f"Volume di atas rata-rata ({_fmt_num(vol_ratio, 1)}x) "
                "— ada peningkatan minat, belum signifikan."
            )
        else:
            tech_parts.append(
                "Volume relatif normal atau di bawah rata-rata "
                "— belum ada konfirmasi partisipasi besar."
            )

    if obv_trend:
        tech_parts.append("OBV (On-Balance Volume) sedang naik, mengindikasikan akumulasi secara kumulatif.")

    # --- Posisi terhadap 52-week high ---
    pct_52w = _num(row.get("pct_from_52w_high"))
    if pct_52w is not None:
        if pct_52w >= -5:
            tech_parts.append(
                f"Harga sangat dekat dengan 52-week high (hanya {_fmt_num(abs(pct_52w), 1)}% di bawah) "
                "— potensi breakout ke all-time high area."
            )
        elif pct_52w >= -15:
            tech_parts.append(
                f"Harga masih {_fmt_num(abs(pct_52w), 1)}% di bawah 52-week high "
                "— sedang mendekati area kritis."
            )
        else:
            tech_parts.append(
                f"Harga masih {_fmt_num(abs(pct_52w), 1)}% di bawah 52-week high "
                "— butuh perjalanan panjang untuk uji level tersebut."
            )

    # ATR Breakout
    if _truthy(row.get("atr_breakout")):
        tech_parts.append(
            "Harga bergerak lebih dari 1.5× ATR di atas penutupan kemarin "
            "— breakout harga yang signifikan secara statistik."
        )

    # ROC
    roc5 = _num(row.get("roc5"))
    if roc5 is not None:
        if roc5 > 5:
            tech_parts.append(f"ROC 5 hari sebesar +{_fmt_num(roc5, 1)}% — momentum jangka sangat pendek kuat.")
        elif roc5 < -5:
            tech_parts.append(f"ROC 5 hari sebesar {_fmt_num(roc5, 1)}% — pelemahan jangka sangat pendek.")

    parts.extend(tech_parts if tech_parts else ["Data teknikal tidak lengkap untuk ticker ini."])

    # === SEKSI 2: NEWS / CATALYST ============================================
    parts.append("")
    parts.append("### 📰 News & Catalyst (3 hari terakhir)")

    news_status  = str(row.get("news_data_status", "")).strip().lower()
    news_count   = _num(row.get("news_count_3d"))
    news_mean    = _num(row.get("news_sentiment_mean"))
    news_pos     = _num(row.get("news_positive_count"))
    news_neg     = _num(row.get("news_negative_count"))
    news_score_v = _num(row.get("news_sentiment_score"))
    news_source  = str(row.get("news_source", "")).strip()
    news_err     = row.get("news_error_message")

    if news_status == "failed":
        err_hint = f" ({news_err})" if news_err else ""
        parts.append(
            f"⚠️ **Data news tidak tersedia** — provider gagal mengambil data{err_hint}. "
            "Skor news diabaikan dalam kalkulasi (dianggap netral 5.0)."
        )
    elif news_status == "empty":
        parts.append(
            "Tidak ada berita yang ditemukan dalam 3 hari terakhir untuk ticker ini. "
            "Sentimen dianggap netral (tidak ada katalis positif maupun negatif)."
        )
    elif news_status == "ok" and news_count is not None:
        n = int(news_count)
        pos = int(news_pos) if news_pos is not None else 0
        neg = int(news_neg) if news_neg is not None else 0
        neu = n - pos - neg

        # Score bar (visual)
        if news_score_v is not None:
            score_bar = _score_bar(news_score_v)
            parts.append(
                f"Ditemukan **{n} artikel** dalam 3 hari terakhir "
                f"(sumber: {news_source or 'yfinance'})."
            )
            parts.append(
                f"Breakdown: 🟢 {pos} positif · 🔴 {neg} negatif · ⚪ {neu} netral  "
                f"→ Skor sentimen: **{_fmt_num(news_score_v, 1)}/10** {score_bar}"
            )

            if news_mean is not None:
                if news_mean > 0.2:
                    parts.append(
                        "Mayoritas berita bersifat positif — ada potensi katalis fundamental "
                        "yang mendukung momentum harga."
                    )
                elif news_mean < -0.2:
                    parts.append(
                        "Mayoritas berita bersifat negatif — waspadai risiko fundamental "
                        "yang dapat menekan harga."
                    )
                else:
                    parts.append("Berita cukup berimbang — tidak ada sentimen kuat ke satu arah.")
        else:
            parts.append(f"Ditemukan {n} artikel tapi skor sentimen tidak dapat dihitung.")
    else:
        # news_data_status kolom belum ada di data ini (scan lama sebelum upgrade)
        parts.append(
            "_Data news tidak tersedia untuk scan ini (format lama). "
            "Jalankan ulang scan untuk mendapatkan status news yang akurat._"
        )

    # === SEKSI 3: FUNDAMENTAL ================================================
    parts.append("")
    parts.append("### 📊 Fundamental")
    parts.append(
        "_Data fundamental (EPS, PBV, DER, ROE, dividend yield) belum terintegrasi "
        "dalam pipeline saat ini. Akan ditambahkan pada iterasi berikutnya._"
    )
    parts.append(
        "Untuk analisis fundamental, silakan cek laporan keuangan terakhir di "
        "[IDX](https://www.idx.co.id) atau platform seperti stockbit.com / RTI Business."
    )

    # === Disclaimer ==========================================================
    parts.append("")
    parts.append(
        "---\n"
        "⚠️ *Ini adalah output analisis teknikal otomatis, bukan rekomendasi investasi. "
        "Selalu lakukan riset sendiri (DYOR) sebelum mengambil keputusan.*"
    )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Level 2: LLM explanation (skeleton — uncomment saat API key tersedia)
# ---------------------------------------------------------------------------

def explain_signal_llm(
    row: pd.Series,
    api_key: str | None = None,
) -> str:
    """Gunakan Claude API untuk narasi yang lebih natural.

    Jika api_key tidak tersedia, fallback otomatis ke explain_signal().

    Args:
        row: Series dengan fitur + signal + ml_prob
        api_key: ANTHROPIC_API_KEY (jika None, baca dari env var)

    Returns:
        String narasi penjelasan
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
    #     max_tokens=400,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # return response.content[0].text.strip()

    # Fallback sampai API diaktifkan
    return explain_signal(row)


def _build_llm_prompt(row: pd.Series) -> str:
    """Build prompt terstruktur 3 dimensi untuk Claude."""
    ticker = row.get("ticker", "?")
    signal = row.get("signal", "NONE")
    total  = _fmt_num(row.get("total_score"), 1)
    close  = _fmt_price(row.get("close"))
    ml_prob = row.get("ml_prob")

    # --- Dimensi 1: Teknikal ---
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

    # --- Dimensi 2: News/Catalyst ---
    news_status = str(row.get("news_data_status", "")).lower()
    if news_status == "failed":
        news_lines = "  Status: GAGAL (provider error) — tidak tersedia"
    elif news_status == "empty":
        news_lines = "  Status: KOSONG — tidak ada berita 3 hari terakhir"
    elif news_status == "ok":
        news_lines = "\n".join([
            f"  Jumlah artikel: {_fmt_num(row.get('news_count_3d'), 0)}",
            f"  Sentimen rata-rata: {_fmt_num(row.get('news_sentiment_mean'), 2)} (-1 s/d 1)",
            f"  Positif: {_fmt_num(row.get('news_positive_count'), 0)} | "
            f"Negatif: {_fmt_num(row.get('news_negative_count'), 0)}",
            f"  Skor sentimen (0-10): {_fmt_num(row.get('news_sentiment_score'), 1)}",
        ])
    else:
        news_lines = "  Status: tidak tersedia (scan lama)"

    # --- Dimensi 3: Fundamental ---
    fundamental_lines = "  Data fundamental belum tersedia dalam pipeline."

    ml_line = (
        f"\nML Probability (return >3% dalam 5 hari): {round(float(ml_prob), 3)}"
        if ml_prob is not None and not pd.isna(ml_prob)
        else ""
    )

    return f"""Kamu adalah analis saham IDX Indonesia. Berikan penjelasan singkat (maksimal 150 kata, dalam bahasa Indonesia) tentang sinyal berikut dalam 3 bagian ringkas.

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
{fundamental_lines}

Jelaskan dalam format:
🔧 **Teknikal:** [1-2 kalimat tentang indikator utama]
📰 **News:** [1 kalimat tentang sentimen berita, atau "tidak ada data" jika gagal/kosong]
📊 **Fundamental:** [1 kalimat jujur bahwa data belum tersedia]
⚠️ **Risiko:** [1 kalimat risiko utama atau konfirmasi yang dibutuhkan]
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

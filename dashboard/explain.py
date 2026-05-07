"""Explanation generator untuk sinyal saham IDX.

Dua level:
  1. explain_signal()     — rule-based, selalu jalan tanpa API key
  2. explain_signal_llm() — Claude API (skeleton, uncomment saat siap)

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

    Tidak butuh API key. Output adalah paragraf 4–8 kalimat.
    """
    signal = str(row.get("signal", "NONE"))
    ticker = str(row.get("ticker", "?"))
    close = _fmt_price(row.get("close"))
    total = _fmt_num(row.get("total_score"), 1)

    parts: list[str] = []

    # --- Kalimat pembuka per signal ---
    opener = {
        "BREAKOUT": f"**{ticker}** mendapat sinyal **BREAKOUT** dengan total skor {total}/10.",
        "PRE_MARKUP": f"**{ticker}** mendapat sinyal **PRE_MARKUP** (kandidat menjelang breakout) dengan skor {total}/10.",
        "WATCH": f"**{ticker}** masuk watchlist dengan skor {total}/10 — belum cukup kuat untuk aksi, tapi layak dipantau.",
        "AVOID": f"**{ticker}** mendapat sinyal **AVOID** (skor {total}/10) — kondisi saat ini tidak mendukung entry.",
        "NONE": f"**{ticker}** tidak menghasilkan sinyal yang jelas (skor {total}/10).",
    }
    parts.append(opener.get(signal, opener["NONE"]))

    # --- Tren MA ---
    ma_full = _truthy(row.get("ma_full_alignment"))
    ma_partial = _truthy(row.get("ma_partial_alignment"))
    ma20 = _num(row.get("ma20"))
    ma50 = _num(row.get("ma50"))
    ma200 = _num(row.get("ma200"))
    close_num = _num(row.get("close"))

    if ma_full:
        parts.append("Tren jangka pendek, menengah, dan panjang selaras (MA20 > MA50 > MA200) — struktur teknikal bullish.")
    elif ma_partial:
        parts.append("MA20 berada di atas MA50, namun MA200 belum dilewati — tren menengah mulai positif tapi tren panjang belum konfirmasi.")
    else:
        if ma200 and close_num and close_num < ma200:
            parts.append(f"Harga ({close}) masih di bawah MA200 ({_fmt_price(ma200)}) — tren jangka panjang masih turun.")
        elif ma50 and close_num and close_num < ma50:
            parts.append(f"Harga masih di bawah MA50 — tren menengah belum mendukung.")

    # --- Momentum RSI ---
    rsi = _num(row.get("rsi14"))
    if rsi is not None:
        if rsi > 80:
            parts.append(f"RSI {_fmt_num(rsi, 1)} — sangat overbought, risiko koreksi jangka pendek tinggi.")
        elif rsi > 70:
            parts.append(f"RSI {_fmt_num(rsi, 1)} — mendekati overbought, momentum kuat tapi perlu hati-hati.")
        elif 50 <= rsi <= 70:
            parts.append(f"RSI {_fmt_num(rsi, 1)} — zona momentum sehat, tidak overbought.")
        elif 40 <= rsi < 50:
            parts.append(f"RSI {_fmt_num(rsi, 1)} — momentum netral, belum ada dorongan kuat ke atas.")
        else:
            parts.append(f"RSI {_fmt_num(rsi, 1)} — oversold, potensi rebound tapi konfirmasi volume diperlukan.")

    # --- Volume ---
    vol_spike = _truthy(row.get("vol_spike"))
    vol_ratio = _num(row.get("vol_ratio_20d"))
    obv_trend = _truthy(row.get("obv_trend"))

    if vol_spike and vol_ratio and vol_ratio >= 2.0:
        parts.append(f"Volume melonjak {_fmt_num(vol_ratio, 1)}x rata-rata 20 hari — sinyal partisipasi pasar yang kuat (akumulasi institusional mungkin terjadi).")
    elif vol_ratio and vol_ratio >= 1.3:
        parts.append(f"Volume di atas rata-rata ({_fmt_num(vol_ratio, 1)}x) — ada peningkatan minat, belum signifikan.")
    else:
        parts.append("Volume relatif normal atau di bawah rata-rata — belum ada konfirmasi partisipasi besar.")

    if obv_trend:
        parts.append("OBV (On-Balance Volume) sedang naik, mengindikasikan akumulasi secara kumulatif.")

    # --- Posisi terhadap 52-week high ---
    pct_52w = _num(row.get("pct_from_52w_high"))
    if pct_52w is not None:
        if pct_52w >= -5:
            parts.append(f"Harga sangat dekat dengan 52-week high (hanya {_fmt_num(abs(pct_52w), 1)}% di bawah) — potensi breakout ke all-time high area.")
        elif pct_52w >= -15:
            parts.append(f"Harga masih {_fmt_num(abs(pct_52w), 1)}% di bawah 52-week high — sedang mendekati area kritis.")
        else:
            parts.append(f"Harga masih {_fmt_num(abs(pct_52w), 1)}% di bawah 52-week high — butuh perjalanan panjang untuk uji level tersebut.")

    # --- ATR Breakout ---
    atr_breakout = _truthy(row.get("atr_breakout"))
    if atr_breakout:
        parts.append("Harga bergerak lebih dari 1.5× ATR di atas penutupan kemarin — breakout harga yang signifikan secara statistik.")

    # --- ROC momentum short-term ---
    roc5 = _num(row.get("roc5"))
    if roc5 is not None:
        if roc5 > 5:
            parts.append(f"ROC 5 hari sebesar +{_fmt_num(roc5, 1)}% — momentum jangka sangat pendek kuat.")
        elif roc5 < -5:
            parts.append(f"ROC 5 hari sebesar {_fmt_num(roc5, 1)}% — pelemahan jangka sangat pendek.")

    # --- Disclaimer ---
    parts.append("⚠️ Ini adalah output analisis teknikal otomatis, bukan rekomendasi investasi. Selalu lakukan riset sendiri sebelum mengambil keputusan.")

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
    #     max_tokens=300,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # return response.content[0].text.strip()

    # Alternatif dengan requests (tanpa SDK):
    #
    # import requests
    # headers = {
    #     "x-api-key": key,
    #     "anthropic-version": "2023-06-01",
    #     "content-type": "application/json",
    # }
    # payload = {
    #     "model": "claude-sonnet-4-6",
    #     "max_tokens": 300,
    #     "messages": [{"role": "user", "content": prompt}],
    # }
    # resp = requests.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers)
    # resp.raise_for_status()
    # return resp.json()["content"][0]["text"].strip()

    # Fallback sampai API diaktifkan
    return explain_signal(row)


def _build_llm_prompt(row: pd.Series) -> str:
    """Build prompt terstruktur untuk Claude."""
    ticker = row.get("ticker", "?")
    signal = row.get("signal", "NONE")
    total = _fmt_num(row.get("total_score"), 1)
    close = _fmt_price(row.get("close"))
    ml_prob = row.get("ml_prob")

    feature_lines = "\n".join([
        f"  RSI 14: {_fmt_num(row.get('rsi14'), 1)}",
        f"  MACD Histogram: {_fmt_num(row.get('macd_histogram'), 3)}",
        f"  Volume Ratio 20d: {_fmt_num(row.get('vol_ratio_20d'), 2)}x",
        f"  % dari 52-week High: {_fmt_num(row.get('pct_from_52w_high'), 1)}%",
        f"  ATR Breakout: {row.get('atr_breakout', False)}",
        f"  MA Full Alignment (bullish): {row.get('ma_full_alignment', False)}",
        f"  OBV Trend Naik: {row.get('obv_trend', False)}",
        f"  ROC 5 hari: {_fmt_num(row.get('roc5'), 1)}%",
    ])

    ml_line = (
        f"\nML Probability (return >3% dalam 5 hari): {round(float(ml_prob), 3)}"
        if ml_prob is not None and not pd.isna(ml_prob)
        else ""
    )

    return f"""Kamu adalah analis teknikal untuk saham IDX Indonesia. Berikan penjelasan singkat (maksimal 120 kata, dalam bahasa Indonesia) tentang sinyal berikut.

Ticker: {ticker}
Harga Penutupan: {close}
Signal: {signal}
Total Score: {total}/10
{ml_line}

Indikator teknikal:
{feature_lines}

Jelaskan:
1. Mengapa ticker ini mendapat sinyal {signal} berdasarkan indikator di atas
2. Satu risiko atau hal yang perlu dikonfirmasi sebelum entry
3. Disclaimer singkat bahwa ini bukan rekomendasi investasi

Ringkas, padat, dan gunakan bahasa yang mudah dipahami investor ritel."""


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

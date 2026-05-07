"""ExplainAgent — narasi singkat per ticker dari Claude API.

Alur:
  1. Ambil baris terkini ticker (fitur + signal + ml_prob)
  2. Build prompt terstruktur
  3. Kirim ke Claude API (claude-sonnet-4-6)
  4. Return teks penjelasan

SAAT INI: explain_signal() mengembalikan string dummy formatted.
Uncomment blok Claude API di bawah untuk integrasi nyata.
"""
import os

import pandas as pd
from loguru import logger

# Model default untuk explain
_CLAUDE_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 300


def explain_signal(
    ticker: str,
    row: pd.Series,
    model_info: dict | None = None,
    config: dict | None = None,
) -> str:
    """Buat narasi singkat untuk satu ticker berdasarkan fitur dan signal.

    Args:
        ticker: kode saham (mis. "BBCA.JK")
        row: Series dengan kolom fitur + signal + ml_prob
        model_info: info tambahan tentang model (opsional; mis. threshold, horizon)
        config: config dict (opsional; untuk ambil API key atau override model)

    Returns:
        String narasi penjelasan (50–150 kata).
    """
    prompt = _build_prompt(ticker, row, model_info or {})

    # --- TODO: Uncomment untuk integrasi Claude API nyata ---
    #
    # import anthropic
    # api_key = os.environ.get("ANTHROPIC_API_KEY") or (config or {}).get("anthropic_api_key")
    # if not api_key:
    #     logger.warning(f"{ticker}: ANTHROPIC_API_KEY tidak ditemukan, skip explain")
    #     return "[explain skipped: no API key]"
    #
    # client = anthropic.Anthropic(api_key=api_key)
    # response = client.messages.create(
    #     model=_CLAUDE_MODEL,
    #     max_tokens=_MAX_TOKENS,
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # return response.content[0].text.strip()

    # Sementara: kembalikan ringkasan formatted dari fitur
    return _format_stub(ticker, row)


def _build_prompt(ticker: str, row: pd.Series, model_info: dict) -> str:
    """Build prompt untuk Claude berdasarkan data ticker."""
    signal = row.get("signal", "NONE")
    close = row.get("close", "N/A")
    total_score = row.get("total_score", "N/A")
    ml_prob = row.get("ml_prob", None)

    key_features = {
        "RSI 14": row.get("rsi14"),
        "MACD Histogram": row.get("macd_histogram"),
        "Volume Ratio (20d)": row.get("vol_ratio_20d"),
        "% from 52w High": row.get("pct_from_52w_high"),
        "ATR Breakout": row.get("atr_breakout"),
        "MA Full Alignment": row.get("ma_full_alignment"),
        "OBV Trend Up": row.get("obv_trend"),
    }

    feature_lines = "\n".join(
        f"  - {k}: {round(float(v), 2) if v is not None and not pd.isna(v) else 'N/A'}"
        for k, v in key_features.items()
    )

    ml_line = f"- ML Probability (return >{model_info.get('target_return_pct', 3)}% in {model_info.get('target_horizon_days', 5)}d): {round(float(ml_prob), 3)}" if ml_prob is not None else ""

    prompt = f"""Kamu adalah analis teknikal untuk saham IDX. Berikan penjelasan singkat (maksimal 100 kata) tentang sinyal berikut.

Ticker: {ticker}
Harga Penutupan: {close}
Signal: {signal}
Total Score: {total_score}/10

Indikator teknikal kunci:
{feature_lines}
{ml_line}

Jelaskan secara singkat:
1. Kenapa ticker ini mendapat signal {signal}
2. Apa yang perlu diperhatikan (risiko atau konfirmasi yang dibutuhkan)
3. Satu kalimat disclaimer bahwa ini bukan rekomendasi investasi.

Gunakan bahasa Indonesia, ringkas dan jelas. Bukan nasihat investasi."""

    return prompt


def _format_stub(ticker: str, row: pd.Series) -> str:
    """Placeholder output sebelum Claude API diaktifkan."""
    signal = row.get("signal", "NONE")
    total = row.get("total_score", 0)
    rsi = row.get("rsi14")
    vol_ratio = row.get("vol_ratio_20d")
    pct_52w = row.get("pct_from_52w_high")
    ml_prob = row.get("ml_prob")

    parts = [f"[{ticker}] Signal: {signal} | Score: {round(float(total), 1)}/10"]

    if rsi is not None and not pd.isna(rsi):
        parts.append(f"RSI: {round(float(rsi), 1)}")
    if vol_ratio is not None and not pd.isna(vol_ratio):
        parts.append(f"Vol Ratio: {round(float(vol_ratio), 2)}x")
    if pct_52w is not None and not pd.isna(pct_52w):
        parts.append(f"52w High: {round(float(pct_52w), 1)}%")
    if ml_prob is not None and not pd.isna(ml_prob):
        parts.append(f"ML Prob: {round(float(ml_prob), 3)}")

    parts.append("(Explain stub — aktifkan Claude API di explain_agent.py)")

    return " | ".join(parts)


def explain_batch(
    df: pd.DataFrame,
    model_info: dict | None = None,
    config: dict | None = None,
    max_explain: int = 10,
) -> pd.DataFrame:
    """Tambahkan kolom 'explanation' ke DataFrame kandidat terkuat.

    Args:
        df: DataFrame dengan kolom signal + fitur
        max_explain: hanya proses N ticker teratas (hemat API call)

    Returns:
        df dengan kolom tambahan 'explanation'
    """
    df = df.copy()
    df["explanation"] = ""

    # Prioritaskan BREAKOUT dan PRE_MARKUP terlebih dahulu
    priority_signals = ["BREAKOUT", "PRE_MARKUP"]
    mask = df["signal"].isin(priority_signals)
    candidates = df[mask].head(max_explain)

    if candidates.empty:
        logger.info("Tidak ada kandidat priority untuk explain")
        return df

    logger.info(f"Generating explanations for {len(candidates)} tickers")
    for idx, row in candidates.iterrows():
        ticker = row.get("ticker", str(idx))
        try:
            explanation = explain_signal(ticker, row, model_info, config)
            df.at[idx, "explanation"] = explanation
        except Exception as e:
            logger.error(f"{ticker}: explain failed — {e}")
            df.at[idx, "explanation"] = f"[explain error: {e}]"

    return df

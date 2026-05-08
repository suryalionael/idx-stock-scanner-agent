"""News Sentiment Pipeline untuk IDX tickers.

Source (MVP): yfinance.Ticker.news  — free, tidak butuh API key.
Sentiment: keyword-based scoring (bilingual ID+EN).

Kolom output yang ditambahkan ke feature DataFrame:
    news_count_3d          — jumlah berita 3 hari terakhir
    news_sentiment_mean    — rata-rata skor per headline (-1 to 1)
    news_positive_count    — jumlah headline positif
    news_negative_count    — jumlah headline negatif
    news_sentiment_score   — skor final 0–10 (5.0 = neutral)

TODO: Ganti `_fetch_yf_news()` dengan source yang lebih kaya
      (mis. CNBC Indonesia scraper, IDX press release RSS, Bloomberg API)
      saat yfinance.Ticker.news tidak stabil.
"""
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Keyword lists — bilingual (Bahasa Indonesia + English)
# ---------------------------------------------------------------------------
_POSITIVE_WORDS = {
    # Indonesia
    "naik", "meningkat", "tumbuh", "positif", "untung", "laba", "profit",
    "rekor", "tinggi", "ekspansi", "akuisisi", "dividen", "buyback",
    "surplus", "optimis", "kuat", "membaik",
    # English
    "up", "rise", "gain", "high", "record", "growth", "profit", "strong",
    "beat", "exceed", "rally", "upgrade", "buy", "bullish", "outperform",
    "dividend", "acquisition", "expansion", "positive",
}

_NEGATIVE_WORDS = {
    # Indonesia
    "turun", "menurun", "rugi", "kerugian", "negatif", "rendah",
    "koreksi", "lemah", "gagal", "tunda", "suspend", "investigasi",
    "penurunan", "defisit", "pesimis", "tertekan",
    # English
    "down", "fall", "drop", "loss", "weak", "miss", "decline", "sell",
    "bearish", "downgrade", "suspend", "investigate", "delay", "risk",
    "concern", "below", "negative", "crash", "default",
}


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------

def score_headline(text: str) -> float:
    """Skor satu headline: +1 positif, -1 negatif, 0 netral."""
    if not text:
        return 0.0
    words = set(text.lower().replace(",", " ").replace(".", " ").split())
    pos_hits = len(words & _POSITIVE_WORDS)
    neg_hits = len(words & _NEGATIVE_WORDS)
    if pos_hits == neg_hits:
        return 0.0
    return 1.0 if pos_hits > neg_hits else -1.0


def sentiment_to_score(mean_sentiment: float) -> float:
    """Konversi mean sentiment (-1 to 1) ke skala 0–10 (5.0 = neutral)."""
    return round(5.0 + mean_sentiment * 5.0, 2)


# ---------------------------------------------------------------------------
# Fetch via yfinance
# ---------------------------------------------------------------------------

def _fetch_yf_news(ticker: str, days: int = 3) -> list[dict]:
    """Ambil berita dari yfinance.Ticker.news dan filter N hari terakhir.

    Returns:
        List of {"title": str, "published": datetime, "publisher": str}
    """
    try:
        import yfinance as yf
        raw_news = yf.Ticker(ticker).news  # list of dicts
    except Exception as e:
        logger.debug(f"{ticker}: yf.news gagal — {e}")
        return []

    if not raw_news:
        return []

    cutoff = datetime.utcnow() - timedelta(days=days)
    results = []
    for item in raw_news:
        try:
            pub_ts = item.get("providerPublishTime") or item.get("publishedAt")
            if pub_ts is None:
                continue
            # providerPublishTime adalah Unix timestamp (int)
            pub_dt = datetime.utcfromtimestamp(int(pub_ts))
            if pub_dt < cutoff:
                continue
            title = item.get("title") or item.get("content", {}).get("title", "")
            publisher = item.get("publisher", "")
            results.append({"title": str(title), "published": pub_dt, "publisher": publisher})
        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_news_sentiment(ticker: str, news_days: int = 3) -> dict:
    """Ambil berita dan hitung sentiment untuk satu ticker.

    Returns:
        dict dengan keys: news_count_3d, news_sentiment_mean,
                          news_positive_count, news_negative_count,
                          news_sentiment_score
    """
    news = _fetch_yf_news(ticker, days=news_days)
    default = {
        "news_count_3d": 0,
        "news_sentiment_mean": 0.0,
        "news_positive_count": 0,
        "news_negative_count": 0,
        "news_sentiment_score": 5.0,  # neutral
    }
    if not news:
        return default

    scores = [score_headline(item["title"]) for item in news]
    mean_score = sum(scores) / len(scores)
    return {
        "news_count_3d": len(scores),
        "news_sentiment_mean": round(mean_score, 3),
        "news_positive_count": sum(1 for s in scores if s > 0),
        "news_negative_count": sum(1 for s in scores if s < 0),
        "news_sentiment_score": sentiment_to_score(mean_score),
    }


def enrich_with_news(
    df_features: pd.DataFrame,
    news_dir: Path | None = None,
    news_days: int = 3,
    save: bool = True,
    scan_date: str | None = None,
) -> pd.DataFrame:
    """Tambahkan kolom news sentiment ke feature DataFrame.

    Dipanggil di run_daily_scan.py setelah build_features.
    Untuk setiap ticker, fetch berita dan tambahkan kolom sentiment.
    Jika fetch gagal, isi dengan nilai neutral (tidak crash pipeline).

    Args:
        df_features: DataFrame dengan kolom 'ticker' (satu baris per ticker)
        news_dir: direktori untuk cache news (opsional)
        news_days: berapa hari terakhir berita yang dihitung
        save: simpan hasil ke news_dir/{scan_date}.parquet

    Returns:
        df_features dengan 5 kolom sentiment tambahan
    """
    NEWS_COLS = [
        "news_count_3d", "news_sentiment_mean",
        "news_positive_count", "news_negative_count",
        "news_sentiment_score",
    ]

    df = df_features.copy()
    results = []

    for ticker in df["ticker"].unique():
        try:
            sentiment = compute_news_sentiment(ticker, news_days)
            sentiment["ticker"] = ticker
            results.append(sentiment)
            logger.debug(f"{ticker}: news={sentiment['news_count_3d']}, score={sentiment['news_sentiment_score']:.1f}")
        except Exception as e:
            logger.warning(f"{ticker}: news sentiment gagal — {e}")
            results.append({"ticker": ticker, **{c: (0 if "count" in c else 5.0) for c in NEWS_COLS}})

    news_df = pd.DataFrame(results)
    df = df.merge(news_df[["ticker"] + NEWS_COLS], on="ticker", how="left")

    # Isi NaN dengan neutral
    for col in NEWS_COLS:
        default_val = 0 if "count" in col else 5.0
        df[col] = df[col].fillna(default_val)

    if save and news_dir and scan_date:
        news_dir.mkdir(parents=True, exist_ok=True)
        path = news_dir / f"{scan_date}.parquet"
        news_df.to_parquet(path, index=False)
        logger.info(f"News sentiment saved → {path}")

    n_positive = (news_df["news_sentiment_score"] > 6).sum()
    n_negative = (news_df["news_sentiment_score"] < 4).sum()
    logger.info(f"News enrichment done: +{n_positive} positive, -{n_negative} negative")
    return df

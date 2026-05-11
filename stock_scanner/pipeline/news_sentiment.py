"""News Sentiment Pipeline — IDX Stock Scanner.

Architecture
------------
Providers (fetch raw articles)
    BaseNewsProvider        ← abstract
    GoogleNewsRssProvider   — free, no API key; primary for IDX/Indonesia news
    YFinanceNewsProvider    — free, no API key; fallback (often empty for .JK)
    RssNewsProvider         — TODO: IDX press release / CNBC Indonesia RSS

Aggregator
    NewsAggregator     — tries providers in priority order, records which
                         succeeded/failed, surfaces explicit status

Output columns added to feature DataFrame
    news_count_3d          — int: articles found in last N days (0 if none)
    news_sentiment_mean    — float | NaN  (NaN when fetch failed)
    news_positive_count    — int
    news_negative_count    — int
    news_sentiment_score   — float 0–10 | NaN  (NaN when fetch failed)
    news_data_status       — "ok" | "none" | "failed"
    news_source            — "google_rss" | "yfinance" | "rss" | "none" | "all_failed"
    news_error_message     — str | None  (populated only on failure)

Status semantics
    "ok"     : fetch succeeded and ≥1 article found
    "none"   : fetch succeeded but 0 articles in the window
                → score 5.0 (valid neutral — no news ≠ failed)
    "failed" : all providers failed → score NaN ← NOT the same as neutral

Downstream contract
    _news_score() in signal_engine.py uses fillna(5.0) so scoring is unaffected.
    Dashboard and explanation code must check news_data_status before trusting score.
"""
from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Keyword lists — bilingual (Bahasa Indonesia + English)
# ---------------------------------------------------------------------------

_POSITIVE_WORDS = {
    "naik", "meningkat", "tumbuh", "positif", "untung", "laba", "profit",
    "rekor", "tinggi", "ekspansi", "akuisisi", "dividen", "buyback",
    "surplus", "optimis", "kuat", "membaik", "capai", "cetak",
    "up", "rise", "gain", "high", "record", "growth", "profit", "strong",
    "beat", "exceed", "rally", "upgrade", "buy", "bullish", "outperform",
    "dividend", "acquisition", "expansion", "positive", "surge", "soar",
}

_NEGATIVE_WORDS = {
    "turun", "menurun", "rugi", "kerugian", "negatif", "rendah",
    "koreksi", "lemah", "gagal", "tunda", "suspend", "investigasi",
    "penurunan", "defisit", "pesimis", "tertekan", "anjlok", "ambles",
    "down", "fall", "drop", "loss", "weak", "miss", "decline", "sell",
    "bearish", "downgrade", "suspend", "investigate", "delay", "risk",
    "concern", "below", "negative", "crash", "default", "fraud", "slump",
}


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class NewsProviderError(Exception):
    """Raised by a provider when fetch fails — lets Aggregator try next provider."""


# ---------------------------------------------------------------------------
# Provider base + implementations
# ---------------------------------------------------------------------------

class BaseNewsProvider(ABC):
    """Interface for a news source.

    Subclasses must raise NewsProviderError (not return empty list) when the
    fetch genuinely fails — empty list is a valid "no news found" result.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and news_source column."""

    @abstractmethod
    def fetch(self, ticker: str, days: int) -> list[dict]:
        """Fetch recent articles for ticker.

        Args:
            ticker : e.g. "BBCA.JK"
            days   : look-back window in days

        Returns:
            List of dicts: {"title": str, "published": datetime, "publisher": str}
            Empty list is valid (no news in window).

        Raises:
            NewsProviderError: on network failure, malformed response, timeout.
        """


# ---------------------------------------------------------------------------
# Provider 1: Google News RSS (primary — works for IDX Indonesian tickers)
# ---------------------------------------------------------------------------

_GOOGLE_NEWS_URL = (
    "https://news.google.com/rss/search"
    "?q={query}&hl=id&gl=ID&ceid=ID:id"
)
_REQUEST_TIMEOUT_SECS = 12


class GoogleNewsRssProvider(BaseNewsProvider):
    """Fetch IDX news via Google News RSS (no API key required).

    Search query: "{ticker_code} saham" where ticker_code strips the .JK suffix
    so that "BBCA.JK" → searches "BBCA saham".

    Known limitations:
    - Rate limits if called too aggressively (adds delay between calls)
    - Pub dates from Google are sometimes approximated (07:00 UTC)
    - Returns articles about the company, not just price/signal news
    """

    def __init__(self, delay_between_calls: float = 0.5) -> None:
        self._delay = delay_between_calls
        self._last_call: float = 0.0

    @property
    def name(self) -> str:
        return "google_rss"

    def fetch(self, ticker: str, days: int) -> list[dict]:
        # Rate limiting
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)

        # Build ticker code (strip .JK suffix for better search)
        code = ticker.split(".")[0]
        query = urllib.parse.quote(f"{code} saham")
        url = _GOOGLE_NEWS_URL.format(query=query)

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; IDX-Scanner/1.0)"},
            )
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECS) as resp:
                xml_data = resp.read()
            self._last_call = time.monotonic()
        except OSError as exc:
            self._last_call = time.monotonic()
            raise NewsProviderError(f"Google News RSS network error: {exc}") from exc
        except Exception as exc:
            self._last_call = time.monotonic()
            raise NewsProviderError(f"Google News RSS request failed: {exc}") from exc

        # Parse XML
        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as exc:
            raise NewsProviderError(f"Google News RSS XML parse error: {exc}") from exc

        items = root.findall(".//item")
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
        results: list[dict] = []
        malformed = 0

        for item in items:
            try:
                title_el = item.find("title")
                pub_el   = item.find("pubDate")
                source_el = item.find("source")

                title = title_el.text.strip() if title_el is not None and title_el.text else ""
                publisher = source_el.text.strip() if source_el is not None and source_el.text else "Google News"

                if pub_el is None or not pub_el.text:
                    malformed += 1
                    continue

                # parsedate_to_datetime handles RFC 2822 dates with timezone
                pub_dt = parsedate_to_datetime(pub_el.text.strip())
                # Ensure timezone-aware comparison
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)

                if pub_dt < cutoff:
                    continue  # outside the look-back window

                results.append({
                    "title":     title,
                    "published": pub_dt.replace(tzinfo=None),  # store as naive UTC
                    "publisher": publisher,
                })
            except Exception:
                malformed += 1
                continue

        if malformed:
            logger.debug("%s: Google RSS — %d malformed items skipped", ticker, malformed)

        return results


# ---------------------------------------------------------------------------
# Provider 2: yfinance (fallback — often empty for .JK tickers)
# ---------------------------------------------------------------------------

class YFinanceNewsProvider(BaseNewsProvider):
    """News via yfinance.Ticker.news — free, no API key required.

    Known failure modes handled explicitly:
    - Returns dict instead of list (yfinance "faulty response" error object)
    - providerPublishTime missing or non-numeric
    - API timeout / network error

    Note: As of 2025, yfinance returns empty list for most .JK tickers.
    Use GoogleNewsRssProvider as primary; this is the fallback.
    """

    @property
    def name(self) -> str:
        return "yfinance"

    def fetch(self, ticker: str, days: int) -> list[dict]:
        try:
            import yfinance as yf
            raw = yf.Ticker(ticker).news
        except Exception as exc:
            raise NewsProviderError(f"yfinance import/call failed: {exc}") from exc

        # ── Guard: yfinance sometimes returns a dict error object ──────────
        if raw is None:
            raise NewsProviderError("yfinance.news returned None")
        if isinstance(raw, dict):
            # e.g. {"faulty": True, "error": "Failed to retrieve the news..."}
            err_msg = raw.get("error") or raw.get("message") or str(list(raw.keys()))
            raise NewsProviderError(f"yfinance returned dict (malformed): {err_msg}")
        if not isinstance(raw, list):
            raise NewsProviderError(
                f"yfinance.news unexpected type {type(raw).__name__}"
            )

        # ── Parse articles ─────────────────────────────────────────────────
        cutoff = datetime.utcnow() - timedelta(days=days)
        results: list[dict] = []
        malformed = 0

        for item in raw:
            try:
                pub_ts = item.get("providerPublishTime") or item.get("publishedAt")
                if pub_ts is None:
                    malformed += 1
                    continue
                pub_dt = datetime.utcfromtimestamp(int(pub_ts))
                if pub_dt < cutoff:
                    continue
                title = (
                    item.get("title")
                    or (item.get("content") or {}).get("title")
                    or ""
                )
                publisher = item.get("publisher", "")
                results.append({
                    "title":     str(title).strip(),
                    "published": pub_dt,
                    "publisher": str(publisher),
                })
            except Exception:
                malformed += 1
                continue

        if malformed:
            logger.debug("%s: yfinance news — %d malformed items skipped", ticker, malformed)

        return results


# ---------------------------------------------------------------------------
# Provider 3: RSS fallback (TODO)
# ---------------------------------------------------------------------------

class RssNewsProvider(BaseNewsProvider):
    """RSS/scraper fallback for IDX news.

    TODO: Implement with one of:
        - IDX press release RSS: https://www.idx.co.id/id/berita/
        - CNBC Indonesia RSS feed
        - Bisnis.com RSS
        - Kontan.co.id

    Pattern to implement:
        import feedparser
        feed = feedparser.parse(f"https://some.rss.url/?q={ticker}")
        for entry in feed.entries:
            ...
    """

    @property
    def name(self) -> str:
        return "rss"

    def fetch(self, ticker: str, days: int) -> list[dict]:
        raise NewsProviderError("RssNewsProvider not yet implemented")


# ---------------------------------------------------------------------------
# Aggregator — tries providers in priority order
# ---------------------------------------------------------------------------

class NewsAggregator:
    """Run providers in order; stop at the first success.

    A provider that raises NewsProviderError is logged and skipped.
    An empty list from a provider is a valid success (no news in window).

    Default provider order:
        1. GoogleNewsRssProvider  — primary (works for IDX .JK tickers)
        2. YFinanceNewsProvider   — fallback (often empty for .JK but kept as safety net)
    """

    def __init__(self, providers: list[BaseNewsProvider] | None = None) -> None:
        if providers is None:
            providers = [
                GoogleNewsRssProvider(),
                YFinanceNewsProvider(),
            ]
        self.providers = providers

    def fetch_with_status(
        self, ticker: str, days: int
    ) -> tuple[list[dict], str, str | None]:
        """
        Returns:
            (articles, source_name, error_message)
            error_message is None when at least one provider succeeded.
        """
        errors: list[str] = []

        for provider in self.providers:
            try:
                articles = provider.fetch(ticker, days)
                logger.debug(
                    "%s: news fetched via %s → %d articles",
                    ticker, provider.name, len(articles),
                )
                return articles, provider.name, None
            except NewsProviderError as exc:
                msg = f"{provider.name}: {exc}"
                logger.debug("%s: news provider failed — %s", ticker, msg)
                errors.append(msg)
                continue

        # All providers failed
        combined_error = " | ".join(errors)
        logger.warning("%s: all news providers failed — %s", ticker, combined_error)
        return [], "all_failed", combined_error


# ---------------------------------------------------------------------------
# Sentiment scoring
# ---------------------------------------------------------------------------

def score_headline(text: str) -> float:
    """Score one headline: +1 positive, -1 negative, 0 neutral."""
    if not text:
        return 0.0
    words = set(text.lower().replace(",", " ").replace(".", " ").split())
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    if pos == neg:
        return 0.0
    return 1.0 if pos > neg else -1.0


def sentiment_to_score(mean_sentiment: float) -> float:
    """Convert mean sentiment (–1 to 1) to 0–10 scale (5.0 = neutral)."""
    return round(5.0 + mean_sentiment * 5.0, 2)


def label_articles(articles: list[dict]) -> list[dict]:
    """Add sentiment_score and sentiment_label to each article dict.

    Args:
        articles: list of {"title": str, "published": datetime, "publisher": str}

    Returns:
        Same list with two extra fields added in-place (returns new list):
            sentiment_score : +1.0 | 0.0 | -1.0
            sentiment_label : "positive" | "neutral" | "negative"
    """
    labeled = []
    for a in articles:
        score = score_headline(a.get("title", ""))
        if score > 0:
            label = "positive"
        elif score < 0:
            label = "negative"
        else:
            label = "neutral"
        labeled.append({**a, "sentiment_score": score, "sentiment_label": label})
    return labeled


# ---------------------------------------------------------------------------
# Public API: per-ticker computation
# ---------------------------------------------------------------------------

_DEFAULT_AGGREGATOR = NewsAggregator()  # module-level singleton


def compute_news_sentiment(
    ticker: str,
    news_days: int = 3,
    aggregator: NewsAggregator | None = None,
) -> dict:
    """Fetch news and compute sentiment metrics for one ticker.

    Returns a dict with the following keys (always present):
        news_count_3d          int   (0 when no news or failure)
        news_sentiment_mean    float | NaN   (NaN on fetch failure)
        news_positive_count    int
        news_negative_count    int
        news_sentiment_score   float | NaN   (NaN on fetch failure)
        news_data_status       str   "ok" | "none" | "failed"
        news_source            str   provider name or "all_failed"
        news_error_message     str | None
    """
    _, stats = _fetch_and_compute(ticker, news_days, aggregator)
    return stats


def _fetch_and_compute(
    ticker: str,
    news_days: int,
    aggregator: NewsAggregator | None,
) -> tuple[list[dict], dict]:
    """Internal: fetch articles, label them, compute stats.

    Returns:
        (labeled_articles, stats_dict)
        labeled_articles is empty list when status != "ok"
    """
    agg = aggregator or _DEFAULT_AGGREGATOR
    articles, source, error_msg = agg.fetch_with_status(ticker, news_days)

    # ── All providers failed → NaN scores, status = "failed" ─────────────
    if source == "all_failed":
        return [], {
            "news_count_3d":        0,
            "news_sentiment_mean":  float("nan"),
            "news_positive_count":  0,
            "news_negative_count":  0,
            "news_sentiment_score": float("nan"),
            "news_data_status":     "failed",
            "news_source":          source,
            "news_error_message":   error_msg,
        }

    # ── Fetch succeeded but 0 articles → valid neutral ───────────────────
    if not articles:
        return [], {
            "news_count_3d":        0,
            "news_sentiment_mean":  0.0,
            "news_positive_count":  0,
            "news_negative_count":  0,
            "news_sentiment_score": 5.0,
            "news_data_status":     "none",
            "news_source":          source,
            "news_error_message":   None,
        }

    # ── Articles found → label + compute sentiment ────────────────────────
    labeled = label_articles(articles)
    scores = [a["sentiment_score"] for a in labeled]
    mean_score = sum(scores) / len(scores)
    stats = {
        "news_count_3d":        len(scores),
        "news_sentiment_mean":  round(mean_score, 3),
        "news_positive_count":  sum(1 for s in scores if s > 0),
        "news_negative_count":  sum(1 for s in scores if s < 0),
        "news_sentiment_score": sentiment_to_score(mean_score),
        "news_data_status":     "ok",
        "news_source":          source,
        "news_error_message":   None,
    }
    return labeled, stats


# ---------------------------------------------------------------------------
# Enrichment — batch over feature DataFrame
# ---------------------------------------------------------------------------

_NEWS_SCORE_COLS = [
    "news_count_3d", "news_sentiment_mean",
    "news_positive_count", "news_negative_count",
    "news_sentiment_score",
]
_NEWS_STATUS_COLS = [
    "news_data_status", "news_source", "news_error_message",
]
_ALL_NEWS_COLS = _NEWS_SCORE_COLS + _NEWS_STATUS_COLS


def enrich_with_news(
    df_features: pd.DataFrame,
    news_dir: Path | None = None,
    news_days: int = 3,
    save: bool = True,
    scan_date: str | None = None,
    aggregator: NewsAggregator | None = None,
) -> pd.DataFrame:
    """Add news sentiment columns to the feature DataFrame.

    NaN policy (important!):
        news_sentiment_score is LEFT AS NaN when fetch failed (status="failed").
        DO NOT fill with 5.0 here — let _news_score() in signal_engine.py
        handle that, so the original NaN is available for the dashboard.

        Status "none" (no articles found) is DIFFERENT from "failed":
        - "none" sets score = 5.0 (valid neutral)
        - "failed" leaves score = NaN (data unavailable)

    Count columns (news_count_3d etc.) are filled with 0 since NaN count = 0.

    Article storage:
        Individual labeled articles are saved to
        {news_dir}/articles/{scan_date}.parquet
        with columns: ticker, title, published, publisher, sentiment_score, sentiment_label
        These are used by the dashboard explanation to show "what the news was about".
    """
    df = df_features.copy()
    results: list[dict] = []
    all_articles: list[dict] = []   # ← collect per-article rows for storage
    failed_tickers: list[str] = []
    t0 = time.monotonic()

    for ticker in df["ticker"].unique():
        try:
            labeled, row = _fetch_and_compute(ticker, news_days, aggregator)
            row["ticker"] = ticker
            results.append(row)
            # Collect articles with ticker tag for article-level storage
            for a in labeled:
                all_articles.append({"ticker": ticker, **a})
            if row["news_data_status"] == "failed":
                failed_tickers.append(ticker)
        except Exception as exc:
            logger.error("%s: unexpected error in _fetch_and_compute — %s", ticker, exc)
            results.append({
                "ticker":               ticker,
                "news_count_3d":        0,
                "news_sentiment_mean":  float("nan"),
                "news_positive_count":  0,
                "news_negative_count":  0,
                "news_sentiment_score": float("nan"),
                "news_data_status":     "failed",
                "news_source":          "error",
                "news_error_message":   str(exc),
            })
            failed_tickers.append(ticker)

    news_df = pd.DataFrame(results)
    df = df.merge(news_df[["ticker"] + _ALL_NEWS_COLS], on="ticker", how="left")

    # Fill counts with 0 (NaN count is meaningless — it's still 0)
    for col in ["news_count_3d", "news_positive_count", "news_negative_count"]:
        df[col] = df[col].fillna(0).astype(int)

    # news_sentiment_score: leave NaN for failed; fill mean NaN with 0.0
    df["news_sentiment_mean"] = df["news_sentiment_mean"].fillna(0.0)

    # Status defaults for any ticker not in results
    df["news_data_status"] = df["news_data_status"].fillna("failed")
    df["news_source"] = df["news_source"].fillna("all_failed")

    elapsed = time.monotonic() - t0
    total = len(df["ticker"].unique())
    ok_count    = (news_df["news_data_status"] == "ok").sum()
    none_count  = (news_df["news_data_status"] == "none").sum()
    fail_count  = len(failed_tickers)

    logger.info(
        "News enrichment done in %.1fs — ok=%d none=%d failed=%d / %d tickers",
        elapsed, ok_count, none_count, fail_count, total,
    )
    if failed_tickers:
        sample = failed_tickers[:5]
        logger.warning(
            "Failed news tickers (%d total), sample: %s%s",
            fail_count,
            ", ".join(sample),
            f" ... (+{fail_count - len(sample)} more)" if fail_count > 5 else "",
        )

    if save and news_dir and scan_date:
        news_dir_p = Path(news_dir)
        news_dir_p.mkdir(parents=True, exist_ok=True)

        # Aggregate stats (one row per ticker)
        path = news_dir_p / f"{scan_date}.parquet"
        news_df.to_parquet(path, index=False)
        logger.info("News sentiment saved → %s", path)

        # Individual labeled articles (one row per article)
        if all_articles:
            articles_dir = news_dir_p / "articles"
            articles_dir.mkdir(parents=True, exist_ok=True)
            articles_df = pd.DataFrame(all_articles)
            # Ensure published is serialisable (convert datetime to str if needed)
            if "published" in articles_df.columns:
                articles_df["published"] = pd.to_datetime(
                    articles_df["published"], errors="coerce"
                )
            art_path = articles_dir / f"{scan_date}.parquet"
            articles_df.to_parquet(art_path, index=False)
            logger.info(
                "News articles saved → %s (%d rows)", art_path, len(articles_df)
            )

    return df

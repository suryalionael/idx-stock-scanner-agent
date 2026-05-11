"""News narrative summarizer for IDX signal explanations.

Takes a list of labeled articles (each with title, sentiment_label, publisher, published)
and produces short bullet-point summaries per sentiment category.

No external NLP dependencies — uses keyword-based topic detection and
heuristic title ranking to select the most informative articles.

Public API
----------
summarize_news_articles(articles, max_per_polarity=5) -> dict
    Returns:
        {
          "positive_summary": list[str],   # 1–3 bullet strings
          "negative_summary": list[str],
          "neutral_summary":  list[str],
        }

format_news_bullets(summary, total_article_count, max_chars=800) -> str
    Returns Markdown text for the news bullet section.
    Respects Telegram's 4096-char limit by trimming neutral bullets first.
"""
from __future__ import annotations

import re
from datetime import datetime

# ---------------------------------------------------------------------------
# Topic detection — bilingual keyword patterns (order matters: first match wins)
# ---------------------------------------------------------------------------

_TOPIC_PATTERNS: list[tuple[str, list[str]]] = [
    # Corporate actions (high signal — check first)
    ("aksi korporasi",
     ["dividen", "dividend", "buyback", "rights issue", "stock split", "rups",
      "akuisisi", "merger", "divestasi", "obligasi", "bond", "right issue",
      "delisting", "relisting", "penawaran umum", "ipo", "secondary offering"]),

    # Earnings & financial results
    ("kinerja keuangan",
     ["laba", "rugi", "profit", "loss", "pendapatan", "revenue", "earning",
      "kinerja", "laporan keuangan", "kuartal", "triwulan", "semester",
      "eps", "net income", "operating income", "ebitda", "margin",
      "pertumbuhan", "penurunan laba", "cetak laba", "raup laba"]),

    # Analyst ratings & recommendations
    ("rekomendasi analis",
     ["rekomendasi", "upgrade", "downgrade", "target price", "analis", "rating",
      "buy", "sell", "hold", "neutral", "outperform", "underperform",
      "coverage", "inisiasi", "price target", "tp "]),

    # Regulation & legal
    ("regulasi & kebijakan",
     ["regulasi", "ojk", "bi ", "pemerintah", "kebijakan", "sanksi",
      "izin", "hukum", "investigasi", "compliance", "kpk", "pengadilan",
      "peraturan", "aturan baru", "tarif", "bea masuk"]),

    # Expansion & operations
    ("ekspansi & bisnis",
     ["ekspansi", "pabrik", "proyek", "kontrak", "kemitraan", "partnership",
      "peluncuran", "produk baru", "kapasitas", "fasilitas", "investasi baru",
      "pembangunan", "tender", "konstruksi"]),

    # Macro / market context
    ("pasar & makro",
     ["ihsg", "jci", "bi rate", "inflasi", "kurs", "rupiah", "dollar",
      "fed rate", "ekonomi", "makro", "global", "resesi", "pertumbuhan ekonomi",
      "pdb", "gdp", "neraca", "current account"]),

    # Foreign & institutional flow
    ("aliran dana asing",
     ["asing", "foreign", "net buy", "net sell", "institusi", "reksadana",
      "fund", "outflow", "inflow", "diborong", "dilepas", "beli asing",
      "jual asing", "kepemilikan asing"]),
]


def _detect_topic(title: str) -> str:
    """Return the best-matching topic label or 'berita lainnya'."""
    t = title.lower()
    for topic, keywords in _TOPIC_PATTERNS:
        for kw in keywords:
            if kw in t:
                return topic
    return "berita lainnya"


def _rank_article(article: dict) -> float:
    """Rank article by informativeness for bullet selection.

    Higher = more informative:
    - Title length (longer = more context)
    - Has a recognized topic (not generic)
    - More recent
    """
    title = article.get("title", "")
    topic = _detect_topic(title)
    has_topic = 1.0 if topic != "berita lainnya" else 0.0
    length_score = min(len(title) / 80, 1.0)  # normalize to [0,1], cap at 80 chars
    # Recency: articles with published datetime get a small boost for being fresh
    pub = article.get("published")
    recency_score = 0.0
    if isinstance(pub, datetime):
        # Prefer articles within last 24h
        try:
            hours_ago = (datetime.utcnow() - pub).total_seconds() / 3600
            recency_score = max(0, 1 - hours_ago / 72)  # decay over 3 days
        except Exception:
            pass
    return has_topic * 0.5 + length_score * 0.3 + recency_score * 0.2


def _shorten_title(title: str, max_len: int = 90) -> str:
    """Trim title to max_len characters, breaking at a word boundary."""
    title = title.strip()
    if len(title) <= max_len:
        return title
    trimmed = title[:max_len].rsplit(" ", 1)[0]
    return trimmed.rstrip(".,;:") + "…"


def _strip_source_suffix(title: str) -> str:
    """Remove ' - Source Name' suffix often appended by Google News."""
    # Pattern: " - SourceName" or " | SourceName" at end of title
    cleaned = re.sub(r"\s*[-|]\s*[A-Za-z0-9\.\s]{3,40}$", "", title).strip()
    return cleaned if len(cleaned) >= 20 else title  # don't over-strip


def _build_bullets(
    articles: list[dict],
    max_pick: int = 5,
    max_bullets: int = 3,
) -> list[str]:
    """Build 1–3 bullet strings from a list of same-polarity articles.

    Strategy:
    1. Sort by rank (most informative first)
    2. Detect topic for each
    3. Group by topic — one bullet per topic, combining articles with same topic
    4. Fall back to shortened title if no topic match
    """
    if not articles:
        return []

    # Sort by informativeness
    ranked = sorted(articles, key=_rank_article, reverse=True)
    top = ranked[:max_pick]

    # Group by topic
    topic_groups: dict[str, list[str]] = {}
    for a in top:
        topic = _detect_topic(a.get("title", ""))
        title = _strip_source_suffix(a.get("title", "")).strip()
        if not title:
            continue
        if topic not in topic_groups:
            topic_groups[topic] = []
        topic_groups[topic].append(title)

    bullets: list[str] = []

    for topic, titles in topic_groups.items():
        if not titles:
            continue
        if len(titles) == 1:
            # Single article for this topic
            short = _shorten_title(titles[0])
            if topic != "berita lainnya":
                bullets.append(f"[{topic}] {short}")
            else:
                bullets.append(short)
        else:
            # Multiple articles on same topic: show topic with count + best title
            best = _shorten_title(titles[0], max_len=70)
            extra = len(titles) - 1
            if topic != "berita lainnya":
                bullets.append(
                    f"[{topic}] {best}"
                    + (f" (+{extra} artikel terkait)" if extra > 0 else "")
                )
            else:
                bullets.append(
                    f"{best}"
                    + (f" (+{extra} artikel serupa)" if extra > 0 else "")
                )

        if len(bullets) >= max_bullets:
            break

    return bullets[:max_bullets]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_news_articles(
    articles: list[dict],
    max_articles_per_polarity: int = 5,
) -> dict:
    """Group articles by sentiment_label and build bullet summaries.

    Args:
        articles: list of dicts with at minimum:
            - title         : str
            - sentiment_label : "positive" | "negative" | "neutral"
            - published     : datetime (optional, used for ranking)
            - publisher     : str (optional)
        max_articles_per_polarity: how many articles to consider per category

    Returns:
        {
          "positive_summary": list[str],  # 0–3 bullet strings
          "negative_summary": list[str],
          "neutral_summary":  list[str],
        }

    Edge cases handled:
        - Empty articles list → all empty lists
        - Missing sentiment_label → defaults to "neutral"
        - Missing title → article skipped
        - Unknown sentiment_label value → treated as "neutral"
    """
    grouped: dict[str, list[dict]] = {"positive": [], "negative": [], "neutral": []}

    for a in articles:
        label = str(a.get("sentiment_label", "neutral")).lower()
        if label not in grouped:
            label = "neutral"
        if a.get("title"):  # skip articles with no title
            grouped[label].append(a)

    return {
        "positive_summary": _build_bullets(
            grouped["positive"], max_articles_per_polarity, max_bullets=3
        ),
        "negative_summary": _build_bullets(
            grouped["negative"], max_articles_per_polarity, max_bullets=3
        ),
        "neutral_summary": _build_bullets(
            grouped["neutral"], max_articles_per_polarity, max_bullets=2
        ),
    }


def format_news_bullets(
    summary: dict,
    total_article_count: int,
    sentiment_score: float | None = None,
    score_bar: str = "",
    source: str = "",
    max_chars: int = 800,
) -> str:
    """Format the full News & Catalyst narrative block.

    Respects max_chars by dropping neutral bullets first, then trimming negatives.

    Args:
        summary         : output from summarize_news_articles()
        total_article_count: total articles found (for header line)
        sentiment_score : 0–10 float (optional)
        score_bar       : pre-built █░ bar string (optional)
        source          : data source label e.g. "google_rss"
        max_chars       : trim if output exceeds this length

    Returns:
        Markdown-formatted string (no leading/trailing newlines).
    """
    pos = summary.get("positive_summary", [])
    neg = summary.get("negative_summary", [])
    neu = summary.get("neutral_summary", [])

    parts: list[str] = []

    # Header
    src_label = source or "google_rss"
    parts.append(f"Ditemukan **{total_article_count} artikel** ({src_label}).")

    # Sentiment score bar
    if sentiment_score is not None:
        n_pos = len([a for a in pos])  # proxied from summary
        parts.append(
            f"Skor sentimen: **{sentiment_score:.1f}/10** {score_bar}"
        )

    def _section(title: str, bullets: list[str]) -> str:
        if not bullets:
            return ""
        lines = "\n".join(f"- {b}" for b in bullets)
        return f"\n**{title}:**\n{lines}"

    pos_block = _section("Positif", pos)
    neg_block = _section("Negatif", neg)
    neu_block = _section("Netral", neu)

    # Build text with all blocks
    body = "".join([pos_block, neg_block, neu_block])
    full = "\n".join(parts) + body

    # Trim if too long: drop neutral first, then trim negative to 1 bullet
    if len(full) > max_chars and neu:
        body_no_neu = "".join([pos_block, neg_block])
        full = "\n".join(parts) + body_no_neu

    if len(full) > max_chars and neg and len(neg) > 1:
        neg_short = _section("Negatif", neg[:1])
        body_trimmed = "".join([pos_block, neg_short])
        full = "\n".join(parts) + body_trimmed

    return full

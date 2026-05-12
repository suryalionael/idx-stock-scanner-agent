"""Daily alert entry point — deep dive + top picks Telegram dispatch.

Combines:
  • 2 deep-dive messages for top BREAKOUT / PRE_MARKUP candidates
  • 1 compact top-picks message with entry/TP/cutloss for top 7 tickers

Usage:
    python -m stock_scanner.alerts.daily_alert
    python -m stock_scanner.alerts.daily_alert --date 2026-05-12
    python -m stock_scanner.alerts.daily_alert --dry-run
    python -m stock_scanner.alerts.daily_alert --dry-run --deep-dive-only
    python -m stock_scanner.alerts.daily_alert --dry-run --top-picks-only

Environment variables required for sending:
    TELEGRAM_BOT_TOKEN   — from @BotFather
    TELEGRAM_CHAT_ID     — your personal chat ID or group/channel ID
"""
from __future__ import annotations

import argparse
import os
import textwrap
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from stock_scanner.alerts.message_builder import (
    build_all_breakout_deep_dive_messages,
    build_telegram_top_picks_message,
    signals_df_to_list,
    split_long_message,
)
from stock_scanner.alerts.telegram_alert import TelegramSender

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT        = Path(__file__).parent.parent.parent
_RANKED_DIR  = _ROOT / "data" / "ranked"
_SIGNALS_DIR = _ROOT / "data" / "signals"
_NEWS_ARTICLES_DIR = _ROOT / "data" / "news" / "articles"

_SEND_DELAY_SEC = 2.0  # polite delay between Telegram messages


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _find_latest_date(max_days_back: int = 7) -> str | None:
    """Return most recent date with a signals or ranked file."""
    today = date.today()
    for delta in range(max_days_back + 1):
        d = (today - timedelta(days=delta)).strftime("%Y-%m-%d")
        if (_SIGNALS_DIR / f"{d}.parquet").exists():
            return d
        if (_RANKED_DIR / f"ranked_{d}.csv").exists():
            return d
    return None


def _load_signals(scan_date: str) -> pd.DataFrame:
    """Load signals DataFrame for a given date.

    Priority: signals/{scan_date}.parquet → ranked_{scan_date}.csv
    """
    parquet = _SIGNALS_DIR / f"{scan_date}.parquet"
    ranked_csv = _RANKED_DIR / f"ranked_{scan_date}.csv"

    if parquet.exists():
        df = pd.read_parquet(parquet)
        logger.info("Loaded signals parquet: %s (%d rows)", parquet.name, len(df))
        return df
    if ranked_csv.exists():
        df = pd.read_csv(ranked_csv)
        logger.info("Loaded ranked CSV: %s (%d rows)", ranked_csv.name, len(df))
        return df

    logger.warning("No signal data found for %s", scan_date)
    return pd.DataFrame()


def _load_articles_by_ticker(scan_date: str) -> dict[str, list[dict]]:
    """Load per-article data grouped by ticker for the given date.

    Returns:
        {ticker_without_JK: [article_dicts, ...]}
    """
    articles_path = _NEWS_ARTICLES_DIR / f"{scan_date}.parquet"
    if not articles_path.exists():
        logger.warning("No articles parquet for %s — news bullets disabled", scan_date)
        return {}

    try:
        df = pd.read_parquet(articles_path)
        if "ticker" not in df.columns:
            return {}
        grouped: dict[str, list[dict]] = {}
        for ticker, grp in df.groupby("ticker"):
            clean = str(ticker).replace(".JK", "").replace(".jk", "")
            grouped[clean] = grp.where(grp.notna(), other=None).to_dict(orient="records")
        logger.info("Loaded articles for %d tickers from %s", len(grouped), articles_path.name)
        return grouped
    except Exception as exc:
        logger.error("Failed to load articles parquet: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def run_daily_alert(
    scan_date: str,
    dry_run: bool = False,
    deep_dive_only: bool = False,
    top_picks_only: bool = False,
    top_n_deep_dive: int = 2,
    top_n_picks: int = 7,
    send_delay: float = _SEND_DELAY_SEC,
) -> None:
    """Load signals and send deep-dive + top-picks alerts.

    Args:
        scan_date       : YYYY-MM-DD
        dry_run         : Print messages to stdout instead of sending.
        deep_dive_only  : Skip top-picks message.
        top_picks_only  : Skip deep-dive messages.
        top_n_deep_dive : Number of deep-dive candidates (default 2).
        top_n_picks     : Number of top picks (default 7).
        send_delay      : Seconds between Telegram sends (default 2).
    """
    logger.info("=== IDX Daily Alert — %s ===", scan_date)

    # 1. Load data
    df = _load_signals(scan_date)
    if df.empty:
        logger.error("No signal data for %s — aborting.", scan_date)
        return

    signals = signals_df_to_list(df, scan_date)
    articles_by_ticker = _load_articles_by_ticker(scan_date)

    # 2. Build messages
    messages: list[str] = []

    if not top_picks_only:
        logger.info("Building deep-dive messages (top %d)...", top_n_deep_dive)
        deep_dives = build_all_breakout_deep_dive_messages(
            signals,
            articles_by_ticker=articles_by_ticker,
            top_n=top_n_deep_dive,
        )
        if deep_dives:
            messages.extend(deep_dives)
            logger.info("  → %d deep-dive message(s) ready.", len(deep_dives))
        else:
            logger.warning("  → No deep-dive candidates found.")

    if not deep_dive_only:
        logger.info("Building top-picks message (top %d)...", top_n_picks)
        top_picks_msg = build_telegram_top_picks_message(
            signals,
            date=scan_date,
            top_n=top_n_picks,
        )
        # Split if it's somehow over limit (shouldn't happen but safety net)
        picks_chunks = split_long_message(top_picks_msg, limit=4000)
        messages.extend(picks_chunks)
        logger.info("  → Top-picks message ready (%d chunk(s)).", len(picks_chunks))

    if not messages:
        logger.warning("No messages to send.")
        return

    # 3. Send or dry-run
    if dry_run:
        _dry_run_print(messages)
        return

    sender = TelegramSender()
    if not sender.bot_token or not sender.chat_id:
        logger.error(
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.\n"
            "  export TELEGRAM_BOT_TOKEN=<token>\n"
            "  export TELEGRAM_CHAT_ID=<chat_id>"
        )
        return

    success_count = 0
    for i, msg in enumerate(messages, 1):
        logger.info("Sending message %d/%d (%d chars)...", i, len(messages), len(msg))
        result = sender.send(msg)
        if result:
            success_count += 1
        else:
            logger.error("  → Failed: %s", result.error)
        if i < len(messages):
            time.sleep(send_delay)

    logger.info(
        "Done — %d/%d messages sent successfully.", success_count, len(messages)
    )


def _dry_run_print(messages: list[str]) -> None:
    """Pretty-print messages to stdout for --dry-run mode."""
    total = len(messages)
    char_count = sum(len(m) for m in messages)
    print(f"\n{'='*60}")
    print(f"DRY RUN — {total} message(s), {char_count} total chars")
    print(f"{'='*60}\n")
    for i, msg in enumerate(messages, 1):
        print(f"{'─'*60}")
        print(f"MESSAGE {i}/{total}  ({len(msg)} chars)")
        print(f"{'─'*60}")
        print(msg)
        print()
    print(f"{'='*60}")
    print("Messages NOT sent (dry-run mode).")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IDX Stock Scanner — Daily Telegram Alert (Deep Dive + Top Picks)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Environment variables:
              TELEGRAM_BOT_TOKEN   Bot token from @BotFather
              TELEGRAM_CHAT_ID     Chat / group / channel ID

            Examples:
              python -m stock_scanner.alerts.daily_alert --dry-run
              python -m stock_scanner.alerts.daily_alert --date 2026-05-12
              python -m stock_scanner.alerts.daily_alert --dry-run --top-picks-only
              python -m stock_scanner.alerts.daily_alert --deep-dive-only
        """),
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Scan date (YYYY-MM-DD). Default: latest available.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print messages to stdout instead of sending.",
    )
    parser.add_argument(
        "--deep-dive-only",
        action="store_true",
        help="Only send deep-dive messages, skip top-picks.",
    )
    parser.add_argument(
        "--top-picks-only",
        action="store_true",
        help="Only send top-picks message, skip deep-dive.",
    )
    parser.add_argument(
        "--top-n-deep-dive",
        type=int,
        default=2,
        help="Number of deep-dive candidates (default: 2).",
    )
    parser.add_argument(
        "--top-n-picks",
        type=int,
        default=7,
        help="Number of top picks (default: 7).",
    )
    args = parser.parse_args()

    scan_date = args.date or _find_latest_date()
    if scan_date is None:
        logger.error(
            "No scan data found. Run: python -m stock_scanner.pipeline.run_daily_scan"
        )
        raise SystemExit(1)

    run_daily_alert(
        scan_date=scan_date,
        dry_run=args.dry_run,
        deep_dive_only=args.deep_dive_only,
        top_picks_only=args.top_picks_only,
        top_n_deep_dive=args.top_n_deep_dive,
        top_n_picks=args.top_n_picks,
    )


if __name__ == "__main__":
    main()

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
    format_swing_full_alert,
    signals_df_to_list,
    split_long_message,
)
from stock_scanner.alerts.scalping_formatter import format_scalping_alert
from stock_scanner.alerts.telegram_alert import TelegramSender
from stock_scanner.pipeline.quality_filters import EXCLUDED_STATUSES

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

    If scalping columns are missing (older files), they are computed on-the-fly.
    """
    from stock_scanner.pipeline.scalping import enrich_df_with_scalping

    parquet = _SIGNALS_DIR / f"{scan_date}.parquet"
    ranked_csv = _RANKED_DIR / f"ranked_{scan_date}.csv"

    if parquet.exists():
        df = pd.read_parquet(parquet)
        logger.info("Loaded signals parquet: %s (%d rows)", parquet.name, len(df))
    elif ranked_csv.exists():
        df = pd.read_csv(ranked_csv)
        logger.info("Loaded ranked CSV: %s (%d rows)", ranked_csv.name, len(df))
    else:
        logger.warning("No signal data found for %s", scan_date)
        return pd.DataFrame()

    # Ensure scalping columns are present (computed during run_daily_scan
    # but may be missing for files generated before this feature was added)
    if "scalping_label" not in df.columns:
        logger.info("Scalping columns missing — computing on-the-fly.")
        df = enrich_df_with_scalping(df)

    # Filter out hard-excluded tickers (excluded_fundamental, excluded_float, excluded_regulatory)
    # Tickers with insufficient_data are kept — they just get flagged.
    if "final_status" in df.columns:
        before = len(df)
        df = df[~df["final_status"].isin(EXCLUDED_STATUSES)]
        removed = before - len(df)
        if removed:
            logger.info("Removed %d excluded_* tickers from alert pool", removed)
    else:
        logger.debug("final_status column not found — quality filter not applied to alert pool")

    return df


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
    top_n_picks: int = 10,
    send_delay: float = _SEND_DELAY_SEC,
    send_scalping: bool = True,
    send_swing_full: bool = True,
    send_daily_report: bool = True,
) -> None:
    """Load signals and send all alert types.

    Alert sequence (in order):
        1. Deep-dive (2 top BREAKOUT/PRE_MARKUP — detailed analysis)
        2. Scalping High alert (format_scalping_alert)
        3. Swing full alert (ALL BREAKOUT + PRE_MARKUP, auto-split)
        4. Top-picks compact list (top N, backward-compat)
        5. Daily report document (Markdown file sent as attachment)

    Args:
        scan_date        : YYYY-MM-DD
        dry_run          : Print messages to stdout instead of sending.
        deep_dive_only   : Only send deep-dive, skip all others.
        top_picks_only   : Only send top-picks compact list.
        top_n_deep_dive  : Number of deep-dive candidates (default 2).
        top_n_picks      : Number of top picks in compact list (default 10).
        send_delay       : Seconds between Telegram sends.
        send_scalping    : Send scalping alert (default True).
        send_swing_full  : Send full swing alert (all BREAKOUT+PRE_MARKUP).
        send_daily_report: Generate and send daily .md report (default True).
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

    # ── Deep dive ──────────────────────────────────────────────────────────
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

    if deep_dive_only:
        # Skip all other alerts if deep-dive-only mode
        if dry_run:
            _dry_run_print(messages)
        else:
            _send_messages(messages, send_delay)
        return

    # ── Scalping alert ─────────────────────────────────────────────────────
    if send_scalping and not top_picks_only:
        logger.info("Building scalping alert...")
        scalp_msgs = format_scalping_alert(signals, scan_date=scan_date)
        if scalp_msgs:
            messages.extend(scalp_msgs)
            logger.info("  → %d scalping message(s) ready.", len(scalp_msgs))
        else:
            logger.info("  → No SCALPING_HIGH candidates.")

    # ── Swing full alert (ALL BREAKOUT + PRE_MARKUP) ───────────────────────
    if send_swing_full and not top_picks_only:
        logger.info("Building swing full alert...")
        swing_msgs = format_swing_full_alert(signals, scan_date=scan_date)
        if swing_msgs:
            messages.extend(swing_msgs)
            logger.info("  → %d swing message(s) ready.", len(swing_msgs))
        else:
            logger.info("  → No swing candidates.")

    # ── Top picks compact (backward-compat, optional) ──────────────────────
    # Only include if swing_full is NOT sent (to avoid duplication)
    if not send_swing_full or top_picks_only:
        logger.info("Building top-picks message (top %d)...", top_n_picks)
        top_picks_msg = build_telegram_top_picks_message(
            signals,
            date=scan_date,
            top_n=top_n_picks,
        )
        picks_chunks = split_long_message(top_picks_msg, limit=4000)
        messages.extend(picks_chunks)
        logger.info("  → Top-picks message ready (%d chunk(s)).", len(picks_chunks))

    # 3. Send text messages (or dry-run print)
    if not messages:
        logger.warning("No messages to send.")
    elif dry_run:
        _dry_run_print(messages)
    else:
        _send_messages(messages, send_delay)

    # ── Daily report document ──────────────────────────────────────────────
    if send_daily_report and not top_picks_only:
        _handle_daily_report(df, signals, scan_date, dry_run, send_delay)


def _handle_daily_report(
    df: "pd.DataFrame",
    signals: list[dict],
    scan_date: str,
    dry_run: bool,
    send_delay: float,
) -> None:
    """Generate daily Markdown report and send as Telegram document."""
    try:
        from stock_scanner.alerts.daily_report import (
            build_report_context,
            generate_daily_report,
        )

        logger.info("Building daily report document...")
        ctx = build_report_context(df, scan_date)
        result = generate_daily_report(ctx)

        report_path    = result["output_path"]
        report_summary = result["report_summary"]

        if dry_run:
            print("\n" + "=" * 60)
            print(f"DAILY REPORT (dry-run) — {report_path}")
            print("=" * 60)
            print("\n--- Telegram Summary ---")
            print(report_summary)
            print("\n--- Markdown Preview (first 80 lines) ---")
            md_preview = "\n".join(result["report_markdown"].split("\n")[:80])
            print(md_preview)
            print("=" * 60)
            return

        sender = TelegramSender()
        if not sender.bot_token or not sender.chat_id:
            logger.error("Cannot send daily report: Telegram credentials not set.")
            return

        # Send summary text first, then file
        time.sleep(send_delay)
        res_text = sender.send(report_summary)
        if not res_text:
            logger.error("Failed to send report summary: %s", res_text.error)

        time.sleep(send_delay)
        caption = f"📄 IDX Daily Report — {scan_date}"
        res_doc = sender.send_document(report_path, caption=caption)
        if res_doc:
            logger.info("Daily report document sent: %s", report_path.name)
        else:
            logger.error("Failed to send report document: %s", res_doc.error)

    except Exception as exc:
        logger.error("Daily report generation/send failed: %s", exc)


def _send_messages(messages: list[str], send_delay: float = _SEND_DELAY_SEC) -> int:
    """Send a list of messages via Telegram. Returns success count."""
    sender = TelegramSender()
    if not sender.bot_token or not sender.chat_id:
        logger.error(
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.\n"
            "  export TELEGRAM_BOT_TOKEN=<token>\n"
            "  export TELEGRAM_CHAT_ID=<chat_id>"
        )
        return 0
    return sender.send_messages_batch(messages, delay_sec=send_delay)


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
        default=10,
        help="Number of top picks in compact list (default: 10).",
    )
    parser.add_argument(
        "--no-scalping",
        action="store_true",
        help="Skip scalping alert.",
    )
    parser.add_argument(
        "--no-swing-full",
        action="store_true",
        help="Skip full swing alert (all BREAKOUT+PRE_MARKUP). Falls back to top-picks list.",
    )
    parser.add_argument(
        "--no-daily-report",
        action="store_true",
        help="Skip daily Markdown report generation and send.",
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
        send_scalping=not args.no_scalping,
        send_swing_full=not args.no_swing_full,
        send_daily_report=not args.no_daily_report,
    )


if __name__ == "__main__":
    main()

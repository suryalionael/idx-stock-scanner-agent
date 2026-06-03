#!/usr/bin/env python3
"""IDX Scanner — one-command daily signal + Telegram alert (non-UI automation).

This is the simple, reliable entrypoint for daily automation. It does two
things in order:

    1. Run the daily scan      (python -m stock_scanner.pipeline.run_daily_scan)
    2. Build + send the morning alert to Telegram

It deliberately reuses the message builder and sender from
``stock_scanner.alerts.telegram_alert`` (only pandas + stdlib urllib) instead of
the heavier multi-message ``daily_alert`` path, so there is minimal failure
surface for a job that just needs to run every morning.

Usage:
    python scripts/run_daily_signal.py                 # scan + send
    python scripts/run_daily_signal.py --dry-run       # scan + print (no send)
    python scripts/run_daily_signal.py --skip-scan     # send using cached data only
    python scripts/run_daily_signal.py --date 2026-06-02 --skip-scan --dry-run
    python scripts/run_daily_signal.py --skip-if-empty # don't send if no priority picks

Environment variables:
    TELEGRAM_BOT_TOKEN   required to send (the actual secret)
    TELEGRAM_CHAT_ID     optional — falls back to the default group if unset
    ANTHROPIC_API_KEY    optional — only used if the explain agent is enabled

Exit codes:
    0  success (sent, or dry-run, or intentionally skipped)
    1  failed to send the alert (e.g. TELEGRAM_BOT_TOKEN missing, Telegram error)
    2  no scan data available at all (nothing to report)

Note: a scan failure is non-fatal — the alert still goes out using the most
recent cached scan (up to 7 days back), matching the GitHub Actions behaviour.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from loguru import logger

# Make the repo root importable when run as `python scripts/run_daily_signal.py`.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from stock_scanner.alerts.telegram_alert import (  # noqa: E402
    TelegramSender,
    _find_latest_date,
    _load_scan,
    _PRIORITY_SIGNALS,
    build_morning_message,
)


def _run_scan() -> bool:
    """Run the daily scan as a subprocess. Returns True on success.

    Non-fatal on failure: the caller will fall back to cached data so an
    alert is still sent.
    """
    cmd = [sys.executable, "-m", "stock_scanner.pipeline.run_daily_scan"]
    logger.info("Starting daily scan: {}", " ".join(cmd))
    try:
        result = subprocess.run(cmd, cwd=str(_ROOT), timeout=3600)
        if result.returncode != 0:
            logger.error("Daily scan exited with code {} (continuing with cached data).",
                         result.returncode)
            return False
        logger.info("Daily scan finished successfully.")
        return True
    except subprocess.TimeoutExpired:
        logger.error("Daily scan timed out after 60 minutes (continuing with cached data).")
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Daily scan crashed: {} (continuing with cached data).", exc)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="IDX Scanner — daily signal scan + Telegram alert (one command).",
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Scan date to report (YYYY-MM-DD). Default: latest available.")
    parser.add_argument("--top-n", type=int, default=7,
                        help="Number of top picks to include (default: 7).")
    parser.add_argument("--skip-scan", action="store_true",
                        help="Skip the scan; send using existing cached data only.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the message to stdout instead of sending to Telegram.")
    parser.add_argument("--skip-if-empty", action="store_true",
                        help="If no priority signals (BREAKOUT/PRE_MARKUP), skip sending.")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("IDX Scanner — daily signal runner")
    logger.info("=" * 60)

    # ── Step 1: Scan (non-fatal on failure) ──────────────────────────────
    if args.skip_scan:
        logger.info("--skip-scan: using existing cached data.")
    else:
        _run_scan()  # failure is logged but non-fatal; we fall back to cache

    # ── Resolve which date to report ─────────────────────────────────────
    scan_date = args.date or _find_latest_date()
    if scan_date is None:
        logger.error(
            "No scan data found at all. Run the scan first: "
            "python -m stock_scanner.pipeline.run_daily_scan"
        )
        return 2
    logger.info("Reporting scan date: {}", scan_date)

    # ── Count priority picks (for logging + --skip-if-empty) ─────────────
    ranked, _ = _load_scan(scan_date)
    n_priority = 0
    if not ranked.empty and "signal" in ranked.columns:
        n_priority = int(ranked["signal"].isin(_PRIORITY_SIGNALS).sum())
    logger.info("Priority signals (BREAKOUT/PRE_MARKUP): {}", n_priority)

    if n_priority == 0 and args.skip_if_empty:
        logger.info("No priority signals and --skip-if-empty set — skipping send.")
        return 0

    # ── Step 2: Build + send ─────────────────────────────────────────────
    message = build_morning_message(scan_date, top_n=args.top_n)

    if args.dry_run:
        print("\n" + "=" * 50)
        print("DRY RUN — message NOT sent to Telegram")
        print("=" * 50)
        print(message)
        print("=" * 50)
        return 0

    sender = TelegramSender()
    result = sender.send(message)
    if result:
        logger.info("Morning alert sent successfully ({} priority pick(s)).", n_priority)
        return 0

    logger.error("Failed to send alert: {}", result.error)
    return 1


if __name__ == "__main__":
    sys.exit(main())

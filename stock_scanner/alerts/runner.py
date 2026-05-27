"""Full daily pipeline runner — single entry point for automation.

Runs in sequence:
    1. Incremental OHLCV update + signal scan (run_daily_scan)
    2. Scalping + swing + daily report Telegram alerts (daily_alert)

Called by scripts/run_daily_alerts.sh. Can also be run manually.

Usage:
    python -m stock_scanner.alerts.runner
    python -m stock_scanner.alerts.runner --dry-run
    python -m stock_scanner.alerts.runner --skip-scan        # alert-only (today's data must exist)
    python -m stock_scanner.alerts.runner --date 2026-05-20 --dry-run

Exit codes:
    0  — success
    1  — scan failed (no signal data produced)
    2  — alert failed (Telegram not reachable or credentials missing)
    3  — health check failed (env vars missing, etc.)

Environment variables:
    TELEGRAM_BOT_TOKEN    required
    TELEGRAM_CHAT_ID      required
    ANTHROPIC_API_KEY     optional (enables Claude AI summary in daily report)
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import traceback
from datetime import date, timedelta
from pathlib import Path

from loguru import logger

_ROOT        = Path(__file__).parent.parent.parent
_RANKED_DIR  = _ROOT / "data" / "ranked"
_SIGNALS_DIR = _ROOT / "data" / "signals"
_REPORTS_DIR = _ROOT / "data" / "reports"


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def check_env_vars(dry_run: bool = False) -> bool:
    """Verify required environment variables are set.

    Returns True if all required vars present (or dry-run mode).
    Logs which vars are missing.
    """
    if dry_run:
        logger.info("Dry-run mode — skipping env var check.")
        return True

    required = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    missing  = [k for k in required if not os.getenv(k)]

    if missing:
        logger.error("Missing required environment variables: %s", ", ".join(missing))
        logger.error(
            "Set them in .env.alerts or export before running:\n"
            "  export TELEGRAM_BOT_TOKEN=<token>\n"
            "  export TELEGRAM_CHAT_ID=<chat_id>"
        )
        return False

    optional = ["ANTHROPIC_API_KEY"]
    for k in optional:
        if not os.getenv(k):
            logger.warning("%s not set — Claude AI summary will use rule-based fallback.", k)

    return True


def check_scan_output(scan_date: str) -> bool:
    """Return True if today's scan data exists."""
    ranked_path = _RANKED_DIR / f"ranked_{scan_date}.csv"
    signals_parquet = _SIGNALS_DIR / f"{scan_date}.parquet"
    return ranked_path.exists() or signals_parquet.exists()


# ---------------------------------------------------------------------------
# Step 1 — Daily scan
# ---------------------------------------------------------------------------

def run_scan(config_path: Path | None = None) -> bool:
    """Run the daily OHLCV + signal scan.

    Returns True on success (output files created).
    """
    import subprocess
    cmd = [sys.executable, "-m", "stock_scanner.pipeline.run_daily_scan"]
    if config_path:
        cmd += ["--config", str(config_path)]

    logger.info("Starting daily scan: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_ROOT),
            capture_output=False,   # let output flow to shell's log
            timeout=3600,           # 60 min max
        )
        if result.returncode != 0:
            logger.error("Daily scan exited with code %d", result.returncode)
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.error("Daily scan timed out after 60 minutes.")
        return False
    except Exception as exc:
        logger.error("Daily scan crashed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Step 2 — Alerts
# ---------------------------------------------------------------------------

def run_alerts(
    scan_date: str,
    dry_run: bool = False,
    send_scalping: bool = True,
    send_swing_full: bool = True,
    send_daily_report: bool = True,
) -> bool:
    """Send all Telegram alerts.

    Returns True on success.
    """
    try:
        from stock_scanner.alerts.daily_alert import run_daily_alert

        logger.info("Sending alerts for %s (dry_run=%s)...", scan_date, dry_run)
        run_daily_alert(
            scan_date=scan_date,
            dry_run=dry_run,
            send_scalping=send_scalping,
            send_swing_full=send_swing_full,
            send_daily_report=send_daily_report,
        )
        return True
    except Exception as exc:
        logger.error("Alert dispatch failed: %s", exc)
        logger.debug(traceback.format_exc())
        return False


# ---------------------------------------------------------------------------
# Step 3 — Publish to GitHub
# ---------------------------------------------------------------------------

def _publish_to_github(scan_date: str) -> None:
    """Commit dan push latest_scan.json ke GitHub remote.

    Non-fatal: jika gagal, log warning saja — scan dianggap berhasil.
    Script bash dipakai agar git identity bisa di-set per-command tanpa
    mengubah global git config.
    """
    import subprocess

    script = _ROOT / "scripts" / "publish_to_github.sh"
    if not script.exists():
        logger.warning("publish_to_github.sh tidak ditemukan — skip GitHub publish.")
        return

    logger.info("Publishing to GitHub: %s", scan_date)
    try:
        result = subprocess.run(
            ["bash", str(script)],
            cwd=str(_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.stdout:
            for line in result.stdout.strip().splitlines():
                logger.info("[publish] %s", line)
        if result.stderr:
            for line in result.stderr.strip().splitlines():
                logger.warning("[publish] %s", line)
        if result.returncode != 0:
            logger.warning("publish_to_github.sh exited %d — dashboard data mungkin stale.", result.returncode)
        else:
            logger.info("GitHub publish selesai.")
    except subprocess.TimeoutExpired:
        logger.warning("GitHub publish timed out (120s) — skip.")
    except Exception as exc:
        logger.warning("GitHub publish gagal (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Error notification
# ---------------------------------------------------------------------------

def notify_failure(step: str, detail: str) -> None:
    """Send a brief failure notification to Telegram admin (best-effort)."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat  = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        return   # credentials missing — can't notify

    try:
        from stock_scanner.alerts.telegram_alert import send_telegram_message
        today = date.today().strftime("%Y-%m-%d")
        msg = (
            f"🚨 <b>IDX Scanner Error</b>\n"
            f"Tanggal : {today}\n"
            f"Step    : {step}\n"
            f"Detail  : <code>{detail[:300]}</code>\n"
            f"<i>Cek log di logs/daily/ untuk detail.</i>"
        )
        send_telegram_message(msg, bot_token=token, chat_id=chat)
        logger.info("Failure notification sent to Telegram.")
    except Exception as exc:
        logger.warning("Could not send failure notification: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run full pipeline. Returns exit code."""
    parser = argparse.ArgumentParser(
        description="IDX Scanner — full daily pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Typical usage by automation (launchd):
              python -m stock_scanner.alerts.runner

            Skip scan (use today's existing data):
              python -m stock_scanner.alerts.runner --skip-scan

            Test without sending:
              python -m stock_scanner.alerts.runner --dry-run

            Backfill a specific date (alert-only):
              python -m stock_scanner.alerts.runner --date 2026-05-20 --skip-scan --dry-run
        """),
    )
    parser.add_argument("--date",       type=str,  default=None,
                        help="Date to process (YYYY-MM-DD). Default: today.")
    parser.add_argument("--skip-scan",  action="store_true",
                        help="Skip run_daily_scan (use existing data for today).")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print everything to stdout — do NOT send to Telegram.")
    parser.add_argument("--no-scalping",      action="store_true")
    parser.add_argument("--no-swing-full",    action="store_true")
    parser.add_argument("--no-daily-report",  action="store_true")
    parser.add_argument("--config",     type=Path, default=None,
                        help="Path to scanner_config.yaml (optional override).")
    args = parser.parse_args()

    scan_date = args.date or date.today().strftime("%Y-%m-%d")
    logger.info("=" * 60)
    logger.info("IDX Scanner Daily Runner — %s", scan_date)
    logger.info("=" * 60)

    # ── Health check ──────────────────────────────────────────────────────
    if not check_env_vars(dry_run=args.dry_run):
        notify_failure("health_check", "Required env vars TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
        return 3

    # ── Step 1: Scan ──────────────────────────────────────────────────────
    if args.skip_scan:
        logger.info("--skip-scan: skipping daily scan.")
    else:
        scan_ok = run_scan(config_path=args.config)
        if not scan_ok:
            notify_failure("daily_scan", "run_daily_scan exited with error. Check logs.")
            return 1

    # Verify output exists
    if not check_scan_output(scan_date):
        # If no data for today, try most recent date within 3 days
        for delta in range(1, 4):
            fallback = (date.today() - timedelta(days=delta)).strftime("%Y-%m-%d")
            if check_scan_output(fallback):
                logger.warning(
                    "No scan data for %s — falling back to %s", scan_date, fallback
                )
                scan_date = fallback
                break
        else:
            err = f"No scan data found for {scan_date} or recent dates."
            logger.error(err)
            notify_failure("data_check", err)
            return 1

    # ── Step 2: Alerts ────────────────────────────────────────────────────
    alert_ok = run_alerts(
        scan_date=scan_date,
        dry_run=args.dry_run,
        send_scalping=not args.no_scalping,
        send_swing_full=not args.no_swing_full,
        send_daily_report=not args.no_daily_report,
    )
    if not alert_ok:
        notify_failure("daily_alert", "Alert dispatch raised an exception. Check logs.")
        return 2

    # ── Step 3: Publish ke GitHub (non-fatal) ────────────────────────────
    if not args.dry_run:
        _publish_to_github(scan_date)
    else:
        logger.info("Dry-run: skip GitHub publish.")

    logger.info("=" * 60)
    logger.info("Pipeline complete — %s", scan_date)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

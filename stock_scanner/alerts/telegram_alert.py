"""Telegram alert sender + morning summary builder.

Usage (standalone):
    python -m stock_scanner.alerts.telegram_alert
    python -m stock_scanner.alerts.telegram_alert --date 2026-05-09
    python -m stock_scanner.alerts.telegram_alert --dry-run

Environment variables required for sending:
    TELEGRAM_BOT_TOKEN   — from @BotFather
    TELEGRAM_CHAT_ID     — your personal chat ID or group/channel ID

Quick-start (get these values):
    1. Message @BotFather on Telegram → /newbot → copy token
    2. Message @userinfobot → copy your chat ID
    3. export TELEGRAM_BOT_TOKEN=... TELEGRAM_CHAT_ID=...
    4. python -m stock_scanner.alerts.telegram_alert --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from stock_scanner.alerts.base import AlertResult, BaseAlertSender
from stock_scanner.utils.trading_calendar import expected_market_date

# WIB is a fixed UTC+7 offset (Indonesia has no DST), so this is always correct
# regardless of the machine/runner timezone — no tzdata dependency needed.
_WIB = timezone(timedelta(hours=7))


def now_wib() -> datetime:
    """Current time in WIB (Asia/Jakarta), timezone-aware."""
    return datetime.now(_WIB)

# ---------------------------------------------------------------------------
# Paths (resolved relative to repo root)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent
_RANKED_DIR = _ROOT / "data" / "ranked"
_SIGNALS_DIR = _ROOT / "data" / "signals"

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_MSG_LEN = 4096  # Telegram hard limit per message

# Fallback chat ID (the IDX Scanner Telegram group). The bot TOKEN is the real
# secret and must always come from TELEGRAM_BOT_TOKEN. The chat ID is not
# sensitive, so we allow a hardcoded fallback to keep automation running even if
# TELEGRAM_CHAT_ID is not explicitly set. Prefer the env var when present.
_DEFAULT_CHAT_ID = "-1003764018733"

# Signal display config
_SIG_EMOJI = {
    "BREAKOUT":   "🟢",
    "PRE_MARKUP": "🔵",
    "WATCH":      "🟠",
    "AVOID":      "🔴",
    "NONE":       "⚪",
}
_PRIORITY_SIGNALS = ["BREAKOUT", "PRE_MARKUP"]


# ---------------------------------------------------------------------------
# TelegramSender
# ---------------------------------------------------------------------------

class TelegramSender(BaseAlertSender):
    """Send messages via Telegram Bot API.

    Args:
        bot_token : Telegram bot token from @BotFather.
        chat_id   : Target chat / group / channel ID.
        parse_mode: 'HTML' (default) or 'Markdown'.
    """

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        parse_mode: str = "HTML",
    ) -> None:
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        # Chat ID resolution order: explicit arg → env var → hardcoded group default.
        env_chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        self.chat_id = chat_id or env_chat or _DEFAULT_CHAT_ID
        if not chat_id and not env_chat:
            logger.warning(
                "TELEGRAM_CHAT_ID not set — falling back to default group {}",
                _DEFAULT_CHAT_ID,
            )
        self.parse_mode = parse_mode

    @property
    def channel_name(self) -> str:
        return "Telegram"

    def send(self, message: str) -> AlertResult:
        """Send one message. Truncates to Telegram's 4096-char limit."""
        if not self.bot_token:
            err = (
                "TELEGRAM_BOT_TOKEN not set. "
                "Export it before sending: export TELEGRAM_BOT_TOKEN=<token>"
            )
            logger.error(err)
            return AlertResult(success=False, channel=self.channel_name, error=err)
        if not self.chat_id:
            err = "No chat ID resolved (TELEGRAM_CHAT_ID empty and no default)."
            logger.error(err)
            return AlertResult(success=False, channel=self.channel_name, error=err)

        # Truncate gracefully
        text = message[:_MAX_MSG_LEN - 3] + "…" if len(message) > _MAX_MSG_LEN else message

        try:
            import urllib.request
            import urllib.parse
            import json

            payload = json.dumps({
                "chat_id":    self.chat_id,
                "text":       text,
                "parse_mode": self.parse_mode,
                "disable_web_page_preview": True,
            }).encode()

            url = _TELEGRAM_API.format(token=self.bot_token)
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read())

            if body.get("ok"):
                logger.info("Telegram: message sent (chat_id={}, len={})", self.chat_id, len(text))
                return AlertResult(success=True, channel=self.channel_name)
            else:
                err = f"Telegram API error: {body}"
                logger.error(err)
                return AlertResult(success=False, channel=self.channel_name, error=err)

        except Exception as exc:
            logger.error("Telegram send failed: {}", exc)
            return AlertResult(success=False, channel=self.channel_name, error=str(exc))


    def send_document(
        self,
        file_path: "Path | str",
        caption: str = "",
    ) -> "AlertResult":
        """Send a local file as a Telegram document (e.g. .md, .pdf).

        Args:
            file_path : Local path to the file to upload.
            caption   : Short caption shown below the file in Telegram (≤1024 chars).

        Returns:
            AlertResult
        """
        from pathlib import Path as _Path
        import urllib.request, urllib.parse, json as _json

        file_path = _Path(file_path)
        if not file_path.exists():
            err = f"File not found: {file_path}"
            logger.error(err)
            return AlertResult(success=False, channel=self.channel_name, error=err)

        if not self.bot_token or not self.chat_id:
            err = "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set."
            logger.error(err)
            return AlertResult(success=False, channel=self.channel_name, error=err)

        try:
            import email.mime.multipart, email.mime.base, email.mime.text
            import email.encoders, io

            url = f"https://api.telegram.org/bot{self.bot_token}/sendDocument"

            # Build multipart form-data manually (no requests / httpx required)
            boundary = "IDXScannerBoundary123456"
            body_parts: list[bytes] = []

            def _field(name: str, value: str) -> bytes:
                return (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                    f"{value}\r\n"
                ).encode()

            body_parts.append(_field("chat_id", str(self.chat_id)))
            if caption:
                body_parts.append(_field("caption", caption[:1024]))

            # File part
            file_content = file_path.read_bytes()
            file_part = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="document"; filename="{file_path.name}"\r\n'
                f"Content-Type: application/octet-stream\r\n\r\n"
            ).encode() + file_content + b"\r\n"
            body_parts.append(file_part)
            body_parts.append(f"--{boundary}--\r\n".encode())

            body = b"".join(body_parts)
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = _json.loads(resp.read())

            if result.get("ok"):
                logger.info(
                    "Telegram: document '%s' sent (chat_id=%s)", file_path.name, self.chat_id
                )
                return AlertResult(success=True, channel=self.channel_name)
            else:
                err = f"Telegram API error: {result}"
                logger.error(err)
                return AlertResult(success=False, channel=self.channel_name, error=err)

        except Exception as exc:
            logger.error("Telegram send_document failed: %s", exc)
            return AlertResult(success=False, channel=self.channel_name, error=str(exc))

    def send_messages_batch(
        self,
        messages: list[str],
        delay_sec: float = 2.0,
    ) -> int:
        """Send multiple messages sequentially with a polite delay.

        Args:
            messages  : List of HTML strings.
            delay_sec : Seconds between messages (default 2.0).

        Returns:
            Number of messages successfully sent.
        """
        import time as _time
        success_count = 0
        for i, msg in enumerate(messages, 1):
            result = self.send(msg)
            if result:
                success_count += 1
            else:
                logger.error("Batch send failed (msg %d/%d): %s", i, len(messages), result.error)
            if i < len(messages):
                _time.sleep(delay_sec)
        logger.info(
            "Batch send: %d/%d messages sent.", success_count, len(messages)
        )
        return success_count


# ---------------------------------------------------------------------------
# Standalone helper (backward-compat / simple usage)
# ---------------------------------------------------------------------------

def send_telegram_message(
    text: str,
    bot_token: str | None = None,
    chat_id: str | None = None,
) -> bool:
    """One-liner helper. Returns True on success.

    Args:
        text      : Message body (HTML markup OK).
        bot_token : Falls back to TELEGRAM_BOT_TOKEN env var.
        chat_id   : Falls back to TELEGRAM_CHAT_ID env var.
    """
    sender = TelegramSender(bot_token=bot_token, chat_id=chat_id)
    result = sender.send(text)
    return bool(result)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _find_latest_date(max_days_back: int = 7) -> str | None:
    """Return the most recent date that has a ranked or signals file."""
    today = date.today()
    for delta in range(max_days_back + 1):
        d = (today - timedelta(days=delta)).strftime("%Y-%m-%d")
        if (_RANKED_DIR / f"ranked_{d}.csv").exists():
            return d
        if (_SIGNALS_DIR / f"{d}.parquet").exists():
            return d
        if (_SIGNALS_DIR / f"{d}.csv").exists():
            return d
    return None


def _load_scan(scan_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ranked_df, all_signals_df) for the given date.

    ranked_df    : BREAKOUT / PRE_MARKUP / WATCH tickers, sorted by score.
    all_signals_df: full universe with signal column (may be empty).
    """
    ranked_path = _RANKED_DIR / f"ranked_{scan_date}.csv"
    signals_parquet = _SIGNALS_DIR / f"{scan_date}.parquet"
    signals_csv = _SIGNALS_DIR / f"{scan_date}.csv"

    ranked = pd.DataFrame()
    signals = pd.DataFrame()

    if ranked_path.exists():
        ranked = pd.read_csv(ranked_path)

    if signals_parquet.exists():
        signals = pd.read_parquet(signals_parquet)
    elif signals_csv.exists():
        signals = pd.read_csv(signals_csv)

    return ranked, signals


# ---------------------------------------------------------------------------
# Message builder
# ---------------------------------------------------------------------------

def _day_name_id(dt: date) -> str:
    """Indonesian day name."""
    names = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
    return names[dt.weekday()]


def build_morning_message(
    scan_date: str,
    top_n: int = 7,
    run_date: date | None = None,
) -> str:
    """Build the morning alert HTML message for Telegram.

    The header explicitly distinguishes the SEND date (today, when the alert is
    delivered) from the MARKET DATA date (the last completed trading session the
    signals are based on), so a Thursday-morning alert about Wednesday's session
    is never mistaken for stale data.

    Args:
        scan_date: 'YYYY-MM-DD' — market data date (last completed trading session).
        top_n    : How many top tickers to list.
        run_date : The send date (defaults to today in WIB).

    Returns:
        HTML-formatted string ready for Telegram parse_mode=HTML.
    """
    ranked, signals = _load_scan(scan_date)

    # Use signals for distribution if available, else ranked
    dist_source = signals if not signals.empty else ranked

    # --- Header: run date (send) vs market data date ---
    if run_date is None:
        run_date = now_wib().date()
    market_dt = datetime.strptime(scan_date, "%Y-%m-%d").date()

    run_str    = f"{_day_name_id(run_date)}, {run_date.strftime('%d %b %Y')}"
    market_str = f"{_day_name_id(market_dt)}, {market_dt.strftime('%d %b %Y')}"

    lines: list[str] = [
        "📊 <b>IDX Morning Alert</b>",
        f"<i>{run_str} — Data market: {market_str}</i>",
        "",
    ]

    # --- Signal distribution ---
    if not dist_source.empty and "signal" in dist_source.columns:
        total = len(dist_source)
        dist = dist_source["signal"].value_counts()

        lines.append("<b>📈 Signal Distribution</b>")
        lines.append(f"<code>Total scanned : {total} ticker</code>")
        for sig in ["BREAKOUT", "PRE_MARKUP", "WATCH", "AVOID", "NONE"]:
            cnt = int(dist.get(sig, 0))
            if cnt == 0 and sig in ("AVOID", "NONE"):
                continue  # skip zero-count low-priority signals to save space
            emoji = _SIG_EMOJI.get(sig, "⚪")
            lines.append(f"<code>{emoji} {sig:<11}: {cnt:>3}</code>")
        lines.append("")
    else:
        lines.append("<i>⚠️ Signal distribution tidak tersedia.</i>")
        lines.append("")

    # --- Top picks ---
    if not ranked.empty:
        # Filter priority signals
        top = ranked[ranked["signal"].isin(_PRIORITY_SIGNALS)].copy()
        has_priority = not top.empty
        if top.empty:
            top = ranked.copy()  # fallback: show all if no priority signals

        sort_col = "enhanced_total_score" if "enhanced_total_score" in top.columns else "total_score"
        if sort_col in top.columns:
            top = top.sort_values(sort_col, ascending=False)

        top = top.head(top_n)

        if has_priority:
            lines.append(f"<b>🏆 Top {len(top)} Pick Hari Ini</b>")
        else:
            lines.append("<i>⚪ Tidak ada sinyal prioritas (BREAKOUT/PRE_MARKUP) hari ini.</i>")
            lines.append(f"<b>👀 Top {len(top)} Watchlist</b>")
        for i, (_, row) in enumerate(top.iterrows(), 1):
            ticker = str(row.get("ticker", "?"))
            sig = str(row.get("signal", ""))
            score = row.get("total_score", 0)
            enh = row.get("enhanced_total_score", None)
            close = row.get("close", None)

            emoji = _SIG_EMOJI.get(sig, "⚪")
            score_str = f"{float(score):.1f}" if pd.notna(score) else "—"
            enh_str = f"→{float(enh):.1f}" if pd.notna(enh) and enh is not None else ""
            close_str = f"Rp{int(close):,}" if pd.notna(close) and close is not None else ""

            line = f"{i}. <b>{ticker}</b>  {emoji} {sig}  <code>{score_str}{enh_str}</code>"
            if close_str:
                line += f"  {close_str}"
            lines.append(line)

            # Optional sub-details
            details: list[str] = []
            rsi = row.get("rsi14")
            vol = row.get("vol_ratio_20d")
            pct52 = row.get("pct_from_52w_high")
            if pd.notna(rsi):
                details.append(f"RSI {float(rsi):.0f}")
            if pd.notna(vol):
                details.append(f"Vol×{float(vol):.1f}")
            if pd.notna(pct52):
                details.append(f"52w {float(pct52):+.1f}%")
            if details:
                lines.append(f"   <i>{'  ·  '.join(details)}</i>")

        lines.append("")
    else:
        lines.append("<i>⚠️ Tidak ada data ranked tersedia untuk tanggal ini.</i>")
        lines.append("")

    # --- Footer ---
    now_str = now_wib().strftime("%H:%M WIB")
    lines += [
        "─────────────────────",
        f"<i>📡 Dikirim: {now_str}</i>",
        "<i>🤖 IDX Scanner Agent</i>",
    ]

    return "\n".join(lines)


def build_stale_message(
    expected_md: date,
    latest_md: date | None,
    run_date: date | None = None,
) -> str:
    """Build a short STATUS message used when the freshest scan is stale.

    Sent INSTEAD of the normal morning alert so the user is explicitly told the
    data for the expected session is not available yet (rather than silently
    receiving an old session's signals).

    Args:
        expected_md: the market date we expected (last completed session).
        latest_md  : the latest market date actually available (or None).
        run_date   : the send date (defaults to today in WIB).
    """
    if run_date is None:
        run_date = now_wib().date()
    now_str = now_wib().strftime("%H:%M WIB")

    run_str = f"{_day_name_id(run_date)}, {run_date.strftime('%d %b %Y')}"
    exp_str = f"{_day_name_id(expected_md)}, {expected_md.strftime('%d %b %Y')}"
    latest_str = (
        f"{_day_name_id(latest_md)}, {latest_md.strftime('%d %b %Y')}"
        if latest_md is not None else "tidak ada"
    )

    return "\n".join([
        "⚠️ <b>IDX Morning Alert — Data Belum Update</b>",
        f"<i>{run_str} · {now_str}</i>",
        "",
        f"<code>Sesi diharapkan : {exp_str}</code>",
        f"<code>Data tersedia   : {latest_str}</code>",
        "",
        "Data sesi bursa terakhir belum masuk (kemungkinan lag penyedia data).",
        "Morning alert normal <b>ditunda</b> agar tidak mengirim data lama.",
        "─────────────────────",
        "<i>🤖 IDX Scanner Agent</i>",
    ])


# ---------------------------------------------------------------------------
# Freshness resolution
# ---------------------------------------------------------------------------

def resolve_alert_target(date_arg: str | None, now: datetime | None = None) -> dict:
    """Decide what the alert should report and whether the data is fresh.

    The freshness check is the root-cause fix for "Thursday morning but
    Scan: 2026-06-02": it compares the latest available scan against the
    EXPECTED market date (last completed trading session) and flags stale data
    so an old session is never sent as a normal morning alert.

    Args:
        date_arg: explicit --date override (YYYY-MM-DD) or None for auto.
        now     : current time (defaults to now in WIB).

    Returns:
        dict with keys:
          run_date  (date)        — send date in WIB
          expected  (date)        — last completed trading session
          scan_date (str | None)  — date string to report
          latest_md (date | None) — latest available market date
          verdict   (str)         — "fresh" | "stale" | "nodata" | "manual"
    """
    if now is None:
        now = now_wib()
    run_date = now.date()
    expected = expected_market_date(now)

    # Explicit override — trust the caller, skip the freshness gate.
    if date_arg:
        return {
            "run_date": run_date, "expected": expected, "scan_date": date_arg,
            "latest_md": datetime.strptime(date_arg, "%Y-%m-%d").date(),
            "verdict": "manual",
        }

    latest = _find_latest_date()
    if latest is None:
        return {
            "run_date": run_date, "expected": expected, "scan_date": None,
            "latest_md": None, "verdict": "nodata",
        }

    latest_md = datetime.strptime(latest, "%Y-%m-%d").date()
    verdict = "fresh" if latest_md >= expected else "stale"
    return {
        "run_date": run_date, "expected": expected, "scan_date": latest,
        "latest_md": latest_md, "verdict": verdict,
    }


def _print_dry(message: str) -> None:
    print("\n" + "=" * 50)
    print("DRY RUN — message NOT sent to Telegram")
    print("=" * 50)
    print(message)
    print("=" * 50)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="IDX Stock Scanner — Morning Alert via Telegram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Environment variables:
              TELEGRAM_BOT_TOKEN   Bot token from @BotFather
              TELEGRAM_CHAT_ID     Chat / group / channel ID (falls back to default group)

            Freshness:
              The alert auto-resolves the latest scan and compares it to the last
              completed trading session (WIB-aware). If the data is older than
              expected it sends a short STALE notice instead of a normal alert.
              Pass --require-fresh to also exit non-zero (2) on stale data.

            Exit codes:
              0  normal alert sent / dry-run / stale notice sent (without --require-fresh)
              1  send failure or no scan data at all
              2  data stale and --require-fresh set (stale notice still sent)

            Examples:
              python -m stock_scanner.alerts.telegram_alert
              python -m stock_scanner.alerts.telegram_alert --dry-run
              python -m stock_scanner.alerts.telegram_alert --require-fresh
              python -m stock_scanner.alerts.telegram_alert --date 2026-05-09
        """),
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Market data date to report (YYYY-MM-DD). Default: latest available. "
                             "Bypasses the freshness check.")
    parser.add_argument("--top-n", type=int, default=7,
                        help="Number of top picks to include (default: 7).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print message to stdout instead of sending to Telegram.")
    parser.add_argument("--skip-if-empty", action="store_true",
                        help="If no priority signals (BREAKOUT/PRE_MARKUP), skip sending the normal alert.")
    parser.add_argument("--require-fresh", action="store_true",
                        help="Exit code 2 if the data is stale (after sending the stale notice). "
                             "Use in CI so stale days show as a failed job.")
    args = parser.parse_args()

    target = resolve_alert_target(args.date)
    run_date, expected = target["run_date"], target["expected"]
    scan_date, latest_md, verdict = target["scan_date"], target["latest_md"], target["verdict"]

    # ── Observability: log every input to the freshness decision ─────────
    logger.info("Run date (WIB)        : {}", run_date)
    logger.info("Expected market date  : {}", expected)
    logger.info("Latest available scan : {}", scan_date or "NONE")
    logger.info("Freshness verdict     : {}", verdict.upper())

    # ── No data at all ───────────────────────────────────────────────────
    if verdict == "nodata":
        logger.error(
            "No scan data found in {} or {}. Run the scan first: "
            "python -m stock_scanner.pipeline.run_daily_scan",
            _RANKED_DIR, _SIGNALS_DIR,
        )
        return 1

    # ── Stale: send a status notice INSTEAD of the normal alert ──────────
    if verdict == "stale":
        logger.warning(
            "STALE DATA: latest scan {} is older than expected session {}. "
            "Withholding normal morning alert.", scan_date, expected,
        )
        msg = build_stale_message(expected, latest_md, run_date)
        if args.dry_run:
            _print_dry(msg)
            return 0
        result = TelegramSender().send(msg)
        if not result:
            logger.error("Failed to send stale notice: {}", result.error)
            return 1
        logger.info("Stale notice sent to Telegram.")
        return 2 if args.require_fresh else 0

    # ── Fresh (or manual --date): build + send the normal alert ──────────
    logger.info("Building morning alert — market data {} (verdict={}).", scan_date, verdict)
    ranked, _ = _load_scan(scan_date)
    n_priority = 0
    if not ranked.empty and "signal" in ranked.columns:
        n_priority = int(ranked["signal"].isin(_PRIORITY_SIGNALS).sum())

    if n_priority == 0 and args.skip_if_empty:
        logger.info("No priority signals and --skip-if-empty set — skipping send.")
        return 0

    message = build_morning_message(scan_date, top_n=args.top_n, run_date=run_date)
    if args.dry_run:
        _print_dry(message)
        return 0

    result = TelegramSender().send(message)
    if result:
        logger.info("Morning alert sent successfully ({} priority pick(s)).", n_priority)
        return 0
    logger.error("Failed to send alert: {}", result.error)
    return 1


if __name__ == "__main__":
    sys.exit(main())

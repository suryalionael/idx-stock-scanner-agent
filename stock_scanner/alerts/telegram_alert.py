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
import textwrap
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from stock_scanner.alerts.base import AlertResult, BaseAlertSender

# ---------------------------------------------------------------------------
# Paths (resolved relative to repo root)
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent
_RANKED_DIR = _ROOT / "data" / "ranked"
_SIGNALS_DIR = _ROOT / "data" / "signals"

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
_MAX_MSG_LEN = 4096  # Telegram hard limit per message

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
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.parse_mode = parse_mode

    @property
    def channel_name(self) -> str:
        return "Telegram"

    def send(self, message: str) -> AlertResult:
        """Send one message. Truncates to Telegram's 4096-char limit."""
        if not self.bot_token or not self.chat_id:
            err = (
                "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set. "
                "Export both env vars before sending."
            )
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
                logger.info("Telegram: message sent (chat_id=%s, len=%d)", self.chat_id, len(text))
                return AlertResult(success=True, channel=self.channel_name)
            else:
                err = f"Telegram API error: {body}"
                logger.error(err)
                return AlertResult(success=False, channel=self.channel_name, error=err)

        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
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


def build_morning_message(scan_date: str, top_n: int = 7) -> str:
    """Build the morning alert HTML message for Telegram.

    Args:
        scan_date: 'YYYY-MM-DD' string — which day's scan to use.
        top_n    : How many top tickers to list.

    Returns:
        HTML-formatted string ready for Telegram parse_mode=HTML.
    """
    ranked, signals = _load_scan(scan_date)

    # Use signals for distribution if available, else ranked
    dist_source = signals if not signals.empty else ranked

    # --- Header ---
    dt = datetime.strptime(scan_date, "%Y-%m-%d").date()
    day_name = _day_name_id(dt)
    date_str = dt.strftime("%d %b %Y")

    lines: list[str] = [
        "📊 <b>IDX Morning Alert</b>",
        f"<i>{day_name}, {date_str} — Scan: {scan_date}</i>",
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
        if top.empty:
            top = ranked.copy()  # fallback: show all if no priority signals

        sort_col = "enhanced_total_score" if "enhanced_total_score" in top.columns else "total_score"
        if sort_col in top.columns:
            top = top.sort_values(sort_col, ascending=False)

        top = top.head(top_n)

        lines.append(f"<b>🏆 Top {len(top)} Pick Hari Ini</b>")
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
    now_str = datetime.now().strftime("%H:%M WIB")
    lines += [
        "─────────────────────",
        f"<i>📡 Dikirim: {now_str}</i>",
        "<i>🤖 IDX Scanner Agent</i>",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="IDX Stock Scanner — Morning Alert via Telegram",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Environment variables:
              TELEGRAM_BOT_TOKEN   Bot token from @BotFather
              TELEGRAM_CHAT_ID     Chat / group / channel ID

            Examples:
              python -m stock_scanner.alerts.telegram_alert
              python -m stock_scanner.alerts.telegram_alert --dry-run
              python -m stock_scanner.alerts.telegram_alert --date 2026-05-09
        """),
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Scan date to use (YYYY-MM-DD). Default: latest available.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=7,
        help="Number of top picks to include (default: 7).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print message to stdout instead of sending to Telegram.",
    )
    args = parser.parse_args()

    # Resolve date
    scan_date = args.date or _find_latest_date()
    if scan_date is None:
        logger.error(
            "No scan data found in %s or %s.\n"
            "Run: python -m stock_scanner.pipeline.run_daily_scan",
            _RANKED_DIR, _SIGNALS_DIR,
        )
        raise SystemExit(1)

    logger.info("Building morning alert for scan date: %s", scan_date)
    message = build_morning_message(scan_date, top_n=args.top_n)

    if args.dry_run:
        print("\n" + "=" * 50)
        print("DRY RUN — message NOT sent to Telegram")
        print("=" * 50)
        print(message)
        print("=" * 50)
        return

    sender = TelegramSender()
    result = sender.send(message)

    if result:
        logger.info("Morning alert sent successfully.")
    else:
        logger.error("Failed to send alert: %s", result.error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

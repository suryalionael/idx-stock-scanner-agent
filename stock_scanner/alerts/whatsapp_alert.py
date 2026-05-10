"""WhatsApp alert sender — stub / placeholder.

TODO: Implement using one of:
    A) Twilio WhatsApp API  (https://www.twilio.com/whatsapp)
    B) 360dialog / WABA     (https://docs.360dialog.com/)
    C) Meta Cloud API       (https://developers.facebook.com/docs/whatsapp/cloud-api)

Quick-start with Twilio (Option A):
    pip install twilio
    export TWILIO_ACCOUNT_SID=...
    export TWILIO_AUTH_TOKEN=...
    export TWILIO_WHATSAPP_FROM=whatsapp:+14155238886   # Twilio sandbox number
    export WHATSAPP_TO=whatsapp:+628xxxxxxxxxx          # your number

Required env vars (when implemented):
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_WHATSAPP_FROM
    WHATSAPP_TO

Usage (once implemented):
    sender = WhatsAppSender()
    sender.send("📊 IDX Morning Alert...")
"""
from __future__ import annotations

import os

from loguru import logger

from stock_scanner.alerts.base import AlertResult, BaseAlertSender


class WhatsAppSender(BaseAlertSender):
    """WhatsApp alert sender via Twilio.

    Status: NOT IMPLEMENTED — returns a friendly error until wired up.

    To implement:
        1. pip install twilio (add to pyproject.toml optional deps)
        2. Uncomment and fill the implementation below
        3. Set the env vars listed above
        4. Wire into the morning alert runner
    """

    def __init__(
        self,
        account_sid: str | None = None,
        auth_token: str | None = None,
        from_: str | None = None,
        to: str | None = None,
    ) -> None:
        self.account_sid = account_sid or os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token  = auth_token  or os.getenv("TWILIO_AUTH_TOKEN", "")
        self.from_       = from_       or os.getenv("TWILIO_WHATSAPP_FROM", "")
        self.to          = to          or os.getenv("WHATSAPP_TO", "")

    @property
    def channel_name(self) -> str:
        return "WhatsApp"

    def send(self, message: str) -> AlertResult:
        """TODO: implement real Twilio send.

        Example implementation (uncomment after pip install twilio):
        ---
        from twilio.rest import Client
        client = Client(self.account_sid, self.auth_token)
        msg = client.messages.create(
            from_=self.from_,
            body=message,
            to=self.to,
        )
        return AlertResult(
            success=msg.status not in ("failed", "undelivered"),
            channel=self.channel_name,
            error=msg.error_message,
        )
        ---
        """
        err = (
            "WhatsAppSender is not yet implemented. "
            "See whatsapp_alert.py for instructions on wiring Twilio."
        )
        logger.warning(err)
        return AlertResult(success=False, channel=self.channel_name, error=err)

"""Abstract base for all alert channels.

To add a new channel (e.g. WhatsApp, Slack, email):
    1. Subclass BaseAlertSender
    2. Implement send() and channel_name
    3. Wire it into the morning alert runner
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


# ---------------------------------------------------------------------------
# Result wrapper
# ---------------------------------------------------------------------------

@dataclass
class AlertResult:
    """Outcome of a single send() call."""
    success: bool
    channel: str
    sent_at: datetime = field(default_factory=datetime.now)
    error: str | None = None

    def __bool__(self) -> bool:
        return self.success


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseAlertSender(ABC):
    """Common interface for all alert delivery channels.

    Usage pattern:
        sender = ConcreteSender(...)
        result = sender.send("hello world")
        if not result:
            logger.error("Alert failed: %s", result.error)
    """

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """Human-readable channel name used in logs, e.g. 'Telegram', 'WhatsApp'."""

    @abstractmethod
    def send(self, message: str) -> AlertResult:
        """Deliver message to the channel.

        Args:
            message: Plain text or channel-specific markup (e.g. HTML for Telegram).

        Returns:
            AlertResult with success flag and optional error description.
        """

    def send_many(self, messages: list[str]) -> list[AlertResult]:
        """Send multiple messages sequentially. Override for batching."""
        return [self.send(m) for m in messages]

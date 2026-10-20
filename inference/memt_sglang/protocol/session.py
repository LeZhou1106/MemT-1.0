"""Session lifecycle errors and acknowledgement states shared with clients."""

from enum import StrEnum


class SessionError(RuntimeError):
    """An operation conflicts with the current session lifecycle."""


class CapacityError(SessionError):
    """The configured session or commit capacity is exhausted."""


class ReceiptStatus(StrEnum):
    PENDING = "pending"
    APPLIED = "applied"
    ABORTED = "aborted"


class TransactionStatus(StrEnum):
    IDLE = "idle"
    OPEN = "open"
    PUBLISHED = "published"
    ABORTED = "aborted"

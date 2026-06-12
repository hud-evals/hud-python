"""Wall-clock helpers for wire timestamps."""

from __future__ import annotations

from datetime import UTC, datetime


def now_iso() -> str:
    """Current time as an ISO-8601 string with a ``Z`` suffix.

    The wire format for step and span timestamps.
    """
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")

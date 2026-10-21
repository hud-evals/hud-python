"""The values exports carry, parsed leniently: a cell that does not parse reads as missing."""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, tzinfo

from dateutil import parser as dateparser

_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %I:%M %p",
    "%d/%b/%y %I:%M %p",
    "%d/%b/%y %H:%M",
)
_DATE_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%m/%d/%y",
    "%Y/%m/%d",
    "%d-%b-%Y",
    "%d-%b-%y",
    "%d %b %Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d/%b/%y",
    "%Y%m%d",
)


def parse_datetime(text: str | None, tz: tzinfo = UTC) -> datetime | None:
    """A timestamp; naive values are taken to be in ``tz``."""
    if text is None or not text.strip():
        return None
    value = text.strip()
    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        for fmt in (*_DATETIME_FORMATS, *_DATE_FORMATS):
            try:
                parsed = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
    if parsed is None:
        try:
            parsed = dateparser.parse(value)
        except (ValueError, OverflowError):
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=tz)


def parse_date(text: str | None) -> date | None:
    parsed = parse_datetime(text)
    return parsed.date() if parsed else None


def parse_number(text: str | None) -> float | None:
    if text is None:
        return None
    value = text.replace(",", "").replace("%", "").strip()
    try:
        return float(value) if value else None
    except ValueError:
        return None


def split_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [part.strip() for part in re.split(r"[;,|]", text) if part.strip()]

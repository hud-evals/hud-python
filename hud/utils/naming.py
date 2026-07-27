"""Names shared across the SDK's surfaces."""

from __future__ import annotations

import re


def normalize_environment_name(name: str, *, default: str = "environment") -> str:
    """Slugify *name* into a valid environment name (lowercase, ``[a-z0-9-]``).

    One implementation, because the name an integration puts on a row has to
    equal the one a deploy registers — two spellings would join silently
    wrong.
    """
    normalized = name.strip().lower()
    normalized = normalized.replace(" ", "-").replace("_", "-")
    normalized = re.sub(r"[^a-z0-9-]", "", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-") or default

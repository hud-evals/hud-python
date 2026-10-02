"""Names shared across the SDK's surfaces."""

from __future__ import annotations

import re


def normalize_environment_name(name: str, *, default: str = "environment") -> str:
    """Slugify ``name`` into a valid environment name (lowercase, ``[a-z0-9-]``).

    Integrations and deploys must both use it, or their names stop matching.
    """
    normalized = name.strip().lower()
    normalized = normalized.replace(" ", "-").replace("_", "-")
    normalized = re.sub(r"[^a-z0-9-]", "", normalized)
    normalized = re.sub(r"-+", "-", normalized)
    return normalized.strip("-") or default

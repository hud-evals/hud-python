"""Names shared across the SDK's surfaces."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


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


def rename_environment_in_source(source: Path, source_name: str, target_name: str) -> None:
    """Rename a template's literal declaration, including nested environment packages."""
    source_name = normalize_environment_name(source_name)
    target_name = normalize_environment_name(target_name)
    declaration = f'Environment(name="{source_name}")'
    candidates = [source / "env.py", source / "environment" / "env.py"]
    matches = [
        path
        for path in candidates
        if path.is_file() and declaration in path.read_text(encoding="utf-8")
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {declaration} declaration in {source}")
    path = matches[0]
    contents = path.read_text(encoding="utf-8")
    if contents.count(declaration) != 1:
        raise ValueError(f"expected one {declaration} declaration in {path}")
    path.write_text(
        contents.replace(declaration, f'Environment(name="{target_name}")'), encoding="utf-8"
    )

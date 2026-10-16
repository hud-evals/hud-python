"""Files a test writes: a project tree, or an image."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def write_files(root: Path, files: Mapping[str, bytes | str]) -> None:
    """Write ``files`` under ``root``; a name ending in ``/`` is an empty directory."""
    root.mkdir(parents=True, exist_ok=True)
    for relative, content in files.items():
        path = root / relative
        if relative.endswith("/"):
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            path.write_text(content)
        else:
            path.write_bytes(content)


def image(
    size: tuple[int, int],
    *,
    format: str = "PNG",
    mode: str = "RGB",
    color: tuple[int, ...] | str = "white",
    exif: Image.Exif | None = None,
) -> bytes:
    """An image of ``size`` encoded as ``format``."""
    buffer = io.BytesIO()
    options: dict[str, Any] = {} if exif is None else {"exif": exif}
    Image.new(mode, size, color).save(buffer, format=format, **options)
    return buffer.getvalue()

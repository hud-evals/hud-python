"""Files a test writes: a project tree, a case directory, or an image."""

from __future__ import annotations

import io
import shutil
import tomllib
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


def read_expected(case: Path) -> dict[str, Any]:
    """The ``expected.toml`` of a case directory, or an empty table when it has none."""
    spec = case / "expected.toml"
    return tomllib.loads(spec.read_text("utf-8")) if spec.is_file() else {}


def assemble_case(case: Path, into: Path, *ignore: str) -> None:
    """Copy the case directory ``case`` to ``into``, symlinks kept as links.

    ``base = "<case>"`` in ``expected.toml`` names a sibling case to start from:
    it is assembled first and ``case``'s files replace its files, so a variant
    holds only what it changes. Neither ``expected.toml`` nor a name matching an
    ``ignore`` pattern is copied.
    """
    base = read_expected(case).get("base")
    if base is not None:
        assemble_case(case.parent / base, into, *ignore)
    shutil.copytree(
        case,
        into,
        symlinks=True,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("expected.toml", "__pycache__", *ignore),
    )


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

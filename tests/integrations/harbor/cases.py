"""Harbor case directories: a tree of files a case starts from, and what it expects."""

from __future__ import annotations

import shutil
import tomllib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


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

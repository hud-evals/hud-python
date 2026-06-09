"""Pure helpers in ``hud.cli.build``: version parsing and bumping."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.cli.build import get_existing_version, increment_version, parse_version

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_version_pads_and_strips_v() -> None:
    assert parse_version("1.2.3") == (1, 2, 3)
    assert parse_version("v2.0") == (2, 0, 0)
    assert parse_version("3") == (3, 0, 0)
    assert parse_version("garbage") == (0, 0, 0)


def test_increment_version() -> None:
    assert increment_version("1.2.3", "patch") == "1.2.4"
    assert increment_version("1.2.3", "minor") == "1.3.0"
    assert increment_version("1.2.3", "major") == "2.0.0"
    assert increment_version("1.2.3") == "1.2.4"  # default is patch


def test_get_existing_version_reads_lock(tmp_path: Path) -> None:
    lock_path = tmp_path / "hud.lock.yaml"
    lock_path.write_text("build:\n  version: 1.2.3\n", encoding="utf-8")
    assert get_existing_version(lock_path) == "1.2.3"


def test_get_existing_version_none_when_missing(tmp_path: Path) -> None:
    assert get_existing_version(tmp_path / "hud.lock.yaml") is None

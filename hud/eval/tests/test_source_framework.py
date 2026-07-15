"""``resolve_source_framework``: Dockerfile.hud marker detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.eval.source_framework import resolve_source_framework

if TYPE_CHECKING:
    from pathlib import Path


def test_resolve_returns_hud_when_dockerfile_hud_present(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile.hud").write_text("FROM python:3.11\n")
    assert resolve_source_framework(tmp_path) == "hud"


def test_resolve_returns_none_without_marker(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text("FROM python:3.11\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert resolve_source_framework(tmp_path) is None


def test_resolve_returns_none_for_empty_dir(tmp_path: Path) -> None:
    assert resolve_source_framework(tmp_path) is None

"""Pure helpers in ``hud.cli.build``: version parsing/bumping + Dockerfile parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.cli.build import (
    extract_env_vars_from_dockerfile,
    get_existing_version,
    increment_version,
    parse_base_image,
    parse_version,
)

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


def test_parse_base_image_first_from_strips_stage(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile"
    df.write_text("# comment\nFROM python:3.11 AS build\nRUN echo hi\n", encoding="utf-8")
    assert parse_base_image(df) == "python:3.11"


def test_parse_base_image_missing_file_is_none(tmp_path: Path) -> None:
    assert parse_base_image(tmp_path / "nope") is None


def test_extract_env_vars_required_runtime_only(tmp_path: Path) -> None:
    df = tmp_path / "Dockerfile.hud"
    df.write_text(
        "FROM python:3.11\n"
        "ARG BUILD_ONLY\n"  # build-time only -> not required
        "ENV NEEDS_VALUE=\n"  # no value -> required
        "ENV HAS_DEFAULT=foo\n"  # has value -> not required
        "ENV BARE_ENV\n",  # no '=' -> required
        encoding="utf-8",
    )
    required, _optional = extract_env_vars_from_dockerfile(df)
    assert "NEEDS_VALUE" in required
    assert "BARE_ENV" in required
    assert "HAS_DEFAULT" not in required
    assert "BUILD_ONLY" not in required  # ARG is build-time, not runtime


def test_get_existing_version_none_when_missing(tmp_path: Path) -> None:
    assert get_existing_version(tmp_path / "hud.lock.yaml") is None

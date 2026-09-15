"""Deploy-time pyproject / Dockerfile lint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.cli.app import app  # noqa: F401
from hud.cli.deploy import (
    EnvironmentSource,
    _validate_dockerfile,
    _validate_environment,
    _validate_pyproject,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def test_no_pyproject_is_clean(tmp_path: Path) -> None:
    assert _validate_pyproject(EnvironmentSource.open(tmp_path)) == []


def test_missing_license_file_is_error(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nlicense = {file = "LICENSE"}\n')

    issues = _validate_pyproject(EnvironmentSource.open(tmp_path))

    assert [i.severity for i in issues] == ["error"]
    assert "License file not found" in issues[0].message


def test_missing_readme_is_warning(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nreadme = "README.md"\n')

    issues = _validate_pyproject(EnvironmentSource.open(tmp_path))

    assert [i.severity for i in issues] == ["warning"]
    assert "Readme file not found" in issues[0].message


def test_all_references_present_is_clean(tmp_path: Path) -> None:
    _write(
        tmp_path / "pyproject.toml",
        '[project]\nname = "x"\nlicense = {file = "LICENSE"}\nreadme = "README.md"\n',
    )
    _write(tmp_path / "LICENSE", "MIT")
    _write(tmp_path / "README.md", "# x")

    assert _validate_pyproject(EnvironmentSource.open(tmp_path)) == []


def test_unparseable_pyproject_is_error(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", "this is not = valid = toml [[[")

    issues = _validate_pyproject(EnvironmentSource.open(tmp_path))

    assert any(i.severity == "error" and "Failed to parse" in i.message for i in issues)


def test_license_not_copied_before_install_is_error(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nlicense = {file = "LICENSE"}\n')
    _write(
        tmp_path / "Dockerfile.hud",
        "FROM python:3.11\nCOPY pyproject.toml ./\nRUN uv sync\nCOPY . .\n",
    )

    issues = _validate_dockerfile(EnvironmentSource.open(tmp_path))

    assert any(i.severity == "error" and "LICENSE" in i.message for i in issues)


def test_full_copy_before_install_is_clean(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nlicense = {file = "LICENSE"}\n')
    _write(tmp_path / "Dockerfile.hud", "FROM python:3.11\nCOPY . .\nRUN uv sync\n")

    assert _validate_dockerfile(EnvironmentSource.open(tmp_path)) == []


def test_copy_continuation_before_install_is_clean(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nlicense = {file = "LICENSE"}\n')
    _write(
        tmp_path / "Dockerfile.hud",
        "FROM python:3.11\nCOPY LICENSE \\\n  ./\nRUN uv sync\n",
    )

    assert _validate_dockerfile(EnvironmentSource.open(tmp_path)) == []


def test_no_dockerfile_is_clean(tmp_path: Path) -> None:
    assert _validate_dockerfile(EnvironmentSource.open(tmp_path)) == []


def test_validate_environment_aggregates(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nlicense = {file = "LICENSE"}\n')
    _write(
        tmp_path / "Dockerfile.hud",
        "FROM python:3.11\nCOPY pyproject.toml ./\nRUN uv sync\nCOPY . .\n",
    )

    issues = _validate_environment(EnvironmentSource.open(tmp_path))
    assert len(issues) >= 2

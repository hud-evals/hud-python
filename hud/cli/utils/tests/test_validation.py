"""``hud.cli.utils.validation`` — pre-deploy checks over an env directory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.cli.utils.validation import (
    ValidationIssue,
    format_validation_issues,
    validate_dockerfile,
    validate_environment,
    validate_pyproject_references,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# ─── validate_pyproject_references ────────────────────────────────────


def test_no_pyproject_is_clean(tmp_path: Path) -> None:
    assert validate_pyproject_references(tmp_path) == []


def test_missing_license_file_is_error(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nlicense = {file = "LICENSE"}\n')

    issues = validate_pyproject_references(tmp_path)

    assert [i.severity for i in issues] == ["error"]
    assert "License file not found" in issues[0].message


def test_missing_readme_is_warning(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nreadme = "README.md"\n')

    issues = validate_pyproject_references(tmp_path)

    assert [i.severity for i in issues] == ["warning"]
    assert "Readme file not found" in issues[0].message


def test_all_references_present_is_clean(tmp_path: Path) -> None:
    _write(
        tmp_path / "pyproject.toml",
        '[project]\nname = "x"\nlicense = {file = "LICENSE"}\nreadme = "README.md"\n',
    )
    _write(tmp_path / "LICENSE", "MIT")
    _write(tmp_path / "README.md", "# x")

    assert validate_pyproject_references(tmp_path) == []


def test_unparseable_pyproject_is_error(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", "this is not = valid = toml [[[")

    issues = validate_pyproject_references(tmp_path)

    assert any(i.severity == "error" and "Failed to parse" in i.message for i in issues)


# ─── validate_dockerfile (copy-order) ─────────────────────────────────


def test_license_not_copied_before_install_is_error(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nlicense = {file = "LICENSE"}\n')
    _write(
        tmp_path / "Dockerfile.hud",
        "FROM python:3.11\nCOPY pyproject.toml ./\nRUN uv sync\nCOPY . .\n",
    )

    issues = validate_dockerfile(tmp_path)

    assert any(i.severity == "error" and "LICENSE" in i.message for i in issues)


def test_full_copy_before_install_is_clean(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nlicense = {file = "LICENSE"}\n')
    _write(tmp_path / "Dockerfile.hud", "FROM python:3.11\nCOPY . .\nRUN uv sync\n")

    # ``COPY . .`` precedes the install, so nothing is missing.
    assert validate_dockerfile(tmp_path) == []


def test_no_dockerfile_is_clean(tmp_path: Path) -> None:
    assert validate_dockerfile(tmp_path) == []


# ─── aggregation + formatting ─────────────────────────────────────────


def test_validate_environment_aggregates(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nlicense = {file = "LICENSE"}\n')
    _write(
        tmp_path / "Dockerfile.hud",
        "FROM python:3.11\nCOPY pyproject.toml ./\nRUN uv sync\nCOPY . .\n",
    )

    issues = validate_environment(tmp_path)
    # one from pyproject (missing LICENSE) + one from dockerfile (copy order)
    assert len(issues) >= 2


def test_format_validation_issues() -> None:
    assert format_validation_issues([]) == ""

    text = format_validation_issues(
        [
            ValidationIssue(severity="error", message="boom", file="pyproject.toml", hint="fix it"),
            ValidationIssue(severity="warning", message="meh"),
        ]
    )
    assert "1 error(s)" in text
    assert "1 warning(s)" in text
    assert "boom" in text
    assert "fix it" in text

"""EnvironmentSource: identity, dockerfile, source files, references, validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.environment.source import EnvironmentSource

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# ─── identity ──────────────────────────────────────────────────────────


def test_environment_name_override() -> None:
    assert EnvironmentSource.open(".").environment_name("Custom Env") == "custom-env"


def test_environment_name_auto(tmp_path: Path) -> None:
    env = tmp_path / "my_env"
    env.mkdir()
    assert EnvironmentSource.open(env).environment_name() == "my-env"


def test_detects_environment_directory(tmp_path: Path) -> None:
    d = tmp_path / "env"
    d.mkdir()
    assert EnvironmentSource.open(d).is_environment is False
    (d / "Dockerfile").write_text("FROM python:3.11")
    assert EnvironmentSource.open(d).is_environment is False
    (d / "pyproject.toml").write_text("[tool.hud]")
    assert EnvironmentSource.open(d).is_environment is True


def test_detects_environment_with_dockerfile_hud(tmp_path: Path) -> None:
    d = tmp_path / "env"
    d.mkdir()
    (d / "Dockerfile.hud").write_text("FROM python:3.11")
    assert EnvironmentSource.open(d).is_environment is False
    (d / "pyproject.toml").write_text("[tool.hud]")
    assert EnvironmentSource.open(d).is_environment is True


def test_prefers_dockerfile_hud(tmp_path: Path) -> None:
    d = tmp_path / "env"
    d.mkdir()
    assert EnvironmentSource.open(d).dockerfile is None
    (d / "Dockerfile").write_text("FROM python:3.11")
    assert EnvironmentSource.open(d).dockerfile == d / "Dockerfile"
    (d / "Dockerfile.hud").write_text("FROM python:3.12")
    assert EnvironmentSource.open(d).dockerfile == d / "Dockerfile.hud"


# ─── dockerfile parsing ────────────────────────────────────────────────


def test_base_image_strips_stage(tmp_path: Path) -> None:
    _write(tmp_path / "Dockerfile", "# comment\nFROM python:3.11 AS build\nRUN echo hi\n")
    assert EnvironmentSource.open(tmp_path).base_image() == "python:3.11"


def test_base_image_without_dockerfile_is_none(tmp_path: Path) -> None:
    assert EnvironmentSource.open(tmp_path).base_image() is None


# ─── source files / hash ───────────────────────────────────────────────


def test_source_hash_changes_with_content(tmp_path: Path) -> None:
    env = tmp_path / "env"
    env.mkdir()
    (env / "Dockerfile").write_text("FROM python:3.11")
    (env / "pyproject.toml").write_text("[tool.hud]\n")
    (env / "server").mkdir()
    (env / "server" / "main.py").write_text("print('hi')\n")

    source = EnvironmentSource.open(env)
    h1 = source.source_hash()
    (env / "server" / "main.py").write_text("print('bye')\n")
    h2 = source.source_hash()
    assert h1 != h2


def test_source_files_sorted(tmp_path: Path) -> None:
    env = tmp_path / "env"
    env.mkdir()
    (env / "Dockerfile").write_text("FROM python:3.11")
    (env / "environment").mkdir()
    (env / "environment" / "a.py").write_text("a")
    (env / "environment" / "b.py").write_text("b")

    source = EnvironmentSource.open(env)
    assert source.source_file_refs() == ["Dockerfile", "environment/a.py", "environment/b.py"]


# ─── Environment("name") references ────────────────────────────────────


def test_finds_positional_name_reference(tmp_path: Path) -> None:
    _write(tmp_path / "env.py", 'env = Environment("foo")\n')

    refs = EnvironmentSource.open(tmp_path).environment_name_references()

    assert len(refs) == 1
    ref = refs[0]
    assert ref.name == "foo"
    assert ref.line == 1
    assert "Environment" in ref.text


def test_finds_single_quotes_and_nested_dirs(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    _write(tmp_path / "sub" / "e.py", "e = Environment('bar')\n")

    names = {ref.name for ref in EnvironmentSource.open(tmp_path).environment_name_references()}

    assert names == {"bar"}


def test_keyword_form_is_not_matched(tmp_path: Path) -> None:
    # Environment(name="kw") is the keyword form — the scanner targets the
    # positional string form, so it should not match.
    _write(tmp_path / "env.py", 'env = Environment(name="kw")\n')

    assert EnvironmentSource.open(tmp_path).environment_name_references() == []


def test_scanner_does_not_rewrite_mismatched_name(tmp_path: Path) -> None:
    env_py = tmp_path / "env.py"
    _write(env_py, 'env = Environment("old-name")\n')

    refs = EnvironmentSource.open(tmp_path).environment_name_references()

    assert refs[0].name == "old-name"
    assert 'Environment("old-name")' in env_py.read_text(encoding="utf-8")


def test_no_references_is_a_pass(tmp_path: Path) -> None:
    _write(tmp_path / "env.py", "x = 1\n")
    assert EnvironmentSource.open(tmp_path).environment_name_references() == []


# ─── validation ────────────────────────────────────────────────────────


def test_no_pyproject_is_clean(tmp_path: Path) -> None:
    assert EnvironmentSource.open(tmp_path).validate_pyproject_references() == []


def test_missing_license_file_is_error(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nlicense = {file = "LICENSE"}\n')

    issues = EnvironmentSource.open(tmp_path).validate_pyproject_references()

    assert [i.severity for i in issues] == ["error"]
    assert "License file not found" in issues[0].message


def test_missing_readme_is_warning(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nreadme = "README.md"\n')

    issues = EnvironmentSource.open(tmp_path).validate_pyproject_references()

    assert [i.severity for i in issues] == ["warning"]
    assert "Readme file not found" in issues[0].message


def test_all_references_present_is_clean(tmp_path: Path) -> None:
    _write(
        tmp_path / "pyproject.toml",
        '[project]\nname = "x"\nlicense = {file = "LICENSE"}\nreadme = "README.md"\n',
    )
    _write(tmp_path / "LICENSE", "MIT")
    _write(tmp_path / "README.md", "# x")

    assert EnvironmentSource.open(tmp_path).validate_pyproject_references() == []


def test_unparseable_pyproject_is_error(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", "this is not = valid = toml [[[")

    issues = EnvironmentSource.open(tmp_path).validate_pyproject_references()

    assert any(i.severity == "error" and "Failed to parse" in i.message for i in issues)


def test_license_not_copied_before_install_is_error(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nlicense = {file = "LICENSE"}\n')
    _write(
        tmp_path / "Dockerfile.hud",
        "FROM python:3.11\nCOPY pyproject.toml ./\nRUN uv sync\nCOPY . .\n",
    )

    issues = EnvironmentSource.open(tmp_path).validate_dockerfile()

    assert any(i.severity == "error" and "LICENSE" in i.message for i in issues)


def test_full_copy_before_install_is_clean(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nlicense = {file = "LICENSE"}\n')
    _write(tmp_path / "Dockerfile.hud", "FROM python:3.11\nCOPY . .\nRUN uv sync\n")

    # ``COPY . .`` precedes the install, so nothing is missing.
    assert EnvironmentSource.open(tmp_path).validate_dockerfile() == []


def test_no_dockerfile_is_clean(tmp_path: Path) -> None:
    assert EnvironmentSource.open(tmp_path).validate_dockerfile() == []


def test_validate_environment_aggregates(tmp_path: Path) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "x"\nlicense = {file = "LICENSE"}\n')
    _write(
        tmp_path / "Dockerfile.hud",
        "FROM python:3.11\nCOPY pyproject.toml ./\nRUN uv sync\nCOPY . .\n",
    )

    issues = EnvironmentSource.open(tmp_path).validate()
    assert len(issues) >= 2

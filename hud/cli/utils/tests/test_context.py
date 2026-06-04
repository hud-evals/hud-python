"""``hud.cli.utils.context`` — build-context tarball + ignore-pattern matching."""

from __future__ import annotations

import tarfile
from typing import TYPE_CHECKING

from hud.cli.utils.context import (
    create_build_context_tarball,
    format_size,
    parse_ignore_file,
    should_ignore,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_ignore_file_skips_comments_and_blanks(tmp_path: Path) -> None:
    f = tmp_path / ".dockerignore"
    f.write_text("# comment\n\n*.pyc\nnode_modules/\n", encoding="utf-8")
    assert parse_ignore_file(f) == ["*.pyc", "node_modules/"]


def test_parse_ignore_file_missing(tmp_path: Path) -> None:
    assert parse_ignore_file(tmp_path / "nope") == []


def test_format_size() -> None:
    assert format_size(500) == "500.0 B"
    assert format_size(1536) == "1.5 KB"
    assert format_size(5 * 1024 * 1024) == "5.0 MB"
    assert format_size(2 * 1024**4) == "2.0 TB"


def test_should_ignore_glob_and_filename(tmp_path: Path) -> None:
    (tmp_path / "a.pyc").touch()
    (tmp_path / "a.py").touch()
    assert should_ignore(tmp_path / "a.pyc", tmp_path, ["*.pyc"]) is True
    assert should_ignore(tmp_path / "a.py", tmp_path, ["*.pyc"]) is False


def test_should_ignore_directory_pattern(tmp_path: Path) -> None:
    (tmp_path / "node_modules").mkdir()
    assert should_ignore(tmp_path / "node_modules", tmp_path, ["node_modules/"]) is True


def test_should_ignore_negation_reincludes(tmp_path: Path) -> None:
    (tmp_path / "keep.pyc").touch()
    assert should_ignore(tmp_path / "keep.pyc", tmp_path, ["*.pyc", "!keep.pyc"]) is False


def test_create_build_context_tarball_excludes_secrets(tmp_path: Path) -> None:
    ctx = tmp_path / "ctx"
    ctx.mkdir()
    (ctx / "main.py").write_text("print('hi')", encoding="utf-8")
    (ctx / ".env").write_text("SECRET=1", encoding="utf-8")
    git = ctx / ".git"
    git.mkdir()
    (git / "config").write_text("x", encoding="utf-8")

    tarball, size, count, duration = create_build_context_tarball(ctx)
    try:
        assert tarball.exists()
        assert size > 0
        assert duration >= 0
        with tarfile.open(tarball) as tar:
            names = tar.getnames()
        assert "main.py" in names
        assert ".env" not in names
        assert not any(n.startswith(".git") for n in names)
        assert count == 1
    finally:
        tarball.unlink(missing_ok=True)

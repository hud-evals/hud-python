"""``hud.cli.utils.name_check`` — scanning ``Environment("name")`` references."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.cli.utils.name_check import find_env_name_references

if TYPE_CHECKING:
    from pathlib import Path


def test_finds_positional_name_reference(tmp_path: Path) -> None:
    (tmp_path / "env.py").write_text('env = Environment("foo")\n', encoding="utf-8")

    refs = find_env_name_references(tmp_path)

    assert len(refs) == 1
    _file_path, line_no, line_text, name = refs[0]
    assert name == "foo"
    assert line_no == 1
    assert "Environment" in line_text


def test_finds_single_quotes_and_nested_dirs(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "e.py").write_text("e = Environment('bar')\n", encoding="utf-8")

    names = {name for *_rest, name in find_env_name_references(tmp_path)}

    assert names == {"bar"}


def test_keyword_form_is_not_matched(tmp_path: Path) -> None:
    # Environment(name="kw") is the keyword form — the scanner targets the
    # positional string form, so it should not match.
    (tmp_path / "env.py").write_text('env = Environment(name="kw")\n', encoding="utf-8")

    assert find_env_name_references(tmp_path) == []


def test_scanner_does_not_rewrite_mismatched_name(tmp_path: Path) -> None:
    env_py = tmp_path / "env.py"
    env_py.write_text('env = Environment("old-name")\n', encoding="utf-8")

    refs = find_env_name_references(tmp_path)

    assert refs[0][3] == "old-name"
    assert 'Environment("old-name")' in env_py.read_text(encoding="utf-8")


def test_no_references_is_a_pass(tmp_path: Path) -> None:
    (tmp_path / "env.py").write_text("x = 1\n", encoding="utf-8")
    assert find_env_name_references(tmp_path) == []

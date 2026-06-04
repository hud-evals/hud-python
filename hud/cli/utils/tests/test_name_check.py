"""``hud.cli.utils.name_check`` — scanning + fixing ``Environment("name")`` references."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hud.cli.utils.name_check import check_and_fix_env_name, find_env_name_references
from hud.utils.hud_console import HUDConsole

if TYPE_CHECKING:
    from pathlib import Path

_console = HUDConsole()


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


def test_check_passes_when_names_match(tmp_path: Path) -> None:
    (tmp_path / "env.py").write_text('env = Environment("match")\n', encoding="utf-8")

    assert check_and_fix_env_name(tmp_path, "match", _console, auto_fix=True) is True


def test_check_and_fix_rewrites_mismatched_name(tmp_path: Path) -> None:
    env_py = tmp_path / "env.py"
    env_py.write_text('env = Environment("old-name")\n', encoding="utf-8")

    result = check_and_fix_env_name(tmp_path, "new-name", _console, auto_fix=True)

    assert result is True
    assert 'Environment("new-name")' in env_py.read_text(encoding="utf-8")
    assert "old-name" not in env_py.read_text(encoding="utf-8")


def test_no_references_is_a_pass(tmp_path: Path) -> None:
    (tmp_path / "env.py").write_text("x = 1\n", encoding="utf-8")
    assert check_and_fix_env_name(tmp_path, "whatever", _console, auto_fix=True) is True

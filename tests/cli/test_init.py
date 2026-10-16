"""``hud init``: copy an example environment from this SDK checkout into a new project."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from tests.support import Hud


def tree(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


@pytest.mark.parametrize(
    ("argv", "directory", "declaration"),
    [
        (["My Cool_Env", "--preset", "blank"], "My Cool_Env", 'Environment(name="my-cool-env")'),
        (["custom", "-t", "cua"], "custom", 'Environment(name="custom")'),
    ],
)
def test_init_copies_the_example_and_names_its_environment(
    hud: Hud, argv: list[str], directory: str, declaration: str
) -> None:
    result = hud("init", *argv, "--dir", "projects", "--json")

    target = hud.cwd / "projects" / directory
    assert result.exit_code == 0, result
    assert result.json == {
        "path": f"projects/{directory}",
        "preset": result.json["preset"],
        "created": True,
    }
    assert declaration in (target / "env.py").read_text()
    assert (target / "README.md").exists()
    assert not any(
        part in {".venv", "__pycache__"} for name in tree(target) for part in name.split("/")
    )


def test_a_dry_run_plans_without_creating(hud: Hud) -> None:
    result = hud("init", "thing", "--dir", "projects", "--dry-run", "--json")

    assert result.exit_code == 0, result
    assert result.json == {
        "dry_run": True,
        "action": "init",
        "path": "projects/thing",
        "preset": "coding",
    }
    assert not (hud.cwd / "projects").exists()


@pytest.mark.parametrize(
    ("setup", "argv", "exit_code", "error", "after"),
    [
        pytest.param(
            {"taken/precious.txt": "data"},
            ["taken", "--preset", "blank"],
            1,
            "conflict",
            {"taken/precious.txt": "data"},
            id="non-empty-directory",
        ),
        pytest.param(
            {"outside.py": "original", "project/env.py": "->outside.py"},
            ["project", "--preset", "blank", "--force"],
            1,
            "failure",
            {"outside.py": "original"},
            id="symlink-inside-target",
        ),
        pytest.param(
            {"outside/": "", "project": "->outside"},
            ["project", "--preset", "blank"],
            1,
            "failure",
            {},
            id="symlinked-target",
        ),
    ],
)
def test_init_refuses_without_touching_existing_files(
    hud: Hud,
    setup: dict[str, str],
    argv: list[str],
    exit_code: int,
    error: str,
    after: dict[str, str],
) -> None:
    for name, content in setup.items():
        path = hud.cwd / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith("/"):
            path.mkdir()
        elif content.startswith("->"):
            path.symlink_to(hud.cwd / content.removeprefix("->"))
        else:
            path.write_text(content)
    before = tree(hud.cwd)

    result = hud("init", *argv, "--json")

    assert (result.exit_code, result.json["error"]) == (exit_code, error), result
    assert tree(hud.cwd) == before
    assert {name: (hud.cwd / name).read_text() for name in after} == after


def test_force_overwrites_an_existing_project(hud: Hud) -> None:
    (hud.cwd / "env").mkdir()
    (hud.cwd / "env" / "env.py").write_text("old")

    result = hud("init", "env", "--preset", "blank", "--force", "--json")

    assert result.exit_code == 0, result
    assert 'Environment(name="env")' in (hud.cwd / "env" / "env.py").read_text()

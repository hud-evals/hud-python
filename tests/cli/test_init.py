"""``hud init``: copy an example environment from this SDK checkout into a new project."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot

if TYPE_CHECKING:
    from pathlib import Path

    from tests.harness import Hud


def tree(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


@pytest.mark.parametrize(
    ("argv", "directory", "declaration"),
    [
        (["berry", "--preset", "blank"], "berry", 'Environment(name="berry")'),
        (["My Cool_Env", "--template", "blank"], "My Cool_Env", 'Environment(name="my-cool-env")'),
        (["--preset", "coding"], "coding", 'Environment(name="coding")'),
        (["my-cool-env"], "my-cool-env", 'Environment(name="my-cool-env")'),
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
    ("setup", "argv", "exit_code", "document", "after"),
    [
        pytest.param(
            {"taken/precious.txt": "data"},
            ["taken", "--preset", "blank"],
            1,
            snapshot(
                {
                    "error": "conflict",
                    "message": "taken already exists and is not empty (use --force)",
                }
            ),
            {"taken/precious.txt": "data"},
            id="non-empty-directory",
        ),
        pytest.param(
            {"outside.py": "original", "project/env.py": "->outside.py"},
            ["project", "--preset", "blank", "--force"],
            1,
            snapshot(
                {
                    "error": "failure",
                    "message": (
                        "Failed to prepare example environment 'blank': cannot copy an example "
                        "environment over symlinks in project"
                    ),
                }
            ),
            {"outside.py": "original"},
            id="symlink-inside-target",
        ),
        pytest.param(
            {"outside/": "", "project": "->outside"},
            ["project", "--preset", "blank"],
            1,
            snapshot(
                {
                    "error": "failure",
                    "message": (
                        "Failed to prepare example environment 'blank': cannot copy an example "
                        "environment over symlink project"
                    ),
                }
            ),
            {},
            id="symlinked-target",
        ),
        pytest.param(
            {},
            ["--preset", "does-not-exist"],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": (
                        "Unknown example environment 'does-not-exist'. Available: coding, cua, "
                        "argument-hints, blank"
                    ),
                }
            ),
            {},
            id="unknown-example",
        ),
        pytest.param(
            {},
            [],
            2,
            snapshot(
                {
                    "error": "usage",
                    "message": (
                        "Nothing to create. Pass a name (hud init my-env) or --template, or run in "
                        "an interactive terminal to choose an example environment."
                    ),
                }
            ),
            {},
            id="nothing-to-create",
        ),
    ],
)
def test_init_refuses_without_touching_existing_files(
    hud: Hud,
    setup: dict[str, str],
    argv: list[str],
    exit_code: int,
    document: dict[str, Any],
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

    assert result.exit_code == exit_code, result
    assert result.json == document
    assert tree(hud.cwd) == before
    assert {name: (hud.cwd / name).read_text() for name in after} == after


def test_force_overwrites_an_existing_project(hud: Hud) -> None:
    (hud.cwd / "env").mkdir()
    (hud.cwd / "env" / "env.py").write_text("old")

    result = hud("init", "env", "--preset", "blank", "--force", "--json")

    assert result.exit_code == 0, result
    assert 'Environment(name="env")' in (hud.cwd / "env" / "env.py").read_text()

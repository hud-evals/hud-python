"""``hud init``: copy an example environment from this SDK checkout into a new project."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import Hud


def tree(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def test_init_copies_the_example_and_names_its_environment(hud: Hud) -> None:
    result = hud("init", "My Cool_Env", "--preset", "blank", "--dir", "projects", "--json")

    target = hud.cwd / "projects" / "My Cool_Env"
    assert result.exit_code == 0, result
    assert 'Environment(name="my-cool-env")' in (target / "env.py").read_text()
    assert not any(
        part in {".venv", "__pycache__"} for name in tree(target) for part in name.split("/")
    )

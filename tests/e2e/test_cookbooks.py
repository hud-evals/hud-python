"""The cookbooks against this checkout's SDK: every module imports, and the local runs complete.

Cookbooks live outside the package and its tests, so an SDK change can break
them silently: a moved import path, a renamed export. Every ``hud`` name a
cookbook imports must exist; every cookbook module must import, unless a
third-party package it needs is not installed; and the cookbooks whose README
runs them with no outside account complete one capped rollout with a scripted
model. Nightly, through the ``e2e`` marker.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from tests.harness import steps

from .lifecycle import assert_clean_lifecycle, evaluated, scripted
from .projects import ROOT

if TYPE_CHECKING:
    from pathlib import Path

    from tests.harness import Hud, HudEnv, Models

COOKBOOKS = ROOT / "cookbooks"
MODULES = sorted(path for path in COOKBOOKS.rglob("*.py") if ".venv" not in path.parts)
# The README commands that need no outside account: (cookbook, task source).
LOCAL_RUNS = [("connect4-selfplay", "env.py"), ("daytona-rl", "tasks.py")]

pytestmark = pytest.mark.e2e


def module_id(path: Path) -> str:
    return str(path.relative_to(COOKBOOKS))


def cookbook_of(path: Path) -> Path:
    return COOKBOOKS / path.relative_to(COOKBOOKS).parts[0]


def search_path(path: Path) -> list[Path]:
    """Where a cookbook's scripts find its sibling modules: their directory and the cookbook's."""
    return list(dict.fromkeys([path.parent, cookbook_of(path)]))


def imports(path: Path) -> list[tuple[str, str | None]]:
    """``(module, name)`` for every import in ``path``; ``name`` is None for ``import module``."""
    found: list[tuple[str, str | None]] = []
    for node in ast.walk(ast.parse(path.read_text("utf-8"))):
        if isinstance(node, ast.Import):
            found.extend((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.extend((node.module, alias.name) for alias in node.names)
    return found


def resolves(module: str, name: str | None) -> bool:
    """``module`` imports and has ``name``, as an attribute or a submodule."""
    imported = importlib.import_module(module)
    if name is None or hasattr(imported, name):
        return True
    return (
        hasattr(imported, "__path__") and importlib.util.find_spec(f"{module}.{name}") is not None
    )


@pytest.mark.parametrize("path", MODULES, ids=module_id)
def test_every_hud_name_a_cookbook_imports_exists(path: Path) -> None:
    hud_imports = [
        (module, name) for module, name in imports(path) if module.split(".")[0] == "hud"
    ]

    missing = [f"{module}:{name}" for module, name in hud_imports if not resolves(module, name)]

    assert missing == []


@pytest.mark.parametrize("path", MODULES, ids=module_id)
def test_a_cookbook_module_imports(path: Path) -> None:
    local = {sibling.stem for directory in search_path(path) for sibling in directory.glob("*.py")}
    third_party = {
        module.split(".")[0]
        for module, _ in imports(path)
        if module.split(".")[0] not in {*sys.stdlib_module_names, "hud", "__future__", *local}
    }
    if missing := sorted(name for name in third_party if importlib.util.find_spec(name) is None):
        pytest.skip(f"needs {', '.join(missing)}")
    program = (
        f"import sys; sys.path[:0] = {[str(p) for p in search_path(path)]!r}; import {path.stem}"
    )

    imported = subprocess.run(
        [sys.executable, "-c", program],
        cwd=cookbook_of(path),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert imported.returncode == 0, imported.stderr


@pytest.mark.parametrize(("cookbook", "source"), LOCAL_RUNS, ids=[run[0] for run in LOCAL_RUNS])
def test_a_cookbook_completes_its_local_run(
    cookbook: str, source: str, models: Models, hud_env: HudEnv, hud: Hud
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.respond(scripted)

    result = hud(
        "eval", source, "claude", "--max-steps", "3", "-y", "--json", cwd=COOKBOOKS / cookbook
    )

    run = evaluated(result)
    assert run["is_error"] is False
    assert_clean_lifecycle(steps(run["trace_id"]))

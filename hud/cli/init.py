"""``hud init``: scaffold a new HUD environment package.

By default (or in a non-interactive shell) it writes a minimal local scaffold —
no network, no API key. With ``--preset`` (or via the interactive picker) it
downloads one of the starter environments from GitHub instead — the same set the
platform's *environments/new* flow offers. See :mod:`hud.cli.presets`.
"""

from __future__ import annotations

import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

import httpx
import typer

from hud.utils.hud_console import HUDConsole

from .presets import ENVIRONMENT_PRESETS, PRESETS_BY_ID, EnvironmentPreset, materialize_preset
from .templates import DOCKERFILE_HUD, ENV_PY, PYPROJECT_TOML, TASKS_PY

_LOCAL_SCAFFOLD = "__local__"


def _python_name(name: str) -> str:
    """Normalize a package name into a Python-identifier-ish env name."""
    name = name.replace("-", "_").replace(" ", "_")
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _resolve_preset(preset: str | None, hud_console: HUDConsole) -> EnvironmentPreset | None:
    """Pick the starter: an explicit ``--preset`` id, an interactive choice, or
    ``None`` for the minimal local scaffold."""
    if preset is not None:
        chosen = PRESETS_BY_ID.get(preset)
        if chosen is None:
            available = ", ".join(PRESETS_BY_ID)
            hud_console.error(f"Unknown preset {preset!r}. Available: {available}")
            raise typer.Exit(1)
        return chosen

    # No flag: pick interactively when we have a TTY, else the local scaffold.
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None

    choices: list[str | dict[str, Any]] = [
        {"name": "Minimal (local scaffold, no download)", "value": _LOCAL_SCAFFOLD},
        *({"name": f"{p.name} — {p.description}", "value": p.id} for p in ENVIRONMENT_PRESETS),
    ]
    selected = hud_console.select("Choose a starter", choices, default=0)
    return None if selected == _LOCAL_SCAFFOLD else PRESETS_BY_ID[selected]


def _write_local_scaffold(target: Path, env_name: str, hud_console: HUDConsole) -> None:
    """Write the bundled minimal env package into ``target``."""
    files = {
        "pyproject.toml": PYPROJECT_TOML.format(name=env_name.replace("_", "-")),
        "env.py": ENV_PY.format(env_name=env_name),
        "tasks.py": TASKS_PY.format(env_name=env_name),
        "Dockerfile.hud": DOCKERFILE_HUD,
    }
    target.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (target / filename).write_text(content)
        hud_console.status_item(filename, "✓")


def init_command(
    name: str = typer.Argument(..., help="Environment name (directory to create)"),
    directory: str = typer.Option(".", "--dir", "-d", help="Parent directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
    preset: str | None = typer.Option(
        None,
        "--preset",
        "-p",
        help="Starter preset to download from GitHub (e.g. blank, coding, browser, "
        "deepresearch, rubrics, remote-browser). Omit for an interactive picker; in a "
        "non-interactive shell, omitting it writes the minimal local scaffold.",
    ),
) -> None:
    """🚀 Create a new HUD environment package.

    [not dim]With no --preset, writes a minimal local scaffold (env.py, tasks.py,
    Dockerfile.hud, pyproject.toml) — or, in a TTY, lets you pick a starter. With
    --preset, downloads that starter from GitHub.

    Examples:
        hud init my-env                  # interactive picker (or local scaffold)
        hud init my-env --preset coding  # download the coding starter
        hud init my-env --dir envs       # create ./envs/my-env[/not dim]
    """
    hud_console = HUDConsole()

    target = Path(directory) / name
    if target.exists() and any(target.iterdir()) and not force:
        hud_console.error(f"{target} already exists and is not empty (use --force)")
        raise typer.Exit(1)

    chosen = _resolve_preset(preset, hud_console)

    hud_console.header(f"HUD Init: {name}")
    if chosen is not None:
        hud_console.info(f"Downloading {chosen.owner}/{chosen.repo} …")
        created = not target.exists()
        try:
            materialize_preset(chosen, target)
        except (httpx.HTTPError, tarfile.TarError, ValueError, OSError) as exc:
            # Don't leave a half-written tree behind — it would trip the
            # non-empty-directory guard on the next run. Only remove a directory
            # this run created (never a dir the user already had).
            if created and target.exists():
                shutil.rmtree(target, ignore_errors=True)
            hud_console.error(f"Failed to fetch preset {chosen.id!r}: {exc}")
            raise typer.Exit(1) from exc
        hud_console.status_item(f"{chosen.owner}/{chosen.repo}", "✓")
    else:
        _write_local_scaffold(target, _python_name(name), hud_console)

    hud_console.section_title("Next Steps")
    hud_console.info("")
    hud_console.command_example(f"cd {target}", "1. Enter the package")
    hud_console.info("")
    if chosen is not None:
        hud_console.info("2. Read the README for this starter's setup + tasks.")
        hud_console.info("")
        hud_console.command_example("hud eval tasks.py claude", "3. Run an agent over the tasks")
        hud_console.info("")
        hud_console.info("4. Deploy for scale")
        hud_console.info("   hud deploy, then run many evals in parallel.")
    else:
        hud_console.info("2. Define task definitions in env.py")
        hud_console.info("   A @env.template is an async generator: it yields a prompt, then")
        hud_console.info("   (after the agent answers) yields a reward.")
        hud_console.info("")
        hud_console.info("3. List the tasks to run in tasks.py")
        hud_console.info("   Call a task with args to bind a runnable Task.")
        hud_console.info("")
        hud_console.command_example("hud eval tasks.py claude", "4. Run an agent over them")
        hud_console.info("")
        hud_console.info("5. Deploy for scale")
        hud_console.info("   hud deploy, then run many evals in parallel.")
    hud_console.info("")
    hud_console.info("Tip: Install the HUD skill so your coding agent can help you build:")
    hud_console.command_example("npx skills add docs.hud.ai", "Install HUD skill")

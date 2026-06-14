"""``hud init``: scaffold a new HUD environment package.

Purely local — writes the v6 template files into a fresh directory. No
network, no API key, no prompts.
"""

from __future__ import annotations

from pathlib import Path

import typer

from hud.utils.hud_console import HUDConsole

from .templates import DOCKERFILE_HUD, ENV_PY, PYPROJECT_TOML, TASKS_PY


def _python_name(name: str) -> str:
    """Normalize a package name into a Python-identifier-ish env name."""
    name = name.replace("-", "_").replace(" ", "_")
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def init_command(
    name: str = typer.Argument(..., help="Environment name (directory to create)"),
    directory: str = typer.Option(".", "--dir", "-d", help="Parent directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
) -> None:
    """🚀 Create a new HUD environment package.

    [not dim]Writes env.py (tasks + capabilities), tasks.py, Dockerfile.hud, and
    pyproject.toml into a new directory.

    Examples:
        hud init my-env             # create ./my-env
        hud init my-env --dir envs  # create ./envs/my-env[/not dim]
    """
    hud_console = HUDConsole()

    target = Path(directory) / name
    if target.exists() and any(target.iterdir()) and not force:
        hud_console.error(f"{target} already exists and is not empty (use --force)")
        raise typer.Exit(1)

    env_name = _python_name(name)
    files = {
        "pyproject.toml": PYPROJECT_TOML.format(name=env_name.replace("_", "-")),
        "env.py": ENV_PY.format(env_name=env_name),
        "tasks.py": TASKS_PY.format(env_name=env_name),
        "Dockerfile.hud": DOCKERFILE_HUD,
    }

    hud_console.header(f"HUD Init: {env_name}")
    target.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (target / filename).write_text(content)
        hud_console.status_item(filename, "✓")

    hud_console.section_title("Next Steps")
    hud_console.info("")
    hud_console.command_example(f"cd {target}", "1. Enter the package")
    hud_console.info("")
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

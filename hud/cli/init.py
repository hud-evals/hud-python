"""``hud init``: scaffold a new HUD environment package.

With no ``NAME`` it shows an interactive picker of the starter environments and
clones the chosen one from GitHub into ``./<repo>`` — the same set the platform's
*environments/new* flow offers (see :mod:`hud.cli.presets`). With a ``NAME`` it
scaffolds into ``./NAME``. The ``blank`` starter is special: it writes the bundled
minimal scaffold (no network, no API key) — a tiny letter-counting task — rather
than downloading anything. That same scaffold is the default in a non-interactive
shell when a ``NAME`` is given. ``--preset`` skips the picker.
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


def _python_name(name: str) -> str:
    """Normalize a package name into a Python-identifier-ish env name."""
    name = name.replace("-", "_").replace(" ", "_")
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _resolve_preset(preset: str | None, hud_console: HUDConsole) -> EnvironmentPreset | None:
    """Pick the starter: an explicit ``--preset`` id, an interactive choice, or
    ``None`` when there's no TTY to prompt.

    The ``blank`` starter (``local=True``) writes the bundled minimal scaffold; the
    rest are downloaded from GitHub, with the repo name becoming the target
    directory when no ``NAME`` is given.
    """
    if preset is not None:
        chosen = PRESETS_BY_ID.get(preset)
        if chosen is None:
            available = ", ".join(PRESETS_BY_ID)
            hud_console.error(f"Unknown preset {preset!r}. Available: {available}")
            raise typer.Exit(1)
        return chosen

    # No flag: pick interactively when we have a TTY, else fall back to the caller's
    # local-scaffold default (only reachable when a NAME was passed).
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None

    choices: list[str | dict[str, Any]] = [
        {"name": f"{p.emoji}  {p.name} — {p.description}", "value": p.id}
        for p in ENVIRONMENT_PRESETS
    ]
    selected = hud_console.select("Choose a template", choices, default=0, spaced=True)
    return PRESETS_BY_ID[selected]


def _ensure_writable(target: Path, force: bool, hud_console: HUDConsole) -> None:
    """Refuse to scaffold into a non-empty directory unless ``--force``."""
    if target.exists() and any(target.iterdir()) and not force:
        hud_console.error(f"{target} already exists and is not empty (use --force)")
        raise typer.Exit(1)


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
    name: str | None = typer.Argument(
        None,
        help="Environment name (directory to create). Omit to pick a template "
        "and clone it into the current directory.",
    ),
    directory: str = typer.Option(".", "--dir", "-d", help="Parent directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
    preset: str | None = typer.Option(
        None,
        "--preset",
        "-p",
        help="Template to use (e.g. blank, browser, cua, deepresearch, coding, ml, "
        "verilog). 'blank' writes a minimal local scaffold; the rest download from "
        "GitHub. Omit for the interactive picker; with a NAME in a non-interactive "
        "shell, omitting it writes the blank local scaffold.",
    ),
) -> None:
    """🚀 Create a new HUD environment package.

    [not dim]With no NAME, pick a template and put it in the current directory
    (as ./<template>). With a NAME, scaffold into ./NAME. The 'blank' template
    writes a minimal local scaffold (env.py, tasks.py, Dockerfile.hud,
    pyproject.toml); every other template downloads from GitHub. Pass --preset to
    skip the picker.

    Examples:
        hud init                          # pick a template → ./<template>
        hud init my-env                   # pick a template → ./my-env
        hud init my-env --preset blank    # minimal local scaffold → ./my-env
        hud init my-env --preset browser  # clone the browser template → ./my-env
        hud init --preset cua             # clone the cua template → ./cua-template[/not dim]
    """
    hud_console = HUDConsole()

    # Fail fast if an explicitly named target is occupied, before any prompt/download.
    explicit_target = Path(directory) / name if name is not None else None
    if explicit_target is not None:
        _ensure_writable(explicit_target, force, hud_console)

    chosen = _resolve_preset(preset, hud_console)

    # ``blank`` (and the non-interactive no-preset default) writes the bundled
    # local scaffold; every other preset is downloaded from GitHub.
    is_download = chosen is not None and not chosen.local

    if explicit_target is not None:
        target = explicit_target
    elif chosen is not None:
        # No name: create ./<repo> (clone target for downloads, or the blank dir).
        target = Path(directory) / chosen.repo
        _ensure_writable(target, force, hud_console)
    else:
        hud_console.error(
            "Nothing to create. Pass a name (hud init my-env), a --preset, "
            "or run in an interactive terminal to pick a template."
        )
        raise typer.Exit(1)

    hud_console.header(f"HUD Init: {target.name}")
    if is_download and chosen is not None:
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
        _write_local_scaffold(target, _python_name(target.name), hud_console)

    hud_console.section_title("Next Steps")
    hud_console.info("")
    hud_console.command_example(f"cd {target}", "1. Enter the package")
    hud_console.info("")
    if is_download:
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

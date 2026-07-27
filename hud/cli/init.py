"""``hud init``: scaffold a new HUD environment package.

With no ``NAME`` it shows an interactive picker of the starter environments and puts
the chosen one in ``./<repo>`` — the same set the platform's *environments/new* flow
offers (see :mod:`hud.cli.presets`). With a ``NAME`` it scaffolds into ``./NAME``.
Vendored starters (``blank``, ``coding``, ``cua``) are copied out of the installed
package and need no network; the rest are downloaded from GitHub. The ``blank``
starter is the default in a non-interactive shell when a ``NAME`` is given.
``--preset`` skips the picker.
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

_BLANK_PRESET = PRESETS_BY_ID["blank"]


def _resolve_preset(preset: str | None, hud_console: HUDConsole) -> EnvironmentPreset | None:
    """Pick the starter: an explicit ``--preset`` id, an interactive choice, or
    ``None`` when there's no TTY to prompt.

    The repo name becomes the target directory when no ``NAME`` is given.
    """
    if preset is not None:
        chosen = PRESETS_BY_ID.get(preset)
        if chosen is None:
            available = ", ".join(PRESETS_BY_ID)
            hud_console.error(f"Unknown preset {preset!r}. Available: {available}")
            raise typer.Exit(1)
        return chosen

    # No flag: pick interactively when we have a TTY, else fall back to the caller's
    # blank-starter default (only reachable when a NAME was passed).
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
        "verilog). 'blank', 'coding' and 'cua' are bundled with the SDK; the rest "
        "download from GitHub. Omit for the interactive picker; with a NAME in a "
        "non-interactive shell, omitting it writes the blank starter.",
    ),
) -> None:
    """🚀 Create a new HUD environment package.

    [not dim]With no NAME, pick a template and put it in the current directory
    (as ./<template>). With a NAME, scaffold into ./NAME. The 'blank', 'coding'
    and 'cua' templates are bundled with the SDK, so they always match the
    installed version; every other template downloads from GitHub. Pass --preset
    to skip the picker.

    Examples:
        hud init                          # pick a template → ./<template>
        hud init my-env                   # pick a template → ./my-env
        hud init my-env --preset blank    # minimal bundled scaffold → ./my-env
        hud init my-env --preset browser  # clone the browser template → ./my-env
        hud init --preset cua             # the bundled cua template → ./cua-template[/not dim]
    """
    hud_console = HUDConsole()

    # Fail fast if an explicitly named target is occupied, before any prompt/download.
    explicit_target = Path(directory) / name if name is not None else None
    if explicit_target is not None:
        _ensure_writable(explicit_target, force, hud_console)

    chosen = _resolve_preset(preset, hud_console)

    if explicit_target is not None:
        target = explicit_target
        # A NAME without a preset in a non-interactive shell: fall back to blank.
        chosen = chosen or _BLANK_PRESET
    elif chosen is not None:
        target = Path(directory) / chosen.repo
        _ensure_writable(target, force, hud_console)
    else:
        hud_console.error(
            "Nothing to create. Pass a name (hud init my-env), a --preset, "
            "or run in an interactive terminal to pick a template."
        )
        raise typer.Exit(1)

    hud_console.header(f"HUD Init: {target.name}")
    if not chosen.vendored:
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
        hud_console.error(f"Failed to create preset {chosen.id!r}: {exc}")
        raise typer.Exit(1) from exc
    for entry in sorted(p.name for p in target.iterdir()):
        hud_console.status_item(entry, "✓")

    has_readme = (target / "README.md").exists()
    hud_console.section_title("Next Steps")
    hud_console.info("")
    hud_console.command_example(f"cd {target}", "1. Enter the package")
    hud_console.info("")
    if has_readme:
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

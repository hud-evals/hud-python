"""``hud init``: start a project from a HUD environment."""

from __future__ import annotations

import shutil
import sys
import tarfile
from pathlib import Path
from typing import Any

import httpx
import typer

from hud.utils.hud_console import HUDConsole
from hud.utils.naming import normalize_environment_name

from .presets import (
    DEFAULT_PRESET,
    ENVIRONMENT_PRESETS,
    PRESETS_BY_ID,
    EnvironmentPreset,
    materialize_preset,
)


def _resolve_preset(preset: str | None, hud_console: HUDConsole) -> EnvironmentPreset | None:
    """Resolve an explicit example environment or ask interactively when possible."""
    if preset is not None:
        chosen = PRESETS_BY_ID.get(preset)
        if chosen is None:
            available = ", ".join(PRESETS_BY_ID)
            hud_console.error(f"Unknown example environment {preset!r}. Available: {available}")
            raise typer.Exit(1)
        return chosen

    # A named non-interactive run uses the default environment in init_command.
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None

    choices: list[str | dict[str, Any]] = [
        {"name": f"{p.name} — {p.description}", "value": p.id} for p in ENVIRONMENT_PRESETS
    ]
    selected = hud_console.select("Choose an example environment", choices, default=0, spaced=True)
    return PRESETS_BY_ID[selected]


def _ensure_writable(target: Path, force: bool, hud_console: HUDConsole) -> None:
    """Refuse to scaffold into a non-empty directory unless ``--force``."""
    if target.exists() and any(target.iterdir()) and not force:
        hud_console.error(f"{target} already exists and is not empty (use --force)")
        raise typer.Exit(1)


def init_command(
    name: str | None = typer.Argument(
        None,
        help="Environment name (directory to create). Omit to choose an example interactively.",
    ),
    directory: str = typer.Option(".", "--dir", "-d", help="Parent directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
    preset: str | None = typer.Option(
        None,
        "--template",
        "--preset",
        "-t",
        "-p",
        help="Example environment to use. Omit to choose interactively; non-interactive runs with "
        "a NAME use coding.",
    ),
) -> None:
    """Create a new HUD environment package.

    [not dim]Choose an example environment and copy it into ./NAME. Examples come from the
    matching HUD SDK source. Pass --template to skip the picker.

    Examples:
        hud init                              # choose an example interactively
        hud init my-env                       # coding → ./my-env
        hud init my-env --template cua        # computer use → ./my-env
        hud init my-env --template blank      # minimal scaffold → ./my-env[/not dim]
    """
    hud_console = HUDConsole()

    # Fail fast if an explicitly named target is occupied, before any prompt/download.
    explicit_target = Path(directory) / name if name is not None else None
    if explicit_target is not None:
        _ensure_writable(explicit_target, force, hud_console)

    chosen = _resolve_preset(preset, hud_console)
    if chosen is None and explicit_target is None:
        hud_console.error(
            "Nothing to create. Pass a name (hud init my-env), a --template, "
            "or run in an interactive terminal to choose an example environment."
        )
        raise typer.Exit(1)
    chosen = chosen or DEFAULT_PRESET

    target = explicit_target or Path(directory) / chosen.id
    if explicit_target is None:
        _ensure_writable(target, force, hud_console)

    hud_console.header(f"HUD Init: {target.name}")
    hud_console.info(f"Preparing the {chosen.name} example from the HUD SDK …")
    created = not target.exists()
    try:
        materialize_preset(chosen, target)
        source_name = normalize_environment_name(chosen.id)
        target_name = normalize_environment_name(target.name)
        if source_name != target_name:
            env_path = target / "env.py"
            contents = env_path.read_text(encoding="utf-8")
            declaration = f'Environment(name="{source_name}")'
            if contents.count(declaration) != 1:
                raise ValueError(f"expected one {declaration} declaration in {env_path}")
            env_path.write_text(
                contents.replace(declaration, f'Environment(name="{target_name}")'),
                encoding="utf-8",
            )
    except (httpx.HTTPError, tarfile.TarError, ValueError, OSError) as exc:
        # Don't leave a half-written tree behind — it would trip the
        # non-empty-directory guard on the next run. Only remove a directory
        # this run created (never a dir the user already had).
        if created and target.exists():
            shutil.rmtree(target, ignore_errors=True)
        hud_console.error(f"Failed to prepare example environment {chosen.id!r}: {exc}")
        raise typer.Exit(1) from exc
    hud_console.status_item(f"environments/{chosen.id}", "✓")

    hud_console.section_title("Next Steps")
    hud_console.info("")
    hud_console.command_example(f"cd {target}", "1. Enter the package")
    hud_console.info("")
    hud_console.info("2. Read the README for this environment's setup + tasks.")
    hud_console.info("")
    hud_console.command_example("hud eval tasks.py claude", "3. Run an agent over the tasks")
    hud_console.info("")
    hud_console.info("4. Deploy for scale")
    hud_console.info("   hud deploy, then run many evals in parallel.")
    hud_console.info("")
    hud_console.info("Tip: Install the HUD skill so your coding agent can help you build:")
    hud_console.command_example("npx skills add docs.hud.ai", "Install HUD skill")

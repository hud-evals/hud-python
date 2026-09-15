"""``hud init``: start a project from a HUD environment."""

from __future__ import annotations

import io
import os
import shutil
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
import typer
from packaging.version import Version

from hud.cli.app import CliError
from hud.utils.hud_console import HUDConsole
from hud.utils.naming import normalize_environment_name
from hud.version import __version__


@dataclass(frozen=True, slots=True)
class EnvironmentPreset:
    name: str
    description: str


ENVIRONMENT_PRESETS: dict[str, EnvironmentPreset] = {
    "coding": EnvironmentPreset(
        "Coding",
        "A repository workspace with a SWE-bench task and hidden-test grading.",
    ),
    "cua": EnvironmentPreset(
        "Computer Use",
        "A virtual Linux desktop with deterministic and model-judged grading.",
    ),
    "argument-hints": EnvironmentPreset(
        "Argument Hints",
        "A prompt, data-file attachments, and rubric grading via console form hints.",
    ),
    "blank": EnvironmentPreset(
        "Blank",
        "A minimal letter-counting task for building an environment from scratch.",
    ),
}
DEFAULT_PRESET_ID = "coding"


def materialize_preset(
    preset_id: str,
    target: Path,
) -> None:
    """Copy an example environment from this checkout or the installed SDK's release tag."""
    if target.is_symlink():
        raise ValueError(f"cannot copy an example environment over symlink {target}")

    repository = Path(__file__).resolve().parents[2]
    local_source = repository / "environments" / preset_id
    if local_source.is_dir():
        if any(path.is_symlink() for path in target.rglob("*")):
            raise ValueError(f"cannot copy an example environment over symlinks in {target}")
        shutil.copytree(
            local_source,
            target,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".venv",
                ".pytest_cache",
                ".ruff_cache",
                "__pycache__",
                "*.pyc",
                "*.pyo",
            ),
        )
        return

    parsed_version = Version(__version__)
    if parsed_version.is_devrelease:
        raise ValueError(
            f"HUD SDK development version {__version__!r} has no matching example archive; "
            "run hud init from a source checkout"
        )

    headers = {}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    ref = f"v{parsed_version.public}"
    url = f"https://codeload.github.com/hud-evals/hud-python/tar.gz/refs/tags/{ref}"
    response = httpx.get(url, headers=headers, follow_redirects=True, timeout=60.0)
    response.raise_for_status()

    target.mkdir(parents=True, exist_ok=True)
    target_root = target.resolve()
    source_parts = ("environments", preset_id)

    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
        for member in archive.getmembers():
            archive_parts = PurePosixPath(member.name).parts[1:]
            if archive_parts[: len(source_parts)] != source_parts:
                continue
            relative_parts = archive_parts[len(source_parts) :]
            if not relative_parts:
                continue

            destination = (target_root / Path(*relative_parts)).resolve()
            try:
                destination.relative_to(target_root)
            except ValueError as exc:
                raise ValueError(f"unsafe path in SDK archive: {member.name!r}") from exc

            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                destination.parent.mkdir(parents=True, exist_ok=True)
                source_file = archive.extractfile(member)
                if source_file is not None:
                    destination.write_bytes(source_file.read())
                    if member.mode & 0o111:
                        destination.chmod(destination.stat().st_mode | (member.mode & 0o111))


def _resolve_preset(
    preset: str | None,
    name: str | None,
    hud_console: HUDConsole,
) -> str:
    """Resolve an explicit example environment or ask interactively when possible."""
    if preset is not None:
        if preset not in ENVIRONMENT_PRESETS:
            available = ", ".join(ENVIRONMENT_PRESETS)
            raise CliError(
                error="usage",
                message=f"Unknown example environment {preset!r}. Available: {available}",
            )
        return preset

    if sys.stdin.isatty() and sys.stdout.isatty():
        choices: list[str | dict[str, Any]] = [
            {"name": f"{example.name} — {example.description}", "value": example_id}
            for example_id, example in ENVIRONMENT_PRESETS.items()
        ]
        return hud_console.select("Choose an example environment", choices, default=0, spaced=True)

    if name is not None:
        return DEFAULT_PRESET_ID
    raise CliError(
        error="usage",
        message="Nothing to create. Pass a name (hud init my-env), a --template, "
        "        or run in an interactive terminal to choose an example environment.",
    )


def init_command(
    name: str | None = typer.Argument(
        None,
        help="Environment name (directory to create). Omit to choose an example interactively.",
    ),
    directory: str = typer.Option(".", "--dir", "-d", help="Parent directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the planned action without making changes."
    ),
    preset: str | None = typer.Option(
        None,
        "--template",
        "--preset",
        "-t",
        "-p",
        help="Example environment to use. Omit to choose interactively; non-interactive runs with "
        "a NAME use coding.",
    ),
) -> Any:
    """Create a new HUD environment package.

    [not dim]Choose an example environment and copy it into ./NAME. Examples come from the
    matching HUD SDK source. Pass --template to skip the picker.

    Examples:
        hud init                              # choose an example interactively
        hud init my-env                       # choose an example → ./my-env
        hud init my-env --template cua        # computer use → ./my-env
        hud init my-env --template blank      # minimal scaffold → ./my-env[/not dim]
    """
    hud_console = HUDConsole()

    if dry_run is True and preset is None:
        if name is None:
            raise CliError(
                error="usage",
                message="Pass --template for a dry run without a name.",
            )
        preset = DEFAULT_PRESET_ID
    preset_id = _resolve_preset(preset, name, hud_console)
    chosen = ENVIRONMENT_PRESETS[preset_id]
    target = Path(directory) / (name if name is not None else preset_id)
    if target.exists() and any(target.iterdir()) and not force:
        raise CliError(
            error="conflict", message=f"{target} already exists and is not empty (use --force)"
        )

    if dry_run is True:
        plan = {"dry_run": True, "action": "init", "path": str(target), "preset": preset_id}
        hud_console.info(f"--dry-run: would create {plan['path']}")
        return plan

    hud_console.header(f"HUD Init: {target.name}")
    hud_console.info(f"Preparing the {chosen.name} example from the HUD SDK …")
    created = not target.exists()
    try:
        materialize_preset(preset_id, target)
        source_name = normalize_environment_name(preset_id)
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
        raise CliError(
            error="failure",
            message=f"Failed to prepare example environment {preset_id!r}: {exc}",
        ) from exc
    hud_console.status_item(f"environments/{preset_id}", "✓")

    saved = {"path": str(target), "preset": preset_id, "created": True}

    hud_console.section_title("Next Steps")
    hud_console.info("")
    hud_console.command_example(f"cd {saved['path']}", "1. Enter the package")
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
    return saved

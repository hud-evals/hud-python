"""``hud init``: start a project from an example HUD environment."""

from __future__ import annotations

import io
import os
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import httpx
import typer
from packaging.version import Version

from hud.cli import CliError
from hud.utils.hud_console import HUDConsole
from hud.utils.naming import normalize_environment_name
from hud.version import __version__

#: ``environments/<id>`` in the SDK repository, with the picker's description.
EXAMPLES: dict[str, str] = {
    "coding": "Coding — A repository workspace with a SWE-bench task and hidden-test grading.",
    "cua": "Computer Use — A virtual Linux desktop with deterministic and model-judged grading.",
    "argument-hints": (
        "Argument Hints — A prompt, data-file attachments, and rubric grading via console "
        "form hints."
    ),
    "blank": "Blank — A minimal letter-counting task for building an environment from scratch.",
}


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

    if preset is None:
        if not dry_run and sys.stdin.isatty() and sys.stdout.isatty():
            preset = hud_console.select(
                "Choose an example environment",
                [{"name": label, "value": example} for example, label in EXAMPLES.items()],
                default=0,
                spaced=True,
            )
        elif name is not None:
            preset = "coding"
        else:
            raise CliError(
                error="usage",
                message="Nothing to create. Pass a name (hud init my-env) or --template, or "
                "run in an interactive terminal to choose an example environment.",
            )
    if preset not in EXAMPLES:
        raise CliError(
            error="usage",
            message=f"Unknown example environment {preset!r}. Available: {', '.join(EXAMPLES)}",
        )

    target = Path(directory) / (name if name is not None else preset)
    if target.exists() and any(target.iterdir()) and not force:
        raise CliError(
            error="conflict", message=f"{target} already exists and is not empty (use --force)"
        )
    if dry_run:
        hud_console.info(f"--dry-run: would create {target}")
        return {"dry_run": True, "action": "init", "path": str(target), "preset": preset}

    hud_console.header(f"HUD Init: {target.name}")
    hud_console.info(f"Preparing the {preset} example from the HUD SDK …")
    created = not target.exists()
    try:
        if target.is_symlink():
            raise ValueError(f"cannot copy an example environment over symlink {target}")
        local_source = Path(__file__).resolve().parents[2] / "environments" / preset
        if local_source.is_dir():
            # A source checkout: copy the example straight from the repository.
            if any(path.is_symlink() for path in target.rglob("*")):
                raise ValueError(f"cannot copy an example environment over symlinks in {target}")
            shutil.copytree(
                local_source,
                target,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns(
                    ".venv", ".pytest_cache", ".ruff_cache", "__pycache__", "*.pyc", "*.pyo"
                ),
            )
        else:
            # An installed SDK: fetch the example from this version's release tag.
            version = Version(__version__)
            if version.is_devrelease:
                raise ValueError(
                    f"HUD SDK development version {__version__!r} has no matching example "
                    "archive; run hud init from a source checkout"
                )
            headers = {}
            if token := os.environ.get("GITHUB_TOKEN"):
                headers["Authorization"] = f"Bearer {token}"
            response = httpx.get(
                f"https://codeload.github.com/hud-evals/hud-python/tar.gz/refs/tags/v{version.public}",
                headers=headers,
                follow_redirects=True,
                timeout=60.0,
            )
            response.raise_for_status()
            target.mkdir(parents=True, exist_ok=True)
            target_root = target.resolve()
            source_parts = ("environments", preset)
            with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
                for member in archive.getmembers():
                    parts = PurePosixPath(member.name).parts[1:]
                    if parts[: len(source_parts)] != source_parts or len(parts) == len(
                        source_parts
                    ):
                        continue
                    destination = (target_root / Path(*parts[len(source_parts) :])).resolve()
                    if not destination.is_relative_to(target_root):
                        raise ValueError(f"unsafe path in SDK archive: {member.name!r}")
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        source_file = archive.extractfile(member)
                        assert source_file is not None
                        destination.write_bytes(source_file.read())
                        if member.mode & 0o111:
                            destination.chmod(destination.stat().st_mode | (member.mode & 0o111))

        source_name = normalize_environment_name(preset)
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
            message=f"Failed to prepare example environment {preset!r}: {exc}",
        ) from exc
    hud_console.status_item(f"environments/{preset}", "✓")

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
    return {"path": str(target), "preset": preset, "created": True}

"""Pluggable format conversion system for HUD.

Converts external benchmark formats (Harbor, Inspect AI, etc.) into
HUD environments + tasksets.

Usage:
    hud convert <path>                    # Auto-detect format
    hud convert <path> --from harbor      # Explicit format
    hud convert <path> --output ./out     # Custom output directory
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import typer

from hud.utils.hud_console import HUDConsole

from .base import BaseConverter, ConvertResult, GeneratedEnvironment

__all__ = [
    "BaseConverter",
    "ConvertResult",
    "GeneratedEnvironment",
    "detect_format",
    "get_converter",
    "list_formats",
    "write_result",
]

LOGGER = logging.getLogger(__name__)

# Shell script extensions that need CRLF -> LF normalization
_SHELL_EXTENSIONS = frozenset({".sh", ".bash", ".zsh", ".ksh"})


def _normalize_line_endings(directory: Path) -> None:
    """Convert CRLF to LF in all shell scripts under a directory.

    Git on Windows with autocrlf=true converts LF to CRLF on checkout.
    Shell scripts with CRLF break on Linux (e.g., shebang errors,
    'set: pipefail\\r: invalid option name').
    """
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix in _SHELL_EXTENSIONS:
            raw = path.read_bytes()
            if b"\r" in raw:
                path.write_bytes(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n"))
                LOGGER.debug("Normalized line endings: %s", path)


# ---------------------------------------------------------------------------
# Converter registry
# ---------------------------------------------------------------------------

# Lazy-loaded to avoid import cost on unrelated CLI commands
_converters: list[BaseConverter] | None = None


def _load_converters() -> list[BaseConverter]:
    global _converters
    if _converters is None:
        from .harbor import HarborConverter

        _converters = [
            HarborConverter(),
            # Future: InspectConverter(), METRConverter(), ...
        ]
    return _converters


def get_converter(name: str) -> BaseConverter | None:
    """Get a converter by its short name (e.g., 'harbor')."""
    for c in _load_converters():
        if c.name == name:
            return c
    return None


def detect_format(path: Path) -> BaseConverter | None:
    """Auto-detect which converter can handle the given path."""
    for c in _load_converters():
        if c.detect(path):
            return c
    return None


def list_formats() -> list[tuple[str, str]]:
    """Return (name, description) pairs for all registered converters."""
    return [(c.name, c.description) for c in _load_converters()]


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def write_result(result: ConvertResult, output_dir: Path) -> Path:
    """Write conversion results to disk.

    Creates the output directory structure:
        output_dir/
        ├── env-name-a/
        │   ├── env.py
        │   ├── Dockerfile.hud
        │   ├── pyproject.toml
        │   └── tasks/
        │       └── <task_id>/  (copied from source, minus environment/ & solution/)
        └── taskset.json

    Returns the path to the generated taskset.json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for env_gen in result.environments:
        env_dir = output_dir / env_gen.name
        env_dir.mkdir(parents=True, exist_ok=True)

        # Write generated files
        (env_dir / "env.py").write_text(env_gen.env_py, encoding="utf-8")
        (env_dir / "Dockerfile.hud").write_text(env_gen.dockerfile, encoding="utf-8")
        (env_dir / "pyproject.toml").write_text(env_gen.pyproject_toml, encoding="utf-8")

        # Copy build context files from source environment/ directory
        # (e.g., warriors/*.red that Harbor Dockerfiles reference via COPY)
        if env_gen.build_context_source and env_gen.build_context_source.is_dir():
            for item in env_gen.build_context_source.iterdir():
                # Skip the Dockerfile itself (we already generated Dockerfile.hud)
                if item.name.lower() in ("dockerfile", "dockerfile.hud"):
                    continue
                dest_item = env_dir / item.name
                if dest_item.exists():
                    if dest_item.is_dir():
                        shutil.rmtree(dest_item)
                    else:
                        dest_item.unlink()
                if item.is_dir():
                    shutil.copytree(item, dest_item)
                else:
                    shutil.copy2(item, dest_item)

        # Copy task data directories (skip environment/ and solution/)
        tasks_dir = env_dir / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)

        for task_id, source_dir in env_gen.task_dirs.items():
            dest = tasks_dir / task_id
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True, exist_ok=True)

            for item in source_dir.iterdir():
                # Skip dirs that are handled by the Dockerfile or ignored
                if item.name in ("environment", "solution"):
                    continue
                if item.is_dir():
                    shutil.copytree(item, dest / item.name)
                else:
                    shutil.copy2(item, dest / item.name)

        # Normalize CRLF -> LF in all shell scripts (fixes Windows git checkout)
        _normalize_line_endings(env_dir)

        LOGGER.info(
            "Wrote environment '%s' with %d task(s)",
            env_gen.name,
            len(env_gen.task_dirs),
        )

    # Write taskset
    taskset_path = output_dir / "taskset.json"
    with open(taskset_path, "w", encoding="utf-8") as f:
        json.dump(result.taskset, f, ensure_ascii=False, indent=2)
        f.write("\n")

    LOGGER.info("Wrote taskset with %d task(s) to %s", len(result.taskset), taskset_path)
    return taskset_path


def convert_command(
    path: str = typer.Argument(
        ..., help="Path to source tasks/dataset directory to convert to HUD format"
    ),
    from_format: str = typer.Option(
        "auto",
        "--from",
        "-f",
        help="Source format (auto, harbor). Use 'auto' to detect automatically.",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (default: ./hud_converted)",
    ),
) -> None:
    """Convert external benchmark formats to HUD environments + tasksets.

    [not dim]Converts tasks from frameworks like Harbor into HUD-compatible
    environments (env.py + Dockerfile.hud) and v5 taskset files.

    Supports pluggable formats. Currently: harbor.

    Examples:
        hud convert ./algotune/                  # Auto-detect, convert dataset
        hud convert ./my-task/ --from harbor      # Explicit format
        hud convert ./dataset/ --output ./out     # Custom output directory[/not dim]
    """
    hud_console = HUDConsole()
    source_path = Path(path).resolve()

    if not source_path.exists():
        hud_console.error(f"Path does not exist: {path}")
        raise typer.Exit(1)

    if from_format == "auto":
        converter = detect_format(source_path)
        if converter is None:
            available = list_formats()
            if not available:
                hud_console.error("No converters registered.")
                raise typer.Exit(1)

            if len(available) == 1:
                converter = get_converter(available[0][0])
                if converter:
                    hud_console.info(f"Using format: {converter.name}")
            else:
                import questionary

                choices = [
                    questionary.Choice(title=f"{name} — {desc}", value=name)
                    for name, desc in available
                ]
                picked = questionary.select(
                    "Could not auto-detect format. Which format is this?",
                    choices=choices,
                ).ask()
                if not picked:
                    raise typer.Exit(1)
                converter = get_converter(picked)

            if converter is None:
                hud_console.error("No converter selected.")
                raise typer.Exit(1)
        else:
            hud_console.info(f"Detected format: {converter.name}")
    else:
        converter = get_converter(from_format)
        if converter is None:
            hud_console.error(f"Unknown format: {from_format}")
            available = list_formats()
            if available:
                hud_console.info("Available formats:")
                for name, desc in available:
                    hud_console.info(f"  {name}: {desc}")
            raise typer.Exit(1)

    try:
        result = converter.convert(source_path)
    except ValueError as e:
        hud_console.error(str(e))
        raise typer.Exit(1) from e
    except Exception as e:
        hud_console.error(f"Conversion failed: {e}")
        raise typer.Exit(1) from e

    output_dir = Path(output) if output else Path("./hud_converted")
    try:
        taskset_path = write_result(result, output_dir.resolve())
    except Exception as e:
        hud_console.error(f"Failed to write output: {e}")
        raise typer.Exit(1) from e

    hud_console.header("Convert Complete")
    hud_console.info("")

    total_tasks = len(result.taskset)
    total_envs = len(result.environments)
    hud_console.success(f"Converted {total_tasks} task(s) into {total_envs} environment(s).")
    hud_console.info("")

    hud_console.section_title("Environments")
    for env_gen in result.environments:
        task_count = len(env_gen.task_dirs)
        hud_console.status_item(env_gen.name, f"{task_count} tasks")
    hud_console.info("")

    hud_console.section_title("Output")
    hud_console.status_item("Directory", str(output_dir.resolve()))
    hud_console.status_item("Taskset", str(taskset_path))
    hud_console.info("")

    hud_console.section_title("Next Steps")
    hud_console.info("")

    hud_console.info("1. Deploy environment(s):")
    if total_envs > 1:
        hud_console.command_example(
            f"hud deploy {output_dir.resolve()} --all",
            f"Deploy all {total_envs} environments",
        )
    else:
        first_env = result.environments[0].name if result.environments else "<env>"
        hud_console.command_example(
            f"hud deploy {output_dir.resolve() / first_env}",
            "Build & deploy to HUD platform",
        )
    hud_console.info("")

    hud_console.info("2. Run evaluation:")
    hud_console.command_example(f"hud eval {taskset_path}", "Run agent against tasks")
    hud_console.info("")

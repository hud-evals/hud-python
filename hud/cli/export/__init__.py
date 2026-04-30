"""HUD → external format exporter.

Public surface:
  * ``export_app`` — Typer subgroup; one subcommand per registered exporter.
  * ``export()`` — programmatic entry used by the platform pipeline.
  * ``write_result()`` — writes an ExportResult to a directory on disk.
  * ``BaseExporter`` / ``ExportInput`` / ``ExportResult`` — data model.

Add a new format by subclassing ``BaseExporter`` and registering it in
``_load_exporters()``. The Typer subcommand is generated automatically.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from hud.utils.hud_console import HUDConsole

from .base import BaseExporter, ExportInput, ExportResult
from .harbor import HarborExporter
from .render import render_prompts_via_image, render_prompts_via_url

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "BaseExporter",
    "ExportInput",
    "ExportResult",
    "export",
    "export_app",
    "get_exporter",
    "list_formats",
    "render_prompts_via_image",
    "render_prompts_via_url",
    "write_result",
]

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_exporters: list[BaseExporter] | None = None


def _load_exporters() -> list[BaseExporter]:
    global _exporters
    if _exporters is None:
        _exporters = [HarborExporter()]
    return _exporters


def get_exporter(name: str) -> BaseExporter | None:
    for e in _load_exporters():
        if e.name == name:
            return e
    return None


def list_formats() -> list[tuple[str, str]]:
    return [(e.name, e.description) for e in _load_exporters()]


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def write_result(result: ExportResult, output_dir: Path) -> Path:
    """Materialize an ExportResult to disk. Returns the output dir."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for rel_path, content in result.files.items():
        target = output_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")
        # Mark shell scripts executable so `./sample-run.sh` works.
        if target.suffix in {".sh", ".bash"}:
            target.chmod(0o755)
    return output_dir


# ---------------------------------------------------------------------------
# Programmatic entry
# ---------------------------------------------------------------------------


def export(format_name: str, inp: ExportInput) -> ExportResult:
    """Run an exporter by name. Used by the platform pipeline."""
    exporter = get_exporter(format_name)
    if exporter is None:
        available = ", ".join(name for name, _ in list_formats())
        raise ValueError(f"Unknown export format: {format_name}. Available: {available}")
    return exporter.export(inp)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


export_app = typer.Typer(
    help="Export HUD taskset+envs to external formats (Harbor)",
    rich_markup_mode="rich",
    no_args_is_help=True,
)


def _resolve_repo_inputs(
    path: Path,
    image_override: str | None,
    tasks_override: str | None,
) -> tuple[list, str, list[str], str | None]:
    """Load tasks and image ref from a local repo.

    Returns (tasks, image_ref, required_env, platform).
    """
    import sys

    from hud.cli.utils.collect import collect_tasks
    from hud.cli.utils.environment import find_dockerfile
    from hud.cli.utils.lockfile import find_lock, get_local_image, load_lock

    if not path.exists() or not path.is_dir():
        raise typer.BadParameter(f"Not a directory: {path}")

    if find_dockerfile(path) is None:
        raise typer.BadParameter(
            f"{path} does not look like a HUD environment (no Dockerfile.hud or Dockerfile found)"
        )

    # Add common Python source roots to sys.path so user task modules
    # can resolve their package imports without requiring hud-python's
    # venv to also have the user's package installed. Covers:
    #   <repo>/        — flat layout (env.py at root, etc.)
    #   <repo>/src     — src layout (hud_controller/, …)
    extra_paths = [str(path), str(path / "src")]
    inserted: list[str] = []
    for p in extra_paths:
        if Path(p).is_dir() and p not in sys.path:
            sys.path.insert(0, p)
            inserted.append(p)
    try:
        tasks_source = tasks_override or str(path)
        tasks = collect_tasks(tasks_source)
    finally:
        for p in inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(p)
    if not tasks:
        raise typer.BadParameter(
            f"No tasks found at {tasks_source}. Pass --tasks <file> if your "
            "taskset lives elsewhere."
        )

    platform: str | None = None
    if image_override:
        image_ref = image_override
        required_env: list[str] = []
    else:
        lock_path = find_lock(path)
        if lock_path is None:
            raise typer.BadParameter(
                f"No hud.lock.yaml found near {path}. Run `hud build` and "
                "`hud push` first, or pass --image <ref>."
            )
        lock_data = load_lock(lock_path)
        image_ref = lock_data.get("images", {}).get("pushed") or get_local_image(lock_data)
        if not image_ref:
            raise typer.BadParameter(
                "Lock file has no pushed/local image. Run `hud push` first, or pass --image <ref>."
            )
        env_block = lock_data.get("environment", {}).get("variables", {}) or {}
        required_env = list(env_block.get("required") or [])
        platform = lock_data.get("build", {}).get("platform")

    return tasks, image_ref, required_env, platform


def _resolve_prompts(
    tasks: list,
    mode: str,
    prompts_file: Path | None,
    *,
    image: str,
    taskset_name: str,
    env_vars: dict[str, str] | None = None,
    platform: str | None = None,
) -> dict[str, str]:
    """Return a slug → rendered-prompt dict per the chosen mode."""
    if prompts_file:
        data = json.loads(prompts_file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise typer.BadParameter(f"--prompts-file must contain a JSON object, got {type(data)}")
        return {str(k): str(v) for k, v in data.items()}

    if mode == "skip":
        return {}
    if mode == "live":
        import asyncio

        return asyncio.run(
            render_prompts_via_image(
                image,
                tasks,
                taskset_name,
                env_vars=env_vars or None,
                platform=platform,
            )
        )
    if mode == "trace":
        # Trace mode reads pre-recorded prompt spans from the platform's
        # trace store. The platform pipeline performs this lookup itself
        # and supplies rendered_prompts via ExportInput; the local CLI
        # has no trace store to query.
        raise NotImplementedError(
            "--render-prompts trace requires the platform pipeline. "
            "Use --render-prompts live (Docker), --prompts-file <json>, "
            "or --render-prompts skip from the CLI."
        )
    if mode == "auto":
        # No trace store available locally; auto == live for the CLI.
        return _resolve_prompts(
            tasks,
            "live",
            None,
            image=image,
            taskset_name=taskset_name,
            env_vars=env_vars,
            platform=platform,
        )
    raise typer.BadParameter(f"Unknown render-prompts mode: {mode}")


def _make_format_command(exporter: BaseExporter) -> Callable[..., None]:
    """Build the per-format Typer command function."""

    def _cmd(
        path: Path = typer.Argument(  # noqa: B008
            Path("."),
            help="Path to the HUD environment repo (default: current directory)",
        ),
        output: Path = typer.Option(  # noqa: B008
            Path("./hud_exported"),
            "--output",
            "-o",
            help="Output directory",
        ),
        taskset: str | None = typer.Option(
            None,
            "--taskset",
            help="Taskset name; reserved for future remote-mode support",
        ),
        tasks: str | None = typer.Option(
            None,
            "--tasks",
            help="Explicit path to a tasks file (.py / .json / .jsonl)",
        ),
        image: str | None = typer.Option(
            None,
            "--image",
            help="Override the base image reference",
        ),
        render_prompts: str = typer.Option(
            "auto",
            "--render-prompts",
            help="How to render prompts: auto (default) | live | trace | skip",
        ),
        prompts_file: Path | None = typer.Option(  # noqa: B008
            None,
            "--prompts-file",
            help="JSON object mapping task slug → rendered prompt",
        ),
        env_var: list[str] | None = typer.Option(  # noqa: B008
            None,
            "--env",
            "-e",
            help="KEY=VALUE env vars for live render container (repeatable)",
        ),
        task_subset: list[str] | None = typer.Option(  # noqa: B008
            None,
            "--task",
            help="Restrict export to specific task slugs (repeatable)",
        ),
    ) -> None:
        f"""Export to {exporter.name}."""  # noqa: B021
        hud_console = HUDConsole()

        if taskset:
            hud_console.error(
                "--taskset (remote mode) is not yet supported in the CLI. "
                "Use the platform pipeline, or run from a local repo."
            )
            raise typer.Exit(1)

        repo_path = path.resolve()

        try:
            collected, image_ref, required_env, platform = _resolve_repo_inputs(
                repo_path, image, tasks
            )
        except typer.BadParameter as exc:
            hud_console.error(str(exc))
            raise typer.Exit(1) from None

        env_vars: dict[str, str] = {}
        for raw in env_var or []:
            if "=" not in raw:
                hud_console.error(f"Invalid --env (expected KEY=VALUE): {raw}")
                raise typer.Exit(1)
            key, value = raw.split("=", 1)
            env_vars[key.strip()] = value

        try:
            rendered = _resolve_prompts(
                collected,
                render_prompts,
                prompts_file,
                image=image_ref,
                taskset_name=taskset or repo_path.name,
                env_vars=env_vars,
                platform=platform,
            )
        except (NotImplementedError, typer.BadParameter) as exc:
            hud_console.error(str(exc))
            raise typer.Exit(1) from None
        except (RuntimeError, OSError) as exc:
            hud_console.error(f"Live render failed: {exc}")
            raise typer.Exit(1) from None

        taskset_name = taskset or repo_path.name

        inp = ExportInput(
            tasks=collected,
            env_image=image_ref,
            env_platform=platform,
            env_required_env=required_env,
            repo_root=repo_path,
            rendered_prompts=rendered,
            taskset_name=taskset_name,
            taskset_id=None,
            task_subset=list(task_subset) if task_subset else None,
        )

        try:
            result = exporter.export(inp)
        except NotImplementedError as exc:
            hud_console.error(str(exc))
            raise typer.Exit(1) from None
        except ValueError as exc:
            hud_console.error(str(exc))
            raise typer.Exit(1) from None

        out_path = write_result(result, output)

        hud_console.header(f"Exported to {exporter.name}")
        hud_console.section_title("Output")
        hud_console.status_item("Directory", str(out_path))
        hud_console.status_item("Tasks", str(result.manifest.get("task_count", 0)))
        hud_console.status_item("Base image", image_ref)
        hud_console.info("")

        hud_console.section_title("Try it")
        hud_console.command_example(
            f"cd {out_path} && {result.manifest['sample_run_command']}",
            "Build and run the first task locally",
        )
        rendered_count = len(rendered)
        total = result.manifest.get("task_count", 0)
        if render_prompts == "skip":
            hud_console.info("")
            hud_console.info(
                "Prompts not rendered (--render-prompts skip). "
                "instruction.md files contain placeholders."
            )
        elif rendered_count < total:
            missing = total - rendered_count
            hud_console.info("")
            hud_console.info(
                f"{missing} of {total} prompt(s) could not be rendered; "
                "those instruction.md files fall back to args.description."
            )

    _cmd.__name__ = f"export_{exporter.name}"
    _cmd.__doc__ = f"Export HUD taskset+envs to {exporter.description}."
    return _cmd


# Auto-register one subcommand per exporter.
for _exp in _load_exporters():
    export_app.command(name=_exp.name)(_make_format_command(_exp))

"""``hud harbor`` — export HUD tasks to Harbor task folders."""

from __future__ import annotations

import asyncio

import typer

from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()


def harbor_command(
    source: str = typer.Argument(
        ...,
        help="Tasks file (.json/.jsonl of {env, task, args}) or a .py source exposing Variants.",
    ),
    out_dir: str = typer.Option(
        "harbor_tasks", "--out", "-o", help="Output directory for the Harbor task folders."
    ),
) -> None:
    """Export HUD tasks to Harbor task folders (deterministic).

    Loads <source> like ``hud eval`` (a JSON/JSONL taskset or a ``.py`` source),
    verifies each env's capabilities are ssh/mcp only, and writes one Harbor task
    folder per task (task + args): ``task.toml`` / ``instruction.md`` /
    ``environment/Dockerfile`` / ``tests/test.sh``. The generated ``test.sh`` grades
    via ``hud client run`` against the env control channel served in the container.
    """
    from hud.eval.harbor import export

    hud_console.header("HUD → Harbor Export")
    try:
        created = asyncio.run(export(source, out_dir))
    except (ValueError, TypeError, FileNotFoundError) as e:
        hud_console.error(str(e))
        raise typer.Exit(1) from e

    if not created:
        hud_console.warning(f"No variants found in {source}")
        raise typer.Exit(1)

    hud_console.success(f"Exported {len(created)} Harbor task(s) to {out_dir}/")
    for task_dir in created:
        hud_console.info(f"  {task_dir.name}")

    hud_console.hint(
        "Grading uses the in-container HUD control channel, so these tasks need "
        "Harbor's default same-container verifier. Don't set [verifier.environment] "
        "in task.toml \u2014 a separate verifier container can't reach the parked run "
        "on 127.0.0.1."
    )

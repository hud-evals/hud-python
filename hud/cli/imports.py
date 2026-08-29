"""``hud import`` command group: import evaluation results from external systems."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Typer inspects annotations at runtime
from typing import Any
from uuid import UUID  # noqa: TC003 - Typer inspects annotations at runtime

import typer

from hud.cli import CLI, CliError
from hud.integrations.harbor import import_results
from hud.settings import settings
from hud.utils.hud_console import HUDConsole
from hud.utils.platform import PlatformClient

import_app = CLI(
    name="import",
    help="Import evaluation results from external systems",
    add_completion=False,
    rich_markup_mode="rich",
)


@import_app.command("harbor")
def import_harbor_command(
    archive: Path | None = typer.Argument(  # noqa: B008
        None,
        help="Completed Harbor job or trial archive.",
    ),
    taskset: str | None = typer.Option(
        None,
        "--taskset",
        help="Taskset name or ID used to attribute imported trials.",
    ),
    data_file_id: UUID | None = typer.Option(  # noqa: B008
        None,
        "--data-file-id",
        help="Reuse an uploaded Data File instead of uploading ARCHIVE.",
    ),
) -> dict[str, Any]:
    """Import a completed Harbor job or trial archive.

    [not dim]Examples:
        hud import harbor harbor-job.zip --taskset my-harbor-tasks
        hud import harbor --data-file-id <data-file-id>   # reuse the uploaded file
        hud import harbor harbor-job.zip --json[/not dim]
    """
    if archive is None and data_file_id is None:
        raise CliError(
            "usage",
            "Pass a Harbor archive or --data-file-id.",
            suggestion="Run 'hud import harbor <archive>' to upload a completed Harbor job.",
        )
    hud_console = HUDConsole()
    hud_console.header("Import Harbor Results", icon="")
    hud_console.progress_message("Submitting Harbor results...")

    result = import_results(
        archive,
        taskset=taskset,
        data_file_id=data_file_id,
        platform=PlatformClient.from_settings(),
    )

    hud_console.success("Harbor import submitted")
    hud_console.key_value_table(
        {
            "Data file": str(result.data_file_id),
            "Job": str(result.job_id),
            "Trace": str(result.trace_id),
        }
    )
    hud_console.link(f"{settings.hud_web_url}/jobs/{result.job_id}")
    return result.model_dump(mode="json")

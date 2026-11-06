"""``hud batch`` — inspect and export a batch's ledger.

    hud batch status batch.py              # rows done / failed, mean reward, resume cursors
    hud batch results batch.py -o out.parquet

A batch runs with ``hud eval batch.py <model> --full`` and resumes by running the
same command again against the same ledger.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.table import Table

from hud.cli import CLI, CliError
from hud.cli.eval import batch_ledger_path
from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()

batch_app = CLI(
    name="batch",
    help="Inspect and export the ledger of a batch (see `hud eval batch.py <model> --full`).",
    add_completion=False,
    rich_markup_mode="rich",
)

SOURCE_HELP = "Python source defining the Batch (a .py file or directory)."
LEDGER_HELP = "SQLite ledger (default .hud/batches/<batch id>.sqlite beside the source)."
NAME_HELP = "Which Batch, when the source defines several."


def _open(source: str, ledger: str | None, name: str | None) -> tuple[Any, Any, Path]:
    from hud.batch import Batch, Ledger

    batch = Batch.from_file(source, name=name)
    path = Path(ledger) if ledger else batch_ledger_path(Path(source).resolve(), batch.id)
    if not path.exists():
        raise CliError(
            "not_found",
            f"No ledger at {path}",
            suggestion=f"Run `hud eval {source} <model> --full` first, or pass --ledger.",
            input={"ledger": str(path)},
        )
    return batch, Ledger(path), path


@batch_app.command("status")
def status_command(
    source: str = typer.Argument(".", help=SOURCE_HELP),
    ledger: str | None = typer.Option(None, "--ledger", help=LEDGER_HELP),
    name: str | None = typer.Option(None, "--name", help=NAME_HELP),
) -> Any:
    """Rows done and failed, the mean reward so far, and where each reader resumes.

    [not dim]Examples:
        hud batch status batch.py
        hud batch status batch.py --json[/not dim]
    """
    batch, store, path = _open(source, ledger, name)
    summary = store.summary(batch.id)
    checkpoints = {
        str(reader): cursor.model_dump(mode="json")
        for reader, cursor in sorted(store.checkpoints(batch.id).items())
    }

    table = Table(title=f"Batch {batch.name}", title_style="bold cyan")
    table.add_column("Field", style="yellow")
    table.add_column("Value", style="green")
    table.add_row("id", batch.id)
    table.add_row("dataset", batch.dataset.uri)
    table.add_row("ledger", str(path))
    table.add_row("completed", str(summary["completed"]))
    table.add_row("failed", str(summary["error"]))
    mean = summary["mean_reward"]
    table.add_row("mean reward", f"{mean:.3f}" if mean is not None else "-")
    for reader, cursor in checkpoints.items():
        table.add_row(
            f"reader {reader}",
            f"file {cursor['file']} group {cursor['row_group']} row {cursor['row']}",
        )
    hud_console.print(table)
    return {
        "batch_id": batch.id,
        "name": batch.name,
        "ledger": str(path),
        **summary,
        "checkpoints": checkpoints,
    }


@batch_app.command("results")
def results_command(
    source: str = typer.Argument(".", help=SOURCE_HELP),
    out: str = typer.Option(..., "--out", "-o", help="Destination: .jsonl or .parquet"),
    ledger: str | None = typer.Option(None, "--ledger", help=LEDGER_HELP),
    name: str | None = typer.Option(None, "--name", help=NAME_HELP),
) -> Any:
    """Export every row's result (reward, evaluation, status, trace id) to a file.

    [not dim]Examples:
        hud batch results batch.py -o results.parquet
        hud batch results batch.py -o results.jsonl --json[/not dim]
    """
    batch, store, _path = _open(source, ledger, name)
    written = store.export(batch.id, out)
    summary = store.summary(batch.id)
    rows = summary["completed"] + summary["error"] + summary["cancelled"]
    hud_console.success(f"Exported {rows} row result(s) to {written}")
    return {"batch_id": batch.id, "path": str(written), "rows": rows}


__all__ = ["batch_app"]

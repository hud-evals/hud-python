"""Rich CLI display for new-flow eval results (``list[Run]``).

Adapted from the legacy ``hud/eval/display.py`` to read :class:`hud.client.Run`
(``reward`` + ``trace.content`` + ``trace.isError`` + ``prompt``) rather than the
legacy ``EvalContext``.
"""

from __future__ import annotations

from statistics import mean, pstdev
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hud.client import Run

_SUCCESS_THRESHOLD = 0.7


def _truncate(text: str | None, max_len: int) -> str:
    if not text:
        return "—"
    text = text.replace("\n", " ").strip()
    return text[: max_len - 2] + ".." if len(text) > max_len else text


def display_runs(
    runs: Sequence[Run],
    *,
    name: str = "",
    elapsed: float | None = None,
    show_details: bool = True,
) -> None:
    """Print a summary (+ per-run details table) for a batch of runs."""
    if not runs:
        print("No results to display")  # noqa: T201
        return

    rewards = [r.reward for r in runs]
    errors = [r for r in runs if r.trace.isError]
    mean_reward = mean(rewards)
    std_reward = pstdev(rewards) if len(rewards) > 1 else 0.0
    success_rate = sum(1 for r in rewards if r > _SUCCESS_THRESHOLD) / len(runs)

    try:
        from rich.table import Table

        from hud.utils.hud_console import HUDConsole

        console = HUDConsole().console  # configured for Windows-safe encoding
    except ImportError:
        print(f"\n{name or 'Eval'}: {len(runs)} runs, mean reward {mean_reward:.3f}")  # noqa: T201
        return

    title = f"'{name}' Results" if name else "Evaluation Complete"
    console.print(f"\n[bold]{title}[/bold]")
    console.print(f"  [dim]Runs:[/dim] {len(runs)}")
    if elapsed:
        rate = len(runs) / elapsed if elapsed > 0 else 0
        console.print(f"  [dim]Time:[/dim] {elapsed:.1f}s ({rate:.1f}/s)")
    console.print(
        f"  [dim]Mean reward:[/dim] [green]{mean_reward:.3f}[/green] +/- {std_reward:.3f}"
    )
    console.print(f"  [dim]Success rate:[/dim] [yellow]{success_rate * 100:.1f}%[/yellow]")
    if errors:
        console.print(f"  [dim]Errors:[/dim] [red]{len(errors)}[/red]")

    if show_details and len(runs) <= 50:
        table = Table(title="Details", show_header=True, header_style="bold")
        table.add_column("#", style="dim", justify="right", width=4)
        table.add_column("Prompt", style="dim", max_width=35)
        table.add_column("Answer", style="dim", max_width=35)
        table.add_column("Reward", justify="right", style="green", width=8)
        table.add_column("", justify="center", width=3)
        for i, run in enumerate(runs):
            if run.trace.isError:
                status = "[red]✗[/red]"
            elif run.reward > _SUCCESS_THRESHOLD:
                status = "[green]✓[/green]"
            else:
                status = "[yellow]○[/yellow]"
            row: list[Any] = [
                str(i),
                _truncate(run.prompt, 35),
                _truncate(run.trace.content, 35),
                f"{run.reward:.3f}",
                status,
            ]
            table.add_row(*row)
        console.print(table)

    if std_reward > 0.3:
        console.print(f"\n[yellow]High variance (std={std_reward:.3f})[/yellow]")
    console.print()


__all__ = ["display_runs"]

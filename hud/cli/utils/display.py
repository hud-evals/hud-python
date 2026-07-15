"""Rich CLI display for new-flow eval results (``list[Run]``).

Adapted from the legacy ``hud/eval/display.py`` to read :class:`hud.eval.Run`
(``reward`` + ``trace.content`` + ``trace.is_error`` + ``prompt``) rather than
the legacy ``EvalContext``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hud.eval.stats import JobStats

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hud.eval.run import Run

_SUCCESS_THRESHOLD = 0.7


def _truncate(text: str | list[Any] | None, max_len: int) -> str:
    if not text:
        return "—"
    if not isinstance(text, str):  # chat-style prompts are message lists
        text = str(text)
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

    stats = JobStats.from_runs(runs)
    rewards = [r.reward for r in runs]
    mean_reward = stats.reward_mean
    std_reward = stats.reward_std
    success_rate = sum(1 for r in rewards if r > _SUCCESS_THRESHOLD) / len(runs)

    try:
        from rich.table import Table

        from hud.utils.hud_console import HUDConsole

        console = HUDConsole().console  # configured for Windows-safe encoding
    except ImportError:
        print(f"\n{name or 'Eval'}: {len(runs)} runs, mean reward {mean_reward:.3f}")  # noqa: T201
        if stats.within_group_reward_std is not None:
            print(  # noqa: T201
                f"Groups with spread: {stats.informative_group_count}/{stats.eligible_group_count}"
            )
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
    if stats.error_count:
        console.print(f"  [dim]Errors:[/dim] [red]{stats.error_count}[/red]")

    within_group_std = stats.within_group_reward_std
    if within_group_std is not None:
        spread_style = (
            "green"
            if stats.informative_group_count == stats.eligible_group_count
            else "yellow"
            if stats.informative_group_count
            else "red"
        )
        console.print(
            f"  [dim]Within-group std:[/dim] [{spread_style}]"
            f"{within_group_std:.3f}[/{spread_style}]"
        )
        console.print(
            f"  [dim]Groups with spread:[/dim] [{spread_style}]"
            f"{stats.informative_group_count}/{stats.eligible_group_count}[/{spread_style}]"
        )
        if stats.constant_group_count:
            console.print(
                f"  [dim]Constant groups:[/dim] {stats.constant_group_count} "
                f"({stats.all_zero_group_count} all-zero, "
                f"{stats.all_one_group_count} all-one)"
            )
        if stats.error_group_count:
            console.print(f"  [dim]Groups with errors:[/dim] [red]{stats.error_group_count}[/red]")

    if show_details and len(runs) <= 50:
        table = Table(title="Details", show_header=True, header_style="bold")
        table.add_column("#", style="dim", justify="right", width=4)
        table.add_column("Prompt", style="dim", max_width=35)
        table.add_column("Answer", style="dim", max_width=35)
        table.add_column("Reward", justify="right", style="green", width=8)
        table.add_column("", justify="center", width=3)
        for i, run in enumerate(runs):
            if run.trace.is_error:
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

    if stats.eligible_group_count and not stats.informative_group_count:
        if std_reward > 0:
            console.print(
                "\n[yellow]Global reward variance comes entirely from differences between groups; "
                "every group is constant.[/yellow]"
            )
        else:
            console.print(
                "\n[yellow]No within-group reward spread; grouped training would produce zero "
                "relative advantage.[/yellow]"
            )
    elif std_reward > 0.3:
        console.print(f"\n[yellow]High variance (std={std_reward:.3f})[/yellow]")
    console.print()


__all__ = ["display_runs"]

"""Display helpers for eval links and job URLs.

Provides consistent, beautiful display for HUD URLs using rich.
"""

from __future__ import annotations

import contextlib
import webbrowser
from statistics import mean, pstdev
from typing import TYPE_CHECKING, Any

from hud.settings import settings

if TYPE_CHECKING:
    from hud.eval.context import EvalContext


def print_link(url: str, title: str, *, open_browser: bool = True) -> None:
    """Print a nicely formatted link with optional browser opening.

    Args:
        url: The URL to display
        title: Title for the panel (e.g., "🔗 Eval Started", "🚀 Job Started")
        open_browser: Whether to open the URL in browser
    """
    # Only print if telemetry is enabled and has API key
    if not (settings.telemetry_enabled and settings.api_key):
        return

    # Open in browser
    if open_browser:
        with contextlib.suppress(Exception):
            webbrowser.open(url, new=2)

    try:
        from rich.align import Align
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        style = "bold underline rgb(108,113,196)"
        link_markup = f"[{style}][link={url}]{url}[/link][/{style}]"

        content = Align.center(link_markup)

        panel = Panel(
            content,
            title=title,
            border_style="rgb(192,150,12)",
            padding=(0, 2),
        )
        console.print(panel)
    except ImportError:
        print(f"{title}: {url}")  # noqa: T201


def print_complete(url: str, name: str, *, error: bool = False) -> None:
    """Print a completion message with link.

    Args:
        url: The URL to display
        name: Name of the eval/job
        error: Whether an error occurred
    """
    # Only print if telemetry is enabled and has API key
    if not (settings.telemetry_enabled and settings.api_key):
        return

    try:
        from rich.console import Console

        console = Console()

        if error:
            console.print(
                f"\n[red]✗ '{name}' failed![/red] [dim]View details at:[/dim] "
                f"[bold link={url}]{url}[/bold link]\n"
            )
        else:
            console.print(
                f"\n[green]✓ '{name}' complete![/green] [dim]View results at:[/dim] "
                f"[bold link={url}]{url}[/bold link]\n"
            )
    except ImportError:
        status = "failed" if error else "complete"
        print(f"\n{name} {status}: {url}\n")  # noqa: T201


def print_eval_stats(
    completed: list[EvalContext],
    name: str = "",
    *,
    elapsed: float | None = None,
    show_details: bool = True,
) -> None:
    """Print statistics for completed evaluations.

    Args:
        completed: List of completed EvalContext objects
        name: Optional name for the evaluation
        elapsed: Optional elapsed time in seconds
        show_details: Whether to show per-eval details table
    """
    if not completed:
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
    except ImportError:
        # Fallback to basic printing
        _print_eval_stats_basic(completed, name, elapsed)
        return

    # Calculate aggregate stats
    rewards = [ctx.reward for ctx in completed if ctx.reward is not None]
    errors = [ctx for ctx in completed if ctx.error is not None]
    durations = [ctx.duration for ctx in completed if ctx.duration > 0]

    mean_reward = mean(rewards) if rewards else 0.0
    std_reward = pstdev(rewards) if len(rewards) > 1 else 0.0
    success_rate = (len(completed) - len(errors)) / len(completed) if completed else 0.0

    # Print summary
    title = f"📊 '{name}' Results" if name else "📊 Eval Results"
    console.print(f"\n[bold]{title}[/bold]")
    console.print(f"  [dim]Evals:[/dim] {len(completed)}")
    if elapsed:
        rate = len(completed) / elapsed if elapsed > 0 else 0
        console.print(f"  [dim]Time:[/dim] {elapsed:.1f}s ({rate:.1f} evals/s)")
    if durations:
        mean_duration = mean(durations)
        console.print(f"  [dim]Avg duration:[/dim] {mean_duration:.2f}s")
    console.print(f"  [dim]Mean reward:[/dim] [green]{mean_reward:.3f}[/green] ± {std_reward:.3f}")
    console.print(f"  [dim]Success rate:[/dim] [yellow]{success_rate * 100:.1f}%[/yellow]")
    if errors:
        console.print(f"  [dim]Errors:[/dim] [red]{len(errors)}[/red]")

    # Show details table if requested and not too many
    if show_details and len(completed) <= 50:
        table = Table(title="Per-Eval Details", show_header=True, header_style="bold")
        table.add_column("#", style="dim", justify="right", width=4)
        table.add_column("Variants", style="cyan", max_width=35)
        table.add_column("Answer", style="white", max_width=25)
        table.add_column("Reward", justify="right", style="green", width=8)
        table.add_column("Duration", justify="right", width=10)
        table.add_column("Status", justify="center", width=8)

        for ctx in completed:
            idx_str = str(ctx.index)
            variants_str = _format_variants(ctx.variants) if ctx.variants else "-"
            answer_str = _truncate(ctx.answer, 30) if ctx.answer else "-"
            reward_str = f"{ctx.reward:.3f}" if ctx.reward is not None else "-"
            duration_str = f"{ctx.duration:.2f}s" if ctx.duration > 0 else "-"

            if ctx.error:
                status = "[red]✗[/red]"
            elif ctx.reward is not None and ctx.reward > 0.7:
                status = "[green]✓[/green]"
            else:
                status = "[yellow]○[/yellow]"

            table.add_row(idx_str, variants_str, answer_str, reward_str, duration_str, status)

        console.print(table)

    # Warn about high variance
    if std_reward > 0.3:
        console.print(f"\n[yellow]⚠️  High variance detected (std={std_reward:.3f})[/yellow]")

    console.print()


def _format_variants(variants: dict[str, Any]) -> str:
    """Format variants dict for display."""
    if not variants:
        return "-"
    parts = [f"{k}={v}" for k, v in variants.items()]
    result = ", ".join(parts)
    return result[:35] + "..." if len(result) > 35 else result


def _truncate(text: str | None, max_len: int) -> str:
    """Truncate text to max length."""
    if not text:
        return "-"
    # Replace newlines with spaces for display
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def _print_eval_stats_basic(
    completed: list[EvalContext],
    name: str,
    elapsed: float | None,
) -> None:
    """Basic stats printing without rich."""
    rewards = [ctx.reward for ctx in completed if ctx.reward is not None]
    errors = [ctx for ctx in completed if ctx.error is not None]

    mean_reward = mean(rewards) if rewards else 0.0
    success_rate = (len(completed) - len(errors)) / len(completed) if completed else 0.0

    title = f"'{name}' Results" if name else "Eval Results"
    print(f"\n{title}")  # noqa: T201
    print(f"  Evals: {len(completed)}")  # noqa: T201
    if elapsed:
        print(f"  Time: {elapsed:.1f}s")  # noqa: T201
    print(f"  Mean reward: {mean_reward:.3f}")  # noqa: T201
    print(f"  Success rate: {success_rate * 100:.1f}%")  # noqa: T201
    if errors:
        print(f"  Errors: {len(errors)}")  # noqa: T201
    print()  # noqa: T201


__all__ = ["print_complete", "print_eval_stats", "print_link"]

"""Display helpers for eval links, job URLs, and result statistics."""

from __future__ import annotations

import contextlib
import webbrowser
from statistics import mean, pstdev
from typing import Any

from hud.settings import settings


def print_link(url: str, title: str, *, open_browser: bool = True) -> None:
    """Print a nicely formatted link with optional browser opening."""
    if not (settings.telemetry_enabled and settings.api_key):
        return

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
        panel = Panel(
            Align.center(link_markup),
            title=title,
            border_style="rgb(192,150,12)",
            padding=(0, 2),
        )
        console.print(panel)
    except ImportError:
        print(f"{title}: {url}")  # noqa: T201


def print_complete(url: str, name: str, *, error: bool = False) -> None:
    """Print a completion message with link."""
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


def print_single_result(
    trace_id: str,
    name: str,
    *,
    reward: float | None = None,
    error: str | None = None,
) -> None:
    """Print a single eval result summary."""
    if not (settings.telemetry_enabled and settings.api_key):
        return

    url = f"https://hud.ai/trace/{trace_id}"

    try:
        from rich.console import Console

        console = Console()

        if error:
            console.print(
                f"\n[red]✗ '{name}' failed![/red]\n"
                f"  [dim]Error:[/dim] [red]{error[:80]}{'...' if len(error) > 80 else ''}[/red]\n"
                f"  [dim]View at:[/dim] [bold link={url}]{url}[/bold link]\n"
            )
        else:
            reward_str = f"{reward:.3f}" if reward is not None else "—"
            reward_color = "green" if reward is not None and reward > 0.7 else "yellow"
            console.print(
                f"\n[green]✓ '{name}' complete![/green]\n"
                f"  [dim]Reward:[/dim] [{reward_color}]{reward_str}[/{reward_color}]\n"
                f"  [dim]View at:[/dim] [bold link={url}]{url}[/bold link]\n"
            )
    except ImportError:
        status = "failed" if error else "complete"
        reward_str = f", reward={reward:.3f}" if reward is not None else ""
        print(f"\n{name} {status}{reward_str}: {url}\n")  # noqa: T201


def display_results(
    results: list[Any],
    *,
    tasks: list[Any] | None = None,
    name: str = "",
    elapsed: float | None = None,
    show_details: bool = True,
) -> None:
    """Display evaluation results in a formatted table.

    Args:
        results: List of EvalContext objects from hud.eval()
        tasks: Optional list of Task objects (for task info in table)
        name: Optional name for the evaluation
        elapsed: Optional elapsed time in seconds
        show_details: Whether to show per-eval details table
    """
    if not results:
        print("No results to display")  # noqa: T201
        return

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
    except ImportError:
        _display_basic(results, name, elapsed)
        return

    # Extract stats from results (EvalContext objects)
    # Use 'or 0' to handle None rewards (scenario failed before returning a reward)
    rewards = [getattr(r, "reward", 0) or 0 for r in results if r is not None]
    errors = [r for r in results if r is not None and getattr(r, "error", None)]
    durations = [getattr(r, "duration", 0) for r in results if getattr(r, "duration", 0) > 0]

    if not rewards:
        console.print("[yellow]No valid results[/yellow]")
        return

    mean_reward = mean(rewards) if rewards else 0.0
    std_reward = pstdev(rewards) if len(rewards) > 1 else 0.0
    success_count = sum(1 for r in rewards if r > 0.7)
    success_rate = success_count / len(results) if results else 0.0

    # Print summary
    title = f"📊 '{name}' Results" if name else "📊 Evaluation Complete"
    console.print(f"\n[bold]{title}[/bold]")
    console.print(f"  [dim]Evals:[/dim] {len(results)}")
    if elapsed:
        rate = len(results) / elapsed if elapsed > 0 else 0
        console.print(f"  [dim]Time:[/dim] {elapsed:.1f}s ({rate:.1f}/s)")
    if durations:
        console.print(f"  [dim]Avg duration:[/dim] {mean(durations):.2f}s")
    console.print(f"  [dim]Mean reward:[/dim] [green]{mean_reward:.3f}[/green] ± {std_reward:.3f}")
    console.print(f"  [dim]Success rate:[/dim] [yellow]{success_rate * 100:.1f}%[/yellow]")
    if errors:
        console.print(f"  [dim]Errors:[/dim] [red]{len(errors)}[/red]")

    # Details table
    if show_details and len(results) <= 50:
        table = Table(title="Details", show_header=True, header_style="bold")
        table.add_column("#", style="dim", justify="right", width=4)

        # Check if we have variants (grouped parallel runs)
        has_variants = any(getattr(r, "variants", None) for r in results if r)
        has_prompts = any(getattr(r, "prompt", None) for r in results if r)
        has_answers = any(getattr(r, "answer", None) for r in results if r)

        if has_variants:
            table.add_column("Variants", style="cyan", max_width=30)
        elif tasks:
            table.add_column("Task", style="cyan", max_width=30)

        if has_prompts:
            table.add_column("Prompt", style="dim", max_width=35)

        if has_answers:
            table.add_column("Answer", style="dim", max_width=35)

        table.add_column("Reward", justify="right", style="green", width=8)
        if durations:
            table.add_column("Time", justify="right", width=8)
        table.add_column("", justify="center", width=3)  # Status icon

        for i, r in enumerate(results):
            if r is None:
                continue

            idx = getattr(r, "index", i)
            reward = getattr(r, "reward", None)
            error = getattr(r, "error", None)
            duration = getattr(r, "duration", 0)
            variants = getattr(r, "variants", None)
            prompt = getattr(r, "prompt", None)
            answer = getattr(r, "answer", None)

            # Status icon
            if error:
                status = "[red]✗[/red]"
            elif reward is not None and reward > 0.7:
                status = "[green]✓[/green]"
            else:
                status = "[yellow]○[/yellow]"

            row = [str(idx)]

            # Variant or task column
            if has_variants:
                row.append(_format_variants(variants))
            elif tasks and i < len(tasks):
                task = tasks[i]
                task_label = _get_task_label(task, i)
                row.append(task_label[:30])

            # Prompt column
            if has_prompts:
                row.append(_truncate(prompt, 35))

            # Answer column
            if has_answers:
                row.append(_truncate(answer, 35))

            # Reward
            row.append(f"{reward:.3f}" if reward is not None else "—")

            # Duration
            if durations:
                row.append(f"{duration:.1f}s" if duration > 0 else "—")

            row.append(status)
            table.add_row(*row)

        console.print(table)

    # Variance warning
    if std_reward > 0.3:
        console.print(f"\n[yellow]⚠️  High variance (std={std_reward:.3f})[/yellow]")

    console.print()


def _display_basic(results: list[Any], name: str, elapsed: float | None) -> None:
    """Fallback display without rich."""
    rewards = [getattr(r, "reward", 0) for r in results if r is not None]
    title = f"'{name}' Results" if name else "Eval Results"
    print(f"\n{title}")  # noqa: T201
    print(f"  Evals: {len(results)}")  # noqa: T201
    if elapsed:
        print(f"  Time: {elapsed:.1f}s")  # noqa: T201
    if rewards:
        print(f"  Mean reward: {mean(rewards):.3f}")  # noqa: T201
    print()  # noqa: T201


def _format_variants(variants: dict[str, Any] | None) -> str:
    """Format variants dict for display."""
    if not variants:
        return "-"
    parts = [f"{k}={v}" for k, v in variants.items()]
    result = ", ".join(parts)
    return result[:28] + ".." if len(result) > 30 else result


def _truncate(text: str | None, max_len: int) -> str:
    """Truncate text to max length."""
    if not text:
        return "-"
    text = text.replace("\n", " ").strip()
    return text[: max_len - 2] + ".." if len(text) > max_len else text


def _get_task_label(task: Any, index: int) -> str:
    """Get a display label for a task."""
    if task is None:
        return f"task_{index}"
    if isinstance(task, dict):
        return task.get("id") or task.get("prompt", "")[:25] or f"task_{index}"
    task_id = getattr(task, "id", None)
    if task_id:
        return task_id
    prompt = getattr(task, "prompt", None) or getattr(task, "scenario", None)
    if prompt:
        return prompt[:25]
    return f"task_{index}"


# Backwards compatibility alias
print_eval_stats = display_results

__all__ = [
    "display_results",
    "print_complete",
    "print_eval_stats",
    "print_link",
    "print_single_result",
]

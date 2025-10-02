"""Inline JSON preview with expandable view for RL flow.

Uses minimal terminal interaction for inline display.
"""

from __future__ import annotations

import json
from typing import Any

from blessed import Terminal
from rich.console import Console
from rich.json import JSON as RichJSON
from rich.panel import Panel
from rich.table import Table


def _mask_secrets(value: Any) -> Any:
    """Recursively mask common secret-looking values."""
    secret_keys = {"authorization", "api-key", "apikey", "token", "secret", "password"}

    def _is_secret_key(k: str) -> bool:
        lowered = k.lower()
        if lowered in secret_keys:
            return True
        return any(s in lowered for s in ["api", "key", "token", "secret", "password"])

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for k, v in value.items():
            if _is_secret_key(str(k)) and isinstance(v, str) and v:
                prefix = v[:4]
                suffix = v[-4:] if len(v) > 8 else ""
                result[k] = f"{prefix}…{suffix}"
            else:
                result[k] = _mask_secrets(v)
        return result
    if isinstance(value, list):
        return [_mask_secrets(v) for v in value]
    return value


def _truncate_value(value: Any, max_len: int = 60) -> str:
    """Truncate a value for preview display."""
    if isinstance(value, str):
        if len(value) > max_len:
            return value[:max_len] + "…"
        return value
    elif isinstance(value, (dict, list)):
        s = json.dumps(value, separators=(",", ":"))
        if len(s) > max_len:
            return s[:max_len] + "…"
        return s
    else:
        return str(value)


def show_json_interactive(
    data: Any,
    *,
    title: str | None = None,
    max_string_len: int = 60,
    prompt: bool = True,
    initial_expanded: bool = False,
) -> None:
    """Display JSON inline with keyboard-based expand/collapse."""
    console = Console()
    safe_data = _mask_secrets(data)

    # Create preview table
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    if title:
        console.print(f"\n[bold cyan]{title}[/bold cyan]")

    # Show preview
    if isinstance(safe_data, dict):
        items = list(safe_data.items())
        for _, (key, value) in enumerate(items[:5]):
            truncated = _truncate_value(value, max_string_len)
            table.add_row(key + ":", truncated)

        if len(items) > 5:
            table.add_row("", f"[dim]... and {len(items) - 5} more items[/dim]")
    else:
        table.add_row("", _truncate_value(safe_data, max_string_len))

    # Display with border
    if not initial_expanded:
        console.print(Panel(table, expand=False, border_style="dim"))
    else:
        # Expanded view
        if title:
            console.rule(f"[bold cyan]{title} (expanded)[/bold cyan]")
        try:
            console.print(RichJSON.from_data(safe_data))
        except Exception:
            console.print(json.dumps(safe_data, indent=2))

    if not prompt:
        console.print()
        return

    # Prompt for expansion (interactive mode)
    console.print("[dim]Press 'e' to expand, Enter to continue[/dim] ", end="")

    try:
        term = Terminal()
        with term.cbreak():
            key = term.inkey(timeout=30)  # 30 second timeout
            if key and key.lower() == "e":
                console.print()  # New line
                if title:
                    console.rule(f"[bold cyan]{title} (expanded)[/bold cyan]")

                try:
                    console.print(RichJSON.from_data(safe_data))
                except Exception:
                    console.print(json.dumps(safe_data, indent=2))

                console.print("\n[dim]Press Enter to continue...[/dim]")
                term.inkey()
    except Exception:
        console.print()  # Ensure we're on a new line
        choice = input().strip().lower()

        if choice == "e":
            if title:
                console.rule(f"[bold cyan]{title} (expanded)[/bold cyan]")

            try:
                console.print(RichJSON.from_data(safe_data))
            except Exception:
                console.print(json.dumps(safe_data, indent=2))

            console.print("\n[dim]Press Enter to continue...[/dim]")
            input()

    console.print()

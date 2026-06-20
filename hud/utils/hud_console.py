"""HUD Console Design System - Consistent styling utilities for CLI output.

This module provides a unified design system for HUD CLI commands,
ensuring consistent colors, formatting, and visual hierarchy across
all commands.

Color Palette:
- Gold (#c0960c): Primary brand color for headers and important elements
- Neutral Grey: Standard text that works on both light and dark backgrounds
- Muted Red: Errors and failures
- Muted Green: Success messages
- Bright Black: Secondary/dimmed information
- Blue-Purple: Links and interactive elements
"""

from __future__ import annotations

import logging
import traceback
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

# HUD Brand Colors - Optimized for both light and dark modes
GOLD = "rgb(192,150,12)"  # #c0960c - Primary brand color
RED = "rgb(205,92,92)"  # Indian red / coral — warm, readable on both backgrounds
GREEN = "rgb(60,140,80)"  # Forest green — rich, not neon
DIM = "bright_black"  # Grey that's visible on both light and dark backgrounds
YELLOW = "rgb(218,165,32)"  # Goldenrod — saturated amber, not raw yellow
TEXT = "bright_white"  # Off-white that's readable on dark, not too bright on light
SECONDARY = "rgb(108,113,196)"  # Muted blue-purple for secondary text


class Symbols:
    """Unicode symbols for consistent CLI output with default colors."""

    # Info/Items - Use for all informational lines (gold)
    ITEM = f"[{GOLD}]•[/{GOLD}]"


class HUDConsole:
    """Design system for HUD CLI output."""

    # Make symbols easily accessible
    sym = Symbols

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize the design system.

        Args:
            logger: Logger to check for log levels. If None, uses the root logger.
        """
        self._stdout_console = Console(stderr=False)
        self._stderr_console = Console(stderr=True)
        self._logger = logger or logging.getLogger()

    def header(self, title: str, icon: str = "🚀", stderr: bool = True) -> None:
        """Print a header panel with gold border.

        Args:
            title: The title text
            icon: Optional emoji icon
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        label = f"{icon} [bold]{title}[/bold]" if icon else f"[bold]{title}[/bold]"
        console.print(Panel.fit(label, border_style=GOLD))

    def section_title(self, title: str, stderr: bool = True) -> None:
        """Print a section title in gold.

        Args:
            title: The section title
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"\n[bold {GOLD}]{title}[/bold {GOLD}]")

    def success(self, message: str, stderr: bool = True) -> None:
        """Print a success message.

        Args:
            message: The success message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{GREEN}]\u2714 {escape(message)}[/{GREEN}]")

    def error(self, message: str, stderr: bool = True) -> None:
        """Print an error message.

        Args:
            message: The error message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        tb = traceback.format_exc()
        escaped_message = escape(message)
        if "NoneType: None" not in tb:
            escaped_tb = escape(tb)
            console.print(
                f"[{RED} not bold]\u2716 {escaped_message}\n{escaped_tb}[/{RED} not bold]"
            )
        else:
            console.print(f"[{RED} not bold]\u2716 {escaped_message}[/{RED} not bold]")

    def warning(self, message: str, stderr: bool = True) -> None:
        """Print a warning message.

        Args:
            message: The warning message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{YELLOW} not bold]\u26a0 {escape(message)}[/{YELLOW} not bold]")

    def info(self, message: str, stderr: bool = True) -> None:
        """Print an info message.

        Args:
            message: The info message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{TEXT} not bold]{escape(message)}[/{TEXT} not bold]")

    def print(self, message: str, stderr: bool = True) -> None:
        """Print a message.

        Args:
            message: The message to print
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(message)

    def dim_info(self, label: str, value: str, stderr: bool = True) -> None:
        """Print dimmed info with a label.

        Args:
            label: The label text
            value: The value text
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(
            f"[{DIM} not bold][default]{escape(label)}[/default][/{DIM} not bold] [default]{escape(value)}[/default]"  # noqa: E501
        )

    def link(self, url: str, stderr: bool = True) -> None:
        """Print an underlined link.

        Args:
            url: The URL to display
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{SECONDARY} underline]{escape(url)}[/{SECONDARY} underline]")

    def key_value_table(
        self, data: dict[str, str | int | float], show_header: bool = False, stderr: bool = True
    ) -> None:
        """Print a key-value table.

        Args:
            data: Dictionary of key-value pairs
            show_header: Whether to show table header
            stderr: If True, output to stderr (default), otherwise stdout
        """
        table = Table(show_header=show_header, box=None, padding=(0, 1))
        table.add_column("Key", style=DIM, no_wrap=True)
        table.add_column("Value")

        for key, value in data.items():
            table.add_row(key, str(value))

        console = self._stderr_console if stderr else self._stdout_console
        console.print(table)

    def progress_message(self, message: str, stderr: bool = True) -> None:
        """Print a progress message.

        Args:
            message: The progress message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{DIM}]{escape(message)}[/{DIM}]")

    def hint(self, hint: str, stderr: bool = True) -> None:
        """Print a hint message.

        Args:
            hint: The hint text
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[rgb(181,137,0)]💡 Hint: {escape(hint)}[/rgb(181,137,0)]")

    def status_item(
        self,
        label: str,
        value: str,
        status: str = "success",
        primary: bool = False,
        stderr: bool = True,
    ) -> None:
        """Print a status item with indicator.

        Args:
            label: The label text
            value: The value text
            status: Status type - "success" (✓), "error" (✗), "warning" (⚠), "info" (•)
            primary: If True, highlight the value as primary
            stderr: If True, output to stderr (default), otherwise stdout
        """
        indicators = {
            "success": f"[{GREEN}]✓[/{GREEN}]",
            "error": f"[{RED}]✗[/{RED}]",
            "warning": "[yellow]⚠[/yellow]",
            "info": f"[{DIM}]•[/{DIM}]",
        }

        indicator = indicators.get(status, indicators["info"])
        console = self._stderr_console if stderr else self._stdout_console

        escaped_label = escape(label)
        escaped_value = escape(value)
        if primary:
            console.print(
                f"{indicator} {escaped_label}: [bold {SECONDARY}]{escaped_value}[/bold {SECONDARY}]"
            )
        else:
            console.print(f"{indicator} {escaped_label}: [{TEXT}]{escaped_value}[/{TEXT}]")

    def command_example(
        self, command: str, description: str | None = None, stderr: bool = True
    ) -> None:
        """Print a command example with cyan highlighting.

        Args:
            command: The command to show
            description: Optional description after the command
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        if description:
            console.print(
                f"  [{SECONDARY}]{command}[/{SECONDARY}]  "
                f"[bright_black]# {description}[/bright_black]"
            )
        else:
            console.print(f"  [{SECONDARY}]{command}[/{SECONDARY}]")

    # Exception rendering utilities
    def render_support_hint(self, stderr: bool = True) -> None:
        """Render a standard support message for users encountering issues."""
        support = (
            "If this looks like an issue with the sdk, please make a github issue at "
            "https://github.com/hud-evals/hud-python/issues"
        )
        self.info(support, stderr=stderr)

    def render_exception(self, error: BaseException, *, stderr: bool = True) -> None:
        """Render exceptions consistently using the HUD design system.

        - Shows exception type and message
        - Displays structured hints if present on the exception (e.g., HudException.hints)
        - Prints a link to open an issue for SDK problems
        """
        from hud.utils.exceptions import HudRequestError  # lazy import: avoid import cycle

        # Header with exception type
        ex_type = type(error).__name__
        message = getattr(error, "message", "") or str(error) or ex_type
        self.error(f"{ex_type}: {message}", stderr=stderr)

        # Specialized details for request errors
        if isinstance(error, HudRequestError):
            details: dict[str, str | int | float] = {}
            if error.status_code is not None:
                details["Status"] = str(error.status_code)
            if error.response_text:
                # Limit very long responses
                text = error.response_text
                details["Response"] = text[:500] + ("..." if len(text) > 500 else "")
            if error.response_json and "Response" not in details:
                details["Response JSON"] = str(error.response_json)
            if details:
                self.key_value_table(details, show_header=False, stderr=stderr)

        # Structured hints, if available
        hints = getattr(error, "hints", None)
        if hints:
            from hud.utils.hints import render_hints  # lazy import: avoid import cycle

            render_hints(hints, design=self)

        # Standard support hint
        self.render_support_hint(stderr=stderr)

    @property
    def console(self) -> Console:
        """Get the stderr console for direct access when needed."""
        return self._stderr_console

    def debug(self, message: str, stderr: bool = True) -> None:
        """Print a debug message only if DEBUG logging is enabled.

        Args:
            message: The debug message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        if self._logger.isEnabledFor(logging.DEBUG):
            self.dim_info(message, "", stderr=stderr)

    def select(
        self,
        message: str,
        choices: list[str | dict[str, Any]] | list[str],
        default: int | None = None,
        spaced: bool = False,
    ) -> str:
        """Interactive selection with arrow key navigation.

        Args:
            message: The prompt message to display
            choices: List of choices. Can be strings or dicts with 'name' and 'value' keys
            default: Default selection (matches against choice name/string)
            spaced: Insert a blank line between choices for a roomier list

        Returns:
            The selected choice value
        """
        import questionary
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys
        from questionary import Style

        # Convert choices to questionary format, optionally interleaving blank
        # (non-selectable) separators so the list breathes.
        q_choices: list[Any] = []

        for choice in choices:
            if spaced and q_choices:
                q_choices.append(questionary.Separator(" "))
            if isinstance(choice, dict):
                name = choice.get("name", str(choice.get("value", "")))
                value = choice.get("value", name)
                q_choices.append(questionary.Choice(title=name, value=value))
            else:
                q_choices.append(choice)

        # Custom style for better visibility of selection
        custom_style = Style(
            [
                ("qmark", "fg:cyan bold"),
                ("question", "bold"),
                ("pointer", "fg:cyan bold"),
                ("highlighted", "fg:cyan bold"),
            ]
        )

        question = questionary.select(
            message,
            choices=q_choices,
            instruction="(Use ↑/↓ arrows, Enter to select, Esc to cancel)",
            style=custom_style,
        )

        # questionary only aborts on Ctrl+C out of the box. Bind Esc to cancel
        # too. Non-eager so it doesn't swallow the Esc-prefixed arrow sequences.
        key_bindings = question.application.key_bindings
        assert isinstance(key_bindings, KeyBindings)

        @key_bindings.add(Keys.Escape)
        def _cancel(event: Any) -> None:
            event.app.exit(result=None)

        result = question.ask()

        # No selection (Ctrl+C or Esc) → cancel the command.
        if result is None:
            import typer

            raise typer.Exit(1)

        return result

    def format_tool_call(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        """Format a tool call in compact HUD style.

        Args:
            name: Tool name
            arguments: Tool arguments dictionary

        Returns:
            Formatted string with Rich markup
        """
        import json

        args_str = ""
        if arguments:
            try:
                # Compact JSON representation
                args_str = json.dumps(arguments, separators=(",", ":"))
                if len(args_str) > 60:
                    args_str = args_str[:57] + "..."
            except (TypeError, ValueError):
                args_str = str(arguments)[:60]

        escaped_name = escape(name)
        escaped_args = escape(args_str)
        return (
            f"[{GOLD}]→[/{GOLD}] [bold {TEXT}]{escaped_name}[/bold {TEXT}]"
            f"[{DIM}]({escaped_args})[/{DIM}]"
        )

    def format_tool_result(self, content: str, is_error: bool = False) -> str:
        """Format a tool result in compact HUD style.

        Args:
            content: Result content (will be truncated if too long)
            is_error: Whether this is an error result

        Returns:
            Formatted string with Rich markup
        """
        # Truncate content if needed
        if len(content) > 80:
            content = content[:77] + "..."

        escaped_content = escape(content)
        # Format with status using HUD colors
        if is_error:
            return f"  [{RED}]✗[/{RED}] [{DIM}]{escaped_content}[/{DIM}]"
        else:
            return f"  [{GREEN}]✓[/{GREEN}] [{TEXT}]{escaped_content}[/{TEXT}]"

    def confirm(self, message: str, default: bool = True) -> bool:
        """Prompt for a yes/no confirmation; Ctrl+C / EOF answers no."""
        import questionary

        return bool(questionary.confirm(message, default=default, qmark="").ask())


# Global design instance for convenience
hud_console = HUDConsole()

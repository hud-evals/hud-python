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
import time
import traceback
from typing import TYPE_CHECKING, Any, Literal, Self

import questionary
import typer
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from rich.status import Status
# HUD Brand Colors - Optimized for both light and dark modes
GOLD = "rgb(192,150,12)"  # #c0960c - Primary brand color
RED = "rgb(220,50,47)"  # Slightly muted red that works on both backgrounds
GREEN = "rgb(133,153,0)"  # Slightly muted green that works on both backgrounds
DIM = "bright_black"  # Grey that's visible on both light and dark backgrounds
YELLOW = "yellow"
TEXT = "bright_white"  # Off-white that's readable on dark, not too bright on light
SECONDARY = "rgb(108,113,196)"  # Muted blue-purple for secondary text


# HUD Symbol System - Minimal 3-category system with default colors
class Symbols:
    """Unicode symbols for consistent CLI output with default colors."""

    # Info/Items - Use for all informational lines (gold)
    ITEM = f"[{GOLD}]•[/{GOLD}]"

    # Status - Use for state/completion (green)
    SUCCESS = f"[{GREEN}]●[/{GREEN}]"

    # Flow/Special - Use for transitions and important notes (gold)
    FLOW = f"[{GOLD}]⟿[/{GOLD}]"


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
        console.print(Panel.fit(f"{icon} [bold]{title}[/bold]", border_style=GOLD))

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
        console.print(f"[{GREEN}]✅ {message}[/{GREEN}]")

    def error(self, message: str, stderr: bool = True) -> None:
        """Print an error message.

        Args:
            message: The error message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        tb = traceback.format_exc()
        if "NoneType: None" not in tb:
            console.print(f"[{RED} not bold]❌ {message}\n{tb}[/{RED} not bold]")
        else:
            console.print(f"[{RED} not bold]❌ {message}[/{RED} not bold]")

    def warning(self, message: str, stderr: bool = True) -> None:
        """Print a warning message.

        Args:
            message: The warning message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"⚠️  [{YELLOW} not bold]{message}[/{YELLOW} not bold]")

    def info(self, message: str, stderr: bool = True) -> None:
        """Print an info message.

        Args:
            message: The info message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{TEXT} not bold]{message}[/{TEXT} not bold]")

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
            f"[{DIM} not bold][default]{label}[/default][/{DIM} not bold] [default]{value}[/default]"  # noqa: E501
        )

    def link(self, url: str, stderr: bool = True) -> None:
        """Print an underlined link.

        Args:
            url: The URL to display
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{SECONDARY} underline]{url}[/{SECONDARY} underline]")

    def json_config(self, json_str: str, stderr: bool = True) -> None:
        """Print JSON configuration with neutral theme.

        Args:
            json_str: JSON string to display
            stderr: If True, output to stderr (default), otherwise stdout
        """
        # Print JSON with neutral grey text
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{TEXT}]{json_str}[/{TEXT}]")

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
        console.print(f"[{DIM}]{message}[/{DIM}]")

    def phase(self, phase_num: int, title: str, stderr: bool = True) -> None:
        """Print a phase header (for debug command).

        Args:
            phase_num: Phase number
            title: Phase title
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"\n{'=' * 80}", style=GOLD)
        console.print(f"[bold {GOLD}]PHASE {phase_num}: {title}[/bold {GOLD}]")
        console.print(f"{'=' * 80}", style=GOLD)

    def command(self, cmd: list[str], stderr: bool = True) -> None:
        """Print a command being executed.

        Args:
            cmd: Command parts as list
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[bold {TEXT}]$ {' '.join(cmd)}[/bold {TEXT}]")

    def hint(self, hint: str, stderr: bool = True) -> None:
        """Print a hint message.

        Args:
            hint: The hint text
            stderr: If True, output to stderr (default), otherwise stdout
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[rgb(181,137,0)]💡 Hint: {hint}[/rgb(181,137,0)]")

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

        if primary:
            console.print(f"{indicator} {label}: [bold {SECONDARY}]{value}[/bold {SECONDARY}]")
        else:
            console.print(f"{indicator} {label}: [{TEXT}]{value}[/{TEXT}]")

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
        try:
            from hud.shared.exceptions import HudRequestError  # lazy import
        except Exception:
            # Keep type available for isinstance guards below without import-time dependency
            HudRequestError = tuple()  # type: ignore

        # Header with exception type
        ex_type = type(error).__name__
        message = getattr(error, "message", "") or str(error) or ex_type
        self.error(f"{ex_type}: {message}", stderr=stderr)

        # Specialized details for request errors
        if isinstance(error, HudRequestError):  # type: ignore[arg-type]
            details: dict[str, str] = {}
            status_code = getattr(error, "status_code", None)
            if status_code is not None:
                details["Status"] = str(status_code)
            response_text = getattr(error, "response_text", None)
            if response_text:
                # Limit very long responses
                trimmed = response_text[:500] + ("..." if len(response_text) > 500 else "")
                details["Response"] = trimmed
            response_json = getattr(error, "response_json", None)
            if response_json and not details.get("Response"):
                details["Response JSON"] = str(response_json)
            if details:
                self.key_value_table(details, show_header=False, stderr=stderr)  # type: ignore

        # Structured hints, if available
        hints = getattr(error, "hints", None)
        if hints:
            try:
                from hud.shared.hints import render_hints  # lazy import

                render_hints(hints, design=self)
            except Exception as render_error:
                self.debug_log(f"Failed to render hints: {render_error}")

        # Standard support hint
        self.render_support_hint(stderr=stderr)

    @property
    def console(self) -> Console:
        """Get the stderr console for direct access when needed."""
        return self._stderr_console

    def set_verbose(self, verbose: bool) -> None:
        """Set the logging level based on verbose flag.

        Args:
            verbose: If True, show INFO level messages. If False, only show WARNING and above.
        """
        if verbose:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.WARNING)

    @property
    def prefix(self) -> str:
        """Get the metadata of the current file."""
        metadata = self._logger.findCaller(stacklevel=3)
        return f"{metadata[0]}:{metadata[1]} in {metadata[2]} | "

    # Logging-aware methods that check logging levels before printing
    def log(
        self,
        message: str,
        level: Literal["info", "debug", "warning", "error"] = "info",
        stderr: bool = True,
    ) -> None:
        """Print a message based on the logging level."""
        prefix = self.prefix
        if level == "info":
            self.info_log(f"{prefix}{message}", stderr=stderr)
        elif level == "debug":
            self.debug_log(f"{prefix}{message}", stderr=stderr)
        elif level == "warning":
            self.warning_log(f"{prefix}{message}", stderr=stderr)
        elif level == "error":
            self.error_log(f"{prefix}{message}", stderr=stderr)

    def debug(self, message: str, stderr: bool = True) -> None:
        """Print a debug message."""
        self.debug_log(message, stderr=stderr)

    def debug_log(self, message: str, stderr: bool = True) -> None:
        """Print a debug message only if DEBUG logging is enabled.

        Args:
            message: The debug message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        if self._logger.isEnabledFor(logging.DEBUG):
            self.dim_info(message, "", stderr=stderr)

    def info_log(self, message: str, stderr: bool = True) -> None:
        """Print an info message only if INFO logging is enabled.

        Args:
            message: The info message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        if self._logger.isEnabledFor(logging.INFO):
            self.info(message, stderr=stderr)

    def progress_log(self, message: str, stderr: bool = True) -> None:
        """Print a progress message only if INFO logging is enabled.

        Args:
            message: The progress message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        if self._logger.isEnabledFor(logging.INFO):
            self.progress_message(message, stderr=stderr)

    def progress(self, initial: str = "", stderr: bool = True) -> _ProgressContext:
        """Create a progress context manager for inline updates.

        Args:
            initial: Initial message to display
            stderr: If True, output to stderr (default), otherwise stdout

        Returns:
            A context manager that provides update() method

        Example:
            with console.progress("Processing...") as progress:
                for i in range(10):
                    progress.update(f"Processing item {i+1}/10")
        """
        return _ProgressContext(
            console=self._stderr_console if stderr else self._stdout_console, initial=initial
        )

    def success_log(self, message: str, stderr: bool = True) -> None:
        """Print a success message only if INFO logging is enabled.

        Args:
            message: The success message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        if self._logger.isEnabledFor(logging.INFO):
            self.success(message, stderr=stderr)

    def warning_log(self, message: str, stderr: bool = True) -> None:
        """Print a warning message only if WARNING logging is enabled.

        Args:
            message: The warning message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        if self._logger.isEnabledFor(logging.WARNING):
            self.warning(message, stderr=stderr)

    def error_log(self, message: str, stderr: bool = True) -> None:
        """Print an error message only if ERROR logging is enabled.

        Args:
            message: The error message
            stderr: If True, output to stderr (default), otherwise stdout
        """
        if self._logger.isEnabledFor(logging.ERROR):
            self.error(message, stderr=stderr)

    def select(
        self,
        message: str,
        choices: list[str | dict[str, Any]] | list[str],
        default: int | None = None,
    ) -> str:
        """Interactive selection with arrow key navigation.

        Args:
            message: The prompt message to display
            choices: List of choices. Can be strings or dicts with 'name' and 'value' keys
            default: Default selection (matches against choice name/string)

        Returns:
            The selected choice value
        """
        # Convert choices to questionary format
        q_choices = []

        for choice in choices:
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

        result = questionary.select(
            message,
            choices=q_choices,
            instruction="(Use ↑/↓ arrows, Enter to select)",
            style=custom_style,
        ).ask()

        # If no selection made (Ctrl+C or ESC), exit
        if result is None:
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

        return f"[{GOLD}]→[/{GOLD}] [bold {TEXT}]{name}[/bold {TEXT}][{DIM}]({args_str})[/{DIM}]"

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

        # Format with status using HUD colors
        if is_error:
            return f"  [{RED}]✗[/{RED}] [{DIM}]{content}[/{DIM}]"
        else:
            return f"  [{GREEN}]✓[/{GREEN}] [{TEXT}]{content}[/{TEXT}]"

    def confirm(self, message: str, default: bool = True) -> bool:
        """Print a confirmation message.

        Args:
            message: The confirmation message
            default: If True, the default choice is True
        """
        return questionary.confirm(message, default=default).ask()

    # Symbol-based output methods
    def symbol(self, symbol: str, message: str, color: str = GOLD, stderr: bool = True) -> None:
        """Print a message with a colored symbol prefix.

        Args:
            symbol: Symbol to use (use Symbols.* constants)
            message: Message text
            color: Color for the symbol (default: gold)
            stderr: If True, output to stderr
        """
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{color}]{symbol}[/{color}] {message}")

    def detail(self, message: str, stderr: bool = True) -> None:
        """Print an indented detail line with gold pointer symbol."""
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"  [{GOLD}]{Symbols.ITEM}[/{GOLD}] {message}")

    def flow(self, message: str, stderr: bool = True) -> None:
        """Print a flow/transition message with wave symbol."""
        self.symbol(Symbols.FLOW, message, GOLD, stderr)

    def note(self, message: str, stderr: bool = True) -> None:
        """Print an important note with asterism symbol."""
        self.symbol(Symbols.ITEM, message, GOLD, stderr)


# Global design instance for convenience
class _ProgressContext:
    """Context manager for inline progress updates."""

    def __init__(self, console: Console, initial: str = "") -> None:
        self.console = console
        self.initial = initial
        self.status: Status | None = None
        self.start_time: float | None = None

    def __enter__(self) -> Self:
        self.status = self.console.status(self.initial)
        self.status.__enter__()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self.status:
            self.status.__exit__(exc_type, exc_val, exc_tb)  # type: ignore

    def update(self, message: str, with_elapsed: bool = True) -> None:
        """Update the progress message.

        Args:
            message: New message to display
            with_elapsed: If True, append elapsed time to message
        """
        if self.status:
            if with_elapsed and self.start_time:
                elapsed = time.time() - self.start_time
                self.status.update(f"{message} [{elapsed:.1f}s]")
            else:
                self.status.update(message)


hud_console = HUDConsole()

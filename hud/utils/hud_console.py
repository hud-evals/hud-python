"""Styled console output for the HUD CLI."""

from __future__ import annotations

import logging
import traceback
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

# Chosen to read on both light and dark terminal backgrounds.
GOLD = "rgb(192,150,12)"
RED = "rgb(205,92,92)"
GREEN = "rgb(60,140,80)"
DIM = "bright_black"
YELLOW = "rgb(218,165,32)"
TEXT = "bright_white"
SECONDARY = "rgb(108,113,196)"


class Symbols:
    """Unicode symbols for consistent CLI output with default colors."""

    ITEM = f"[{GOLD}]•[/{GOLD}]"


class HUDConsole:
    """Design system for HUD CLI output."""

    sym = Symbols

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._stdout_console = Console(stderr=False)
        self._stderr_console = Console(stderr=True)
        self._logger = logger or logging.getLogger()

    def header(self, title: str, icon: str = "🚀", stderr: bool = True) -> None:
        """Print a header panel with gold border."""
        console = self._stderr_console if stderr else self._stdout_console
        label = f"{icon} [bold]{title}[/bold]" if icon else f"[bold]{title}[/bold]"
        console.print(Panel.fit(label, border_style=GOLD))

    def section_title(self, title: str, stderr: bool = True) -> None:
        """Print a section title in gold."""
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"\n[bold {GOLD}]{title}[/bold {GOLD}]")

    def success(self, message: str, stderr: bool = True) -> None:
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{GREEN}]\u2714 {escape(message)}[/{GREEN}]")

    def error(self, message: str, stderr: bool = True) -> None:
        """Print an error message, followed by the traceback of any exception being handled."""
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
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{YELLOW} not bold]\u26a0 {escape(message)}[/{YELLOW} not bold]")

    def info(self, message: str, stderr: bool = True) -> None:
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{TEXT} not bold]{escape(message)}[/{TEXT} not bold]")

    def print(self, message: Any = "", stderr: bool = True, **kwargs: Any) -> None:
        console = self._stderr_console if stderr else self._stdout_console
        console.print(message, **kwargs)

    def dim_info(self, label: str, value: str, stderr: bool = True) -> None:
        """Print a dimmed ``label`` and its ``value`` on one line."""
        console = self._stderr_console if stderr else self._stdout_console
        console.print(
            f"[{DIM} not bold]{escape(label)}[/{DIM} not bold] [default]{escape(value)}[/default]"
        )

    def link(self, url: str, stderr: bool = True) -> None:
        """Print an underlined link."""
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{SECONDARY} underline]{escape(url)}[/{SECONDARY} underline]")

    def key_value_table(
        self, data: dict[str, str | int | float], show_header: bool = False, stderr: bool = True
    ) -> None:
        table = Table(show_header=show_header, box=None, padding=(0, 1))
        table.add_column("Key", style=DIM, no_wrap=True)
        table.add_column("Value")

        for key, value in data.items():
            table.add_row(key, str(value))

        console = self._stderr_console if stderr else self._stdout_console
        console.print(table)

    def progress_message(self, message: str, stderr: bool = True) -> None:
        console = self._stderr_console if stderr else self._stdout_console
        console.print(f"[{DIM}]{escape(message)}[/{DIM}]")

    def hint(self, hint: str, stderr: bool = True) -> None:
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
        """Print ``label: value`` after a glyph for ``status``: success, error, warning or info."""
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
        """Print an indented command, with ``description`` as a trailing ``#`` comment."""
        console = self._stderr_console if stderr else self._stdout_console
        if description:
            console.print(
                f"  [{SECONDARY}]{command}[/{SECONDARY}]  "
                f"[bright_black]# {description}[/bright_black]"
            )
        else:
            console.print(f"  [{SECONDARY}]{command}[/{SECONDARY}]")

    def render_support_hint(self, stderr: bool = True) -> None:
        """Render a standard support message for users encountering issues."""
        support = (
            "If this looks like an issue with the sdk, please make a github issue at "
            "https://github.com/hud-evals/hud-python/issues"
        )
        self.info(support, stderr=stderr)

    def render_exception(self, error: BaseException, *, stderr: bool = True) -> None:
        """Print an exception with its request details, its ``hints``, and the support hint."""
        from hud.utils.exceptions import HudRequestError  # lazy import: avoid import cycle

        ex_type = type(error).__name__
        message = getattr(error, "message", "") or str(error) or ex_type
        self.error(f"{ex_type}: {message}", stderr=stderr)

        if isinstance(error, HudRequestError):
            details: dict[str, str | int | float] = {}
            if error.status_code is not None:
                details["Status"] = str(error.status_code)
            if error.response_text:
                text = error.response_text
                details["Response"] = text[:500] + ("..." if len(text) > 500 else "")
            if error.response_json and "Response" not in details:
                details["Response JSON"] = str(error.response_json)
            if details:
                self.key_value_table(details, show_header=False, stderr=stderr)

        hints = getattr(error, "hints", None)
        if hints:
            from hud.utils.hints import render_hints  # lazy import: avoid import cycle

            render_hints(hints, design=self)

        self.render_support_hint(stderr=stderr)

    @property
    def stdout(self) -> Console:
        return self._stdout_console

    @property
    def console(self) -> Console:
        return self._stderr_console

    def debug(self, message: str, stderr: bool = True) -> None:
        """Print a debug message only if DEBUG logging is enabled."""
        if self._logger.isEnabledFor(logging.DEBUG):
            self.dim_info(message, "", stderr=stderr)

    def select(
        self,
        message: str,
        choices: list[str | dict[str, Any]] | list[str],
        spaced: bool = False,
    ) -> str:
        """Prompt for one of ``choices`` and return its value; Esc or Ctrl+C raises ``typer.Exit``.

        A choice is a string or a ``{"name": ..., "value": ...}`` dict.
        """
        import questionary
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.keys import Keys
        from questionary import Style

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

        if result is None:
            import typer

            raise typer.Exit(1)

        return result

    def format_tool_call(self, name: str, arguments: dict[str, Any] | str | None = None) -> str:
        """Format a tool call as compact Rich markup."""
        import json

        args_str = ""
        if arguments:
            try:
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
        """Format a tool result as compact Rich markup."""
        if len(content) > 80:
            content = content[:77] + "..."

        escaped_content = escape(content)
        if is_error:
            return f"  [{RED}]✗[/{RED}] [{DIM}]{escaped_content}[/{DIM}]"
        else:
            return f"  [{GREEN}]✓[/{GREEN}] [{TEXT}]{escaped_content}[/{TEXT}]"

    def confirm(self, message: str, default: bool = True) -> bool:
        """Prompt for a yes/no confirmation; Ctrl+C / EOF answers no."""
        import questionary

        return bool(questionary.confirm(message, default=default, qmark="").ask())


hud_console = HUDConsole()

"""Git clone wrapper with quiet mode and rich output.

This module provides a CLI command that wraps 'git clone' to provide a better user experience:
- Runs git clone quietly (no verbose output)
- Displays a rich formatted success message
- Shows optional tutorial/getting started messages from the cloned repository

Usage:
    hud clone https://github.com/user/repo.git

The clone command will look for a [tool.hud.clone] section in the cloned repository's
pyproject.toml file. If found, it will display the configured message after cloning.

Configuration in pyproject.toml:
    [tool.hud.clone]
    title = "🚀 My Project"  # Optional title for the message panel
    style = "blue"           # Optional border style (any Rich color)

    # Option 1: Plain text message with Rich markup support
    message = "[bold]Welcome![/bold] Run [cyan]pip install -e .[/cyan] to start."

    # Option 2: Markdown formatted message
    markdown = "## Welcome\\n\\nThis supports **markdown** formatting."

    # Option 3: Step-by-step instructions
    steps = [
        "Install dependencies: [green]pip install -e .[/green]",
        "Run tests: [green]pytest[/green]",
        "Start coding!"
    ]

Rich Markup Examples:
    - Colors: [red]text[/red], [green]text[/green], [blue]text[/blue], [cyan]text[/cyan]
    - Styles: [bold]text[/bold], [italic]text[/italic], [underline]text[/underline]
    - Combined: [bold cyan]text[/bold cyan]
    - Background: [on red]text[/on red]
    - Links: [link=https://example.com]clickable[/link]

See Rich documentation for more markup options: https://rich.readthedocs.io/en/stable/markup.html
"""

from __future__ import annotations

import logging
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)

console = Console()


def clone_repository(url: str) -> tuple[bool, str]:
    """
    Clone a git repository quietly and return status.

    Args:
        url: Git repository URL

    Returns:
        Tuple of (success, directory_path or error_message)
    """
    # Extract repo name from URL
    repo_name = Path(url).stem
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    # Build git clone command (simple, no options)
    cmd = ["git", "clone", "--quiet", url]

    try:
        # Run git clone with quiet flag
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        # Get the absolute path of the cloned directory
        clone_path = Path(repo_name).resolve()

        return True, str(clone_path)

    except subprocess.CalledProcessError as e:
        error_msg = (
            e.stderr.strip() if e.stderr else f"Git clone failed with exit code {e.returncode}"
        )
        return False, error_msg
    except Exception as e:
        return False, f"Unexpected error: {e!s}"


def get_clone_message(clone_path: str) -> dict[str, Any] | None:
    """
    Look for a clone message configuration in the repository's pyproject.toml or .hud.toml.

    Checks for:
    1. [tool.hud.clone] section in pyproject.toml
    2. [clone] section in .hud.toml

    Args:
        clone_path: Path to the cloned repository

    Returns:
        Dictionary with message configuration or None
    """
    repo_path = Path(clone_path)

    # Check pyproject.toml first
    pyproject_path = repo_path / "pyproject.toml"
    if pyproject_path.exists():
        try:
            with open(pyproject_path, "rb") as f:
                data = tomllib.load(f)
                if "tool" in data and "hud" in data["tool"] and "clone" in data["tool"]["hud"]:
                    return data["tool"]["hud"]["clone"]
        except Exception:
            logger.warning("Failed to load clone config from %s", pyproject_path)

    # Check .hud.toml as fallback
    hud_toml_path = repo_path / ".hud.toml"
    if hud_toml_path.exists():
        try:
            with open(hud_toml_path, "rb") as f:
                data = tomllib.load(f)
                if "clone" in data:
                    return data["clone"]
        except Exception:
            logger.warning("Failed to load clone config from %s", hud_toml_path)

    return None


def print_tutorial(clone_config: dict[str, Any] | None = None) -> None:
    """Print a rich formatted success message with optional tutorial."""
    # Display custom message if configured
    if clone_config:
        # Handle different message formats
        if "message" in clone_config:
            # Message with Rich markup support
            console.print(
                Panel(
                    clone_config["message"],  # Rich will parse markup automatically
                    title=clone_config.get("title", "📋 Getting Started"),
                    border_style=clone_config.get("style", "blue"),
                    padding=(1, 2),
                )
            )
        elif "markdown" in clone_config:
            # Markdown message
            console.print(
                Panel(
                    Markdown(clone_config["markdown"]),
                    title=clone_config.get("title", "📋 Getting Started"),
                    border_style=clone_config.get("style", "blue"),
                    padding=(1, 2),
                )
            )
        elif "steps" in clone_config:
            # Step-by-step instructions
            title = clone_config.get("title", "📋 Getting Started")
            console.print(f"\n[bold]{title}[/bold]\n")
            for i, step in enumerate(clone_config["steps"], 1):
                console.print(f"  {i}. {step}")
            console.print()


def print_error(error_msg: str) -> None:
    """Print a rich formatted error message."""
    console.print(
        Panel(
            Text(f"❌ {error_msg}", style="red"),
            title="[bold red]Clone Failed[/bold red]",
            border_style="red",
            padding=(1, 2),
        )
    )


def clone_command(
    url: str = typer.Argument(
        ...,
        help="Git repository URL to clone",
    ),
) -> None:
    """🚀 Clone a git repository quietly with a pretty output.

    [not dim]This command wraps 'git clone' with the --quiet flag and displays
    a rich formatted success message. If the repository contains a clone
    message in pyproject.toml, it will be displayed as a tutorial.

    Configure clone messages in your repository's pyproject.toml:

    [tool.hud.clone]
    title = "🚀 My Project"
    message = "Thanks for cloning! Run 'pip install -e .' to get started."

    # Or use markdown format:
    # markdown = "## Welcome!\\n\\nHere's how to get started..."
    # style = "cyan"

    Examples:
        hud clone https://github.com/user/repo.git[/not dim]
    """
    success, result = clone_repository(url)

    if success:
        clone_config = get_clone_message(result)
        print_tutorial(clone_config)
    else:
        print_error(result)
        raise typer.Exit(1)


def quickstart_command() -> None:
    """Quickstart with evaluating an agent!"""
    clone_command("https://github.com/hud-evals/quickstart.git")

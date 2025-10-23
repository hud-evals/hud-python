"""HUD CLI - Command-line interface for MCP environment analysis and debugging."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from hud.types import AgentType

from . import list_func as list_module
from .analyze import (
    analyze_environment,
    analyze_environment_from_config,
    analyze_environment_from_mcp_config,
)
from .build import build_command
from .clone import clone_repository, get_clone_message, print_error, print_tutorial
from .debug import debug_mcp_stdio
from .dev import run_mcp_dev_server

# Import new commands
from .init import create_environment
from .pull import pull_command
from .push import push_command
from .remove import remove_command
from .utils.config import set_env_values
from .utils.cursor import get_cursor_config_path, list_cursor_servers, parse_cursor_config
from .utils.logging import CaptureLogger

# Create the main Typer app
app = typer.Typer(
    name="hud",
    help="🚀 HUD CLI for MCP environment analysis and debugging",
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,  # Disable Rich's verbose tracebacks
)

console = Console()

# Standard support hint appended to error outputs
SUPPORT_HINT = (
    "If this looks like an issue with the sdk, please make a github issue at "
    "https://github.com/hud-evals/hud-python/issues"
)


# Capture IMAGE and any following Docker args as a single variadic argument list.
@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def analyze(
    params: list[str] = typer.Argument(  # type: ignore[arg-type]  # noqa: B008
        None,  # Optional positional arguments
        help="Docker image followed by optional Docker run arguments (e.g., 'hud-image:latest -e KEY=value')",  # noqa: E501
    ),
    config: Path = typer.Option(  # noqa: B008
        None,
        "--config",
        "-c",
        help="JSON config file with MCP configuration",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    cursor: str | None = typer.Option(
        None,
        "--cursor",
        help="Analyze a server from Cursor config",
    ),
    output_format: str = typer.Option(
        "interactive",
        "--format",
        "-f",
        help="Output format: interactive, json, markdown",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output (shows tool schemas)",
    ),
    live: bool = typer.Option(
        False,
        "--live",
        help="Run container for live analysis (slower but more accurate)",
    ),
) -> None:
    """🔍 Analyze MCP environment - discover tools, resources, and capabilities.

    By default, uses cached metadata for instant results.
    Use --live to run the container for real-time analysis.

    Examples:
        hud analyze hudpython/test_init      # Fast metadata inspection
        hud analyze my-env --live            # Full container analysis
        hud analyze --config mcp-config.json # From MCP config
        hud analyze --cursor text-2048-dev   # From Cursor config
    """
    if config:
        # Load config from JSON file (always live for configs)
        asyncio.run(analyze_environment_from_config(config, output_format, verbose))
    elif cursor:
        # Parse cursor config (always live for cursor)
        command, error = parse_cursor_config(cursor)
        if error or command is None:
            console.print(f"[red]❌ {error or 'Failed to parse cursor config'}[/red]")
            raise typer.Exit(1)
        # Convert to MCP config
        mcp_config = {
            "local": {"command": command[0], "args": command[1:] if len(command) > 1 else []}
        }
        asyncio.run(analyze_environment_from_mcp_config(mcp_config, output_format, verbose))
    elif params:
        image, *docker_args = params
        if live or docker_args:  # If docker args provided, assume live mode
            # Build Docker command from image and args
            from .utils.docker import build_run_command

            docker_cmd = build_run_command(image, docker_args)
            asyncio.run(analyze_environment(docker_cmd, output_format, verbose))
        else:
            # Fast mode - analyze from metadata
            from .utils.metadata import analyze_from_metadata

            asyncio.run(analyze_from_metadata(image, output_format, verbose))
    else:
        console.print("[red]Error: Must specify either a Docker image, --config, or --cursor[/red]")
        console.print("\nExamples:")
        console.print("  hud analyze hudpython/test_init       # Fast metadata analysis")
        console.print("  hud analyze my-env --live             # Live container analysis")
        console.print("  hud analyze --config mcp-config.json  # From config file")
        console.print("  hud analyze --cursor my-server        # From Cursor")
        raise typer.Exit(1)


# Same variadic approach for debug.
@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def debug(
    params: list[str] = typer.Argument(  # type: ignore[arg-type]  # noqa: B008
        None,
        help="Docker image, environment directory, or config file followed by optional Docker arguments",  # noqa: E501
    ),
    config: Path | None = typer.Option(  # noqa: B008
        None,
        "--config",
        "-c",
        help="JSON config file with MCP configuration",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    cursor: str | None = typer.Option(
        None,
        "--cursor",
        help="Debug a server from Cursor config",
    ),
    build: bool = typer.Option(
        False,
        "--build",
        "-b",
        help="Build image before debugging (for directory mode)",
    ),
    max_phase: int = typer.Option(
        5,
        "--max-phase",
        "-p",
        min=1,
        max=5,
        help="Maximum debug phase (1-5)",
    ),
) -> None:
    """🐛 Debug MCP environment - test initialization, tools, and readiness.

    Examples:
        hud debug .                              # Debug current directory
        hud debug environments/browser           # Debug specific directory
        hud debug . --build                      # Build then debug
        hud debug hud-text-2048:latest          # Debug Docker image
        hud debug my-mcp-server:v1 -e API_KEY=xxx
        hud debug --config mcp-config.json
        hud debug --cursor text-2048-dev
        hud debug . --max-phase 3               # Stop after phase 3
    """
    # Import here to avoid circular imports
    from hud.utils.hud_console import HUDConsole

    from .utils.environment import (
        build_environment,
        get_image_name,
        image_exists,
        is_environment_directory,
    )

    hud_console = HUDConsole()

    # Determine the command to run
    command = None
    docker_args = []

    if config:
        # Load config from JSON file
        with open(config) as f:
            mcp_config = json.load(f)

        # Extract command from first server in config
        server_name = next(iter(mcp_config.keys()))
        server_config = mcp_config[server_name]
        command = [server_config["command"], *server_config.get("args", [])]
    elif cursor:
        # Parse cursor config
        command, error = parse_cursor_config(cursor)
        if error or command is None:
            console.print(f"[red]❌ {error or 'Failed to parse cursor config'}[/red]")
            raise typer.Exit(1)
    elif params:
        first_param = params[0]
        docker_args = params[1:] if len(params) > 1 else []

        # Check if it's a directory
        if Path(first_param).exists() and is_environment_directory(first_param):
            # Directory mode - like hud dev
            directory = first_param

            # Get or generate image name
            image_name, source = get_image_name(directory)

            if source == "auto":
                hud_console.info(f"Auto-generated image name: {image_name}")

            # Build if requested or if image doesn't exist
            if build or not image_exists(image_name):
                if not build and not image_exists(image_name):
                    if typer.confirm(f"Image {image_name} not found. Build it now?"):
                        build = True
                    else:
                        raise typer.Exit(1)

                if build and not build_environment(directory, image_name):
                    raise typer.Exit(1)

            # Build Docker command with folder-mode envs
            from .utils.docker import create_docker_run_command

            command = create_docker_run_command(
                image_name, docker_args=docker_args, env_dir=directory
            )
        else:
            # Assume it's an image name
            image = first_param
            from .utils.docker import build_run_command

            # Image-only mode: do not auto-inject local .env
            command = build_run_command(image, docker_args)
    else:
        console.print(
            "[red]Error: Must specify a directory, Docker image, --config, or --cursor[/red]"
        )
        console.print("\nExamples:")
        console.print("  hud debug .                      # Debug current directory")
        console.print("  hud debug environments/browser   # Debug specific directory")
        console.print("  hud debug hud-text-2048:latest  # Debug Docker image")
        console.print("  hud debug --config mcp-config.json")
        console.print("  hud debug --cursor my-server")
        raise typer.Exit(1)

    # Create logger and run debug
    logger = CaptureLogger(print_output=True)
    phases_completed = asyncio.run(debug_mcp_stdio(command, logger, max_phase=max_phase))

    # Show summary using design system
    from hud.utils.hud_console import HUDConsole

    hud_console = HUDConsole()

    hud_console.info("")  # Empty line
    hud_console.section_title("Debug Summary")

    if phases_completed == max_phase:
        hud_console.success(f"All {max_phase} phases completed successfully!")
        if max_phase == 5:
            hud_console.info("Your MCP server is fully functional and ready for production use.")
    else:
        hud_console.warning(f"Completed {phases_completed} out of {max_phase} phases")
        hud_console.info("Check the errors above for troubleshooting.")

    # Exit with appropriate code
    if phases_completed < max_phase:
        raise typer.Exit(1)


@app.command()
def cursor_list() -> None:
    """📋 List all MCP servers configured in Cursor."""
    console.print(Panel.fit("📋 [bold cyan]Cursor MCP Servers[/bold cyan]", border_style="cyan"))

    servers, error = list_cursor_servers()

    if error:
        console.print(f"[red]❌ {error}[/red]")
        raise typer.Exit(1)

    if not servers:
        console.print("[yellow]No servers found in Cursor config[/yellow]")
        return

    # Display servers in a table
    table = Table(title="Available Servers")
    table.add_column("Server Name", style="cyan")
    table.add_column("Command Preview", style="dim")

    config_path = get_cursor_config_path()
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            mcp_servers = config.get("mcpServers", {})

            for server_name in servers:
                server_config = mcp_servers.get(server_name, {})
                command = server_config.get("command", "")
                args = server_config.get("args", [])

                # Create command preview
                if args:
                    preview = f"{command} {' '.join(args[:2])}"
                    if len(args) > 2:
                        preview += " ..."
                else:
                    preview = command

                table.add_row(server_name, preview)

    console.print(table)
    console.print(f"\n[dim]Config location: {config_path}[/dim]")
    console.print(
        "\n[green]Tip:[/green] Use [cyan]hud debug --cursor <server-name>[/cyan] to debug a server"
    )


@app.command()
def version() -> None:
    """Show HUD CLI version."""
    try:
        from hud import __version__

        console.print(f"HUD CLI version: [cyan]{__version__}[/cyan]")
    except ImportError:
        console.print("HUD CLI version: [cyan]unknown[/cyan]")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def dev(
    params: list[str] = typer.Argument(  # type: ignore[arg-type]  # noqa: B008
        None,
        help="Module path or extra Docker args (when using --docker)",
    ),
    docker: bool = typer.Option(
        False,
        "--docker",
        help="Run in Docker with volume mounts for hot-reload (for complex environments)",
    ),
    stdio: bool = typer.Option(
        False,
        "--stdio",
        help="Use stdio transport (default: HTTP)",
    ),
    port: int = typer.Option(8765, "--port", "-p", help="HTTP server port (ignored for stdio)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed logs"),
    inspector: bool = typer.Option(
        False, "--inspector", help="Launch MCP Inspector (HTTP mode only)"
    ),
    interactive: bool = typer.Option(
        False, "--interactive", help="Launch interactive testing mode (HTTP mode only)"
    ),
    watch: list[str] = typer.Option(  # noqa: B008
        None,
        "--watch",
        help="Additional directories to watch for changes (default: current directory)",
    ),
) -> None:
    """🔥 Development mode - run MCP server with hot-reload.

    TWO MODES:

    1. Python Module:
       hud dev                    # Auto-detects module
       hud dev server.main        # Explicit module

    2. Docker with Volume Mounts (Complex environments like 'browser'):
       hud dev --docker           # Auto-detects image from hud.lock.yaml
       hud dev --docker -p 8080:8080  # With extra Docker args

    The server must define 'mcp' in its __init__.py or main.py.

    Examples:
        hud dev                      # Auto-detect in current directory
        hud dev controller           # Run specific module
        hud dev --inspector          # Launch MCP Inspector
        hud dev --interactive        # Launch interactive testing mode
        hud dev --stdio              # Use stdio transport
        hud dev --watch ../shared    # Watch additional directories

    For environment backend servers, use uvicorn directly:
        uvicorn server:app --reload
    """
    # Extract module from params if provided (first param when not --docker)
    module = params[0] if params and not docker else None
    docker_args = params if docker else []

    run_mcp_dev_server(
        module,
        stdio,
        port,
        verbose,
        inspector,
        interactive,
        watch,
        docker=docker,
        docker_args=docker_args,
    )


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    params: list[str] = typer.Argument(  # type: ignore[arg-type]  # noqa: B008
        None,
        help="Docker image followed by optional Docker run arguments "
        "(e.g., 'my-image:latest -e KEY=value')",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Run locally with Docker (default: remote via mcp.hud.so)",
    ),
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport protocol: stdio (default) or http",
    ),
    port: int = typer.Option(
        8765,
        "--port",
        "-p",
        help="Port for HTTP transport (ignored for stdio)",
    ),
    url: str = typer.Option(
        None,
        "--url",
        help="Remote MCP server URL (default: HUD_MCP_URL or mcp.hud.so)",
    ),
    api_key: str | None = typer.Option(
        None,
        "--api-key",
        help="API key for remote server (default: HUD_API_KEY env var)",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Run ID for tracking (remote only)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
) -> None:
    """🚀 Run Docker image as MCP server.

    A simple wrapper around 'docker run' that can launch images locally or remotely.
    By default, runs remotely via mcp.hud.so. Use --local to run with local Docker.

    For local Python development with hot-reload, use 'hud dev' instead.

    Examples:
        hud run my-image:latest                    # Run remotely (default)
        hud run my-image:latest --local            # Run with local Docker
        hud run my-image:latest -e KEY=value       # Remote with env vars
        hud run my-image:latest --local -e KEY=val # Local with env vars
        hud run my-image:latest --transport http   # Use HTTP transport
    """
    if not params:
        console.print("[red]❌ Docker image is required[/red]")
        console.print("\nExamples:")
        console.print("  hud run my-image:latest              # Run remotely (default)")
        console.print("  hud run my-image:latest --local      # Run with local Docker")
        console.print("\n[yellow]For local Python development:[/yellow]")
        console.print("  hud dev                              # Run with hot-reload")
        raise typer.Exit(1)

    image = params[0]
    docker_args = params[1:] if len(params) > 1 else []

    # Check if user accidentally passed a module path
    from pathlib import Path

    if not any(c in image for c in [":", "/"]) and (
        Path(image).is_dir() or Path(image).is_file() or "." in image
    ):
        console.print(f"[yellow]⚠️  '{image}' looks like a module path, not a Docker image[/yellow]")
        console.print("\n[green]For local Python development, use:[/green]")
        console.print(f"  hud dev {image}")
        console.print("\n[green]For Docker images:[/green]")
        console.print("  hud run my-image:latest")
        raise typer.Exit(1)

    # Default to remote if not explicitly local
    is_local = local

    if is_local:
        # Local Docker execution
        from .utils.runner import run_mcp_server

        run_mcp_server(image, docker_args, transport, port, verbose, interactive=False)
    else:
        # Remote execution via proxy
        from .utils.remote_runner import run_remote_server

        # Get URL from options or environment
        if not url:
            from hud.settings import settings

            url = settings.hud_mcp_url

        run_remote_server(image, docker_args, transport, port, url, api_key, run_id, verbose)


@app.command()
def clone(
    url: str = typer.Argument(
        ...,
        help="Git repository URL to clone",
    ),
) -> None:
    """🚀 Clone a git repository quietly with a pretty output.

    This command wraps 'git clone' with the --quiet flag and displays
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
        hud clone https://github.com/user/repo.git
    """
    # Run the clone
    success, result = clone_repository(url)

    if success:
        # Look for clone message configuration
        clone_config = get_clone_message(result)
        print_tutorial(clone_config)
    else:
        print_error(result)
        raise typer.Exit(1)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def build(
    params: list[str] = typer.Argument(  # type: ignore[arg-type]  # noqa: B008
        None,
        help="Environment directory followed by optional arguments (e.g., '. -e API_KEY=secret')",
    ),
    tag: str | None = typer.Option(
        None, "--tag", "-t", help="Docker image tag (default: from pyproject.toml)"
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Build without Docker cache"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    platform: str | None = typer.Option(
        None, "--platform", help="Set Docker target platform (e.g., linux/amd64)"
    ),
) -> None:
    """🏗️ Build a HUD environment and generate lock file.

    This command:
    - Builds a Docker image from your environment
    - Analyzes the MCP server to extract metadata
    - Generates a hud.lock.yaml file for reproducibility

    Examples:
        hud build                    # Build current directory
        hud build environments/text_2048 -e API_KEY=secret
        hud build . --tag my-env:v1.0 -e VAR1=value1 -e VAR2=value2
        hud build . --no-cache       # Force rebuild
    """
    # Parse directory and extra arguments
    if params:
        directory = params[0]
        extra_args = params[1:] if len(params) > 1 else []
    else:
        directory = "."
        extra_args = []

    # Parse environment variables from extra args
    env_vars = {}
    i = 0
    while i < len(extra_args):
        if extra_args[i] == "-e" and i + 1 < len(extra_args):
            # Parse -e KEY=VALUE format
            env_arg = extra_args[i + 1]
            if "=" in env_arg:
                key, value = env_arg.split("=", 1)
                env_vars[key] = value
            i += 2
        elif extra_args[i].startswith("--env="):
            # Parse --env=KEY=VALUE format
            env_arg = extra_args[i][6:]  # Remove --env=
            if "=" in env_arg:
                key, value = env_arg.split("=", 1)
                env_vars[key] = value
            i += 1
        elif extra_args[i] == "--env" and i + 1 < len(extra_args):
            # Parse --env KEY=VALUE format
            env_arg = extra_args[i + 1]
            if "=" in env_arg:
                key, value = env_arg.split("=", 1)
                env_vars[key] = value
            i += 2
        else:
            i += 1

    build_command(directory, tag, no_cache, verbose, env_vars, platform)


@app.command()
def push(
    directory: str = typer.Argument(".", help="Environment directory containing hud.lock.yaml"),
    image: str | None = typer.Option(None, "--image", "-i", help="Override registry image name"),
    tag: str | None = typer.Option(
        None, "--tag", "-t", help="Override tag (e.g., 'v1.0', 'latest')"
    ),
    sign: bool = typer.Option(
        False, "--sign", help="Sign the image with cosign (not yet implemented)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """📤 Push HUD environment to registry.

    Reads hud.lock.yaml from the directory and pushes to registry.
    Auto-detects your Docker username if --image not specified.

    Examples:
        hud push                     # Push with auto-detected name
        hud push --tag v1.0          # Push with specific tag
        hud push . --image myuser/myenv:v1.0
        hud push --yes               # Skip confirmation
    """
    push_command(directory, image, tag, sign, yes, verbose)


@app.command()
def pull(
    target: str = typer.Argument(..., help="Image reference or lock file to pull"),
    lock_file: str | None = typer.Option(
        None, "--lock", "-l", help="Path to lock file (if target is image ref)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    verify_only: bool = typer.Option(
        False, "--verify-only", help="Only verify metadata without pulling"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """📥 Pull HUD environment from registry with metadata preview.

    Shows environment details before downloading.

    Examples:
        hud pull hud.lock.yaml               # Pull from lock file
        hud pull myuser/myenv:latest        # Pull by image reference
        hud pull myuser/myenv --verify-only # Check metadata only
    """
    pull_command(target, lock_file, yes, verify_only, verbose)


@app.command(name="list")
def list_environments(
    filter_name: str | None = typer.Option(
        None, "--filter", "-f", help="Filter environments by name (case-insensitive)"
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
    show_all: bool = typer.Option(False, "--all", "-a", help="Show all columns including digest"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """📋 List all HUD environments in local registry.

    Shows environments pulled with 'hud pull' stored in ~/.hud/envs/

    Examples:
        hud list                    # List all environments
        hud list --filter text      # Filter by name
        hud list --json            # Output as JSON
        hud list --all             # Show digest column
        hud list --verbose         # Show full descriptions
    """
    list_module.list_command(filter_name, json_output, show_all, verbose)


@app.command()
def remove(
    target: str | None = typer.Argument(
        None, help="Environment to remove (digest, name, or 'all' for all environments)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """🗑️ Remove HUD environments from local registry.

    Removes environment metadata from ~/.hud/envs/
    Note: This does not remove the Docker images.

    Examples:
        hud remove abc123              # Remove by digest
        hud remove text_2048           # Remove by name
        hud remove hudpython/test_init # Remove by full name
        hud remove all                 # Remove all environments
        hud remove all --yes           # Remove all without confirmation
    """
    remove_command(target, yes, verbose)


@app.command()
def init(
    name: str = typer.Argument(None, help="Environment name (default: current directory name)"),
    preset: str | None = typer.Option(
        None,
        "--preset",
        "-p",
        help="Preset to use: blank, deep-research, browser. If omitted, you'll choose interactively.",  # noqa: E501
    ),
    directory: str = typer.Option(".", "--dir", "-d", help="Target directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
) -> None:
    """🚀 Initialize a new HUD environment with minimal boilerplate.

    Creates a working MCP environment with:
    - Dockerfile for containerization
    - pyproject.toml for dependencies
    - Minimal MCP server with context
    - Required setup/evaluate tools

    Examples:
        hud init                    # Use current directory name
        hud init my-env             # Create in ./my-env/
        hud init my-env --dir /tmp  # Create in /tmp/my-env/
    """
    create_environment(name, directory, force, preset)


@app.command()
def quickstart() -> None:
    """
    Quickstart with evaluating an agent!
    """
    # Just call the clone command with the quickstart URL
    clone("https://github.com/hud-evals/quickstart.git")


@app.command()
def eval(
    source: str | None = typer.Argument(
        None,
        help=(
            "HuggingFace dataset (e.g. 'hud-evals/SheetBench-50') or task JSON file. "
            "If not provided, looks for task.json in current directory."
        ),
    ),
    agent: str | None = typer.Argument(
        None,
        help=(
            "Agent backend to use (claude, openai, vllm, or litellm). If not provided, will prompt interactively."  # noqa: E501
        ),
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Run the entire dataset (omit for single-task debug mode)",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model name for the chosen agent",
    ),
    allowed_tools: str | None = typer.Option(
        None,
        "--allowed-tools",
        help="Comma-separated list of allowed tools",
    ),
    max_concurrent: int = typer.Option(
        30,
        "--max-concurrent",
        help="Maximum concurrent tasks (1-200 recommended, prevents rate limits)",
    ),
    max_steps: int | None = typer.Option(
        None,
        "--max-steps",
        help="Maximum steps per task (default: 10 for single, 50 for full)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output from the agent",
    ),
    very_verbose: bool = typer.Option(
        False,
        "--very-verbose",
        "-vv",
        help="Enable debug-level logs for maximum visibility",
    ),
    vllm_base_url: str | None = typer.Option(
        None,
        "--vllm-base-url",
        help="Base URL for vLLM server (when using --agent vllm)",
    ),
    group_size: int = typer.Option(
        1,
        "--group-size",
        help="Number of times to run each task (similar to RL training)",
    ),
    integration_test: bool = typer.Option(
        False,
        "--integration-test",
        help=(
            "Run integration_test_tool, where problem is setup, "
            "actions are applied, and evaluation is performed, without "
            "spinning up an agent"
        ),
    ),
) -> None:
    """🚀 Run evaluation on datasets or individual tasks with agents."""
    from hud.settings import settings
    from hud.utils.hud_console import HUDConsole

    hud_console = HUDConsole()

    if integration_test:
        agent = AgentType.INTEGRATION_TEST

    # If no source provided, reuse RL helper to find a tasks file interactively
    if source is None:
        try:
            from hud.cli.utils.tasks import find_tasks_file

            source = find_tasks_file(None, msg="Select a tasks file to run")
            hud_console.success(f"Selected: {source}")
        except (FileNotFoundError, Exception):
            hud_console.error(
                "No source provided and no task/eval JSON files found in current directory"
            )
            hud_console.info(
                "Usage: hud eval <source> or create a task JSON file (e.g., task.json, tasks.jsonl)"
            )
            raise typer.Exit(1) from None

    # Import eval_command lazily to avoid importing agent dependencies
    try:
        from .eval import eval_command, get_available_models
    except ImportError as e:
        hud_console.error(
            "Evaluation dependencies are not installed. "
            "Please install with: pip install 'hud-python[agent]'"
        )
        raise typer.Exit(1) from e

    # If no agent specified, fetch available models and prompt for selection
    base_model = None
    if agent is None:
        # Get available HUD models first
        hud_models = get_available_models()

        # Build choices starting with HUD models
        choices = []

        # Add HUD models as agent choices
        for hud_model in hud_models:
            model_name = hud_model["name"]
            base_model = hud_model["base_model"]
            vllm_status = " ⚡" if hud_model.get("vllm_url") else ""
            choices.append({"name": f"{model_name}{vllm_status}", "value": f"{model_name}"})

        # Add standard agent choices
        choices.extend(
            [
                {"name": "Claude 4 Sonnet", "value": AgentType.CLAUDE},
                {"name": "OpenAI Computer Use", "value": AgentType.OPENAI},
                {"name": "vLLM (Local Server)", "value": AgentType.VLLM},
                {"name": "LiteLLM (Multi-provider)", "value": AgentType.LITELLM},
            ]
        )

        agent = hud_console.select("Select an agent to use:", choices=choices, default=0)

    # Handle HUD model selection
    if agent and agent not in [e.value for e in AgentType]:
        # Find remote model name
        model = agent
        if not vllm_base_url:
            vllm_base_url = f"{settings.hud_rl_url}/models/{model}/vllm"

        # Set model to base model for the vllm endpoint
        if not base_model:
            hud_models = get_available_models()
            for hud_model in hud_models:
                if hud_model["name"] == model:
                    base_model = hud_model["base_model"]
                    break
        if not base_model:
            hud_console.error(f"Model {model} not found")
            raise typer.Exit(1)
        model = base_model
        agent = AgentType.VLLM  # Use vLLM backend for HUD models
        hud_console.info(f"Using HUD model: {model} (trained on {base_model})")

    # Validate agent choice
    valid_agents = [e.value for e in AgentType]
    if agent not in valid_agents:
        hud_console.error(f"Invalid agent: {agent}. Must be one of: {', '.join(valid_agents)}")
        raise typer.Exit(1)

    # Type narrowing: agent is now guaranteed to be an AgentType value after validation
    agent = AgentType(agent)

    # Run the command
    eval_command(
        source=source,
        full=full,
        agent=agent,
        model=model,
        allowed_tools=allowed_tools,
        max_concurrent=max_concurrent,
        max_steps=max_steps,
        verbose=verbose,
        very_verbose=very_verbose,
        vllm_base_url=vllm_base_url,
        group_size=group_size,
        integration_test=integration_test,
    )


@app.command()
def get(
    dataset_name: str = typer.Argument(
        ..., help="HuggingFace dataset name (e.g., 'hud-evals/browser-2048-tasks')"
    ),
    split: str = typer.Option(
        "train", "--split", "-s", help="Dataset split to download (train/test/validation)"
    ),
    output: Path | None = typer.Option(  # noqa: B008
        None, "--output", "-o", help="Output filename (defaults to dataset_name.jsonl)"
    ),
    limit: int | None = typer.Option(
        None, "--limit", "-l", help="Limit number of examples to download"
    ),
    format: str = typer.Option(
        "json",
        "--format",
        "-f",
        help="Output format: json (list) or jsonl (one task per line)",
    ),
) -> None:
    """📥 Download a HuggingFace dataset and save it as JSONL."""
    from hud.cli.get import get_command

    get_command(
        dataset_name=dataset_name,
        split=split,
        output=output,
        limit=limit,
        format=format,
    )


@app.command()
def rl(
    tasks_file: str | None = typer.Argument(
        None,
        help=(
            "Path to tasks file (JSON/JSONL) or HuggingFace dataset name. "
            "If not provided, looks for tasks.json or tasks.jsonl in current directory."
        ),
    ),
    model: str | None = typer.Argument(
        None,
        help="Model to train from https://hud.so/models (default: interactive selection)",
    ),
    config_file: Path | None = typer.Option(  # noqa: B008
        None,
        "--config",
        "-c",
        help="Path to existing configuration file",
    ),
    output_dir: str = typer.Option(
        "checkpoints",
        "--output-dir",
        "-o",
        help="Output directory for checkpoints",
    ),
    restart: bool = typer.Option(
        False,
        "--restart",
        help="Restart the vLLM server before training",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose output",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Run training locally instead of using remote API server",
    ),
    no_ddp: bool = typer.Option(
        False,
        "--no-ddp",
        help="Disable DDP even with multiple GPUs",
    ),
    ddp_gpus: str | None = typer.Option(
        None,
        "--ddp-gpus",
        help="Specific GPUs for DDP (e.g., '0,1,2,3')",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Auto-accept all prompts and use defaults (lazy mode)",
    ),
    vllm_gpu: int | None = typer.Option(
        None,
        "--vllm-gpu",
        help="Specific GPU for vLLM server",
    ),
    vllm_gpu_count: int = typer.Option(
        1,
        "--vllm-gpu-count",
        help="Number of GPUs for vLLM server",
    ),
    skip_vllm_startup: bool = typer.Option(
        False,
        "--skip-vllm-startup",
        help="Skip the vLLM server startup",
    ),
) -> None:
    """🎯 Run GRPO reinforcement learning training on tasks."""
    # Import from the rl module
    from .rl import rl_command

    rl_command(
        tasks_file=tasks_file,
        model=model,
        config_file=config_file,
        output_dir=output_dir,
        restart=restart,
        verbose=verbose,
        local=local,
        no_ddp=no_ddp,
        ddp_gpus=ddp_gpus,
        vllm_gpu=vllm_gpu,
        vllm_gpu_count=vllm_gpu_count,
        yes=yes,
        skip_vllm_startup=skip_vllm_startup,
    )


@app.command()
def set(
    assignments: list[str] = typer.Argument(  # type: ignore[arg-type]  # noqa: B008
        ..., help="One or more KEY=VALUE pairs to persist in ~/.hud/.env"
    ),
) -> None:
    """Persist API keys or other variables for HUD to use by default.

    Examples:
        hud set ANTHROPIC_API_KEY=sk-... OPENAI_API_KEY=sk-...

    Values are stored in ~/.hud/.env and are loaded by hud.settings with
    the lowest precedence (overridden by process env and project .env).
    """
    from hud.utils.hud_console import HUDConsole

    hud_console = HUDConsole()

    updates: dict[str, str] = {}
    for item in assignments:
        if "=" not in item:
            hud_console.error(f"Invalid assignment (expected KEY=VALUE): {item}")
            raise typer.Exit(1)
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            hud_console.error(f"Invalid key in assignment: {item}")
            raise typer.Exit(1)
        updates[key] = value

    path = set_env_values(updates)
    hud_console.success("Saved credentials to user config")
    hud_console.info(f"Location: {path}")


def main() -> None:
    """Main entry point for the CLI."""
    # Check for updates (including on --version command)
    # Skip only on help-only commands
    if not (len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"])):
        from .utils.version_check import display_update_prompt

        display_update_prompt()

    # Handle --version flag before Typer parses args
    if "--version" in sys.argv:
        try:
            from hud import __version__

            console.print(f"HUD CLI version: [cyan]{__version__}[/cyan]")
        except ImportError:
            console.print("HUD CLI version: [cyan]unknown[/cyan]")
        return

    try:
        # Show header for main help
        if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"]):
            console.print(
                Panel.fit(
                    "[bold cyan]🚀 HUD CLI[/bold cyan]\nMCP Environment Analysis & Debugging",
                    border_style="cyan",
                )
            )
            console.print("\n[yellow]Quick Start:[/yellow]")
            console.print(
                "  1. Create a new environment: [cyan]hud init my-env && cd my-env[/cyan]"
            )
            console.print("  2. Develop with hot-reload: [cyan]hud dev --interactive[/cyan]")
            console.print("  3. Build for production: [cyan]hud build[/cyan]")
            console.print("  4. Share your environment: [cyan]hud push[/cyan]")
            console.print("  5. Get shared environments: [cyan]hud pull <org/name:tag>[/cyan]")
            console.print("  6. Run and test: [cyan]hud run <image>[/cyan]")
            console.print("\n[yellow]Datasets & RL Training:[/yellow]")
            console.print("  1. Get dataset: [cyan]hud get hud-evals/browser-2048-tasks[/cyan]")
            console.print(
                "  2. Create dataset: [cyan]hud hf tasks.json --name my-org/my-tasks[/cyan]"
            )
            console.print(
                "  3. Start training: [cyan]hud rl browser-2048-tasks.jsonl --local[/cyan]"
            )
            console.print(
                "  4. Custom model: [cyan]hud rl tasks.jsonl --model meta-llama/Llama-3.2-3B --local[/cyan]"  # noqa: E501
            )
            console.print(
                "  5. Restart server: [cyan]hud rl tasks.jsonl --restart --local[/cyan]\n"
            )

        app()
    except typer.Exit as e:
        # Append SDK support hint for non-zero exits
        try:
            exit_code = getattr(e, "exit_code", 0)
        except Exception:
            exit_code = 1
        if exit_code != 0:
            from hud.utils.hud_console import hud_console

            hud_console.info(SUPPORT_HINT)
        raise
    except Exception:
        raise


if __name__ == "__main__":
    main()

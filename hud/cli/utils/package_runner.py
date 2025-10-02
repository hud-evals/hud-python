"""Run Python modules or commands as MCP servers.

This module handles direct execution of MCP servers, including:
- Python modules with an 'mcp' attribute
- External commands via FastMCP proxy
- Auto-reload functionality for development

For Docker container execution, see hud dev command.
"""

from __future__ import annotations

import importlib
import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

logger = logging.getLogger(__name__)


async def run_package_as_mcp(
    command: str | list[str],
    transport: str = "stdio",
    port: int = 8765,
    verbose: bool = False,
    reload: bool = False,
    watch_paths: list[str] | None = None,
    server_attr: str = "mcp",
    **extra_kwargs: Any,
) -> None:
    """Run a command as an MCP server.

    Can run:
    - Python modules: 'controller' (imports and looks for mcp attribute)
    - Python -m commands: 'python -m controller'
    - Docker commands: 'docker run -it my-mcp-server'
    - Any executable: './my-mcp-binary'

    Args:
        command: Command to run (string or list)
        transport: Transport type ("stdio" or "http")
        port: Port for HTTP transport
        verbose: Enable verbose logging
        reload: Enable auto-reload on file changes
        watch_paths: Paths to watch for changes (defaults to ['.'])
        **extra_kwargs: Additional arguments
    """
    # Set up logging
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Handle reload mode
    if reload:
        if watch_paths is None:
            watch_paths = ["."]

        # Detect external command vs module reliably.
        # If command is a string and contains spaces (e.g., "uv run python -m controller")
        # treat as external command. Otherwise, detect common launchers or paths.
        is_external_cmd = False
        if isinstance(command, list):
            is_external_cmd = True
        elif isinstance(command, str):
            stripped = command.strip()
            if " " in stripped or any(
                stripped.startswith(x)
                for x in ["python", "uv ", "docker", "./", "/", ".\\", "C:\\"]
            ):
                is_external_cmd = True

        if is_external_cmd:
            # External command - pass command list directly
            cmd_list = shlex.split(command) if isinstance(command, str) else command
            run_with_reload(cmd_list, watch_paths, verbose)
        else:
            # Python module - use sys.argv approach
            run_with_reload(None, watch_paths, verbose)
        return

    # Determine if it's a module import or a command
    if isinstance(command, str) and not any(
        command.startswith(x) for x in ["python", "docker", "./", "/", ".\\", "C:\\"]
    ):
        # Treat as Python module for backwards compatibility
        logger.info("Importing module: %s", command)
        module = importlib.import_module(command)

        # Look for server attribute in the module
        if not hasattr(module, server_attr):
            logger.error(
                "Module '%s' does not have an '%s' attribute (MCPServer instance)",
                command,
                server_attr,
            )
            sys.exit(1)

        server = getattr(module, server_attr)

        # Configure server options
        run_kwargs = {
            "transport": transport,
            "show_banner": False,
        }

        if transport == "http":
            # FastMCP expects port/path directly
            run_kwargs["port"] = port
            run_kwargs["path"] = "/mcp"

        # Merge any extra kwargs
        run_kwargs.update(extra_kwargs)

        # Run the server
        logger.info("Running %s on %s transport", server.name, transport)
        await server.run_async(**run_kwargs)
    else:
        # Run as external command using shared proxy utility
        # Parse command if string
        cmd_list = shlex.split(command) if isinstance(command, str) else command

        # Replace 'python' with the current interpreter to preserve venv
        if cmd_list[0] == "python":
            cmd_list[0] = sys.executable
            logger.info("Replaced 'python' with: %s", sys.executable)

        logger.info("Running command: %s", " ".join(cmd_list))

        # Create MCP config for the command
        config = {
            "mcpServers": {
                "default": {
                    "command": cmd_list[0],
                    "args": cmd_list[1:] if len(cmd_list) > 1 else [],
                    # transport defaults to stdio
                }
            }
        }

        # Create proxy server
        proxy = FastMCP.as_proxy(config, name=f"HUD Run - {cmd_list[0]}")

        # Run the proxy
        await proxy.run_async(
            transport=transport if transport == "http" or transport == "stdio" else None,
            port=port if transport == "http" else None,
            show_banner=False,
            **extra_kwargs,
        )


def run_with_reload(
    target_func: Any,
    watch_paths: list[str],
    verbose: bool = False,
) -> None:
    """Run a function or command with file watching and auto-reload.

    Args:
        target_func: Function to run (sync) or command list
        watch_paths: Paths to watch for changes
        verbose: Enable verbose logging
    """
    try:
        import watchfiles
    except ImportError:
        logger.error("watchfiles is required for --reload. Install with: pip install watchfiles")
        sys.exit(1)

    # Resolve watch paths
    resolved_paths = []
    for path_str in watch_paths:
        path = Path(path_str).resolve()
        if path.is_file():
            # Watch the directory containing the file
            resolved_paths.append(str(path.parent))
        else:
            resolved_paths.append(str(path))

    def run_and_restart() -> None:
        """Run the target function in a loop, restarting on file changes."""

        process = None

        def handle_signal(signum: int, frame: Any) -> None:
            """Handle signals by terminating the subprocess."""
            if process:
                process.terminate()
            sys.exit(0)

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

        stop_event = threading.Event()  # Define stop_event at the start

        while True:
            # Run the target function or command
            if target_func is None:
                # Use sys.argv approach for Python modules
                child_args = [a for a in sys.argv[1:] if a != "--reload"]
                # If first arg is already 'run', don't inject it again
                if child_args and child_args[0] == "run":
                    cmd = [sys.executable, "-m", "hud", *child_args]
                else:
                    cmd = [sys.executable, "-m", "hud", "run", *child_args]
            elif isinstance(target_func, list):
                # It's a command list
                cmd = target_func
            else:
                # It's a callable - run it directly
                target_func()
                # Wait for file changes before restarting
                stop_event.wait()
                continue

            if verbose:
                logger.info("Starting process: %s", " ".join(cmd))

            process = subprocess.Popen(cmd, env=os.environ)  # noqa: S603

            # Watch for changes
            try:
                # Use a proper threading.Event for stop_event as required by watchfiles
                stop_event = threading.Event()

                def _wait_and_set(
                    stop_event: threading.Event, process: subprocess.Popen[bytes]
                ) -> None:
                    try:
                        if process is not None:
                            process.wait()
                    finally:
                        stop_event.set()

                threading.Thread(
                    target=_wait_and_set, args=(stop_event, process), daemon=True
                ).start()

                for changes in watchfiles.watch(*resolved_paths, stop_event=stop_event):
                    logger.info("Raw changes detected: %s", changes)
                    # Filter for relevant file types
                    relevant_changes = [
                        (change_type, path)
                        for change_type, path in changes
                        if any(path.endswith(ext) for ext in [".py", ".json", ".toml", ".yaml"])
                        and "__pycache__" not in path
                        and not Path(path).name.startswith(".")
                    ]

                    if relevant_changes:
                        logger.info("File changes detected, restarting server...")
                        if verbose:
                            for change_type, path in relevant_changes:
                                logger.debug("  %s: %s", change_type, path)

                        # Terminate the process
                        if process is not None:
                            process.terminate()
                        try:
                            if process is not None:
                                process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            if process is not None:
                                process.kill()
                                process.wait()

                        # Brief pause before restart
                        time.sleep(0.1)
                        break
                    else:
                        logger.debug("Changes detected but filtered out: %s", changes)
            except KeyboardInterrupt:
                # Handle Ctrl+C gracefully
                if process:
                    process.terminate()
                    process.wait()
                break

    # Always act as the parent. The child is launched without --reload,
    # so it won't re-enter this function.

    run_and_restart()

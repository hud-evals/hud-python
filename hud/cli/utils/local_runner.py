"""Run local Python files as MCP servers with reload support."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import subprocess
import sys
from pathlib import Path

from hud.utils.hud_console import HUDConsole

hud_console = HUDConsole()


async def run_with_reload(
    server_file: str,
    transport: str = "stdio",
    port: int | None = None,
    verbose: bool = False,
    extra_args: list[str] | None = None,
) -> None:
    """Run Python server with auto-reload functionality."""
    try:
        import watchfiles
    except ImportError:
        hud_console.error(
            "watchfiles is required for --reload. Install with: pip install watchfiles"
        )
        sys.exit(1)

    # Parse server file path
    if ":" in server_file:
        file_path, _ = server_file.split(":", 1)
    else:
        file_path = server_file

    file_path = Path(file_path).resolve()
    if not file_path.exists():
        hud_console.error(f"Server file not found: {file_path}")
        sys.exit(1)

    # Watch the directory containing the server file (like uvicorn)
    watch_dir = file_path.parent

    # Build command
    cmd = [sys.executable, "-m", "fastmcp.cli", "run", server_file]
    if transport:
        cmd.extend(["--transport", transport])
    if port and transport == "http":
        cmd.extend(["--port", str(port)])
    cmd.append("--no-banner")
    if verbose:
        cmd.extend(["--log-level", "DEBUG"])
    if extra_args:
        cmd.append("--")
        cmd.extend(extra_args)

    # Filter for Python files and important config files
    def should_reload(change: watchfiles.Change, path: str) -> bool:
        path_obj = Path(path)
        # Ignore common non-code files
        if any(part.startswith(".") for part in path_obj.parts):
            return False
        if "__pycache__" in path_obj.parts:
            return False
        return path_obj.suffix in {".py", ".json", ".toml", ".yaml", ".yml"}

    process = None

    async def run_server() -> int:
        """Run the server process."""
        nonlocal process

        # For stdio transport, we need special handling to preserve stdout
        if transport == "stdio":
            # All server logs must go to stderr
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=sys.stdin,
                stdout=sys.stdout,  # Direct passthrough for MCP
                stderr=sys.stderr,  # Logs and errors
                env=env,
            )
        else:
            # For HTTP transport, normal subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=sys.stdin,
                stdout=sys.stdout,
                stderr=sys.stderr,
            )

        return await process.wait()

    async def stop_server() -> None:
        """Stop the server process gracefully."""
        if process and process.returncode is None:
            if sys.platform == "win32":
                # Windows: terminate directly
                process.terminate()
            else:
                # Unix: send SIGINT for hot reload
                process.send_signal(signal.SIGINT)

            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(process.wait(), timeout=5.0)
            except TimeoutError:
                # Force kill if not responding
                if verbose:
                    hud_console.warning("Server didn't stop gracefully, forcing shutdown...")
                process.kill()
                await process.wait()

    # Initial server start
    server_task = asyncio.create_task(run_server())

    try:
        # Watch for file changes
        async for changes in watchfiles.awatch(watch_dir):
            # Check if any change should trigger reload
            if any(should_reload(change, path) for change, path in changes):
                changed_files = [
                    path for _, path in changes if should_reload(watchfiles.Change.modified, path)
                ]
                if verbose:
                    for file in changed_files[:3]:  # Show first 3 files
                        hud_console.info(f"File changed: {Path(file).relative_to(watch_dir)}")
                    if len(changed_files) > 3:
                        hud_console.info(f"... and {len(changed_files) - 3} more files")

                hud_console.info("🔄 Reloading server...")

                # Stop current server
                await stop_server()

                # Small delay to ensure clean restart
                await asyncio.sleep(0.1)

                # Start new server
                server_task = asyncio.create_task(run_server())

    except KeyboardInterrupt:
        hud_console.info("\n👋 Shutting down...")
        await stop_server()
    except Exception as e:
        if verbose:
            hud_console.error(f"Reload error: {e}")
        await stop_server()
        raise
    finally:
        # Ensure server is stopped
        if server_task and not server_task.done():
            server_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await server_task


def run_local_server(
    server_file: str,
    transport: str = "stdio",
    port: int | None = None,
    verbose: bool = False,
    reload: bool = False,
    extra_args: list[str] | None = None,
) -> None:
    """Run a local Python file as an MCP server."""
    if reload:
        # Run with reload support
        asyncio.run(
            run_with_reload(
                server_file,
                transport=transport,
                port=port,
                verbose=verbose,
                extra_args=extra_args,
            )
        )
    else:
        # Run directly without reload
        cmd = [sys.executable, "-m", "fastmcp.cli", "run", server_file]
        if transport:
            cmd.extend(["--transport", transport])
        if port and transport == "http":
            cmd.extend(["--port", str(port)])
        cmd.append("--no-banner")
        if verbose:
            cmd.extend(["--log-level", "DEBUG"])
        if extra_args:
            cmd.append("--")
            cmd.extend(extra_args)

        try:
            result = subprocess.run(cmd)  # noqa: S603
            sys.exit(result.returncode)
        except KeyboardInterrupt:
            hud_console.info("\n👋 Shutting down...")
            sys.exit(0)

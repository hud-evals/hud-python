"""Common server utilities for HUD CLI."""

from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import FastMCP

from hud.utils.hud_console import HUDConsole

from .docker import generate_container_name, remove_container


class MCPServerManager:
    """Manages MCP server lifecycle and configuration."""

    def __init__(self, image: str, docker_args: list[str] | None = None) -> None:
        """Initialize server manager.

        Args:
            image: Docker image name
            docker_args: Additional Docker arguments
        """
        self.image = image
        self.docker_args = docker_args or []
        self.console = HUDConsole()
        self.container_name = self._generate_container_name()

    def _generate_container_name(self) -> str:
        """Generate a unique container name from image."""
        return generate_container_name(self.image)

    def cleanup_container(self) -> None:
        """Remove any existing container with the same name."""
        remove_container(self.container_name)

    def build_docker_command(
        self,
        extra_args: list[str] | None = None,
        entrypoint: list[str] | None = None,
    ) -> list[str]:
        """Build Docker run command.

        Args:
            extra_args: Additional arguments to add before image
            entrypoint: Custom entrypoint override

        Returns:
            Complete docker command as list
        """
        cmd = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--name",
            self.container_name,
        ]

        # Add extra args (like volume mounts, env vars)
        if extra_args:
            cmd.extend(extra_args)

        # Add user-provided docker args
        cmd.extend(self.docker_args)

        # Add entrypoint if specified
        if entrypoint:
            cmd.extend(["--entrypoint", entrypoint[0]])

        # Add image
        cmd.append(self.image)

        # Add entrypoint args if specified
        if entrypoint and len(entrypoint) > 1:
            cmd.extend(entrypoint[1:])

        return cmd

    def create_mcp_config(self, docker_cmd: list[str]) -> dict[str, Any]:
        """Create MCP configuration for stdio transport.

        Args:
            docker_cmd: Docker command to run

        Returns:
            MCP configuration dict
        """
        return {
            "mcpServers": {
                "default": {
                    "command": docker_cmd[0],
                    "args": docker_cmd[1:] if len(docker_cmd) > 1 else [],
                    # transport defaults to stdio
                }
            }
        }

    def create_proxy(self, config: dict[str, Any], name: str | None = None) -> FastMCP:
        """Create FastMCP proxy server.

        Args:
            config: MCP configuration
            name: Optional server name

        Returns:
            FastMCP proxy instance
        """
        proxy_name = name or f"HUD Server - {self.image}"
        return FastMCP.as_proxy(config, name=proxy_name)

    async def run_http_server(
        self,
        proxy: FastMCP,
        port: int,
        verbose: bool = False,
        path: str = "/mcp",
    ) -> None:
        """Run HTTP server with proper shutdown handling.

        Args:
            proxy: FastMCP proxy instance
            port: Port to listen on
            verbose: Enable verbose logging
            path: URL path for MCP endpoint
        """
        # Set up logging
        import logging
        import os

        os.environ["FASTMCP_DISABLE_BANNER"] = "1"

        if not verbose:
            logging.getLogger("fastmcp").setLevel(logging.ERROR)
            logging.getLogger("mcp").setLevel(logging.ERROR)
            logging.getLogger("uvicorn").setLevel(logging.ERROR)
            logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
            logging.getLogger("uvicorn.error").setLevel(logging.ERROR)

            import warnings

            warnings.filterwarnings("ignore", category=DeprecationWarning)

        try:
            await proxy.run_async(
                transport="http",
                host="0.0.0.0",  # noqa: S104
                port=port,
                path=path,
                log_level="error" if not verbose else "info",
                show_banner=False,
            )
        except asyncio.CancelledError:
            pass  # Normal cancellation
        except Exception as e:
            if verbose:
                self.console.error(f"Server error: {e}")
            raise


async def run_server_with_interactive(
    server_manager: MCPServerManager,
    port: int,
    verbose: bool = False,
) -> None:
    """Run server with interactive testing mode.

    Args:
        server_manager: Server manager instance
        port: Port to listen on
        verbose: Enable verbose logging
    """
    from .interactive import run_interactive_mode
    from .logging import find_free_port

    hud_console = HUDConsole()

    # Find available port
    actual_port = find_free_port(port)
    if actual_port is None:
        hud_console.error(f"No available ports found starting from {port}")
        return

    if actual_port != port:
        hud_console.warning(f"Port {port} in use, using port {actual_port} instead")

    # Clean up any existing container
    server_manager.cleanup_container()

    # Build docker command
    docker_cmd = server_manager.build_docker_command()

    # Create MCP config
    config = server_manager.create_mcp_config(docker_cmd)

    # Create proxy
    proxy = server_manager.create_proxy(config, f"HUD Interactive - {server_manager.image}")

    # Show header
    hud_console.info("")  # Empty line
    hud_console.header("HUD MCP Server - Interactive Mode", icon="🎮")

    # Show configuration
    hud_console.section_title("Server Information")
    hud_console.info(f"Image: {server_manager.image}")
    hud_console.info(f"Port: {actual_port}")
    hud_console.info(f"URL: http://localhost:{actual_port}/mcp")
    hud_console.info(f"Container: {server_manager.container_name}")
    hud_console.info("")

    # Create event to signal server is ready
    server_ready = asyncio.Event()
    server_task = None

    async def start_server() -> None:
        """Start the proxy server."""
        nonlocal server_task
        try:
            # Signal that we're ready before starting
            server_ready.set()
            await server_manager.run_http_server(proxy, actual_port, verbose)
        except asyncio.CancelledError:
            pass

    try:
        # Start server in background
        server_task = asyncio.create_task(start_server())

        # Wait for server to be ready
        await server_ready.wait()
        await asyncio.sleep(0.5)  # Give it a moment to fully start

        # Run interactive mode
        server_url = f"http://localhost:{actual_port}/mcp"
        await run_interactive_mode(server_url, verbose=verbose)

    except KeyboardInterrupt:
        hud_console.info("\n👋 Shutting down...")
    finally:
        # Cancel server task
        if server_task and not server_task.done():
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                hud_console.error("Server task cancelled")

        # Clean up container
        server_manager.cleanup_container()

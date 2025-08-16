"""MCP Server mode for HUD CLI - exposes debug and analyze as MCP tools."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

from .cursor import parse_cursor_config
from .debug import debug_mcp_stdio
from .utils import CaptureLogger


def create_mcp_server() -> FastMCP:
    """Create and configure the HUD MCP server."""
    mcp = FastMCP(
        name="hud-cli",
    )

    @mcp.tool()
    async def debug_docker_image(
        image: str, docker_args: list[str] | None = None, max_phase: int = 5
    ) -> list[TextContent]:
        """
        Debug a Docker-based MCP environment.

        Args:
            image: Docker image name (e.g., 'hud-text-2048:latest')
            docker_args: Additional docker arguments (e.g., ['-e', 'KEY=value'])
            max_phase: Maximum phase to run (1-5, default 5)

        Returns:
            Debug output showing test phases
        """
        # Build docker command
        command = ["docker", "run", "--rm", "-i"] + (docker_args or []) + [image]

        # Create logger in capture mode
        logger = CaptureLogger(print_output=False)

        # Run debug
        phases_completed = await debug_mcp_stdio(command, logger, max_phase=max_phase)

        # Add summary
        output = logger.get_output()
        output += f"\n\n✅ Completed {phases_completed}/{max_phase} phases successfully"

        return [TextContent(text=output, type="text")]

    @mcp.tool()
    async def debug_cursor_config(server_name: str, max_phase: int = 5) -> list[TextContent]:
        """
        Debug a server from Cursor's MCP configuration.

        Args:
            server_name: Name of server in .cursor/mcp.json
            max_phase: Maximum phase to run (1-5, default 5)

        Returns:
            Debug output showing test phases
        """
        # Parse cursor config
        command, error = parse_cursor_config(server_name)

        if error or command is None:
            return [TextContent(text=f"❌ {error or 'Failed to parse cursor config'}", type="text")]

        # Create logger in capture mode
        logger = CaptureLogger(print_output=False)

        # Run debug
        phases_completed = await debug_mcp_stdio(command, logger, max_phase=max_phase)

        # Add summary
        output = logger.get_output()
        output += f"\n\n✅ Completed {phases_completed}/{max_phase} phases successfully"

        return [TextContent(text=output, type="text")]

    @mcp.tool()
    async def debug_config(config: dict, max_phase: int = 5) -> list[TextContent]:
        """
        Debug an MCP environment from a configuration object.

        Args:
            config: MCP configuration dict with server definitions
            max_phase: Maximum phase to run (1-5, default 5)

        Returns:
            Debug output showing test phases
        """
        # Extract command from first server in config
        server_name = next(iter(config.keys()))
        server_config = config[server_name]
        command = [server_config["command"], *server_config.get("args", [])]

        # Create logger in capture mode
        logger = CaptureLogger(print_output=False)

        # Run debug
        phases_completed = await debug_mcp_stdio(command, logger, max_phase=max_phase)

        # Add summary
        output = logger.get_output()
        output += f"\n\n✅ Completed {phases_completed}/{max_phase} phases successfully"

        return [TextContent(text=output, type="text")]

    @mcp.tool()
    async def analyze_docker_image(
        image: str, docker_args: list[str] | None = None, verbose: bool = False
    ) -> list[TextContent]:
        """
        Analyze a Docker-based MCP environment to discover tools and resources.
        Note: The environment must pass debug phase 3 (Tool Discovery) for this to work.

        Args:
            image: Docker image name (e.g., 'hud-text-2048:latest')
            docker_args: Additional docker arguments (e.g., ['-e', 'KEY=value'])
            verbose: Include detailed tool schemas

        Returns:
            Analysis results as JSON
        """
        # Build docker command
        docker_cmd = ["docker", "run", "--rm", "-i"] + (docker_args or []) + [image]

        # Convert to MCP config
        mcp_config = {
            "local": {
                "command": docker_cmd[0],
                "args": docker_cmd[1:] if len(docker_cmd) > 1 else [],
            }
        }

        try:
            # Note: This is a bit of a hack - we're calling the internal function
            # In a real implementation, we'd refactor to have a shared core function
            from hud.clients import MCPClient

            client = MCPClient(mcp_config=mcp_config, verbose=verbose)
            await client.initialize()
            analysis = await client.analyze_environment()
            await client.close()

            # Return as JSON
            return [TextContent(text=json.dumps(analysis, indent=2), type="text")]

        except Exception as e:
            return [
                TextContent(
                    text=f"❌ Analysis failed: {e}\n\nMake sure the environment passes debug phase 3 first.",  # noqa: E501
                    type="text",
                )
            ]

    @mcp.tool()
    async def analyze_cursor_config(server_name: str, verbose: bool = False) -> list[TextContent]:
        """
        Analyze a server from Cursor's MCP configuration.
        Note: The environment must pass debug phase 3 (Tool Discovery) for this to work.

        Args:
            server_name: Name of server in .cursor/mcp.json
            verbose: Include detailed tool schemas

        Returns:
            Analysis results as JSON
        """
        # Parse cursor config
        command, error = parse_cursor_config(server_name)

        if error or command is None:
            return [TextContent(text=f"❌ {error or 'Failed to parse cursor config'}", type="text")]

        # Convert to MCP config
        mcp_config = {
            "local": {"command": command[0], "args": command[1:] if len(command) > 1 else []}
        }

        try:
            from hud.clients import MCPClient

            client = MCPClient(mcp_config=mcp_config, verbose=verbose)
            await client.initialize()
            analysis = await client.analyze_environment()
            await client.close()

            # Return as JSON
            return [TextContent(text=json.dumps(analysis, indent=2), type="text")]

        except Exception as e:
            return [
                TextContent(
                    text=f"❌ Analysis failed: {e}\n\nMake sure the environment passes debug phase 3 first.",  # noqa: E501
                    type="text",
                )
            ]

    @mcp.tool()
    async def analyze_config(config: dict, verbose: bool = False) -> list[TextContent]:
        """
        Analyze an MCP environment from a configuration object.
        Note: The environment must pass debug phase 3 (Tool Discovery) for this to work.

        Args:
            config: MCP configuration dict with server definitions
            verbose: Include detailed tool schemas

        Returns:
            Analysis results as JSON
        """
        try:
            from hud.clients import MCPClient

            client = MCPClient(mcp_config=config, verbose=verbose)
            await client.initialize()
            analysis = await client.analyze_environment()
            await client.close()

            # Return as JSON
            return [TextContent(text=json.dumps(analysis, indent=2), type="text")]

        except Exception as e:
            return [
                TextContent(
                    text=f"❌ Analysis failed: {e}\n\nMake sure the environment passes debug phase 3 first.",  # noqa: E501
                    type="text",
                )
            ]

    @mcp.tool()
    async def list_cursor_servers() -> list[TextContent]:
        """
        List all MCP servers configured in Cursor.

        Returns:
            List of available server names
        """
        from .cursor import list_cursor_servers as _list_cursor_servers

        servers, error = _list_cursor_servers()

        if error is not None:
            return [TextContent(text=f"❌ {error}", type="text")]

        if not servers:
            return [TextContent(text="No servers found in Cursor config", type="text")]

        # Format as a nice list
        output = "📋 Available Cursor MCP Servers:\n\n"
        for server in servers:
            output += f"  • {server}\n"

        return [TextContent(text=output, type="text")]

    return mcp


def run_mcp_server() -> None:
    """Run the HUD MCP server."""
    mcp = create_mcp_server()
    mcp.run()

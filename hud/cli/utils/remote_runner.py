"""Remote runner for HUD MCP servers."""

from __future__ import annotations

import asyncio
import os
import sys

import click
from fastmcp import FastMCP

from hud.settings import settings


def parse_headers(header_args: list[str]) -> dict[str, str]:
    """Parse header arguments into a dictionary.

    Args:
        header_args: List of header strings in format "Key:Value" or "Key=Value"

    Returns:
        Dictionary of headers
    """
    headers = {}
    for header in header_args:
        if ":" in header:
            key, value = header.split(":", 1)
        elif "=" in header:
            key, value = header.split("=", 1)
        else:
            click.echo(f"⚠️  Invalid header format: {header} (use Key:Value or Key=Value)")
            continue

        headers[key.strip()] = value.strip()

    return headers


def parse_env_vars(env_args: list[str]) -> dict[str, str]:
    """Parse environment variable arguments into headers.

    Args:
        env_args: List of env var strings in format "KEY=VALUE"

    Returns:
        Dictionary of headers with Env- prefix
    """
    env_headers = {}
    for env in env_args:
        if "=" not in env:
            click.echo(f"⚠️  Invalid env format: {env} (use KEY=VALUE)")
            continue

        key, value = env.split("=", 1)
        # Convert KEY_NAME to Env-Key-Name header format
        # e.g., API_KEY=xxx becomes Env-Api-Key: xxx
        # e.g., OPENAI_API_KEY=xxx becomes Env-Openai-Api-Key: xxx
        header_parts = key.split("_")
        header_key = f"Env-{'-'.join(part.capitalize() for part in header_parts)}"
        env_headers[header_key] = value

    return env_headers


def build_remote_headers(
    image: str,
    env_args: list[str],
    header_args: list[str],
    api_key: str | None = None,
    run_id: str | None = None,
) -> dict[str, str]:
    """Build headers for remote MCP server.

    Args:
        image: Docker image name
        env_args: Environment variable arguments
        header_args: Additional header arguments
        api_key: API key (from env or arg)
        run_id: Run ID (optional)

    Returns:
        Complete headers dictionary
    """
    headers = {}

    # Required headers
    headers["Mcp-Image"] = image

    # API key
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Run ID if provided
    if run_id:
        headers["Run-Id"] = run_id

    # Environment variables as headers
    env_headers = parse_env_vars(env_args)
    headers.update(env_headers)

    # Additional headers
    extra_headers = parse_headers(header_args)
    headers.update(extra_headers)

    return headers


def run_remote_stdio(
    url: str,
    headers: dict[str, str],
    verbose: bool = False,
) -> None:
    """Run remote MCP server with stdio transport."""
    # CRITICAL: Configure ALL output to go to stderr to keep stdout clean for MCP protocol
    import logging
    import warnings

    # Force all output to stderr
    sys.stdout = sys.stderr

    # Always disable FastMCP banner for stdio
    os.environ["FASTMCP_DISABLE_BANNER"] = "1"

    # Configure root logger to use stderr
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    if not verbose:
        # Suppress all logs and warnings for clean stdio
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.CRITICAL)
        root_logger.addHandler(stderr_handler)
        root_logger.setLevel(logging.CRITICAL)

        # Set all known loggers to CRITICAL
        for logger_name in ["fastmcp", "mcp", "httpx", "httpcore", "anyio", "asyncio", "uvicorn"]:
            logging.getLogger(logger_name).setLevel(logging.CRITICAL)

        # Suppress warnings
        warnings.filterwarnings("ignore")
    else:
        # Only show important logs to stderr
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        root_logger.addHandler(stderr_handler)
        root_logger.setLevel(logging.INFO)

    async def run() -> None:
        # Save the real stdout before we redirected it
        real_stdout = sys.__stdout__

        if verbose:
            click.echo(f"🔗 Connecting to: {url}", err=True)
            click.echo(f"📦 Image: {headers.get('Mcp-Image', 'unknown')}", err=True)
            click.echo(f"🔑 Headers: {list(headers.keys())}", err=True)

        # Create proxy configuration
        proxy_config = {
            "mcpServers": {
                "remote": {"transport": "streamable-http", "url": url, "headers": headers}
            }
        }

        try:
            # Restore stdout for the proxy to use
            sys.stdout = real_stdout

            # Create proxy that forwards remote HTTP to local stdio
            proxy = FastMCP.as_proxy(proxy_config, name="HUD Remote Proxy")

            # Run with stdio transport - this will handle stdin/stdout properly
            await proxy.run_async(transport="stdio", show_banner=False)
        except Exception as e:
            # Ensure errors go to stderr
            sys.stdout = sys.stderr
            if verbose:
                import traceback

                click.echo(f"❌ Proxy error: {e}", err=True)
                click.echo(traceback.format_exc(), err=True)
            raise

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        if verbose:
            click.echo("\n✅ Remote proxy stopped", err=True)
        sys.exit(0)
    except Exception as e:
        if verbose:
            click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


async def run_remote_http(
    url: str,
    headers: dict[str, str],
    port: int,
    verbose: bool = False,
) -> None:
    """Run remote MCP server with HTTP transport."""
    from .logging import find_free_port

    # Find available port
    actual_port = find_free_port(port)
    if actual_port is None:
        click.echo(f"❌ No available ports found starting from {port}")
        return

    if actual_port != port:
        click.echo(f"⚠️  Port {port} in use, using port {actual_port} instead")

    # Suppress logs unless verbose
    if not verbose:
        import logging
        import os

        os.environ["FASTMCP_DISABLE_BANNER"] = "1"
        logging.getLogger("fastmcp").setLevel(logging.ERROR)
        logging.getLogger("mcp").setLevel(logging.ERROR)
        logging.getLogger("uvicorn").setLevel(logging.ERROR)
        logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
        logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
        logging.getLogger("httpx").setLevel(logging.ERROR)
        logging.getLogger("httpcore").setLevel(logging.ERROR)

    # Create the MCP config for the proxy
    config = {"remote": {"transport": "streamable-http", "url": url, "headers": headers}}

    # Create proxy that forwards remote HTTP to local HTTP
    proxy = FastMCP.as_proxy(config, name="HUD Remote Proxy")

    click.echo(f"🌐 Starting HTTP proxy on port {actual_port}")
    click.echo(f"🔗 Server URL: http://localhost:{actual_port}/mcp")
    click.echo(f"☁️  Proxying to: {url}")
    click.echo("⏹️  Press Ctrl+C to stop")

    try:
        # Run with HTTP transport
        await proxy.run_async(
            transport="http",
            host="0.0.0.0",  # noqa: S104
            port=actual_port,
            path="/mcp",
            log_level="error" if not verbose else "info",
            show_banner=False,
        )
    except KeyboardInterrupt:
        click.echo("\n👋 Shutting down...")


def run_remote_server(
    image: str,
    docker_args: list[str],
    transport: str,
    port: int,
    url: str,
    api_key: str | None,
    run_id: str | None,
    verbose: bool,
) -> None:
    """Run remote MCP server via proxy.

    Args:
        image: Docker image name
        docker_args: Docker-style arguments (-e, -h)
        transport: Output transport (stdio or http)
        port: Port for HTTP transport
        url: Remote MCP server URL
        api_key: API key for authentication
        run_id: Optional run ID
        verbose: Show detailed logs
    """
    # Parse docker args into env vars and headers
    env_args = []
    header_args = []

    i = 0
    while i < len(docker_args):
        arg = docker_args[i]

        if arg == "-e" and i + 1 < len(docker_args):
            env_args.append(docker_args[i + 1])
            i += 2
        elif arg == "-h" and i + 1 < len(docker_args):
            header_args.append(docker_args[i + 1])
            i += 2
        else:
            click.echo(f"⚠️  Unknown argument: {arg}", err=True)
            i += 1

    # Get API key from env if not provided
    if not api_key:
        api_key = settings.api_key
        if not api_key:
            click.echo(
                "❌ API key required. Set HUD_API_KEY in your environment or run: hud set HUD_API_KEY=your-key-here",  # noqa: E501
                err=True,
            )
            sys.exit(1)

    # Build headers
    headers = build_remote_headers(image, env_args, header_args, api_key, run_id)

    if verbose:
        click.echo(f"🔧 Remote URL: {url}", err=True)
        click.echo(f"📦 Image: {image}", err=True)
        click.echo(f"🔑 Headers: {list(headers.keys())}", err=True)

    # Run based on transport
    if transport == "stdio":
        run_remote_stdio(url, headers, verbose)
    else:
        asyncio.run(run_remote_http(url, headers, port, verbose))

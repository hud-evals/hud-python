"""Debug command implementation for MCP environments."""

# ruff: noqa: G004
from __future__ import annotations

import asyncio
import json
import subprocess
import threading
import time

from rich.console import Console

from hud.clients import MCPClient
from hud.utils.hud_console import HUDConsole

from .utils.logging import CaptureLogger, Colors, analyze_error_for_hints

console = Console()


async def debug_mcp_stdio(command: list[str], logger: CaptureLogger, max_phase: int = 5) -> int:
    """
    Debug any stdio-based MCP server step by step.

    Args:
        command: Command and arguments to run the MCP server
        logger: CaptureLogger instance for output
        max_phase: Maximum phase to run (1-5, default 5 for all phases)

    Returns:
        Number of phases completed (0-5)
    """
    # Create hud_console instance for initial output (before logger takes over)
    if logger.print_output:
        hud_console = HUDConsole()
        hud_console.header("MCP Server Debugger", icon="🔍")
        hud_console.dim_info("Command:", " ".join(command))
        hud_console.dim_info("Time:", time.strftime("%Y-%m-%d %H:%M:%S"))

        # Explain color coding using Rich formatting
        hud_console.info("\nColor Key:")
        console.print("  [bold]■[/bold] Commands (bold)")
        console.print("  [rgb(192,150,12)]■[/rgb(192,150,12)] STDIO (MCP protocol)")
        console.print("  [dim]■[/dim] STDERR (server logs)")
        console.print("  [green]■[/green] Success messages")
        console.print("  [red]■[/red] Error messages")
        console.print("  ■ Info messages")

    phases_completed = 0
    total_phases = 5
    start_time = time.time()

    # Phase 1: Basic Server Test
    logger.phase(1, "Basic Server Startup Test")

    try:
        # Test if command runs at all
        test_cmd = command + (["echo", "Server OK"] if "docker" in command[0] else [])
        logger.command([*test_cmd[:3], "..."] if len(test_cmd) > 3 else test_cmd)

        result = subprocess.run(  # noqa: S603, ASYNC221
            command[:1],
            capture_output=True,
            text=True,
            timeout=2,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode == 0 or "usage" in result.stderr.lower():
            logger.success("Command executable found")
            phases_completed = 1
        else:
            logger.error(f"Command failed with exit code {result.returncode}")
            if result.stderr:
                logger._log(
                    f"Error output: {result.stderr}", Colors.RED if logger.print_output else ""
                )
                hint = analyze_error_for_hints(result.stderr)
                if hint:
                    logger.hint(hint)
            logger.progress_bar(phases_completed, total_phases)
            return phases_completed

        # Check if we should stop here
        if max_phase <= 1:
            logger.info(f"Stopping at phase {max_phase} as requested")
            logger.progress_bar(phases_completed, total_phases)
            return phases_completed

    except FileNotFoundError:
        logger.error(f"Command not found: {command[0]}")
        logger.hint("Ensure the command is installed and in PATH")
        logger.progress_bar(phases_completed, total_phases)
        return phases_completed
    except Exception as e:
        logger.error(f"Startup test failed: {e}")
        logger.progress_bar(phases_completed, total_phases)
        return phases_completed

    # Phase 2: MCP Initialize Test
    logger.phase(2, "MCP Server Initialize Test")

    logger.info("STDIO is used for MCP protocol, STDERR for server logs")

    init_request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"roots": {"listChanged": True}},
            "clientInfo": {"name": "DebugClient", "version": "1.0.0"},
        },
    }

    try:
        logger.command(command)
        logger.stdio(f"Sending: {json.dumps(init_request)}")

        proc = subprocess.Popen(  # noqa: S603, ASYNC220
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",  # Replace invalid chars with � on Windows
        )

        # Ensure pipes are available
        if proc.stdin is None or proc.stdout is None or proc.stderr is None:
            raise RuntimeError("Failed to create subprocess pipes")

        # Send initialize
        proc.stdin.write(json.dumps(init_request) + "\n")
        proc.stdin.flush()

        # Collect stderr in background
        stderr_lines = []

        def read_stderr() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                line = line.rstrip()
                if line:
                    logger.stderr(line)
                    stderr_lines.append(line)

        stderr_thread = threading.Thread(target=read_stderr)
        stderr_thread.daemon = True
        stderr_thread.start()

        # Wait for response
        response = None
        start = time.time()
        while time.time() - start < 15:
            line = proc.stdout.readline()
            if line:
                try:
                    response = json.loads(line)
                    if response.get("id") == 1:
                        logger.stdio(f"Received: {json.dumps(response)}")
                        break
                except Exception as e:
                    logger.error(f"Failed to parse MCP response: {e}")
                    logger.error(f"Raw output that caused the error: {line!r}")
                    logger.hint("This usually means non-JSON output is being sent to STDOUT")
                    logger.hint("Common causes:")
                    logger.hint("  - Print statements in your server code")
                    logger.hint("  - Library warnings (use warnings.filterwarnings)")
                    logger.hint("  - Import-time output from dependencies")
                    phases_completed = 1  # Mark as failed
                    break  # Stop trying to parse

        if response and "result" in response:
            logger.success("MCP server initialized successfully")
            server_info = response["result"].get("serverInfo", {})
            logger.info(
                f"Server: {server_info.get('name', 'Unknown')} v{server_info.get('version', '?')}"
            )

            # Show capabilities
            caps = response["result"].get("capabilities", {})
            if caps:
                logger.info(f"Capabilities: {', '.join(caps.keys())}")
            phases_completed = 2
        else:
            logger.error("No valid MCP response received")

            # Analyze stderr for hints
            if stderr_lines:
                all_stderr = "\n".join(stderr_lines)
                hint = analyze_error_for_hints(all_stderr)
                if hint:
                    logger.hint(hint)
            else:
                logger.hint("""MCP requires clean stdout. Ensure:
   - All print() statements use file=sys.stderr
   - Logging is configured to use stderr
   - No libraries are printing to stdout""")

            logger.progress_bar(phases_completed, total_phases)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            return phases_completed

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

        # Check if we should stop here
        if phases_completed >= max_phase:
            logger.info(f"Stopping at phase {max_phase} as requested")
            logger.progress_bar(phases_completed, total_phases)
            return phases_completed

    except Exception as e:
        logger.error(f"MCP test failed: {e}")
        hint = analyze_error_for_hints(str(e))
        if hint:
            logger.hint(hint)
        logger.progress_bar(phases_completed, total_phases)
        return phases_completed

    # Phase 3: Tool Discovery
    logger.phase(3, "MCP Tool Discovery Test")

    client = None
    try:
        # Create MCP config for the command
        mcp_config = {
            "test": {"command": command[0], "args": command[1:] if len(command) > 1 else []}
        }

        logger.command(command)
        logger.info("Creating MCP client via hud...")

        client = MCPClient(mcp_config=mcp_config, verbose=False, auto_trace=False)
        await client.initialize()

        # Wait for initialization
        logger.info("Waiting for server initialization...")
        await asyncio.sleep(5)

        # Get tools
        tools = await client.list_tools()

        if tools:
            logger.success(f"Found {len(tools)} tools")

            # Check for lifecycle tools
            tool_names = [t.name for t in tools]
            has_setup = "setup" in tool_names
            has_evaluate = "evaluate" in tool_names

            logger.info(
                f"Lifecycle tools: setup={'✅' if has_setup else '❌'}, evaluate={'✅' if has_evaluate else '❌'}"  # noqa: E501
            )

            # Check for interaction tools
            interaction_tools = [
                name
                for name in tool_names
                if name in ["computer", "playwright", "click", "type", "interact", "move"]
            ]
            if interaction_tools:
                logger.info(f"Interaction tools: {', '.join(interaction_tools)}")

            # List all tools
            logger.info(f"All tools: {', '.join(tool_names)}")

            # Try to list resources
            try:
                resources = await client.list_resources()
                if resources:
                    logger.info(
                        f"Found {len(resources)} resources: {', '.join(str(r.uri) for r in resources[:3])}..."  # noqa: E501
                    )
            except Exception as e:
                logger.error(f"Failed to list resources: {e}")

            phases_completed = 3

        else:
            logger.error("No tools found")
            logger.hint("""No tools found. Ensure:
   - @mcp.tool() decorator is used on functions
   - Tools are registered before mcp.run()
   - No import errors preventing tool registration""")
            logger.progress_bar(phases_completed, total_phases)
            return phases_completed

        # Check if we should stop here
        if phases_completed >= max_phase:
            logger.info(f"Stopping at phase {max_phase} as requested")
            logger.progress_bar(phases_completed, total_phases)
            return phases_completed

        # Phase 4: Remote Deployment Readiness
        logger.phase(4, "Remote Deployment Readiness")

        # Test if setup/evaluate exist
        if "setup" in tool_names:
            try:
                logger.info("Testing setup tool...")
                await client.call_tool(name="setup", arguments={})
                logger.success("Setup tool responded")
            except Exception as e:
                logger.info(f"Setup tool test: {e}")

        if "evaluate" in tool_names:
            try:
                logger.info("Testing evaluate tool...")
                await client.call_tool(name="evaluate", arguments={})
                logger.success("Evaluate tool responded")
            except Exception as e:
                logger.info(f"Evaluate tool test: {e}")

        # Performance check
        init_time = time.time() - start_time
        logger.info(f"Total initialization time: {init_time:.2f}s")

        if init_time > 30:
            logger.error("Initialization took >30s - may be too slow")
            logger.hint("Consider optimizing startup time")

        phases_completed = 4

        # Check if we should stop here
        if phases_completed >= max_phase:
            logger.info(f"Stopping at phase {max_phase} as requested")
            logger.progress_bar(phases_completed, total_phases)
            return phases_completed

        # Phase 5: Concurrent Clients
        logger.phase(5, "Concurrent Clients Testing")

        concurrent_clients = []
        try:
            logger.info("Creating 3 concurrent MCP clients...")

            for i in range(3):
                client_config = {
                    f"test_concurrent_{i}": {
                        "command": command[0],
                        "args": command[1:] if len(command) > 1 else [],
                    }
                }

                concurrent_client = MCPClient(
                    mcp_config=client_config, verbose=False, auto_trace=False
                )
                await concurrent_client.initialize()
                concurrent_clients.append(concurrent_client)
                logger.info(f"Client {i + 1} connected")

            logger.success("All concurrent clients connected")

            # Clean shutdown
            for i, c in enumerate(concurrent_clients):
                await c.shutdown()
                logger.info(f"Client {i + 1} disconnected")

            phases_completed = 5

        except Exception as e:
            logger.error(f"Concurrent test failed: {e}")
        finally:
            for c in concurrent_clients:
                try:
                    await c.shutdown()
                except Exception as e:
                    logger.error(f"Failed to close client: {e}")

    except Exception as e:
        logger.error(f"Tool discovery failed: {e}")
        logger.progress_bar(phases_completed, total_phases)
        return phases_completed
    finally:
        # Ensure client is closed even on exceptions
        if client:
            try:
                await client.shutdown()
            except Exception:
                logger.error("Failed to close client")

    logger.progress_bar(phases_completed, total_phases)
    return phases_completed

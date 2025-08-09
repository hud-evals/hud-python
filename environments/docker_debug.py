#!/usr/bin/env python3
"""
Universal MCP Server Debugger

Works with any stdio-based MCP server (Docker, Python, Node, etc.)

Usage modes:
1. Direct command:  python docker_debug.py --cmd "python -m my_server"
2. Docker shorthand: python docker_debug.py my-image:latest [docker-args...]
3. Cursor config:   python docker_debug.py --cursor server-name
4. As MCP server:   python docker_debug.py --mcp
"""

import asyncio
import json
import logging
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Optional, List, Tuple
from io import StringIO
import shlex

# ANSI color codes for better visual clarity
# Enable ANSI colors on Windows
if sys.platform == "win32":
    import os

    os.system("")  # Enable ANSI escape sequences on Windows


class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    GOLD = "\033[33m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


class CaptureLogger:
    """Logger that can both print and capture output"""

    def __init__(self, print_output: bool = True):
        self.print_output = print_output
        self.buffer = StringIO()
        self.logger = logging.getLogger(__name__)

        # Configure base logger
        if print_output:
            logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
        else:
            # In MCP mode, don't print to stderr
            logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[])

    def _log(self, message: str, color: str = ""):
        """Internal log method that handles both printing and capturing"""
        if self.print_output:
            if color:
                self.logger.info(f"{color}{message}{Colors.ENDC}")
            else:
                self.logger.info(message)

        # Always capture (without ANSI codes)
        clean_msg = self._strip_ansi(message)
        self.buffer.write(clean_msg + "\n")

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes from text"""
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    def timestamp(self) -> str:
        """Get minimal timestamp HH:MM:SS"""
        return datetime.now().strftime("%H:%M:%S")

    def phase(self, phase_num: int, title: str):
        """Log a phase header"""
        self._log(f"\n{'=' * 80}", Colors.CYAN if self.print_output else "")
        self._log(
            f"PHASE {phase_num}: {title}", Colors.BOLD + Colors.CYAN if self.print_output else ""
        )
        self._log(f"{'=' * 80}\n", Colors.CYAN if self.print_output else "")

    def command(self, cmd: list):
        """Log the command being executed"""
        self._log(f"$ {' '.join(cmd)}", Colors.BOLD if self.print_output else "")

    def success(self, message: str):
        """Log a success message"""
        self._log(f"✅ {message}", Colors.GREEN if self.print_output else "")

    def error(self, message: str):
        """Log an error message"""
        self._log(f"❌ {message}", Colors.RED if self.print_output else "")

    def info(self, message: str):
        """Log an info message"""
        self._log(f"[{self.timestamp()}] {message}")

    def stdio(self, message: str):
        """Log STDIO communication"""
        self._log(f"[STDIO] {message}", Colors.GOLD if self.print_output else "")

    def stderr(self, message: str):
        """Log STDERR output"""
        self._log(f"[STDERR] {message}", Colors.GRAY if self.print_output else "")

    def hint(self, hint: str):
        """Log a hint message"""
        self._log(f"\n💡 Hint: {hint}", Colors.YELLOW if self.print_output else "")

    def progress_bar(self, completed: int, total: int):
        """Show a visual progress bar"""
        filled = "█" * completed
        empty = "░" * (total - completed)
        percentage = (completed / total) * 100

        self._log(
            f"\nProgress: [{filled}{empty}] {completed}/{total} phases ({percentage:.0f}%)",
            Colors.BOLD if self.print_output else "",
        )

        if completed == 0:
            self._log("Failed at Phase 1 - Server startup", Colors.RED if self.print_output else "")
        elif completed == 1:
            self._log(
                "Failed at Phase 2 - MCP initialization", Colors.YELLOW if self.print_output else ""
            )
        elif completed == 2:
            self._log(
                "Failed at Phase 3 - Tool discovery", Colors.YELLOW if self.print_output else ""
            )
        elif completed == 3:
            self._log(
                "Failed at Phase 4 - Remote deployment readiness",
                Colors.YELLOW if self.print_output else "",
            )
        elif completed == 4:
            self._log(
                "Failed at Phase 5 - Concurrent clients & resources",
                Colors.YELLOW if self.print_output else "",
            )
        elif completed == 5:
            self._log(
                "All phases completed successfully!", Colors.GREEN if self.print_output else ""
            )

    def get_output(self) -> str:
        """Get the captured output"""
        return self.buffer.getvalue()


# Hint registry with patterns and priorities
HINT_REGISTRY = [
    {
        "patterns": [r"Can't connect to display", r"X11", r"DISPLAY.*not set", r"Xlib.*error"],
        "priority": 10,
        "hint": """GUI environment needs X11. Common fixes:
   - Start Xvfb before importing GUI libraries in your entrypoint
   - Use a base image with X11 pre-configured (e.g., hudpython/novnc-base)
   - Delay GUI imports until after X11 is running""",
    },
    {
        "patterns": [r"ModuleNotFoundError", r"ImportError", r"No module named"],
        "priority": 9,
        "hint": """Missing Python dependencies. Check:
   - Is pyproject.toml complete with all dependencies?
   - Did 'pip install' run successfully?
   - For editable installs, is the package structure correct?""",
    },
    {
        "patterns": [r"json\.decoder\.JSONDecodeError", r"Expecting value.*line.*column"],
        "priority": 8,
        "hint": """Invalid JSON-RPC communication. Check:
   - MCP server is using proper JSON-RPC format
   - No debug prints are corrupting stdout
   - Character encoding is UTF-8""",
    },
    {
        "patterns": [r"Permission denied", r"EACCES", r"Operation not permitted"],
        "priority": 7,
        "hint": """Permission issues. Try:
   - Check file permissions in container/environment
   - Running with appropriate user
   - Using --privileged flag if absolutely needed (Docker)""",
    },
    {
        "patterns": [r"Cannot allocate memory", r"killed", r"OOMKilled"],
        "priority": 6,
        "hint": """Resource limits exceeded. Consider:
   - Increasing memory limits
   - Optimizing memory usage in your code
   - Checking for memory leaks""",
    },
    {
        "patterns": [r"bind.*address already in use", r"EADDRINUSE", r"port.*already allocated"],
        "priority": 5,
        "hint": """Port conflict detected. Options:
   - Use a different port
   - Check if another process is running
   - Ensure proper cleanup in previous runs""",
    },
    {
        "patterns": [r"FileNotFoundError", r"No such file or directory"],
        "priority": 4,
        "hint": """File or directory missing. Check:
   - All required files exist
   - Working directory is set correctly
   - File paths are correct for the environment""",
    },
    {
        "patterns": [r"Traceback.*most recent call last", r"Exception"],
        "priority": 2,
        "hint": """Server crashed during startup. Common causes:
   - Missing environment variables
   - Import errors in your module
   - Initialization code failing""",
    },
    {
        "patterns": [r"timeout", r"timed out"],
        "priority": 1,
        "hint": """Server taking too long to start. Consider:
   - Using initialization wrappers for heavy setup
   - Moving slow operations to setup() tool
   - Checking for deadlocks or infinite loops""",
    },
]


def analyze_error_for_hints(error_text: str) -> Optional[str]:
    """Analyze error text and return the highest priority matching hint."""
    if not error_text:
        return None

    matches = []
    for hint_data in HINT_REGISTRY:
        for pattern in hint_data["patterns"]:
            if re.search(pattern, error_text, re.IGNORECASE):
                matches.append((hint_data["priority"], hint_data["hint"]))
                break

    if matches:
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[0][1]

    return None


async def debug_mcp_stdio(command: List[str], logger: CaptureLogger, max_phase: int = 5) -> int:
    """
    Debug any stdio-based MCP server step by step.

    Args:
        command: Command and arguments to run the MCP server
        logger: CaptureLogger instance for output
        max_phase: Maximum phase to run (1-5, default 5 for all phases)

    Returns:
        Number of phases completed (0-5)
    """

    logger._log(f"\n🔍 MCP Server Debugger", Colors.BOLD if logger.print_output else "")
    logger._log(f"Command: {' '.join(command)}", Colors.GRAY if logger.print_output else "")
    logger._log(
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        Colors.GRAY if logger.print_output else "",
    )

    # Explain color coding (only in print mode)
    if logger.print_output:
        logger._log(f"\nColor Key:", Colors.BOLD if logger.print_output else "")
        logger._log(f"  {Colors.BOLD}■{Colors.ENDC} Commands (bold)")
        logger._log(f"  {Colors.GOLD}■{Colors.ENDC} STDIO (MCP protocol)")
        logger._log(f"  {Colors.GRAY}■{Colors.ENDC} STDERR (server logs)")
        logger._log(f"  {Colors.GREEN}■{Colors.ENDC} Success messages")
        logger._log(f"  {Colors.RED}■{Colors.ENDC} Error messages")
        logger._log(f"  ■ Info messages")

    phases_completed = 0
    total_phases = 5
    start_time = time.time()

    # Phase 1: Basic Server Test
    logger.phase(1, "Basic Server Startup Test")

    try:
        # Test if command runs at all
        test_cmd = command + (["echo", "Server OK"] if "docker" in command[0] else [])
        logger.command(test_cmd[:3] + ["..."] if len(test_cmd) > 3 else test_cmd)

        result = subprocess.run(command[:1], capture_output=True, text=True, timeout=2)

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

        proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Send initialize
        proc.stdin.write(json.dumps(init_request) + "\n")
        proc.stdin.flush()

        # Collect stderr in background
        import threading

        stderr_lines = []

        def read_stderr():
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
                except:
                    continue

        # Give stderr thread time to collect output
        time.sleep(0.5)

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
            proc.wait(timeout=5)
            return phases_completed

        proc.terminate()
        proc.wait(timeout=5)

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

    try:
        from hud.client import MCPClient

        # Create MCP config for the command
        mcp_config = {
            "test": {"command": command[0], "args": command[1:] if len(command) > 1 else []}
        }

        logger.command(command)
        logger.info("Creating MCP client via hud...")

        client = MCPClient(mcp_config=mcp_config, verbose=False)
        await client.initialize()

        # Wait for initialization
        logger.info("Waiting for server initialization...")
        await asyncio.sleep(5)

        # Get tools
        tools = client.get_available_tools()

        if tools:
            logger.success(f"Found {len(tools)} tools")

            # Check for lifecycle tools
            tool_names = [t.name for t in tools]
            has_setup = "setup" in tool_names
            has_evaluate = "evaluate" in tool_names

            logger.info(
                f"Lifecycle tools: setup={'✅' if has_setup else '❌'}, evaluate={'✅' if has_evaluate else '❌'}"
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
                        f"Found {len(resources)} resources: {', '.join(str(r.uri) for r in resources[:3])}..."
                    )
            except:
                pass

            phases_completed = 3

        else:
            logger.error("No tools found")
            logger.hint("""No tools found. Ensure:
   - @mcp.tool() decorator is used on functions
   - Tools are registered before mcp.run()
   - No import errors preventing tool registration""")
            await client.close()
            logger.progress_bar(phases_completed, total_phases)
            return phases_completed

        # Phase 4: Remote Deployment Readiness
        logger.phase(4, "Remote Deployment Readiness")

        # Test if setup/evaluate exist
        if "setup" in tool_names:
            try:
                logger.info("Testing setup tool...")
                setup_result = await client.call_tool("setup", {})
                logger.success("Setup tool responded")
            except Exception as e:
                logger.info(f"Setup tool test: {e}")

        if "evaluate" in tool_names:
            try:
                logger.info("Testing evaluate tool...")
                eval_result = await client.call_tool("evaluate", {})
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

                concurrent_client = MCPClient(mcp_config=client_config, verbose=False)
                await concurrent_client.initialize()
                concurrent_clients.append(concurrent_client)
                logger.info(f"Client {i + 1} connected")

            logger.success("All concurrent clients connected")

            # Clean shutdown
            for i, c in enumerate(concurrent_clients):
                await c.close()
                logger.info(f"Client {i + 1} disconnected")

            phases_completed = 5

        except Exception as e:
            logger.error(f"Concurrent test failed: {e}")
        finally:
            for c in concurrent_clients:
                try:
                    await c.close()
                except:
                    pass

        await client.close()

    except Exception as e:
        logger.error(f"Tool discovery failed: {e}")
        logger.progress_bar(phases_completed, total_phases)
        return phases_completed

    logger.progress_bar(phases_completed, total_phases)
    return phases_completed


def parse_cursor_config(server_name: str) -> Optional[Tuple[List[str], str]]:
    """
    Parse cursor config to get command for a server.

    Returns:
        Tuple of (command_list, error_message) or None if successful
    """
    from pathlib import Path

    # Find cursor config
    cursor_config_path = Path.home() / ".cursor" / "mcp.json"
    if not cursor_config_path.exists():
        # Try Windows path
        cursor_config_path = Path(os.environ.get("USERPROFILE", "")) / ".cursor" / "mcp.json"

    if not cursor_config_path.exists():
        return None, f"Cursor config not found at {cursor_config_path}"

    try:
        with open(cursor_config_path) as f:
            config = json.load(f)

        servers = config.get("mcpServers", {})
        if server_name not in servers:
            available = ", ".join(servers.keys())
            return None, f"Server '{server_name}' not found. Available: {available}"

        server_config = servers[server_name]
        command = server_config.get("command", "")
        args = server_config.get("args", [])

        # Combine command and args
        full_command = [command] + args

        # Handle reloaderoo wrapper
        if command == "npx" and "reloaderoo" in args and "--" in args:
            # Extract the actual command after --
            dash_index = args.index("--")
            full_command = args[dash_index + 1 :]

        return full_command, None

    except Exception as e:
        return None, f"Error reading config: {e}"


def run_as_mcp_server():
    """Run docker_debug as an MCP server."""
    from mcp.server.fastmcp import FastMCP
    from mcp.types import TextContent

    # Initialize MCP server
    mcp = FastMCP(
        name="mcp-debugger",
    )

    @mcp.tool()
    async def debug_stdio_server(
        command: str, args: list[str] = None, max_phase: int = 1
    ) -> list[TextContent]:
        """
        Debug any stdio-based MCP server.

        Args:
            command: Command to run (e.g., 'python', 'node', 'docker')
            args: List of arguments for the command
            max_phase: Maximum phase to run (1-3, default 1 for quick test)

        Returns:
            Debug output showing test phases up to max_phase
        """
        # Build full command
        full_command = [command] + (args or [])

        # Create logger in capture mode
        logger = CaptureLogger(print_output=False)

        # Cap max_phase at 3 for MCP mode
        max_phase = min(max_phase, 3)

        # Run debug
        await debug_mcp_stdio(full_command, logger, max_phase=max_phase)

        # Return captured output
        return [TextContent(text=logger.get_output(), type="text")]

    @mcp.tool()
    async def debug_docker_image(
        image: str, docker_args: list[str] = None, max_phase: int = 1
    ) -> list[TextContent]:
        """
        Debug a Docker-based MCP server.

        Args:
            image: Docker image name
            docker_args: Additional docker arguments (e.g., ['-e', 'KEY=value'])
            max_phase: Maximum phase to run (1-3, default 1 for quick test)

        Returns:
            Debug output showing test phases up to max_phase
        """
        # Build docker command
        command = ["docker", "run", "--rm", "-i"] + (docker_args or []) + [image]

        # Create logger in capture mode
        logger = CaptureLogger(print_output=False)

        # Cap max_phase at 3 for MCP mode
        max_phase = min(max_phase, 3)

        # Run debug
        await debug_mcp_stdio(command, logger, max_phase=max_phase)

        # Return captured output
        return [TextContent(text=logger.get_output(), type="text")]

    @mcp.tool()
    async def debug_cursor_config(server_name: str, max_phase: int = 1) -> list[TextContent]:
        """
        Debug a server from Cursor's MCP configuration.

        Args:
            server_name: Name of server in .cursor/mcp.json
            max_phase: Maximum phase to run (1-3, default 1 for quick test)

        Returns:
            Debug output showing test phases up to max_phase
        """
        # Parse cursor config
        command, error = parse_cursor_config(server_name)

        if error:
            return [TextContent(text=f"❌ {error}", type="text")]

        # Create logger in capture mode
        logger = CaptureLogger(print_output=False)

        # Cap max_phase at 3 for MCP mode
        max_phase = min(max_phase, 3)

        # Run debug
        await debug_mcp_stdio(command, logger, max_phase=max_phase)

        # Return captured output
        return [TextContent(text=logger.get_output(), type="text")]

    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    import warnings
    import gc

    # Parse arguments
    if len(sys.argv) < 2:
        print("Universal MCP Server Debugger")
        print("=" * 40)
        print("\nUsage modes:")
        print('  1. Direct command:   python docker_debug.py --cmd "python -m my_server"')
        print("  2. Docker shorthand: python docker_debug.py my-image:latest [docker-args...]")
        print("  3. Cursor config:    python docker_debug.py --cursor server-name")
        print("  4. As MCP server:    python docker_debug.py --mcp")
        print("\nExamples:")
        print('  python docker_debug.py --cmd "python -m hud_controller.server"')
        print('  python docker_debug.py --cmd "node server.js"')
        print("  python docker_debug.py hud-text-2048:dev")
        print("  python docker_debug.py my-image:latest -e API_KEY=xxx")
        print("  python docker_debug.py --cursor text-2048-dev")
        print("  python docker_debug.py --mcp")
        sys.exit(1)

    # Suppress cleanup warnings
    warnings.filterwarnings("ignore", category=ResourceWarning)

    # Mode 4: Run as MCP server
    if sys.argv[1] == "--mcp":
        run_as_mcp_server()

    # Mode 1: Direct command mode
    elif sys.argv[1] == "--cmd":
        if len(sys.argv) < 3:
            print("Error: --cmd requires a command string")
            sys.exit(1)

        # Parse the command (handle quoted strings)
        command = shlex.split(sys.argv[2])

        # Create logger in print mode
        logger = CaptureLogger(print_output=True)

        # Run debug
        asyncio.run(debug_mcp_stdio(command, logger))

    # Mode 3: Cursor config mode
    elif sys.argv[1] == "--cursor":
        if len(sys.argv) < 3:
            print("Error: --cursor requires a server name")
            sys.exit(1)

        server_name = sys.argv[2]
        command, error = parse_cursor_config(server_name)

        if error:
            print(f"❌ {error}")
            sys.exit(1)

        # Create logger in print mode
        logger = CaptureLogger(print_output=True)

        # Run debug
        asyncio.run(debug_mcp_stdio(command, logger))

    # Mode 2: Docker shorthand (backward compatible)
    else:
        # Assume it's a docker image
        docker_image = sys.argv[1]
        docker_args = sys.argv[2:] if len(sys.argv) > 2 else []

        # Build docker command
        command = ["docker", "run", "--rm", "-i"] + docker_args + [docker_image]

        # Create logger in print mode
        logger = CaptureLogger(print_output=True)

        # Run debug
        asyncio.run(debug_mcp_stdio(command, logger))

    # Force cleanup
    gc.collect()

"""CLI utilities - logging, colors, and error analysis."""

from __future__ import annotations

import re
import sys
from datetime import datetime
from io import StringIO

# Enable ANSI colors on Windows
if sys.platform == "win32":
    import os

    os.system("")  # Enable ANSI escape sequences on Windows # noqa: S607 S605


class Colors:
    """ANSI color codes for terminal output - optimized for both light and dark modes."""

    HEADER = "\033[95m"  # Light magenta
    BLUE = "\033[94m"  # Light blue
    CYAN = "\033[96m"  # Light cyan
    GREEN = "\033[92m"  # Light green
    YELLOW = "\033[93m"  # Light yellow
    GOLD = "\033[33m"  # Gold/orange
    RED = "\033[91m"  # Light red
    GRAY = "\033[37m"  # Light gray
    ENDC = "\033[0m"  # Reset
    BOLD = "\033[1m"  # Bold


class CaptureLogger:
    """Logger that can both print and capture output."""

    def __init__(self, print_output: bool = True) -> None:
        self.print_output = print_output
        self.buffer = StringIO()

    def _log(self, message: str, color: str = "") -> None:
        """Internal log method that handles both printing and capturing."""
        if self.print_output:
            if color:
                print(f"{color}{message}{Colors.ENDC}")  # noqa: T201
            else:
                print(message)  # noqa: T201

        # Always capture (without ANSI codes)
        clean_msg = self._strip_ansi(message)
        self.buffer.write(clean_msg + "\n")

    def _strip_ansi(self, text: str) -> str:
        """Remove ANSI escape codes from text."""
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        return ansi_escape.sub("", text)

    def timestamp(self) -> str:
        """Get minimal timestamp HH:MM:SS."""
        return datetime.now().strftime("%H:%M:%S")

    def phase(self, phase_num: int, title: str) -> None:
        """Log a phase header."""
        self._log(f"\n{'=' * 80}", Colors.GOLD if self.print_output else "")
        self._log(
            f"PHASE {phase_num}: {title}", Colors.BOLD + Colors.GOLD if self.print_output else ""
        )
        self._log(f"{'=' * 80}\n", Colors.GOLD if self.print_output else "")

    def command(self, cmd: list) -> None:
        """Log the command being executed."""
        self._log(f"$ {' '.join(cmd)}", Colors.BOLD if self.print_output else "")

    def success(self, message: str) -> None:
        """Log a success message."""
        self._log(f"✅ {message}", Colors.GREEN if self.print_output else "")

    def error(self, message: str) -> None:
        """Log an error message."""
        self._log(f"❌ {message}", Colors.RED if self.print_output else "")

    def info(self, message: str) -> None:
        """Log an info message."""
        self._log(f"[{self.timestamp()}] {message}")

    def stdio(self, message: str) -> None:
        """Log STDIO communication."""
        self._log(f"[STDIO] {message}", Colors.GOLD if self.print_output else "")

    def stderr(self, message: str) -> None:
        """Log STDERR output."""
        self._log(f"[STDERR] {message}", Colors.GRAY if self.print_output else "")

    def hint(self, hint: str) -> None:
        """Log a hint message."""
        self._log(f"\n💡 Hint: {hint}", Colors.YELLOW if self.print_output else "")

    def progress_bar(self, completed: int, total: int) -> None:
        """Show a visual progress bar."""
        filled = "█" * completed
        empty = "░" * (total - completed)
        percentage = (completed / total) * 100

        self._log(
            f"\nProgress: [{filled}{empty}] {completed}/{total} phases ({percentage:.0f}%)",
            Colors.BOLD if self.print_output else "",
        )

        phase_messages = {
            0: ("Failed at Phase 1 - Server startup", Colors.RED),
            1: ("Failed at Phase 2 - MCP initialization", Colors.YELLOW),
            2: ("Failed at Phase 3 - Tool discovery", Colors.YELLOW),
            3: ("Failed at Phase 4 - Remote deployment readiness", Colors.YELLOW),
            4: ("Failed at Phase 5 - Concurrent clients & resources", Colors.YELLOW),
            5: ("All phases completed successfully!", Colors.GREEN),
        }

        if completed in phase_messages:
            msg, color = phase_messages[completed]
            self._log(msg, color if self.print_output else "")

    def get_output(self) -> str:
        """Get the captured output."""
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


def analyze_error_for_hints(error_text: str | None) -> str | None:
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


def find_free_port(start_port: int = 8765, max_attempts: int = 100) -> int | None:
    """Find a free port starting from the given port.

    Args:
        start_port: Port to start searching from
        max_attempts: Maximum number of ports to try

    Returns:
        Available port number or None if no ports found
    """
    import socket

    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Try to bind to the port
                s.bind(("", port))
                s.close()
                return port
            except OSError:
                # Port is in use, try next one
                continue
    return None


def is_port_free(port: int) -> bool:
    """Check if a specific port is free.

    Args:
        port: Port number to check

    Returns:
        True if port is free, False otherwise
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", port))
            s.close()
            return True
        except OSError:
            return False

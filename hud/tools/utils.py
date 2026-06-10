from __future__ import annotations

import asyncio
import subprocess

# Default timeout for running commands
DEFAULT_TIMEOUT = 10.0


async def run(
    command: str | list[str],
    input: str | None = None,
    timeout: float | None = DEFAULT_TIMEOUT,  # noqa: ASYNC109
) -> tuple[int, str, str]:
    """
    Run a command asynchronously and return the result.

    Args:
        command: Command to run (string or list of strings)
        input: Optional input to send to stdin
        timeout: Timeout in seconds

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    if isinstance(command, str):
        proc = await asyncio.create_subprocess_shell(
            command,
            stdin=subprocess.PIPE if input else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    else:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdin=subprocess.PIPE if input else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    stdout, stderr = await asyncio.wait_for(
        proc.communicate(input=input.encode() if input else None), timeout=timeout
    )

    return proc.returncode or 0, stdout.decode(), stderr.decode()


def maybe_truncate(text: str, max_length: int = 2048 * 10) -> str:
    """Truncate output if too long."""
    return text if len(text) <= max_length else text[:max_length] + "... (truncated)"

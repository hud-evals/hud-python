"""WebSocket log streaming for remote builds."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any

import websockets
from websockets.exceptions import ConnectionClosed

from hud.utils.exceptions import HudRequestError
from hud.utils.hud_console import HUDConsole

if TYPE_CHECKING:
    from hud.utils.platform import PlatformClient


async def stream_build_logs(
    platform: PlatformClient,
    build_id: str,
    console: HUDConsole | None = None,
    max_reconnects: int = 3,
) -> str:
    """Stream build logs from the HUD backend via WebSocket."""
    if console is None:
        console = HUDConsole()

    ws_url = platform.api_url.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url.rstrip('/')}/builds/{build_id}/logs?api_key={platform.api_key}"

    final_status = "UNKNOWN"
    reconnect_count = 0
    last_log_count = 0

    while reconnect_count <= max_reconnects:
        try:
            console.info("Connecting to build logs stream...")
            async with websockets.connect(
                ws_url,
                ping_interval=30,
                ping_timeout=10,
            ) as websocket:
                reconnect_count = 0  # Reset on successful connect

                async for message in websocket:
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")

                        if msg_type == "status":
                            # Initial connection status
                            console.info(data.get("message", "Connected"))

                        elif msg_type == "status_update":
                            # Build status update
                            status = data.get("status", "")
                            if status == "IN_PROGRESS":
                                # Don't spam status updates
                                pass
                            else:
                                console.info(f"Build status: {status}")

                        elif msg_type == "log":
                            # Actual log line
                            log_message = data.get("message", "")
                            timestamp = data.get("timestamp")
                            if log_message:
                                _print_log_line(console, log_message, timestamp)
                            last_log_count += 1

                        elif msg_type == "complete":
                            # Build completed
                            final_status = data.get("final_status", "UNKNOWN")
                            completion_msg = data.get("message", f"Build {final_status}")
                            console.info(completion_msg)
                            return final_status

                        elif msg_type == "error":
                            # Error from server
                            error_msg = data.get("error", "Unknown error")
                            console.error(f"Build error: {error_msg}")
                            return "FAILED"

                    except json.JSONDecodeError:
                        # Non-JSON message, print as-is
                        console.info(str(message))

        except ConnectionClosed as e:
            if e.code == 4003:
                # Authentication or access error
                console.error(f"Access denied: {e.reason}")
                return "FAILED"

            reconnect_count += 1
            if reconnect_count <= max_reconnects:
                wait_time = min(2**reconnect_count, 30)
                console.warning(
                    f"Connection closed, reconnecting in {wait_time}s... "
                    f"(attempt {reconnect_count}/{max_reconnects})"
                )
                await asyncio.sleep(wait_time)
            else:
                console.error("Max reconnection attempts reached")
                return "UNKNOWN"

        except Exception as e:
            reconnect_count += 1
            if reconnect_count <= max_reconnects:
                wait_time = min(2**reconnect_count, 30)
                console.warning(
                    f"Connection error: {e}. Reconnecting in {wait_time}s... "
                    f"(attempt {reconnect_count}/{max_reconnects})"
                )
                await asyncio.sleep(wait_time)
            else:
                console.error(f"Failed to stream logs: {e}")
                return "UNKNOWN"

    return final_status


def _print_log_line(
    console: HUDConsole,
    message: str,
    timestamp: str | int | None = None,
) -> None:
    """Print a log line with optional timestamp formatting.

    Args:
        console: HUDConsole for output
        message: Log message to print
        timestamp: Optional timestamp (ISO string or Unix ms)
    """
    # Strip trailing whitespace/newlines from message
    message = message.rstrip()

    # Format timestamp if provided
    prefix = ""
    if timestamp:
        try:
            if isinstance(timestamp, int):
                # Unix timestamp in milliseconds
                dt = datetime.fromtimestamp(timestamp / 1000)
                prefix = f"[{dt.strftime('%H:%M:%S')}] "
            elif isinstance(timestamp, str):
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                prefix = f"[{dt.strftime('%H:%M:%S')}] "
        except Exception:  # noqa: S110
            pass  # Timestamp parsing is best-effort

    # Colorize based on content - be smart about what's actually an error
    lower_msg = message.lower()

    # Patterns that indicate error handling, not actual errors
    is_error_handling = any(
        pattern in lower_msg
        for pattern in [
            "2>/dev/null",
            "|| true",
            "|| echo",
            "|| :",
            "if [",
            "if aws",
            "if docker",
            "--quiet",
        ]
    )

    # Actual error indicators (not just containing "error" somewhere)
    stripped_msg = message.strip()
    is_actual_error = not is_error_handling and (
        lower_msg.startswith(("error:", "error "))
        or "exit status 1" in lower_msg
        or "exit code: 1" in lower_msg
        or "command did not exit successfully" in lower_msg
        or "failed to" in lower_msg
        or ": FAILED" in message  # Case-sensitive for status
        or "State: FAILED" in message
        or stripped_msg.startswith(("OSError:", "Exception:"))
    )

    if is_actual_error:
        console.error(f"{prefix}{message}")
    elif "warning" in lower_msg or "warn:" in lower_msg:
        console.warning(f"{prefix}{message}")
    elif "success" in lower_msg or "completed successfully" in lower_msg:
        console.success(f"{prefix}{message}")
    else:
        console.info(f"{prefix}{message}")


async def poll_build_status(
    platform: PlatformClient,
    build_id: str,
    console: HUDConsole | None = None,
    poll_interval: float = 5.0,
    max_wait: float = 3600.0,
) -> dict[str, Any]:
    """Poll for build status as a fallback when WebSocket is not available."""
    if console is None:
        console = HUDConsole()

    start_time = asyncio.get_event_loop().time()
    last_status = ""

    while True:
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > max_wait:
            console.error(f"Build timed out after {max_wait}s")
            return {"status": "TIMED_OUT"}

        try:
            data = await platform.aget(f"/builds/{build_id}/status")

            status = data.get("status", "")
            if status != last_status:
                console.info(f"Build status: {status}")
                last_status = status

            if status in ["SUCCEEDED", "FAILED", "STOPPED", "TIMED_OUT"]:
                return data

        except HudRequestError as e:
            console.warning(f"Status check failed: {e.status_code or e}")
        except Exception as e:
            console.warning(f"Status check error: {e}")

        await asyncio.sleep(poll_interval)

#!/usr/bin/env python3
"""Harbor entrypoint for HUD-exported task images.

Two jobs:

1. Start ``hud dev`` in the background so the harbor agent's
   ``[[environment.mcp_servers]]`` config can connect to it.

2. Register the scenario session against that ``hud dev`` and persist
   the session id to ``/tmp/.hud_scenario_session`` so the harbor
   verifier's ``hud scenario grade`` can resume the *same* session
   later — without re-running ``setup_task`` and clobbering the
   agent's filesystem edits.

The trick is calling MCP's lower-level ``streamablehttp_client`` with
``terminate_on_close=False`` here. fastmcp's ``Client`` defaults to
sending a DELETE on close, which would kill the session as soon as our
``setup`` call returns — leaving the verifier with a "Session
terminated" error. With ``terminate_on_close=False`` the session stays
alive on the server after we disconnect, until either the verifier
explicitly closes it (default fastmcp behaviour) or the container
shuts down.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


async def _wait_port(port: int, timeout: int = 180) -> bool:  # noqa: ASYNC109
    """Return True once ``localhost:port`` accepts TCP connections.

    Default 180s because some env packages do significant work at
    import time. A too-short timeout here makes the entrypoint barrel into
    ``_register_session`` before the MCP server is listening.
    """
    for _ in range(timeout * 2):
        try:
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
            return True
        except OSError:
            await asyncio.sleep(0.5)
    return False


async def _register_session(url: str, scenario: str, args_dict: dict) -> str | None:
    """Run scenario setup against ``url`` and return the resulting MCP
    session id. The session is intentionally NOT terminated on close.
    """
    # Imported lazily so a missing fastmcp install doesn't crash before
    # we've at least booted ``hud dev``.
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    args_for_mcp = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in args_dict.items()}

    async with (
        streamablehttp_client(url, terminate_on_close=False) as (
            read_stream,
            write_stream,
            get_session_id,
        ),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        await session.get_prompt(scenario, arguments=args_for_mcp)
        return get_session_id()


def _build_launch_command(port: int) -> list[str]:
    """Return the env's launch command, with ``--stdio`` rewritten to
    ``--port <port>`` so MCP speaks HTTP for Harbor's compose driver.

    ``HUD_LAUNCH_COMMAND`` (JSON list) is the env image's full original
    CMD, captured at build time and threaded through the export's lock
    yaml. Required for envs that wrap ``hud dev`` in a multi-process
    startup. Falls back to a bare ``hud dev env:env`` when not set.
    """
    raw = os.environ.get("HUD_LAUNCH_COMMAND")
    if raw:
        cmd = json.loads(raw)
        rewritten: list[str] = []
        for arg in cmd:
            if arg == "--stdio":
                rewritten.extend(["--port", str(port)])
            elif "--stdio" in arg:
                # ``sh -c`` shell-string forms — rewrite inside the string
                rewritten.append(arg.replace("--stdio", f"--port {port}"))
            else:
                rewritten.append(str(arg))
        return rewritten
    return ["hud", "dev", "env:env", "--port", str(port)]


def _spawn_hud_dev(port: int) -> subprocess.Popen[bytes]:
    cmd = _build_launch_command(port)
    with open("/tmp/hud-dev.log", "wb") as log_fd:  # noqa: S108
        # ``start_new_session=True`` puts the launch tree in its own
        # process group so envs that fork sub-processes (uvicorn, Xvfb,
        # chrome, ...) survive ``os.execvp`` into Harbor's compose
        # ``sleep infinity`` below — and also so a future graceful
        # shutdown could ``killpg`` the whole tree if needed.
        return subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


_KEEPALIVE_INTERVAL_SEC = 30
_KEEPALIVE_LOG_PATH = "/logs/agent/keepalive.log"


def _spawn_session_keepalive(url: str, session_id: str) -> subprocess.Popen[bytes]:
    """Background-ping the registered MCP session to keep it alive.

    The agent runs on its OWN MCP session (claude-code/codex/etc. each
    open one when they connect), so the session entrypoint registered
    here sits idle the entire time the agent is doing work. Without
    keepalive, the verifier's ``hud scenario grade`` fails with
    ``McpError: Session terminated`` because the session was reaped
    mid-trial. A cheap ``list_tools`` every 30s keeps it warm.

    The script runs as a detached subprocess so PID 1 is free to
    ``exec`` into Harbor's compose command. ``streamablehttp_client``
    is used with ``terminate_on_close=False`` so the keepalive doesn't
    accidentally DELETE the session it's trying to preserve. Logs are
    timestamped + flushed eagerly so a missing keepalive process is
    obvious in /logs/keepalive.log.
    """
    script = (
        "import asyncio, sys, time, os\n"
        "URL = " + repr(url) + "\n"
        "SID = " + repr(session_id) + "\n"
        "INTERVAL = " + repr(_KEEPALIVE_INTERVAL_SEC) + "\n"
        "def _log(msg):\n"
        "    sys.stdout.write(f\"[{time.strftime('%H:%M:%S')}] keepalive: {msg}\\n\")\n"
        "    sys.stdout.flush()\n"
        '_log(f"started; session={SID[:8]}... interval={INTERVAL}s")\n'
        "async def _ping_loop():\n"
        "    from mcp import ClientSession\n"
        "    from mcp.client.streamable_http import streamablehttp_client\n"
        "    n = 0\n"
        "    while True:\n"
        "        try:\n"
        "            async with streamablehttp_client(\n"
        '                URL, terminate_on_close=False, headers={"mcp-session-id": SID}\n'
        "            ) as (r, w, _gid):\n"
        "                async with ClientSession(r, w) as session:\n"
        "                    await session.initialize()\n"
        "                    tools = await session.list_tools()\n"
        "                    n += 1\n"
        '                    _log(f"ping #{n} ok ({len(tools.tools)} tools)")\n'
        "        except Exception as exc:\n"
        '            _log(f"ping #{n} failed: {type(exc).__name__}: {exc}")\n'
        "        await asyncio.sleep(INTERVAL)\n"
        "asyncio.run(_ping_loop())\n"
    )
    # Open the log under /logs for post-mortem visibility; if /logs
    # isn't writable yet (Harbor sometimes lazily mounts), fall back
    # to /tmp.
    try:
        log_fd = open(_KEEPALIVE_LOG_PATH, "wb")  # noqa: SIM115
    except OSError:
        log_fd = open("/tmp/hud-keepalive.log", "wb")  # noqa: S108, SIM115
    return subprocess.Popen(
        ["python3", "-u", "-c", script],  # noqa: S607
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="entrypoint: %(message)s",
        stream=sys.stderr,
    )
    port = int(os.environ.get("HUD_MCP_PORT", "8765"))
    scenario = os.environ.get("HUD_TASK_SCENARIO") or ""
    args_json = os.environ.get("HUD_TASK_ARGS") or "{}"
    url = f"http://localhost:{port}/mcp"

    _spawn_hud_dev(port)

    if not asyncio.run(_wait_port(port)):
        logger.error("hud dev did not bind :%d", port)

    if scenario:
        # Bake-time ``hud scenario setup`` already wrote a session id
        # into the image at /tmp/.hud_scenario_session. That id belongs
        # to a server that no longer exists. If runtime registration
        # below fails (slow env init, transient error, etc.) we want
        # the verifier's ``hud scenario grade`` to fall through to a
        # *new* session rather than try to resume the dead bake-time
        # one and crash with "Session terminated". So clear the stale
        # value upfront and only repopulate on a confirmed re-register.
        session_file = Path("/tmp/.hud_scenario_session")  # noqa: S108
        session_file.unlink(missing_ok=True)
        try:
            try:
                args_dict = json.loads(args_json) if args_json else {}
            except json.JSONDecodeError as exc:
                logger.error("invalid HUD_TASK_ARGS: %s", exc)
                args_dict = {}
            session_id = asyncio.run(
                _register_session(url, scenario, args_dict if isinstance(args_dict, dict) else {})
            )
            if session_id:
                session_file.write_text(session_id)
                logger.info("scenario session %s", session_id)
                _spawn_session_keepalive(url, session_id)
                logger.info("session keepalive started (every %ds)", _KEEPALIVE_INTERVAL_SEC)
            else:
                logger.error("scenario setup returned no session id")
        except Exception as exc:
            logger.error("scenario setup failed: %s", exc)

    # Hand off PID 1 to whatever harbor's compose ``command:`` was
    # (typically ``sh -c "sleep infinity"``).
    cmd = sys.argv[1:]
    if cmd:
        os.execvp(cmd[0], cmd)  # noqa: S606
    else:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        signal.pause()


if __name__ == "__main__":
    _main()

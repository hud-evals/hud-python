"""A HUD runtime that runs each rollout in a fresh Daytona sandbox.

A runtime is any callable that takes a ``Task`` and returns an async context
manager yielding a ``hud.runtime.Runtime``: the address of the environment's
control channel. HUD connects there, runs the rollout, and exits the context,
which is where the runtime tears its host down. This one, per rollout:

1. builds the snapshot from ``image`` the first time its name is missing,
2. creates a sandbox from the snapshot,
3. starts ``hud serve`` in it, streaming its output to this terminal,
4. forwards a local port to the control channel over SSH, since Daytona exposes
   sandbox ports only as HTTPS previews,

and unwinds in reverse when the rollout ends, deleting the sandbox last.

    runtime = DaytonaRuntime(snapshot_name(), image=Image.from_dockerfile("Dockerfile.hud"))
    job = await fix_calc().run(agent, runtime=runtime)

The snapshot name is its identity: an existing snapshot is reused as is, so name
it after what goes into it (``snapshot.py``). Credentials come from the
environment: ``DAYTONA_API_KEY``, or ``DAYTONA_JWT_TOKEN`` with
``DAYTONA_ORGANIZATION_ID``.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import asyncssh
from daytona import (
    AsyncDaytona,
    AsyncSandbox,
    CreateSandboxFromSnapshotParams,
    CreateSnapshotParams,
    DaytonaNotFoundError,
    Image,
    Resources,
    SessionExecuteRequest,
)
from hud.runtime import Runtime

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from hud import Task

logger = logging.getLogger(__name__)

SSH_GATEWAY = "ssh.app.daytona.io"
SERVE_SESSION = "hud-serve"
SERVE_PORT = 8765
# Dockerfile.hud's CMD, started by hand: a snapshot's CMD is not the sandbox's main process, and
# a Daytona session starts in ~, not the image's WORKDIR. ``uv run`` puts the project venv on
# PATH, which the agent's shells inherit, so the task's ``pytest`` resolves.
SERVE_COMMAND = f"cd /app && uv run hud serve env:env --host 0.0.0.0 --port {SERVE_PORT}"
# A sandbox stops, and being ephemeral is deleted, after this long without a refresh. An open SSH
# forward does not count as activity, so the runtime refreshes it while the rollout runs; once
# this process is gone, a forgotten sandbox stops billing within the interval.
AUTO_STOP_MINUTES = 10
REFRESH_EVERY_S = 60


@dataclass(frozen=True, kw_only=True)
class DaytonaEndpoint(Runtime):
    """The runtime ``DaytonaRuntime`` yields: the control channel and the ``sandbox`` behind it
    (bench/warmpool.py reads its region)."""

    sandbox: AsyncSandbox


async def keep_alive(sandbox: AsyncSandbox) -> None:
    """Refresh ``sandbox``'s activity until cancelled; one failed refresh is not yet a stop."""
    while True:
        try:
            await sandbox.refresh_activity()
        except Exception:
            logger.warning("could not refresh Daytona sandbox %s", sandbox.id, exc_info=True)
        await asyncio.sleep(REFRESH_EVERY_S)


class DaytonaRuntime:
    """Places each rollout in a fresh sandbox from snapshot ``snapshot``, built from ``image``.

    ``resources`` sizes the snapshot when it is built; Daytona fixes a snapshot's
    resources then, so a different sizing needs a different snapshot name. A task
    that sets ``runtime_config`` is refused rather than run without it.
    """

    def __init__(
        self,
        snapshot: str,
        *,
        image: Image,
        resources: Resources | None = None,
        startup_timeout: float = 120,
    ) -> None:
        self.snapshot = snapshot
        self.image = image
        self.resources = resources
        self.startup_timeout = startup_timeout
        self.snapshot_ready = False
        self.snapshot_lock = asyncio.Lock()

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncGenerator[Runtime]:
        if task.runtime_config:
            raise ValueError(
                f"DaytonaRuntime sizes sandboxes through its snapshot; task {task.slug!r} sets runtime_config"
            )
        async with AsyncDaytona() as daytona, AsyncExitStack() as teardown:
            # 1. The snapshot, built once across concurrent rollouts.
            async with self.snapshot_lock:
                if not self.snapshot_ready:
                    try:
                        await daytona.snapshot.get(self.snapshot)
                    except DaytonaNotFoundError:
                        await daytona.snapshot.create(
                            CreateSnapshotParams(
                                name=self.snapshot, image=self.image, resources=self.resources
                            )
                        )
                    self.snapshot_ready = True

            # 2. A sandbox for this rollout, kept alive while it runs. Ephemeral: deleted
            # when it stops, and some regions allow no other kind.
            sandbox = await daytona.create(
                CreateSandboxFromSnapshotParams(
                    snapshot=self.snapshot, ephemeral=True, auto_stop_interval=AUTO_STOP_MINUTES
                ),
                timeout=self.startup_timeout,
            )
            teardown.push_async_callback(daytona.delete, sandbox)
            teardown.callback(asyncio.create_task(keep_alive(sandbox)).cancel)

            # 3. The server, in a session; HUD retries its handshake until it is up.
            await sandbox.process.create_session(SERVE_SESSION)
            command = await sandbox.process.execute_session_command(
                SERVE_SESSION, SessionExecuteRequest(command=SERVE_COMMAND, run_async=True)
            )
            output = asyncio.create_task(
                sandbox.process.get_session_command_logs_async(
                    SERVE_SESSION,
                    command.cmd_id,
                    partial(print, end="", flush=True),
                    partial(print, end="", flush=True, file=sys.stderr),
                )
            )
            teardown.callback(output.cancel)

            # 4. A local port forwarded to the server through Daytona's SSH gateway.
            access = await sandbox.create_ssh_access(expires_in_minutes=24 * 60)
            ssh = await teardown.enter_async_context(
                asyncssh.connect(SSH_GATEWAY, username=access.token, known_hosts=None)
            )
            listener = await ssh.forward_local_port("127.0.0.1", 0, "127.0.0.1", SERVE_PORT)
            yield DaytonaEndpoint(f"tcp://127.0.0.1:{listener.get_port()}", sandbox=sandbox)

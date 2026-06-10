"""Runtime + providers: how an execution substrate comes up.

A :class:`Runtime` is pure data — the connectable address of a substrate
serving the HUD control channel (``url`` + connection ``params``). A
*provider* is the scheduler half of placement: called with the task row it is
placing (the request — env name, args, whatever the row carries), it brings up
one fresh substrate for it and yields its ``Runtime`` (single-use
acquisitions, so per-rollout isolation is structural)::

    Provider = Callable[[Task], AbstractAsyncContextManager[Runtime]]

- :func:`spawn` — the local provider: each acquisition runs a subprocess
  serving the row's env from a ``.py`` source (uvicorn-shaped; the path is
  always given, never recovered from a live object).
- :func:`provision` — the HUD-hosted provider (control-plane spinup; not
  wired yet).
- ``Runtime(url)`` — the ``nullcontext`` of providers: called with any row it
  yields itself with a no-op lifecycle, i.e. a *borrowed, shared* substrate
  provisioned elsewhere, by explicit choice.

Per-task heterogeneity (this row on 1 GPU, that one on 4, different images)
is therefore just a provider that reads the row — the eval engine consumes
exactly this contract (``(on or provision())(task)``); new infra means a new
provider, never a new engine branch.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from .server import PORT_ANNOUNCEMENT, bind

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.eval.task import Task

    from .env import Environment

#: Provider contract: called with the task row being placed, acquires one
#: fresh substrate for it.
Provider: TypeAlias = Callable[["Task"], AbstractAsyncContextManager["Runtime"]]


@dataclass(frozen=True)
class Runtime:
    """The connectable address of a provisioned substrate.

    ``url`` is the control-channel address (``tcp://127.0.0.1:7000`` for a
    local process, ``tcp://sandbox-abc.hud.so:443`` for a hosted box);
    ``params`` carries connection-time data a transport may need (auth token,
    sandbox id). Constructed directly, it is also a provider — the borrowed,
    shared case: it ignores the placement request and yields itself with a
    no-op lifecycle, since whoever provisioned the substrate owns its
    teardown.
    """

    url: str
    params: dict[str, Any] = field(default_factory=dict)

    def __call__(self, task: Task) -> AbstractAsyncContextManager[Runtime]:
        return nullcontext(self)


def spawn(
    path: str | Path,
    *,
    env: str | None = None,
    ready_timeout: float = 120.0,
) -> Provider:
    """The local provider: serve the placed row's env from *path* in a child process.

    Each acquisition runs ``python -m hud.environment.server <path> --env
    name`` — the same serving entry point a container CMD runs — on an
    ephemeral loopback port, yields its :class:`Runtime`, and terminates the
    child on exit. *path* is a ``.py`` file or a directory of them. The served
    env is the placed task's ``env.name`` (so a mixed-env taskset works
    against one source), unless *env* pins one explicitly; placing a row whose
    env the source does not define fails loudly in the child.

    The child's working directory is the source's directory, so sibling
    imports and relative data paths resolve; ``@env.initialize`` daemons start
    in the child and die with it. Because the source is re-imported in the
    child, a script spawning itself (``spawn(__file__)``) must keep top-level
    run calls under ``if __name__ == "__main__":``.
    """
    source = Path(path).resolve()

    @asynccontextmanager
    async def acquire(task: Task) -> AsyncIterator[Runtime]:
        if not source.exists():
            raise FileNotFoundError(f"spawn: source not found: {source}")
        cmd = [sys.executable, "-m", "hud.environment.server", str(source)]
        cmd += ["--env", env or task.env.name]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            cwd=source if source.is_dir() else source.parent,
        )
        try:
            port = await asyncio.wait_for(_read_port(proc, source), ready_timeout)
            assert proc.stdout is not None
            drain = asyncio.create_task(_drain(proc.stdout))
            try:
                yield Runtime(f"tcp://127.0.0.1:{port}")
            finally:
                drain.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await drain
        finally:
            await _terminate(proc)

    return acquire


def provision(**opts: Any) -> Provider:
    """The HUD-hosted provider: one substrate per acquisition, by the row's env name.

    Not wired to the platform control plane yet; acquiring raises a precise
    error naming the placements that work today.
    """

    @asynccontextmanager
    async def acquire(task: Task) -> AsyncIterator[Runtime]:
        raise NotImplementedError(
            f"HUD-hosted provisioning (env {task.env.name!r}) is not wired up yet. "
            "Pass a placement instead: on=spawn('path/to/env.py') to serve a local "
            "source, or on=Runtime(url) to attach to an already-served env."
        )
        yield  # pragma: no cover - generator shape for the asynccontextmanager contract

    return acquire


@asynccontextmanager
async def _local(env: Environment) -> AsyncIterator[Runtime]:
    """Substrate-side serving: a live env owned by *this* process, as a runtime.

    Not a placement the engine offers (the orchestrator never serves an env
    in-process), so deliberately not a ``Provider`` — it serves a live object,
    not a placed row. Code already running *inside* a placed substrate adapts
    it (``AgentTool`` sub-rollouts: ``on=lambda _: _local(env)``); test
    harnesses enter it directly.
    """
    await env.start()
    server = await bind(env, "127.0.0.1", 0)
    host, port = server.sockets[0].getsockname()[:2]
    serve_task = asyncio.create_task(server.serve_forever())
    try:
        yield Runtime(f"tcp://{host}:{port}")
    finally:
        serve_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serve_task
        server.close()
        with contextlib.suppress(Exception):
            await server.wait_closed()
        await env.stop()


async def _read_port(proc: asyncio.subprocess.Process, source: Path) -> int:
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            raise RuntimeError(
                f"spawned env exited with code {await proc.wait()} before serving "
                f"(source: {source}); see its stderr above",
            )
        text = line.decode("utf-8", "replace").strip()
        if text.startswith(PORT_ANNOUNCEMENT):
            return int(text.removeprefix(PORT_ANNOUNCEMENT))


async def _drain(stream: asyncio.StreamReader) -> None:
    """Keep consuming the child's stdout so it never blocks on a full pipe."""
    while await stream.read(65536):
        pass


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), 10.0)
    except TimeoutError:
        proc.kill()
        await proc.wait()


__all__ = ["Provider", "Runtime", "provision", "spawn"]

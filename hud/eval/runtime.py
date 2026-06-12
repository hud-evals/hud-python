"""Runtime + providers: how an execution substrate comes up.

A :class:`Runtime` is pure data — the connectable address of a substrate
serving the HUD control channel (``url`` + connection ``params``). A
:class:`Provider` is the scheduler half of placement: called with the task
row it is placing (the request — env name, args, whatever the row carries),
it brings up one fresh substrate for it and yields its ``Runtime``
(single-use acquisitions, so per-rollout isolation is structural).

- :class:`LocalRuntime` — the local provider: each acquisition runs a subprocess
  serving the row's env from a ``.py`` source (uvicorn-shaped; the path is
  always given, never recovered from a live object).
- :class:`DockerRuntime` — the container provider: each acquisition ``docker run``s
  an image whose CMD serves the control channel.
- :class:`HUDRuntime` — the HUD-hosted provider (control-plane spinup; not
  wired yet).
- ``Runtime(url)`` — the ``nullcontext`` of providers: called with any row it
  yields itself with a no-op lifecycle, i.e. a *borrowed, shared* substrate
  provisioned elsewhere, by explicit choice.

The contract is structural (anything callable as ``(task) -> async context
manager of Runtime``), so a provider can be a class holding real state — a
platform session, an image cache, a warm pool — or just a closure. Per-task
heterogeneity (this row on 1 GPU, that one on 4, different images) is
therefore just a provider that reads the row — the eval engine consumes
exactly this contract (``(runtime or HUDRuntime())(task)``); new infra means
a new provider, never a new engine branch.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from hud.environment.env import Environment

    from .task import Task


class Provider(Protocol):
    """Placement contract: called with the task row being placed, acquire one
    fresh substrate for it and yield its :class:`Runtime`."""

    def __call__(self, task: Task, /) -> AbstractAsyncContextManager[Runtime]: ...


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


class LocalRuntime:
    """The local provider: serve the placed row's env from *path* in a child process.

    Each acquisition runs ``python -m hud.environment.server <path> --env
    name`` — the same serving entry point a container CMD runs — on an
    ephemeral loopback port, yields its :class:`Runtime`, and terminates the
    child on exit. *path* is a ``.py`` file or a directory of them. The served
    env is the placed task's ``env`` name (so a mixed-env taskset works
    against one source), unless *env* pins one explicitly; placing a row whose
    env the source does not define fails loudly in the child.

    The child's working directory is the source's directory, so sibling
    imports and relative data paths resolve; ``@env.initialize`` daemons start
    in the child and die with it. Because the source is re-imported in the
    child, a script spawning itself (``LocalRuntime(__file__)``) must keep top-level
    run calls under ``if __name__ == "__main__":``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        env: str | None = None,
        ready_timeout: float = 120.0,
    ) -> None:
        self.source = Path(path).resolve()
        self.env = env
        self.ready_timeout = ready_timeout

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        if not self.source.exists():
            raise FileNotFoundError(f"LocalRuntime: source not found: {self.source}")
        cmd = [sys.executable, "-m", "hud.environment.server", str(self.source)]
        cmd += ["--env", self.env or task.env]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            cwd=self.source if self.source.is_dir() else self.source.parent,
        )
        try:
            port = await asyncio.wait_for(_read_port(proc, self.source), self.ready_timeout)
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


class DockerRuntime:
    """The container provider: each acquisition ``docker run``s a fresh *image*.

    The image's CMD serves the env's control channel on *port* inside the
    container (the scaffolded ``Dockerfile.hud`` serves 8765). Each
    acquisition publishes that port on an ephemeral loopback port, yields its
    :class:`Runtime`, and force-removes the container on exit. *run_args* are
    extra ``docker run`` flags (``-e``, ``--gpus``, volumes); per-task
    heterogeneity (this row on one image, that row on another) is a custom
    provider reading the row.

    Acquisition returns as soon as the port mapping exists — the env may
    still be importing behind it. Protocol-level readiness is the client's
    job: ``connect`` retries the handshake until the channel answers.
    """

    def __init__(self, image: str, *, port: int = 8765, run_args: Sequence[str] = ()) -> None:
        self.image = image
        self.port = port
        self.run_args = tuple(run_args)

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        out, _ = await _docker(
            "run", "--detach", *self.run_args, "--publish", f"127.0.0.1::{self.port}", self.image
        )
        container = out.strip()
        try:
            mapping, _ = await _docker("port", container, str(self.port))
            if not mapping.strip():
                logs_out, logs_err = await _docker("logs", "--tail", "40", container, check=False)
                raise RuntimeError(
                    f"container for image {self.image!r} exited before serving port "
                    f"{self.port}:\n{(logs_err or logs_out).strip()}",
                )
            host_port = int(mapping.strip().splitlines()[0].rsplit(":", 1)[1])
            yield Runtime(f"tcp://127.0.0.1:{host_port}")
        finally:
            # check=False: teardown must not shadow the run's own error, and
            # rm -f only fails when the daemon itself is broken.
            await _docker("rm", "--force", container, check=False)


async def _docker(*args: str, check: bool = True) -> tuple[str, str]:
    """Run a docker CLI command and return decoded ``(stdout, stderr)``."""
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if check and proc.returncode != 0:
        detail = err.decode("utf-8", "replace").strip() or out.decode("utf-8", "replace").strip()
        raise RuntimeError(f"docker {' '.join(args)} failed ({proc.returncode}): {detail}")
    return out.decode("utf-8", "replace"), err.decode("utf-8", "replace")


class HUDRuntime:
    """The HUD-hosted provider: one substrate per acquisition, by the row's env name.

    The instance is where the platform session will live (auth, sandbox
    handles) once control-plane spinup is wired; until then acquiring raises
    a precise error naming the placements that work today.
    """

    def __init__(self, **opts: Any) -> None:
        self.opts = opts

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        raise NotImplementedError(
            f"HUD-hosted provisioning (env {task.env!r}) is not wired up yet. "
            "Pass a placement instead: runtime=LocalRuntime('path/to/env.py') to serve a "
            "local source, or runtime=Runtime(url) to attach to an already-served env."
        )
        yield  # pragma: no cover - generator shape for the asynccontextmanager contract


@asynccontextmanager
async def _local(env: Environment) -> AsyncIterator[Runtime]:
    """Substrate-side serving: a live env owned by *this* process, as a runtime.

    Not a placement the engine offers (the orchestrator never serves an env
    in-process), so deliberately not a ``Provider`` — it serves a live object,
    not a placed row. Code already running *inside* a placed substrate adapts
    it (``AgentTool`` sub-rollouts: ``runtime=lambda _: _local(env)``); test
    harnesses enter it directly.
    """
    from hud.environment.server import bind

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
    # Imported lazily: a module-level import would pre-load hud.environment.server
    # in every `python -m hud.environment.server` child, tripping runpy's
    # found-in-sys.modules RuntimeWarning on each spawned rollout.
    from hud.environment.server import PORT_ANNOUNCEMENT

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


__all__ = ["DockerRuntime", "HUDRuntime", "LocalRuntime", "Provider", "Runtime"]

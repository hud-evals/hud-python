"""Provider: server placement — where the env's control channel comes from.

A :class:`Provider` brings up the *server* (the env's control channel) for one
rollout and yields its connectable :class:`Runtime`; the agent loop drives it
from this process (:func:`hud.eval.run.rollout`). The channel is location
transparent, so "co-located" (loopback) and "split" (agent here, env
elsewhere) are the same code, differing only in the url.

- :class:`LocalRuntime` — runs a subprocess serving the row's env from a ``.py``
  source (the path is always given, never recovered from a live object).
- :class:`DockerRuntime` — ``docker run``s an image whose CMD serves the channel.
- ``Runtime(url)`` — the ``nullcontext`` of providers: yields itself, a
  *borrowed, shared* substrate provisioned elsewhere (env served anywhere —
  a cloud sandbox, another host — that this process connects to).

The provider contract is structural (anything callable as ``(task) -> async
context manager of Runtime``), so per-task heterogeneity (this row on 1 GPU,
that one on 4, different images) is just a provider that reads the row.

The *other* placement — :class:`HUDRuntime`, running the whole rollout off-box
on a HUD sandbox — also lives here; the scheduler (:meth:`Taskset.run`)
chooses between it and a provider. A hosted box's own driver is
itself a ``Provider`` (its ``DockerRuntime``) driven by the same ``rollout``
atom — co-location all the way down.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
import uuid
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from hud.types import Step
from hud.utils.platform import PlatformClient

from .run import Grade, Run

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from hud.agents.base import Agent
    from hud.environment.env import Environment

    from .task import Task

logger = logging.getLogger("hud.eval.runtime")


class Provider(Protocol):
    """Server placement: called with the task row being placed, acquire one
    fresh env substrate for it and yield its connectable :class:`Runtime`.

    A provider brings up the *server* (the env's control channel) wherever it
    lives — a local subprocess, a container, a cloud sandbox — and the agent
    loop drives it from this process (:func:`hud.eval.run.rollout`). The
    channel is location-transparent, so "co-located" (loopback) and "split"
    (agent here, env elsewhere) are the same code, differing only in the url.
    """

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
            # Start child in its own session for clean signal handling.
            start_new_session=True,
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
    # Child leads its own group (pgid == pid), so SIGTERM the whole group to
    # reap env-spawned grandchildren too, then SIGKILL stragglers. Windows has
    # no killpg — fall back to the direct child.
    with contextlib.suppress(ProcessLookupError):
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), 10.0)
    except TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            if hasattr(os, "killpg"):
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        await proc.wait()


#: Platform trace statuses that end a hosted rollout.
_TERMINAL_TRACE_STATUSES = frozenset({"completed", "error", "cancelled"})


class HUDRuntime:
    """HUD-hosted placement: runs the rollout on a leased box and returns its ``Run``.

    The *client-elsewhere* placement. Where a :class:`Provider` yields a channel
    this process drives, ``HUDRuntime`` runs the whole rollout off-box: the
    platform leases an instance, brings the env's container up on it, and runs
    the agent right next to it (the instance-side driver is just
    :func:`hud.eval.run.rollout` over a ``DockerRuntime`` — co-location all the
    way down). This process only submits the rollout and polls the trace to
    completion, folding the result into a :class:`~hud.eval.run.Run`. Because
    the agent runs remotely, its identity travels via :func:`_agent_spec`.

    ``run_timeout`` bounds one rollout end to end, including instance
    provisioning (a cold EC2 boot plus image pull), queueing, and the agent
    run itself. A local cancel (Ctrl-C) requests a platform-side cancel before
    propagating, so abandoned rollouts do not hold instances open.
    """

    def __init__(self, *, poll_interval: float = 5.0, run_timeout: float = 3600.0) -> None:
        self.poll_interval = poll_interval
        self.run_timeout = run_timeout

    async def run(
        self,
        task: Task,
        agent: Agent,
        *,
        job_id: str,
        group_id: str | None = None,
        trace_id: str | None = None,
    ) -> Run:
        """Submit one rollout, await its terminal trace, and fold it into a ``Run``.

        The platform owns the trace lifecycle (the instance-side driver reports
        enter/exit and streams telemetry), so this never double-reports.
        Failures isolating one rollout from its batch (submit rejected, the
        env/model unresolved) surface as :meth:`Run.failed`; a timeout or a
        local cancel propagate, having first asked the platform to release the
        lease.
        """
        trace_id = trace_id or uuid.uuid4().hex
        try:
            state = await self._submit_and_await(
                task, agent, job_id=job_id, group_id=group_id, trace_id=trace_id
            )
        except (TimeoutError, asyncio.CancelledError):
            raise
        except Exception as exc:
            logger.warning("hosted rollout failed to launch: %s", exc)
            run = Run.failed(str(exc))
        else:
            run = self._fold(state, trace_id)
        run.trace.trace_id = trace_id
        run.job_id = job_id
        run.group_id = group_id
        return run

    async def _submit_and_await(
        self,
        task: Task,
        agent: Agent,
        *,
        job_id: str,
        group_id: str | None,
        trace_id: str,
    ) -> dict[str, Any]:
        spec_of = getattr(agent, "hosted_spec", None)
        if not callable(spec_of):
            raise ValueError(
                f"hosted execution requires a gateway agent that can serialize its "
                f"identity (Claude/OpenAI/Gemini/OpenAIChat); got {type(agent).__name__}"
            )
        spec = spec_of()
        platform = PlatformClient.from_settings()
        if not platform.api_key:
            raise RuntimeError("HUD-hosted execution requires HUD_API_KEY")
        payload: dict[str, Any] = {
            # The SDK's hex ids travel as canonical UUID strings.
            "trace_id": str(uuid.UUID(trace_id)),
            "job_id": str(uuid.UUID(job_id)),
            "env": task.env,
            "task": task.id,
            "args": task.args,
            "agent": spec,
        }
        if group_id is not None:
            payload["group_id"] = group_id
        await platform.apost("/rollouts/submit", json=payload)
        try:
            return await self._await_terminal(platform, payload["trace_id"])
        except asyncio.CancelledError:
            await self._cancel(platform, payload["trace_id"])
            raise

    @staticmethod
    def _fold(state: dict[str, Any], trace_id: str) -> Run:
        """Build the local view of a remotely-executed rollout from its trace state."""
        run = Run(None, "", {})
        # The poll loop only returns terminal states, so the status is one of
        # the trace vocabulary; anything else would be a platform bug.
        status = state.get("status")
        run.trace.status = status if status in ("completed", "error", "cancelled") else "error"
        error = state.get("error")
        if error:
            run.record(Step(source="system", error=str(error)))
        reward = state.get("reward")
        run.grade = Grade(
            reward=float(reward) if reward is not None else 0.0,
            is_error=status == "error",
            content=str(error) if error else None,
        )
        run._runtime = f"hud://trace/{trace_id}"
        return run

    async def _await_terminal(self, platform: PlatformClient, trace_id: str) -> dict[str, Any]:
        loop = asyncio.get_event_loop()
        deadline = loop.time() + self.run_timeout
        while True:
            state: dict[str, Any] = await platform.aget(f"/trace/{trace_id}")
            if state.get("status") in _TERMINAL_TRACE_STATUSES:
                return state
            if loop.time() >= deadline:
                await self._cancel(platform, trace_id)
                raise TimeoutError(
                    f"hosted rollout {trace_id} did not finish within "
                    f"{self.run_timeout:.0f}s (status: {state.get('status')})"
                )
            await asyncio.sleep(self.poll_interval)

    async def _cancel(self, platform: PlatformClient, trace_id: str) -> None:
        # The platform also bounds instances by max runtime; this just releases
        # the lease promptly. Never shadow the caller's outcome.
        try:
            await platform.apost("/rollouts/cancel", json={"trace_id": trace_id})
        except Exception as exc:
            logger.warning("hosted rollout %s cancel failed: %s", trace_id, exc)


__all__ = [
    "DockerRuntime",
    "HUDRuntime",
    "LocalRuntime",
    "Provider",
    "Runtime",
]

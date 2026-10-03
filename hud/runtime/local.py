"""Local in-process and subprocess runtime providers."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import sys
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING
from weakref import WeakValueDictionary

from hud.environment import load_environment
from hud.environment.env import Environment
from hud.environment.server import PORT_ANNOUNCEMENT
from hud.environment.session import Session
from hud.utils.process import (
    OUTPUT_DRAIN_TIMEOUT,
    create_process_group_exec,
    finish_output,
    output_writer,
    stream_output,
)

from .core import Runtime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from hud.capabilities import Connection
    from hud.environment.egress import WorkspaceRoute
    from hud.eval.run import EnvironmentSession
    from hud.eval.task import Task


# Acquisitions hold their locks; completed environments are not retained here.
_ENVIRONMENT_LOCKS: WeakValueDictionary[Environment, asyncio.Lock] = WeakValueDictionary()


@dataclass(frozen=True)
class _InProcessRuntime(Runtime):
    """An environment started in this process."""

    env: Environment = field(kw_only=True, repr=False, compare=False)

    @asynccontextmanager
    async def connect(
        self,
        *,
        session_id: str | None = None,
        lease: float | None = None,
        workspace_routes: Sequence[WorkspaceRoute] = (),
        connections: Sequence[Connection] = (),
    ) -> AsyncIterator[EnvironmentSession]:
        if session_id is not None or lease is not None:
            raise ValueError("an in-process session ends with this block; none outlives it")
        session = Session(self.env)
        try:
            session.bind(connections=connections, workspace_routes=workspace_routes)
            yield session
        finally:
            await session.close()


class LocalRuntime:
    """The local provider: run an environment in this process, with no control channel.

    ``source`` is a ``.py`` file or directory loaded per acquisition (``env`` picks
    one of several environments, defaulting to the task's), a live ``Environment``,
    or a ``(task) -> Environment`` constructor called per acquisition. The yielded
    runtime's url, ``local://<env name>``, names the environment.

    ``ready_timeout`` bounds ``@env.initialize``. Acquisitions of one environment
    instance are serialized. Hooks share this event loop, so blocking env code stalls
    concurrent rollouts; ``SubprocessRuntime`` isolates them in a process.
    """

    def __init__(
        self,
        source: str | Path | Environment | Callable[[Task], Environment],
        *,
        env: str | None = None,
        ready_timeout: float = 120.0,
    ) -> None:
        self.ready_timeout = ready_timeout
        self._source_dir: Path | None = None
        if isinstance(source, Environment):
            if env is not None:
                raise TypeError("LocalRuntime: env= applies only to source paths")
            self._build: Callable[[Task], Environment] = lambda _task: source
        elif isinstance(source, (str, Path)):
            path, pinned = Path(source).resolve(), env
            self._source_dir = path if path.is_dir() else path.parent

            def _load(task: Task) -> Environment:
                return load_environment(path, name=pinned or task.env)

            self._build: Callable[[Task], Environment] = _load
        elif callable(source):
            if env is not None:
                raise TypeError("LocalRuntime: env= applies only to source paths")
            self._build = source
        else:
            raise TypeError(
                f"LocalRuntime: expected a source path, an Environment, or a "
                f"(task) -> Environment constructor; got {source!r}"
            )

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        if task.runtime_config is not None:
            raise ValueError("LocalRuntime does not support task runtime_config")
        # Importable for the whole acquisition so templates can lazily import siblings;
        # one insert and one remove per acquisition keeps sys.path balanced.
        if self._source_dir is not None:
            sys.path.insert(0, str(self._source_dir))
        try:
            try:
                env = self._build(task)
            except RuntimeError as e:
                # The source ran an event loop at import, usually an unguarded run call.
                if "running event loop" not in str(e):
                    raise
                raise RuntimeError(
                    "the env source ran async code while being imported to place a "
                    'rollout — guard top-level run calls with `if __name__ == "__main__":`'
                ) from e
            if not isinstance(env, Environment):
                raise TypeError(f"LocalRuntime: constructor returned {env!r}, not an Environment")

            async with _ENVIRONMENT_LOCKS.setdefault(env, asyncio.Lock()):
                try:
                    await asyncio.wait_for(env.start(), self.ready_timeout)
                    yield _InProcessRuntime(f"local://{env.name}", env=env)
                finally:
                    await env.stop()
        finally:
            if self._source_dir is not None:
                with contextlib.suppress(ValueError):
                    sys.path.remove(str(self._source_dir))


class SubprocessRuntime:
    """The child-process provider: serve the placed row's env from ``source``.

    ``source`` is a ``.py`` file, a directory, or a live ``Environment`` (served from
    the file declaring its templates). Each acquisition serves the task's env, unless
    ``env`` pins one, from a child in the source's directory, terminated on exit. The
    child re-imports the source, so a script spawning itself must guard top-level run
    calls with ``if __name__ == "__main__":``.
    """

    def __init__(
        self,
        source: str | Path | Environment,
        *,
        env: str | None = None,
        ready_timeout: float = 120.0,
    ) -> None:
        if isinstance(source, Environment):
            if env is not None:
                raise TypeError("SubprocessRuntime: env= applies only to source paths")
            files = {Path(inspect.getfile(t.func)).resolve() for t in source.tasks.values()}
            if len(files) != 1:
                raise ValueError(
                    f"SubprocessRuntime: Environment {source.name!r} must declare its "
                    "@env.template tasks in exactly one source file to be served from source; "
                    "serve a source path instead"
                )
            self.source = files.pop()
            self.env: str | None = source.name
        else:
            self.source = Path(source).resolve()
            self.env = env
        self.ready_timeout = ready_timeout

    @asynccontextmanager
    async def __call__(self, task: Task) -> AsyncIterator[Runtime]:
        if task.runtime_config is not None:
            raise ValueError("SubprocessRuntime does not support task runtime_config")
        if not self.source.exists():
            raise FileNotFoundError(f"SubprocessRuntime: source not found: {self.source}")
        cmd = [sys.executable, "-m", "hud.environment.server", str(self.source)]
        cmd += ["--env", self.env or task.env]
        proc = await create_process_group_exec(
            *cmd,
            term_timeout=10.0,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.source if self.source.is_dir() else self.source.parent,
        )
        output = proc.stdout
        error = proc.stderr
        assert output is not None
        assert error is not None
        output_tail: deque[str] = deque(maxlen=50)
        error_tail: deque[str] = deque(maxlen=50)
        stdout_drain: asyncio.Task[None] | None = None

        def capture_error(text: str) -> None:
            error_tail.append(text.strip()[-1024:])

        stderr_drain = asyncio.create_task(stream_output(error, sys.stderr, capture=capture_error))
        try:
            write_stdout, finish_stdout = output_writer(
                sys.stdout,
                capture=lambda text: output_tail.append(text.strip()[-1024:]),
            )

            async def read_port() -> int | None:
                """The port the child announces, or ``None`` once stdout ends without one."""
                line_continues = False
                while True:
                    eof = False
                    try:
                        line = await output.readuntil()
                    except asyncio.LimitOverrunError as exc:
                        line = await output.read(exc.consumed)
                        line_continues = True
                    except asyncio.IncompleteReadError as exc:
                        line = exc.partial
                        eof = True
                    if not line:
                        return None
                    if not line_continues:
                        text = line.decode("utf-8", "replace").strip()
                        if text.startswith(PORT_ANNOUNCEMENT):
                            return int(text.removeprefix(PORT_ANNOUNCEMENT))
                    write_stdout(line)
                    line_continues = not line.endswith(b"\n")
                    if eof:
                        return None

            reading = asyncio.create_task(read_port())
            exited = asyncio.create_task(proc.wait())
            try:
                async with asyncio.timeout(self.ready_timeout):
                    await asyncio.wait((reading, exited), return_when=asyncio.FIRST_COMPLETED)
                    if not reading.done():
                        # The child exited without announcing a port, and a process
                        # it started may still hold stdout open: take what is there.
                        await asyncio.wait((reading,), timeout=OUTPUT_DRAIN_TIMEOUT)
            finally:
                exited.cancel()
                reading.cancel()
                await asyncio.gather(reading, exited, return_exceptions=True)
                finish_stdout()
            port = None if reading.cancelled() else reading.result()
            if port is None:
                # Stdout ends as the child exits; give it a moment to be reaped
                # so the error reports its exit code rather than a closed pipe.
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(1.0):
                        await proc.wait()
                await finish_output(stderr_drain)
                tail = "\n".join((*output_tail, *error_tail)).strip()
                detail = f":\n{tail}" if tail else " (no output captured)"
                status = (
                    f"exited with code {proc.returncode}"
                    if proc.returncode is not None
                    else "closed stdout"
                )
                raise RuntimeError(
                    f"spawned env {status} before serving (source: {self.source}){detail}"
                )

            stdout_drain = asyncio.create_task(stream_output(output, sys.stdout))
            yield Runtime(f"tcp://127.0.0.1:{port}")
        finally:
            await proc.terminate()
            await finish_output(
                *(task for task in (stdout_drain, stderr_drain) if task is not None)
            )

"""Local in-process and subprocess runtime providers."""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from hud.utils.process import (
    create_process_group_exec,
    stream_output,
    write_output,
)

from .core import Runtime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from hud.environment.env import Environment
    from hud.eval.task import Task


class LocalRuntime:
    """The local provider: serve a fresh env per rollout, in this process.

    *source* points at the env in whatever form you have:

    - a ``.py`` file or directory — imported fresh per acquisition (sibling
      imports resolve); *env* pins one name when several are declared,
      defaulting to the placed task's env
    - a live :class:`~hud.environment.Environment` — served directly, one
      acquisition at a time
    - a ``(task) -> Environment`` callable — called per acquisition with the
      placed row

    ::

        runtime = LocalRuntime("env.py")
        runtime = LocalRuntime(env)
        runtime = LocalRuntime(lambda task: build_env(task.env))

    ``ready_timeout`` bounds ``@env.initialize`` startup. Source paths and
    constructors create a fresh environment per acquisition. Hooks share this
    process's event loop, so blocking env code stalls concurrent rollouts —
    use :class:`SubprocessRuntime` or :class:`DockerRuntime` for process
    isolation, and ``Runtime(url)`` to attach to a substrate served elsewhere.
    """

    def __init__(
        self,
        source: str | Path | Environment | Callable[[Task], Environment],
        *,
        env: str | None = None,
        ready_timeout: float = 120.0,
    ) -> None:
        from hud.environment.env import Environment as _Environment

        self.ready_timeout = ready_timeout
        self._source_dir: Path | None = None
        self._live_lock: asyncio.Lock | None = None
        if isinstance(source, _Environment):
            if env is not None:
                raise TypeError("LocalRuntime: env= applies only to source paths")
            self._build: Callable[[Task], Environment] = lambda _task: source
            self._live_lock = asyncio.Lock()
        elif isinstance(source, (str, Path)):
            path, pinned = Path(source).resolve(), env
            self._source_dir = path if path.is_dir() else path.parent
            from hud.environment import load_environment

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
        from hud.environment.env import Environment as _Environment

        if task.runtime_config is not None:
            raise ValueError("LocalRuntime does not support task runtime_config")
        # The source dir stays importable for the whole acquisition, not just
        # the initial import, so a template can lazily import a sibling
        # module at run time (as it could under the child-process runtime).
        # Always insert-and-remove one entry: balanced under concurrency.
        if self._live_lock is not None:
            await self._live_lock.acquire()
        if self._source_dir is not None:
            sys.path.insert(0, str(self._source_dir))
        try:
            try:
                env = self._build(task)
            except RuntimeError as e:
                # The source ran an event loop at import — usually an unguarded
                # top-level run call; name the actual mistake.
                if "running event loop" not in str(e):
                    raise
                raise RuntimeError(
                    "the env source ran async code while being imported to place a "
                    'rollout — guard top-level run calls with `if __name__ == "__main__":`'
                ) from e
            if not isinstance(env, _Environment):
                raise TypeError(f"LocalRuntime: constructor returned {env!r}, not an Environment")
            from hud.environment.server import _shutdown, bind

            try:
                await asyncio.wait_for(env.start(), self.ready_timeout)
                server = await bind(env, "127.0.0.1", 0)
                host, port = server.sockets[0].getsockname()[:2]
                serve_task = asyncio.create_task(server.serve_forever())
                try:
                    yield Runtime(f"tcp://{host}:{port}")
                finally:
                    serve_task.cancel()
                    await _shutdown(server)
                    with contextlib.suppress(asyncio.CancelledError):
                        await serve_task
            finally:
                await env.stop()
        finally:
            if self._source_dir is not None:
                with contextlib.suppress(ValueError):
                    sys.path.remove(str(self._source_dir))
            if self._live_lock is not None:
                self._live_lock.release()


class SubprocessRuntime:
    """The child-process provider: serve the placed row's env from *path*.

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
    child, a script spawning itself (``SubprocessRuntime(__file__)``) must keep
    top-level run calls under ``if __name__ == "__main__":``.
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
            stderr=asyncio.subprocess.STDOUT,
            cwd=self.source if self.source.is_dir() else self.source.parent,
        )
        assert proc.stdout is not None
        output = proc.stdout
        output_tail: deque[str] = deque(maxlen=50)
        drain: asyncio.Task[None] | None = None
        try:
            from hud.environment.server import PORT_ANNOUNCEMENT

            port = None
            async with asyncio.timeout(self.ready_timeout):
                while line := await output.readline():
                    text = line.decode("utf-8", "replace").strip()
                    if text.startswith(PORT_ANNOUNCEMENT):
                        port = int(text.removeprefix(PORT_ANNOUNCEMENT))
                        break
                    output_tail.append(text)
                    write_output(sys.stdout, line)
            if port is None:
                code = await proc.wait()
                tail = "\n".join(output_tail).strip()
                detail = f":\n{tail}" if tail else " (no output captured)"
                raise RuntimeError(
                    f"spawned env exited with code {code} before serving "
                    f"(source: {self.source}){detail}"
                )

            drain = asyncio.create_task(stream_output(output, sys.stdout))
            yield Runtime(f"tcp://127.0.0.1:{port}")
        finally:
            await proc.terminate()
            if drain is not None:
                await asyncio.gather(drain, return_exceptions=True)

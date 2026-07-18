"""``RobotEndpoint`` — the env server's handle on a sim process.

A bridge always lives in the sim's own process (see :mod:`~.bridge`); the
endpoint is the JSON-RPC client that drives it through episodes (``reset`` /
``result``) and mints its capability — and, when it spawned the process, owns
its lifecycle. Two ways to build one, identical methods either way:

- **Spawned** — :meth:`RobotEndpoint.spawn`: fork the sim program, read its
  announced control port, tear the process down on :meth:`stop`
  (``env.gym(...)`` builds this).
- **Attached** — :meth:`RobotEndpoint.remote`: dial a sim process something
  else runs (another container, a warm Isaac kept alive across env-server
  restarts); :meth:`stop` only drops the link, never the process.

Control plane only: the agent's step/observation loop tunnels straight to the
bridge's ``robot`` WebSocket, and templates drive episodes through the handle::

    sim = env.gym(make_env)


    @env.template(id="pawn_lift")
    async def pawn_lift(task: str = "solo_pawn_lift", seed: int = 0, num_envs: int = 1):
        yield {"prompt": await sim.reset(task=task, seed=seed, num_envs=num_envs)}
        yield await sim.result()
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any

from hud.environment.utils import read_frame, send_frame
from hud.utils.process import create_process_group_exec

from .bridge import PORT_ANNOUNCEMENT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hud.capabilities import Capability
    from hud.utils.process import ProcessGroup


class RobotEndpoint:
    """Drive a simulation bridge living in another process.

    Build with :meth:`spawn` (own the sim process) or :meth:`remote` (attach to
    one served elsewhere); :meth:`start` brings the link up either way.
    """

    def __init__(
        self,
        *,
        cmd: Sequence[str] | None = None,
        host: str | None = None,
        port: int | None = None,
        connect_timeout_s: float = 240.0,
    ) -> None:
        self._cmd = list(cmd) if cmd is not None else None  # set => spawned mode
        self._host = host
        self._port = port
        self._connect_timeout_s = connect_timeout_s
        self._proc: ProcessGroup | None = None
        self._forward: asyncio.Task[None] | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    @classmethod
    def spawn(cls, cmd: Sequence[str], *, connect_timeout_s: float = 240.0) -> RobotEndpoint:
        """An endpoint that forks *cmd* (a sim program; see :mod:`~.bridge`) and owns it."""
        return cls(cmd=cmd, connect_timeout_s=connect_timeout_s)

    @classmethod
    def remote(cls, host: str, port: int, *, connect_timeout_s: float = 240.0) -> RobotEndpoint:
        """An endpoint attached to a sim process something else runs."""
        return cls(host=host, port=port, connect_timeout_s=connect_timeout_s)

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Bring the link up: fork the sim program (spawned mode) and connect."""
        if self._cmd is not None and self._proc is None:
            self._proc = await create_process_group_exec(
                *self._cmd,
                term_timeout=10.0,
                stdout=asyncio.subprocess.PIPE,  # for the port announcement; stderr inherits
            )
            self._host = "127.0.0.1"
            self._port = await asyncio.wait_for(
                self._read_announced_port(), self._connect_timeout_s
            )
            # Keep passing the sim's stdout through so its logs stay visible
            # (and the pipe never fills and blocks the child).
            assert self._proc.stdout is not None
            self._forward = asyncio.create_task(_forward_lines(self._proc.stdout))
        await self._connect()

    async def stop(self) -> None:
        """Drop the link; tear the sim process down when this endpoint spawned it."""
        if self._forward is not None:
            self._forward.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._forward
            self._forward = None
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
            self._reader = self._writer = None
        if self._proc is not None:  # owned: SIGTERM the sim's whole process group
            await self._proc.terminate()
            self._proc = None

    async def _read_announced_port(self) -> int:
        """The sim program's ``HUD_SIM_PORT=`` line, passing boot logs through."""
        assert self._proc is not None and self._proc.stdout is not None
        while line := await self._proc.stdout.readline():
            text = line.decode("utf-8", "replace").rstrip()
            if text.startswith(PORT_ANNOUNCEMENT):
                return int(text.removeprefix(PORT_ANNOUNCEMENT))
            print(text, flush=True)
        code = await self._proc.wait()
        raise RuntimeError(f"sim process exited with code {code} before announcing its port")

    async def _connect(self, retry_every: float = 2.0) -> None:
        """Dial the control channel, retrying until the sim serves (it may boot slowly)."""
        assert self._host is not None and self._port is not None
        try:
            async with asyncio.timeout(self._connect_timeout_s):
                while True:
                    if self._proc is not None and self._proc.returncode is not None:
                        raise RuntimeError(f"sim process exited with code {self._proc.returncode}")
                    try:
                        self._reader, self._writer = await asyncio.open_connection(
                            self._host, self._port
                        )
                        return
                    except OSError:
                        await asyncio.sleep(retry_every)
        except TimeoutError as exc:
            raise TimeoutError(
                f"timed out connecting to sim control at {self._host}:{self._port} "
                f"after {self._connect_timeout_s}s"
            ) from exc

    # ── the control surface ───────────────────────────────────────────────

    async def url(self) -> str:
        """The bridge's ``ws://`` address — the robot capability's url."""
        return (await self._call("url"))["url"]

    async def contract(self) -> dict[str, Any]:
        """The env's self-describing wire contract, read from the bridge."""
        return (await self._call("contract"))["contract"]

    async def capability(self, name: str = "robot") -> Capability:
        """The concrete ``robot`` capability — publish it from an ``@env.initialize`` hook."""
        from hud.capabilities import Capability

        return Capability.robot(name=name, url=await self.url(), contract=await self.contract())

    async def reset(self, **task_args: Any) -> str:
        """Start a new episode; return the task prompt."""
        return (await self._call("reset", task_args))["prompt"]

    async def result(self, **extra: Any) -> dict[str, Any]:
        """The episode score dict, merged with any caller ``extra`` metadata."""
        res = {**(await self._call("result")), **extra}
        print(
            f"[env] result: success={res.get('success')} "
            f"total_reward={res.get('total_reward', 0.0):.3f}",
            flush=True,
        )
        return res

    async def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        # Strictly request/reply, one call at a time, so a constant id is enough.
        if self._writer is None or self._reader is None:
            raise RuntimeError("not connected; call start() first")
        await send_frame(
            self._writer, {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        )
        msg = await read_frame(self._reader)
        if msg is None:
            raise ConnectionError(f"connection closed awaiting {method!r} reply")
        if "error" in msg:
            raise RuntimeError(f"{method} failed: {msg['error']['message']}")
        return msg["result"]


async def _forward_lines(stream: asyncio.StreamReader) -> None:
    while line := await stream.readline():
        print(line.decode("utf-8", "replace").rstrip(), flush=True)


__all__ = ["RobotEndpoint"]

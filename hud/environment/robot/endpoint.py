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
    async def pawn_lift(task: str = "solo_pawn_lift", seed: int = 0):
        ep = await sim.reset(task=task, seed=seed)  # {prompt, token}
        yield {"prompt": ep["prompt"]}
        yield await sim.result()

A single-env sim has one slot, so the template may omit the token as above. A
vectorized sim (``num_envs > 1``) must thread it through so each session grades
its own slot: publish it as this capability's per-episode binding,
``yield {"prompt": ..., "bindings": {"robot": {"token": ep["token"]}}}`` (the
agent reads it off ``run.bindings``), and ``sim.result(token=ep["token"])``.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import sys
from typing import TYPE_CHECKING, Any

from hud.environment.env import current_session_id
from hud.environment.utils import read_frame, send_frame
from hud.utils.process import create_process_group_exec

from .bridge import _HUD_STATE, PORT_ANNOUNCEMENT, RobotBridge

if TYPE_CHECKING:
    from collections.abc import Sequence

    from hud.capabilities import Capability
    from hud.utils.process import ProcessGroup


def _bridge_init_kwargs(bridge: RobotBridge) -> dict[str, Any]:
    """JSON-safe kwargs for the spawned child from a declaration instance.

    Subclass ``__init__`` params ride through as ctor kwargs (host/port skipped —
    the child binds its own address). Declaration-time attrs set *after* init
    (``bridge.contract = ...``, ``num_envs``, ``metadata``) are packed under
    ``__hud_state__`` so the child can apply them without every subclass
    accepting those names as kwargs.
    """
    base = set(inspect.signature(RobotBridge.__init__).parameters) - {"self"}
    out: dict[str, Any] = {}
    for name, param in inspect.signature(type(bridge).__init__).parameters.items():
        if name == "self" or name in base or name == _HUD_STATE:
            continue
        if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
            continue
        attr = name if hasattr(bridge, name) else f"_{name}"
        if not hasattr(bridge, attr):
            continue
        value = getattr(bridge, attr)
        # Callables stay parent-side; anything else must serialize or spawn
        # would silently boot the child with a different config.
        if callable(value):
            continue
        try:
            json.dumps(value)
        except TypeError as exc:
            raise TypeError(
                f"bridge ctor param {name!r} is not JSON-serializable for spawn: {value!r}"
            ) from exc
        out[name] = value
    # Attrs often assigned on the declaration instance, not as ctor params.
    state: dict[str, Any] = {}
    if bridge.contract:
        state["contract"] = bridge.contract
    if bridge.metadata:
        state["metadata"] = bridge.metadata
    if bridge.num_envs != 1:
        state["num_envs"] = bridge.num_envs
    if state:
        out[_HUD_STATE] = state
    return out


class RobotEndpoint:
    """Drive a simulation bridge living in another process.

    Build with a bridge class/instance (``start()`` spawns its process),
    :meth:`spawn` (explicit argv), or :meth:`remote` (attach).
    """

    def __init__(
        self,
        bridge: type[RobotBridge] | RobotBridge | None = None,
        *,
        cmd: Sequence[str] | None = None,
        host: str | None = None,
        port: int | None = None,
        connect_timeout_s: float = 900.0,
    ) -> None:
        if bridge is not None:
            # Child re-imports the class; instance ctor kwargs ride along as --init JSON.
            cls = bridge if isinstance(bridge, type) else type(bridge)
            name = getattr(cls, "__qualname__", "")
            if not name or "." in name or "<" in name:
                raise ValueError(f"bridge class must be module-level, got {bridge!r}")
            cmd = [
                sys.executable,
                "-m",
                "hud.environment.robot.bridge",
                f"{inspect.getfile(cls)}:{name}",
            ]
            if isinstance(bridge, RobotBridge) and (kwargs := _bridge_init_kwargs(bridge)):
                cmd += ["--init", json.dumps(kwargs)]
        self._cmd = list(cmd) if cmd is not None else None  # set => spawned mode

        self._host = host
        self._port = port
        self._connect_timeout_s = connect_timeout_s
        self._proc: ProcessGroup | None = None
        self._forward: asyncio.Task[None] | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        # N sessions share this one TCP link; serialize send+read so replies don't cross.
        self._lock = asyncio.Lock()
        # Open slot token per control session; "" = already freed via result().
        self._claims: dict[str, str] = {}

    @classmethod
    def spawn(cls, cmd: Sequence[str], *, connect_timeout_s: float = 900.0) -> RobotEndpoint:
        """An endpoint that forks *cmd* (a sim program; see :mod:`~.bridge`) and owns it."""
        return cls(cmd=cmd, connect_timeout_s=connect_timeout_s)

    @classmethod
    def remote(cls, host: str, port: int, *, connect_timeout_s: float = 900.0) -> RobotEndpoint:
        """An endpoint attached to a sim process something else runs."""
        return cls(host=host, port=port, connect_timeout_s=connect_timeout_s)

    def attach(self, env: Any) -> RobotEndpoint:
        """Hook slot release into *env* task teardown (``env.gym`` does this)."""
        env._on_task_teardown.append(self.release_claim)
        return self

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Bring the link up: fork the sim program (spawned mode) and connect."""
        try:
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
        except BaseException:
            await self.stop()  # don't orphan a spawned child after a failed boot
            raise

    async def stop(self) -> None:
        """Drop the link; tear the sim process down when this endpoint spawned it."""
        # Free slots cancel/bye failed to release — while the control link is up.
        await self._release_outstanding_claims()
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
        """The env's self-describing wire contract, read from the bridge.

        Under lazy spawn this may build the env and derive ``contract.json`` when
        none was pre-written — so a capability published at initialize is complete.
        """
        return (await self._call("contract"))["contract"]

    async def capabilities(self, name: str = "robot") -> list[Capability]:
        """Everything the bridge serves — the ``robot`` wire named *name*, plus any
        extras it adds itself (e.g. MCP tools from the sim process).

        Publish from an ``@env.initialize`` hook (``env.gym`` does this).
        """
        from hud.capabilities import Capability

        published = await self._call("capabilities", {"name": name})
        return [Capability.from_manifest(c) for c in published["capabilities"]]

    async def reset(self, **task_args: Any) -> dict[str, Any]:
        """Claim a slot for a new episode; return ``{"prompt", "token"}``."""
        while True:
            try:
                ep = await self._call("reset", task_args)
            except RuntimeError as exc:
                # Batch full: drop the lock between tries so a peer can result.
                if "slots are claimed" not in str(exc):
                    raise
                await asyncio.sleep(0.05)
                continue
            session_id, token = current_session_id.get(), ep.get("token")
            if session_id is not None and isinstance(token, str):
                self._claims[session_id] = token
            return ep

    async def result(self, *, token: str | None = None, **extra: Any) -> dict[str, Any]:
        """This slot's score dict (frees the slot), merged with any caller ``extra``.

        ``token`` may be omitted on a single-env bridge (one claimed slot);
        vectorized envs must pass the token from :meth:`reset`.
        """
        res = {**(await self._call("result", {"token": token})), **extra}
        if (session_id := current_session_id.get()) is not None:
            self._claims[session_id] = ""  # freed; disconnect/cancel must not re-result
        print(
            f"[env] result: success={res.get('success')} "
            f"total_reward={res.get('total_reward', 0.0):.3f}",
            flush=True,
        )
        return res

    async def release_claim(self) -> None:
        """Free this session's slot if ``result`` never ran (cancel / bye / drop)."""
        session_id = current_session_id.get()
        if session_id is None:
            return
        token = self._claims.get(session_id)
        if token is None:
            return
        if not token:  # already freed via result()
            self._claims.pop(session_id, None)
            return
        if await self._result_with_retry(token):
            self._claims.pop(session_id, None)
            # else keep the claim — stop() drains leftovers while the link is up

    async def _release_outstanding_claims(self) -> None:
        """Best-effort free of every tracked slot (last chance before the link drops)."""
        for session_id, token in list(self._claims.items()):
            if not token:
                self._claims.pop(session_id, None)
                continue
            if await self._result_with_retry(token):
                self._claims.pop(session_id, None)

    async def _result_with_retry(
        self, token: str, *, attempts: int = 3, timeout_s: float = 10.0
    ) -> bool:
        """``result`` RPC with short retries; teardown only runs once per cancel path."""
        # Each attempt is bounded: a wedged sim can hold the link open yet never
        # answer, and shutdown must still finish (stop() owns a child to terminate).
        for attempt in range(attempts):
            try:
                await asyncio.wait_for(self._call("result", {"token": token}), timeout_s)
                return True
            except Exception:
                if attempt + 1 < attempts:
                    await asyncio.sleep(0.05)
        return False

    async def _call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        # One in-flight RPC: N sessions share this link; constant id is enough under the lock.
        if self._writer is None or self._reader is None:
            raise RuntimeError("not connected; call start() first")
        async with self._lock:
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

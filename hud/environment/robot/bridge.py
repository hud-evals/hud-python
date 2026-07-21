"""Sim-process ``robot`` bridge: WebSocket obs/action + JSON-RPC control.

Subclass :class:`RobotBridge` (``reset`` / ``step`` / ``get_observation``);
:func:`serve_bridge` is the blocking entry. Wire is scalar openpi per claimed
slot; the sim may be vectorized internally (barrier step, fan-out). Sim owns
main; serving runs on a background loop and queues touches via
:meth:`RobotBridge._run_on_sim` (thread-affine / Isaac). Gym path:
:class:`~.gym.GymBridge`.
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import secrets
import signal
import sys
import threading
from abc import ABC, abstractmethod
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import websockets
import websockets.exceptions

# The openpi/0 wire codec is defined alongside the agent-side client; reuse it so both
# ends of the protocol stay in lockstep (env -> capabilities is the correct direction).
from hud.capabilities.robot import _packb, _unpackb
from hud.environment.utils import error, read_frame, reply, send_frame

if TYPE_CHECKING:
    from collections.abc import Callable

#: Line the sim program prints once its control channel is bound; the spawning
#: RobotEndpoint reads it from this process's stdout.
PORT_ANNOUNCEMENT = "HUD_SIM_PORT="


@dataclass
class _Slot:
    """One sim slot: claimed by a control-plane token, driven by one WS connection."""

    index: int
    token: str | None = None
    ws: Any = None
    action: np.ndarray | None = None
    idle: bool = False  # hold next step (dialing timed out, or WS dropped)
    # Touched this episode; after release stays unreclaimable until the next global reset.
    used: bool = False


@dataclass
class _SlotRegistry:
    """Free/claimed slots for one bridge. All-free → next reset is global."""

    slots: list[_Slot] = field(default_factory=list)

    def configure(self, n: int) -> None:
        self.slots = [_Slot(index=i) for i in range(n)]

    @property
    def all_free(self) -> bool:
        return all(s.token is None for s in self.slots)

    def free_slot(self) -> _Slot | None:
        # v1: a freed slot is only reclaimable after all slots free (whole-batch episodes).
        return next((s for s in self.slots if s.token is None and not s.used), None)

    def resolve(self, token: str | None) -> _Slot:
        """Token → its claimed slot; ``None`` binds the sole claimed slot (single-env)."""
        if token is None:
            claimed = self.claimed()
            if len(claimed) != 1:
                raise ValueError(
                    f"tokenless claim needs exactly one claimed slot, found {len(claimed)}; "
                    "pass the token from reset()"
                )
            return claimed[0]
        slot = next((s for s in self.slots if s.token == token), None)
        if slot is None:
            raise ValueError(f"unknown episode token: {token!r}")
        return slot

    def claimed(self) -> list[_Slot]:
        return [s for s in self.slots if s.token is not None]

    def claim(self, slot: _Slot) -> str:
        token = f"slot-{slot.index}-{secrets.token_hex(4)}"
        slot.token = token
        slot.used = True
        slot.action = None
        slot.idle = False
        slot.ws = None
        return token

    def release(self, slot: _Slot) -> None:
        slot.token = None
        slot.ws = None
        slot.action = None
        slot.idle = False
        # keep used=True until configure() on the next global reset


class RobotBridge(ABC):
    """Serves a sim over ``robot`` WebSocket + a JSON-RPC control side channel.

    **Subclass contract:** implement :meth:`reset`, :meth:`step`, and
    :meth:`get_observation`, and set ``self.contract`` (the wire contract the
    capability publishes) before serving. The base owns the WebSocket serve
    loop and the control listener; subclasses own the sim and set ``num_envs``
    (default 1). All three hooks run on the sim thread via :meth:`_run_on_sim`
    (under :func:`serve_bridge`, that is main — safe for Isaac / thread-affine sims).

    - :meth:`reset` initialises the sim for a new episode and returns the task
      prompt. The base resets scoring state.
    - :meth:`step` advances the sim by one batched action ``[N, A]`` (N=1 ok).
    - :meth:`get_observation` returns ``(data, terminated)`` with ``[N, ...]``
      arrays and an ``[N]`` terminated mask (N=1 ok), or ``None`` if not ready.
    - :meth:`result_slots` returns one score dict per slot.
    """

    #: Seconds to wait for a *still-dialing* claimed slot before stepping with a
    #: hold. Connected slots never hit this — slow inference must not advance the
    #: sim with zeros.
    step_timeout: float = 30.0

    def __init__(self, *, host: str = "127.0.0.1", port: int = 0) -> None:
        # Loopback + ephemeral by default; the concrete address is published in the
        # manifest post-``start()`` and tunneled, so no env manages bridge ports.
        self._host = host
        self._port = port
        self._server: Any = None
        # Connect-time metadata frame (sent first on each connection); subclasses may set it.
        self.metadata: dict[str, Any] = {}
        #: The env's self-describing wire contract (features, control_rate); the
        #: endpoint fetches it over the control channel to mint the capability.
        self.contract: dict[str, Any] = {}
        # Sim touches queue to the thread that owns the simulator (see _run_on_sim).
        self._sim_q: queue.Queue[tuple[Callable[[], Any], Future[Any]]] = queue.Queue()
        self._sim_ident: int | None = None
        self.num_envs: int = 1
        self._registry = _SlotRegistry()
        self._registry.configure(1)
        self._tick_task: asyncio.Task[None] | None = None
        self._action_event = asyncio.Event()
        # Episode scoring read by ``result_slots``; single-env subclasses update these
        # in ``reset``/``step`` (batched bridges override result_slots instead).
        self.task_description: str = ""
        self.total_reward: float = 0.0
        self.success: bool = False
        self.terminated: bool = False
        # kwargs of the current batch's global reset; later claims must match.
        self._episode_kwargs: dict[str, Any] = {}

    async def _run_on_sim(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run ``fn(*args, **kwargs)`` on the sim thread, awaited on the caller's loop.

        Before :func:`serve_bridge` binds a sim thread (tests, in-loop use) calls
        execute inline on the caller — the degenerate single-thread case.
        """
        if self._sim_ident is None or threading.get_ident() == self._sim_ident:
            return fn(*args, **kwargs)  # not serving yet, or already on the sim thread
        fut: Future[Any] = Future()
        self._sim_q.put((lambda: fn(*args, **kwargs), fut))
        return await asyncio.wrap_future(fut)

    async def _claim_episode(self, **kwargs: Any) -> dict[str, Any]:
        """Control-plane reset: global sim reset when all free, else claim a free slot."""
        if self._registry.all_free:
            self.total_reward = 0.0
            self.success = False
            self.terminated = False
            # Same sim-thread hop as step/obs — custom bridges must not touch Isaac here
            # on the background serve loop.
            self.task_description = await self._run_on_sim(self.reset, **kwargs)
            self._episode_kwargs = kwargs
            self._registry.configure(self.num_envs)
            slot = self._registry.slots[0]
        else:
            # Lockstep sim: one global reset serves the whole batch, so a later claim
            # cannot get its own task/seed — reject differing kwargs instead of
            # silently running the first claim's task.
            if kwargs and kwargs != self._episode_kwargs:
                raise ValueError(
                    f"slots share one batch reset ({self._episode_kwargs}); a concurrent "
                    f"claim cannot use different task kwargs ({kwargs}) — group tasks with "
                    "identical args, or use one sim per distinct task/seed"
                )
            slot = self._registry.free_slot()
            if slot is None:
                raise RuntimeError(f"all {self.num_envs} slots are claimed")
        token = self._registry.claim(slot)
        return {"prompt": self.task_description, "token": token}

    async def _release_episode(self, token: str | None) -> dict[str, Any]:
        """Control-plane result: this slot's score, then free it.

        Scores are written in ``step`` on the sim thread — read them there too
        so grading never races a mid-step update on the serve loop.
        """
        slot = self._registry.resolve(token)
        grade = (await self._run_on_sim(self.result_slots))[slot.index]
        self._registry.release(slot)
        return grade

    @abstractmethod
    def reset(self, **kwargs: Any) -> str:
        """Reset the sim for a new episode; return the task prompt.

        Always invoked on the sim thread (via :meth:`_run_on_sim`) — implement
        this as ordinary sync code, same as :meth:`step` / :meth:`get_observation`.
        """

    @abstractmethod
    def step(self, action: np.ndarray) -> None:
        """Advance the sim by one batched action ``[N, A]``."""

    @abstractmethod
    def get_observation(self) -> tuple[dict[str, np.ndarray], np.ndarray] | None:
        """Return ``(data[N, ...], terminated[N])``, or ``None`` if not ready."""

    def result_slots(self) -> list[dict[str, Any]]:
        """One score dict per slot. Default: the scalar episode grade for every slot
        (vectorized bridges with per-slot scoring override this).

        Invoked on the sim thread via :meth:`_run_on_sim` (same as ``step``).
        """
        grade = {
            "score": 1.0 if self.success else 0.0,
            "success": bool(self.success),
            "total_reward": float(self.total_reward),
        }
        return [dict(grade) for _ in range(self.num_envs)]

    def hold_action(self) -> np.ndarray:
        """Action used for idle/stalled slots at a barrier step (zeros)."""
        action = next(
            (f for f in self.contract.get("features", {}).values() if f.get("role") == "action"),
            {},
        )
        # Derived contracts carry only per-dim names, no shape; the label count is the dim.
        shape = action.get("shape") or (len(action.get("names") or []) or 1,)
        return np.zeros(shape, dtype=np.float32)

    @property
    def url(self) -> str:
        """The bridge's concrete ``ws://`` address — publish this in the manifest.

        With an ephemeral port (the default) the address only exists once
        :meth:`start` has bound the socket, so publish after ``await bridge.start()``.
        """
        if self._port == 0:
            raise RuntimeError("bridge bound to an ephemeral port; call start() before reading url")
        return f"ws://{self._host}:{self._port}"

    async def start(self) -> None:
        # Idempotent: a long-lived bridge serves sequential agents, so re-``start`` (e.g. a
        # second run against the same sim) is a no-op rather than an EADDRINUSE rebind.
        if self._server is not None:
            return
        self._registry.configure(self.num_envs)
        # No keepalive: a lockstep step can block the sim (and the GIL) for
        # minutes during heavy resets; ping timeouts would sever a healthy run.
        self._server = await websockets.serve(
            self._handle_client,
            self._host,
            self._port,
            max_size=None,
            reuse_address=True,
            ping_interval=None,
        )
        if self._port == 0:
            self._port = self._server.sockets[0].getsockname()[1]
        self._tick_task = asyncio.create_task(self._tick_loop())
        print(f"[env] robot listening on ws://{self._host}:{self._port}", flush=True)

    async def stop(self) -> None:
        if self._tick_task is not None:
            self._tick_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._tick_task
            self._tick_task = None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_client(self, ws: Any) -> None:
        """One agent connection: claim a slot, then feed actions into the barrier."""
        slot: _Slot | None = None
        try:
            # Connect-time metadata frame; claim_required lets clients without a
            # token fail fast instead of waiting forever for an observation.
            await ws.send(_packb({**self.metadata, "claim_required": True}))
            # Fail fast on a client that never claims (it would otherwise deadlock:
            # we wait for the claim frame, it waits for an observation).
            raw = await asyncio.wait_for(ws.recv(), timeout=self.step_timeout)
            if isinstance(raw, str):
                raise RuntimeError(raw)
            claim = _unpackb(raw)
            if not isinstance(claim, dict) or "claim" not in claim:
                raise ValueError('first frame must be {"claim": <token or None>}')
            # A None claim binds the sole claimed slot — single-env agents skip
            # the token plumbing; ambiguous (vectorized) claims error instead.
            slot = self._registry.resolve(claim["claim"])
            if slot.ws is not None:
                raise RuntimeError(f"slot {slot.index} already has a live connection")
            slot.ws = ws
            slot.action = None
            slot.idle = False
            # First frame after claim: one sim read, then this slot's scalar row.
            await self._send_slot_observation(slot, await self._run_on_sim(self.get_observation))
            async for frame in ws:
                action = _unpackb(frame)["actions"]  # codec already returns an ndarray
                slot.action = np.asarray(action, dtype=np.float32)
                slot.idle = False
                self._action_event.set()
        except websockets.exceptions.ConnectionClosed:
            pass
        except Exception:
            import traceback

            with contextlib.suppress(Exception):
                await ws.send(traceback.format_exc())
            raise
        finally:
            if slot is not None and slot.ws is ws:
                slot.ws = None
                slot.action = None
                # Idle, not pending: a dropped connection must not stall the barrier
                # for the other slots (a reconnect clears idle again).
                slot.idle = True
                self._action_event.set()  # wake the barrier so it doesn't wait on us

    async def _tick_loop(self) -> None:
        """Gather claimed slots' actions (or dialing-timeout → hold), step once, fan out."""
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._action_event.wait(), timeout=self.step_timeout)
            self._action_event.clear()
            claimed = self._registry.claimed()
            # No live connection at all → don't step; episodes must not advance unseen.
            if not any(s.ws is not None for s in claimed):
                continue
            # Barrier over every *claimed* slot: dialing (no WS yet) blocks too, so
            # its episode can't advance before the first observation. step_timeout
            # only holds still-dialing slots — a connected agent mid-inference waits.
            deadline = asyncio.get_running_loop().time() + self.step_timeout
            while True:
                pending = [s for s in claimed if s.action is None and not s.idle]
                if not pending:
                    break
                dialing = [s for s in pending if s.ws is None]
                connected_pending = [s for s in pending if s.ws is not None]
                if dialing:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        for s in dialing:
                            s.idle = True  # never connected → hold so the batch can move
                        if not connected_pending:
                            break
                        # Connected agents still deciding — keep waiting without a deadline.
                        continue
                    self._action_event.clear()
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self._action_event.wait(), timeout=remaining)
                else:
                    # Only live agents pending: wait for an action (or a disconnect).
                    self._action_event.clear()
                    await self._action_event.wait()
                claimed = self._registry.claimed()
                if not any(s.ws is not None for s in claimed):
                    break
            if not any(s.ws is not None for s in claimed):
                continue
            hold = self.hold_action()
            # Stack one row per sim slot (idle/unconnected claimed → hold; free → hold).
            rows = []
            for s in self._registry.slots:
                row = s.action if s in claimed and s.action is not None else hold
                rows.append(np.asarray(row, dtype=np.float32).reshape(-1))
                s.action = None
            actions = np.stack(rows)
            # One step + one batched obs on the sim thread, then fan-out (not N reads).
            await self._run_on_sim(self.step, actions)
            batch = await self._run_on_sim(self.get_observation)
            for s in claimed:
                await self._send_slot_observation(s, batch)

    async def _send_slot_observation(
        self,
        slot: _Slot,
        batch: tuple[dict[str, np.ndarray], np.ndarray] | None,
    ) -> None:
        """Fan one scalar obs frame to a claimed connection from a batched read."""
        if slot.ws is None or batch is None:
            return
        data, terminated = batch
        i = slot.index
        # Slice the [N, ...] batch down to this slot's scalar row.
        msg = {
            **{
                k: (v[i] if getattr(v, "ndim", 0) >= 1 and len(v) == self.num_envs else v)
                for k, v in data.items()
            },
            "terminated": bool(np.asarray(terminated).reshape(-1)[i]),
        }
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            await slot.ws.send(_packb(msg))

    # ── the control side channel (driven by a RobotEndpoint) ────────────────────

    async def serve_control(self, host: str = "127.0.0.1", port: int = 0) -> asyncio.Server:
        """Serve the JSON-RPC side channel: ``url`` / ``contract`` / ``reset`` / ``result``."""
        return await asyncio.start_server(self._handle_control, host, port)

    async def _handle_control(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        with contextlib.suppress(ConnectionResetError, asyncio.IncompleteReadError):
            while (msg := await read_frame(reader)) is not None:
                try:
                    result = await self._dispatch_control(msg["method"], msg.get("params") or {})
                    await send_frame(writer, reply(msg["id"], result))
                except Exception as exc:  # surface to the caller, keep serving the link
                    await send_frame(writer, error(msg["id"], -32000, str(exc)))
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()

    async def ensure_contract(self) -> dict[str, Any]:
        """Return the wire contract, deriving it when a subclass can.

        Default is the already-set ``self.contract``. Lazy-spawn bridges
        (:class:`~.gym.GymBridge`) override to build the env and derive when
        no pre-written ``contract.json`` was loaded at start — so
        ``endpoint.capability()`` publishes a real manifest, not ``{}``.
        """
        return self.contract

    async def _dispatch_control(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "url":
            return {"url": self.url}
        if method == "contract":
            # May trigger env build under lazy spawn when the contract is still empty.
            return {"contract": await self.ensure_contract()}
        if method == "reset":
            return await self._claim_episode(**params)
        if method == "result":
            token = params.get("token")
            if token is not None and not isinstance(token, str):
                raise ValueError("result: 'token' must be a string when given")
            return await self._release_episode(token)
        raise ValueError(f"unknown method {method!r}")


# ── the sim program shape (custom sims call serve_bridge last) ────────────────


def serve_bridge(bridge: RobotBridge, *, host: str = "127.0.0.1", port: int = 0) -> None:
    """Serve *bridge*, blocking for the process's lifetime — the sim program's last call.

    Serving — the robot WebSocket and the control side channel (port announced
    on stdout as ``HUD_SIM_PORT=<port>``) — runs on a background loop thread;
    the calling (main) thread becomes the sim thread, executing every touch the
    bridge queues (see :meth:`RobotBridge._run_on_sim`). SIGTERM / Ctrl-C cancel
    serving; the sim keeps draining through teardown (``stop()`` touches it too).
    """
    # Claim main as the sim thread before serving starts, so no touch slips
    # through inline on the wrong thread.
    bridge._sim_ident = threading.get_ident()
    loop = asyncio.new_event_loop()
    done = threading.Event()
    serve_task: asyncio.Task[Any] | None = None

    async def _serve() -> None:
        nonlocal serve_task
        serve_task = asyncio.current_task()
        await bridge.start()  # WS bound first, so `url` is concrete before any control call
        server = await bridge.serve_control(host, port)
        print(f"{PORT_ANNOUNCEMENT}{server.sockets[0].getsockname()[1]}", flush=True)
        try:
            await asyncio.Event().wait()  # until SIGTERM / Ctrl-C cancels
        finally:
            server.close()
            await bridge.stop()

    def _thread() -> None:
        asyncio.set_event_loop(loop)
        try:
            with contextlib.suppress(asyncio.CancelledError):
                loop.run_until_complete(_serve())
        finally:
            done.set()

    def _cancel(*_: object) -> None:
        if serve_task is not None:
            loop.call_soon_threadsafe(serve_task.cancel)

    with contextlib.suppress(ValueError):  # signals are main-thread-only; tests may not be
        signal.signal(signal.SIGTERM, _cancel)
    thread = threading.Thread(target=_thread, name="hud-serve", daemon=True)
    thread.start()

    # The sim loop: execute queued touches until serving exits, pumping Kit when
    # Omniverse is loaded (it needs continuous main-thread updates).
    while not done.is_set():
        try:
            # Re-checked every pass: Kit may finish loading after serving starts.
            kit = sys.modules.get("omni.kit.app")
            touches: list[tuple[Callable[[], Any], Future[Any]]] = []
            with contextlib.suppress(queue.Empty):
                touches.append(bridge._sim_q.get(timeout=0.002 if kit else 0.05))
                while True:  # drain the backlog behind the first touch
                    touches.append(bridge._sim_q.get_nowait())
            for fn, fut in touches:
                if fut.set_running_or_notify_cancel():
                    try:
                        fut.set_result(fn())
                    except BaseException as exc:  # propagate to the awaiting caller
                        fut.set_exception(exc)
            if kit:
                kit.get_app().update()
        except KeyboardInterrupt:  # Ctrl-C: cancel serving, keep draining through teardown
            _cancel()
    thread.join(timeout=30)


def main() -> None:
    """Child entry: ``python -m hud.environment.robot.bridge path.py:Class [--init JSON]``."""
    import json

    from hud.utils.modules import load_module

    path, _, name = sys.argv[1].rpartition(":")
    kwargs: dict[str, Any] = {}
    if len(sys.argv) >= 4 and sys.argv[2] == "--init":
        kwargs = json.loads(sys.argv[3])
    serve_bridge(getattr(load_module(path), name)(**kwargs))


if __name__ == "__main__":
    main()


__all__ = ["PORT_ANNOUNCEMENT", "RobotBridge", "serve_bridge"]

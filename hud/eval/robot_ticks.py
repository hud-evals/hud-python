"""Rollout-level recorder for the ticks an env plays itself.

``DirectControl`` publishes every tick it plays on a ``ticks/0`` capability: the robot
contract, then one msgpack-numpy observation per tick, closed when the episode ends.
This records each stream to the rollout's trace as numeric state plus one H.264 video
per camera, so the env never uploads telemetry.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import websockets
import websockets.exceptions

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from websockets.asyncio.client import ClientConnection

    from hud.clients.client import HudClient
    from hud.telemetry.robot.recorder import TraceRecorder

logger = logging.getLogger("hud.eval.robot_ticks")

# How long exit waits for the env to end a stream before cutting it.
_DRAIN_TIMEOUT = 10.0


@asynccontextmanager
async def record_played_ticks(client: HudClient, trace_id: str) -> AsyncGenerator[None]:
    """Record every ``ticks/0`` stream the env publishes to ``trace_id``.

    Enter once the task has started and exit after grading: the env ends each stream
    when the episode ends, and exit waits for that, so the video keeps every tick played.
    Recording needs the ``robot`` extra; without it the rollout runs unrecorded.
    """
    assert client.manifest is not None
    names = [b.name for b in client.manifest.bindings if b.protocol == "ticks/0"]
    if not names:
        yield
        return
    try:
        # The ``robot`` extra (numpy, msgpack-numpy, PyAV), needed once an env publishes ticks.
        from hud.capabilities.robot import unpackb
        from hud.telemetry.robot.recorder import TraceRecorder
    except ImportError as exc:
        logger.warning("not recording played ticks; install hud[robot]: %s", exc)
        yield
        return

    async def record(stream: ClientConnection, recorder: TraceRecorder) -> None:
        tick = 0
        try:
            async for frame in stream:
                # Off the loop: an encoder that keeps every frame blocks while its queue drains.
                await asyncio.to_thread(recorder.record_observation, unpackb(frame), tick=tick)
                tick += 1
        except websockets.exceptions.ConnectionClosedError as exc:
            logger.warning("ticks stream dropped after %d ticks: %s", tick, exc)
        finally:
            await asyncio.to_thread(recorder.close)  # flushes the video tails

    recording: dict[asyncio.Task[None], ClientConnection] = {}
    for name in names:
        try:
            stream = await websockets.connect(
                client.binding(name).url, max_size=None, ping_interval=None
            )
            # Subscribed once the contract arrives, so no tick played after this is missed.
            contract = unpackb(await stream.recv())
        except (OSError, websockets.exceptions.WebSocketException) as exc:
            logger.warning("not recording %s: %s", name, exc)
            continue
        recorder = TraceRecorder(
            trace_id=trace_id,
            fps=max(1, round(contract["control_rate"])),
            obs_space={n: f for n, f in contract["features"].items() if f["role"] == "observation"},
            sim_clock=True,
            keep_every_video_frame=True,
        )
        recording[asyncio.create_task(record(stream, recorder))] = stream
    try:
        yield
    finally:
        if recording:
            _, cut = await asyncio.wait(recording, timeout=_DRAIN_TIMEOUT)
            for task in cut:
                logger.warning("the env did not end a ticks stream; cutting it")
                await recording[task].close()
            await asyncio.gather(*recording)


__all__ = ["record_played_ticks"]

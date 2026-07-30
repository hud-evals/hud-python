"""Robot policy adapters exercised through the public openpi wire boundary."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import websockets
from openpi_client import msgpack_numpy  # pyright: ignore[reportMissingTypeStubs]

from hud.agents.robot import LeRobotAdapter
from hud.capabilities.base import Capability
from hud.capabilities.robot import RobotClient

codec: Any = msgpack_numpy


class _TensorProbe:
    def __init__(self, array: np.ndarray[Any, Any]) -> None:
        self.array = array

    def permute(self, *_axes: int) -> _TensorProbe:
        return self

    def float(self) -> _TensorProbe:
        return self

    def __truediv__(self, _value: float) -> _TensorProbe:
        return self


async def test_lerobot_adapter_copies_read_only_wire_arrays_before_torch(
    monkeypatch: Any,
) -> None:
    state = np.arange(8, dtype=np.float32)
    camera = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    async def serve_observation(socket: Any) -> None:
        await socket.send(codec.packb({"name": "wire-probe"}))
        await socket.send(
            codec.packb({"state": state, "camera": camera, "terminated": False}),
        )
        await socket.wait_closed()

    server = await websockets.serve(serve_observation, "127.0.0.1", 0)
    sockets = server.sockets
    assert sockets
    capability = Capability.robot(
        url=f"ws://127.0.0.1:{sockets[0].getsockname()[1]}",
        contract={
            "features": {
                "state": {"role": "observation", "type": "state"},
                "camera": {"role": "observation", "type": "rgb"},
                "action": {"role": "action", "type": "action"},
            },
        },
    )

    client = await RobotClient.connect(capability)
    try:
        async with asyncio.timeout(1):
            observation = await client.get_observation()

        wire_state = observation["data"]["state"]
        wire_camera = observation["data"]["camera"]
        assert not wire_state.flags.writeable
        assert not wire_camera.flags.writeable

        arrays_given_to_torch: list[np.ndarray[Any, Any]] = []

        def from_numpy(array: np.ndarray[Any, Any]) -> _TensorProbe:
            arrays_given_to_torch.append(array)
            return _TensorProbe(array)

        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=from_numpy))

        action_space, observation_space = client.spaces()
        adapter = LeRobotAdapter(model_image_keys=["observation.images.camera"])
        adapter.bind(action_space, observation_space)
        adapter.adapt_observation(observation, "move the object")

        assert [array.flags.writeable for array in arrays_given_to_torch] == [True, True]
        np.testing.assert_array_equal(arrays_given_to_torch[0], state)
        np.testing.assert_array_equal(arrays_given_to_torch[1], camera)
    finally:
        await client.close()
        server.close()
        await server.wait_closed()

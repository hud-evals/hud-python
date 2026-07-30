"""Robot slot lifecycle through the public control and WebSocket protocols."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from hud.capabilities.robot import RobotClient
from hud.environment.env import current_session_id
from hud.environment.robot import RobotBridge, RobotEndpoint
from hud.environment.robot import bridge as bridge_module

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class _ProbeBridge(RobotBridge):
    step_timeout = 0.1

    def __init__(
        self,
        *,
        num_envs: int = 1,
        terminate_after: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.num_envs = num_envs
        self.contract = {
            "control_rate": 10,
            "features": {
                "state": {
                    "role": "observation",
                    "dtype": "float32",
                    "shape": [1],
                    "names": ["position"],
                },
                "action": {
                    "role": "action",
                    "dtype": "float32",
                    "shape": [1],
                    "names": ["move"],
                },
            },
        }
        self.terminate_after = terminate_after or {}
        self.reset_calls: list[dict[str, Any]] = []
        self.step_calls: list[np.ndarray] = []
        self.result_calls = 0
        self.result_called = asyncio.Event()
        self._state = np.zeros((num_envs, 1), dtype=np.float32)
        self._slot_steps = np.zeros(num_envs, dtype=np.int64)
        self._terminated = np.zeros(num_envs, dtype=bool)

    def reset(self, **kwargs: Any) -> str:
        self.reset_calls.append(dict(kwargs))
        self._state.fill(0)
        self._slot_steps.fill(0)
        self._terminated.fill(False)
        return str(kwargs.get("task", "default"))

    def step(self, action: np.ndarray) -> None:
        self.step_calls.append(np.array(action, copy=True))
        self._state += 1
        self._slot_steps += 1
        for index, limit in self.terminate_after.items():
            self._terminated[index] |= self._slot_steps[index] >= limit

    def get_observation(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        return {"state": self._state.copy()}, self._terminated.copy()

    def result_slots(self) -> list[dict[str, Any]]:
        self.result_calls += 1
        self.result_called.set()
        return [
            {
                "score": float(self._terminated[index]),
                "success": bool(self._terminated[index]),
                "total_reward": float(self._terminated[index]),
                "steps": int(self._slot_steps[index]),
            }
            for index in range(self.num_envs)
        ]


class _YieldingResetBridge(_ProbeBridge):
    async def _run_on_sim(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if fn == self.reset:
            await asyncio.sleep(0)
        return await super()._run_on_sim(fn, *args, **kwargs)


class _BlockedResultBridge(_ProbeBridge):
    def __init__(self) -> None:
        super().__init__()
        self.result_started = asyncio.Event()
        self.finish_result = asyncio.Event()

    async def _run_on_sim(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if fn == self.result_slots:
            result = await super()._run_on_sim(fn, *args, **kwargs)
            self.result_started.set()
            await self.finish_result.wait()
            return result
        return await super()._run_on_sim(fn, *args, **kwargs)


class _BlockedStepBridge(_ProbeBridge):
    def __init__(self, *, num_envs: int = 2) -> None:
        super().__init__(num_envs=num_envs)
        self.step_finished = asyncio.Event()
        self.release_step = asyncio.Event()
        self._block_next_step = True

    async def _run_on_sim(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        result = await super()._run_on_sim(fn, *args, **kwargs)
        if fn == self.step and self._block_next_step:
            self._block_next_step = False
            self.step_finished.set()
            await self.release_step.wait()
        return result


@asynccontextmanager
async def _running_bridge(bridge: RobotBridge) -> AsyncIterator[RobotEndpoint]:
    await bridge.start()
    control = await bridge.serve_control()
    sockets = control.sockets
    assert sockets
    endpoint = RobotEndpoint.remote("127.0.0.1", sockets[0].getsockname()[1])
    await endpoint.start()
    try:
        yield endpoint
    finally:
        async with asyncio.timeout(1):
            await endpoint.stop()
        control.close()
        async with asyncio.timeout(1):
            await control.wait_closed()
        async with asyncio.timeout(1):
            await bridge.stop()


async def _observation(client: RobotClient, *, within: float = 1.0) -> dict[str, Any]:
    async with asyncio.timeout(within):
        return await client.get_observation()


async def _close_client(client: RobotClient) -> None:
    with suppress(TimeoutError):
        async with asyncio.timeout(0.5):
            await client.close()


async def test_concurrent_control_clients_claim_distinct_slots_from_one_reset() -> None:
    bridge = _YieldingResetBridge(num_envs=2)
    await bridge.start()
    control = await bridge.serve_control()
    sockets = control.sockets
    assert sockets
    port = sockets[0].getsockname()[1]
    first_endpoint = RobotEndpoint.remote("127.0.0.1", port)
    second_endpoint = RobotEndpoint.remote("127.0.0.1", port)
    await asyncio.gather(first_endpoint.start(), second_endpoint.start())
    clients: list[RobotClient] = []
    try:
        first_episode, second_episode = await asyncio.gather(
            first_endpoint.reset(task="same"),
            second_endpoint.reset(task="same"),
        )
        assert first_episode["token"] != second_episode["token"]
        assert bridge.reset_calls == [{"task": "same"}]

        capability = (await first_endpoint.capabilities())[0]
        clients = list(
            await asyncio.gather(
                RobotClient.connect(capability, token=first_episode["token"]),
                RobotClient.connect(capability, token=second_episode["token"]),
            )
        )
        observations = await asyncio.gather(*(_observation(client) for client in clients))
        assert all(not observation["terminated"] for observation in observations)
    finally:
        await asyncio.gather(*(_close_client(client) for client in clients))
        await asyncio.gather(first_endpoint.stop(), second_endpoint.stop())
        control.close()
        async with asyncio.timeout(1):
            await control.wait_closed()
        async with asyncio.timeout(1):
            await bridge.stop()


async def test_result_and_next_reset_are_atomic_across_control_clients() -> None:
    bridge = _BlockedResultBridge()
    await bridge.start()
    control = await bridge.serve_control()
    sockets = control.sockets
    assert sockets
    port = sockets[0].getsockname()[1]
    first_endpoint = RobotEndpoint.remote("127.0.0.1", port)
    second_endpoint = RobotEndpoint.remote("127.0.0.1", port)
    await asyncio.gather(first_endpoint.start(), second_endpoint.start())
    result_task: asyncio.Task[dict[str, Any]] | None = None
    reset_task: asyncio.Task[dict[str, Any]] | None = None
    try:
        episode = await first_endpoint.reset(task="first")
        result_task = asyncio.create_task(first_endpoint.result(token=episode["token"]))
        await bridge.result_started.wait()
        reset_task = asyncio.create_task(second_endpoint.reset(task="second"))
        await asyncio.sleep(0)
        assert not reset_task.done()

        bridge.finish_result.set()
        result, next_episode = await asyncio.gather(result_task, reset_task)
        assert result["steps"] == 0
        assert next_episode["prompt"] == "second"
        assert bridge.reset_calls == [{"task": "first"}, {"task": "second"}]
    finally:
        bridge.finish_result.set()
        pending = [task for task in (result_task, reset_task) if task is not None]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.gather(first_endpoint.stop(), second_endpoint.stop())
        control.close()
        async with asyncio.timeout(1):
            await control.wait_closed()
        async with asyncio.timeout(1):
            await bridge.stop()


async def test_result_discards_an_action_that_arrives_after_grading_starts() -> None:
    bridge = _BlockedResultBridge()

    async with _running_bridge(bridge) as endpoint:
        episode = await endpoint.reset()
        capability = (await endpoint.capabilities())[0]
        client = await RobotClient.connect(capability, token=episode["token"])
        result_task: asyncio.Task[dict[str, Any]] | None = None
        try:
            await _observation(client)
            result_task = asyncio.create_task(endpoint.result(token=episode["token"]))
            await bridge.result_started.wait()

            await client.send_action(np.array([1.0], dtype=np.float32))
            await asyncio.sleep(bridge.step_timeout)
            assert bridge.step_calls == []

            bridge.finish_result.set()
            result = await asyncio.wait_for(result_task, timeout=1)
            assert result["steps"] == 0
            await asyncio.sleep(bridge.step_timeout)
            assert bridge.step_calls == []
        finally:
            bridge.finish_result.set()
            if result_task is not None:
                await asyncio.gather(result_task, return_exceptions=True)
            await _close_client(client)


async def test_result_waits_for_an_in_flight_step_and_grades_it() -> None:
    bridge = _BlockedStepBridge(num_envs=1)

    async with _running_bridge(bridge) as endpoint:
        episode = await endpoint.reset()
        capability = (await endpoint.capabilities())[0]
        client = await RobotClient.connect(capability, token=episode["token"])
        result_task: asyncio.Task[dict[str, Any]] | None = None
        try:
            await _observation(client)
            await client.send_action(np.array([1.0], dtype=np.float32))
            await bridge.step_finished.wait()

            result_task = asyncio.create_task(endpoint.result(token=episode["token"]))
            await asyncio.sleep(bridge.step_timeout)
            assert not result_task.done()

            bridge.release_step.set()
            result = await asyncio.wait_for(result_task, timeout=1)
            assert result["steps"] == 1
            assert len(bridge.step_calls) == 1
        finally:
            bridge.release_step.set()
            if result_task is not None:
                await asyncio.gather(result_task, return_exceptions=True)
            await _close_client(client)


async def test_cancelled_reset_drains_its_reply_and_releases_the_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _ProbeBridge()
    reset_reply_blocked = asyncio.Event()
    release_reset_reply = asyncio.Event()
    result_reply_blocked = asyncio.Event()
    release_result_reply = asyncio.Event()
    blocked_reset = False
    blocked_result = False
    original_send_frame = bridge_module.send_frame

    async def pause_reset_reply(writer: Any, message: dict[str, Any]) -> None:
        nonlocal blocked_reset, blocked_result
        result = message.get("result")
        if not blocked_reset and isinstance(result, dict) and isinstance(result.get("token"), str):
            blocked_reset = True
            reset_reply_blocked.set()
            await release_reset_reply.wait()
        elif not blocked_result and isinstance(result, dict) and "steps" in result:
            blocked_result = True
            result_reply_blocked.set()
            await release_result_reply.wait()
        await original_send_frame(writer, message)

    monkeypatch.setattr(bridge_module, "send_frame", pause_reset_reply)
    reset_task: asyncio.Task[dict[str, Any]] | None = None
    session = current_session_id.set("cancelled-reset")
    try:
        async with _running_bridge(bridge) as endpoint:
            reset_task = asyncio.create_task(endpoint.reset(task="cancelled"))
            await reset_reply_blocked.wait()
            reset_task.cancel()
            await asyncio.sleep(0)
            assert not reset_task.done()

            release_reset_reply.set()
            await result_reply_blocked.wait()
            reset_task.cancel()
            await asyncio.sleep(0)
            assert not reset_task.done()

            release_result_reply.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(reset_task, timeout=1)

            await endpoint.release_claim()
            assert bridge.result_calls == 1
            assert (await endpoint.contract())["features"] == bridge.contract["features"]

            next_episode = await asyncio.wait_for(endpoint.reset(task="after"), timeout=1)
            assert next_episode["prompt"] == "after"
            await endpoint.result(token=next_episode["token"])
    finally:
        current_session_id.reset(session)
        release_reset_reply.set()
        release_result_reply.set()
        if reset_task is not None and not reset_task.done():
            reset_task.cancel()
            await asyncio.gather(reset_task, return_exceptions=True)


async def test_cancelled_result_drains_its_reply_and_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _ProbeBridge()
    reply_blocked = asyncio.Event()
    release_reply = asyncio.Event()
    blocked_once = False
    original_send_frame = bridge_module.send_frame

    async def pause_result_reply(writer: Any, message: dict[str, Any]) -> None:
        nonlocal blocked_once
        result = message.get("result")
        if not blocked_once and isinstance(result, dict) and "steps" in result:
            blocked_once = True
            reply_blocked.set()
            await release_reply.wait()
        await original_send_frame(writer, message)

    monkeypatch.setattr(bridge_module, "send_frame", pause_result_reply)
    result_task: asyncio.Task[dict[str, Any]] | None = None
    session = current_session_id.set("cancelled-result")
    try:
        async with _running_bridge(bridge) as endpoint:
            episode = await endpoint.reset(task="first")
            result_task = asyncio.create_task(endpoint.result(token=episode["token"]))
            await reply_blocked.wait()
            result_task.cancel()
            await asyncio.sleep(0)
            assert not result_task.done()

            release_reply.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(result_task, timeout=1)

            await endpoint.release_claim()
            assert bridge.result_calls == 1
            assert (await endpoint.contract())["features"] == bridge.contract["features"]

            next_episode = await asyncio.wait_for(endpoint.reset(task="after"), timeout=1)
            assert next_episode["prompt"] == "after"
            await endpoint.result(token=next_episode["token"])
    finally:
        current_session_id.reset(session)
        release_reply.set()
        if result_task is not None and not result_task.done():
            result_task.cancel()
            await asyncio.gather(result_task, return_exceptions=True)


async def test_cancelled_hung_rpc_poisoning_releases_server_owned_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _ProbeBridge()
    await bridge.start()
    control = await bridge.serve_control()
    sockets = control.sockets
    assert sockets
    port = sockets[0].getsockname()[1]
    endpoint = RobotEndpoint.remote("127.0.0.1", port)
    replacement = RobotEndpoint.remote("127.0.0.1", port)
    await endpoint.start()

    reply_blocked = asyncio.Event()
    release_reply = asyncio.Event()
    blocked_once = False
    original_send_frame = bridge_module.send_frame

    async def hang_reset_reply(writer: Any, message: dict[str, Any]) -> None:
        nonlocal blocked_once
        result = message.get("result")
        if not blocked_once and isinstance(result, dict) and isinstance(result.get("token"), str):
            blocked_once = True
            reply_blocked.set()
            await release_reply.wait()
        await original_send_frame(writer, message)

    monkeypatch.setattr(bridge_module, "send_frame", hang_reset_reply)
    reset_task = asyncio.create_task(endpoint.reset(task="hung"))
    queued_call: asyncio.Task[dict[str, Any]] | None = None
    try:
        await reply_blocked.wait()
        queued_call = asyncio.create_task(endpoint.contract())
        await asyncio.sleep(0)
        assert not queued_call.done()
        reset_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(reset_task, timeout=2)
        with pytest.raises(RuntimeError, match="not connected"):
            await queued_call

        release_reply.set()
        await asyncio.wait_for(bridge.result_called.wait(), timeout=1)
        assert bridge.result_calls == 1

        await replacement.start()
        next_episode = await asyncio.wait_for(replacement.reset(task="after"), timeout=1)
        assert next_episode["prompt"] == "after"
        await replacement.result(token=next_episode["token"])
    finally:
        release_reply.set()
        if not reset_task.done():
            reset_task.cancel()
            await asyncio.gather(reset_task, return_exceptions=True)
        if queued_call is not None and not queued_call.done():
            queued_call.cancel()
            await asyncio.gather(queued_call, return_exceptions=True)
        await endpoint.stop()
        await replacement.stop()
        control.close()
        async with asyncio.timeout(1):
            await control.wait_closed()
        async with asyncio.timeout(1):
            await bridge.stop()


async def test_result_rejects_a_token_owned_by_another_session() -> None:
    bridge = _ProbeBridge(num_envs=2)

    async with _running_bridge(bridge) as endpoint:
        first_context = current_session_id.set("first-session")
        try:
            first = await endpoint.reset(task="same")
        finally:
            current_session_id.reset(first_context)

        second_context = current_session_id.set("second-session")
        try:
            second = await endpoint.reset(task="same")
        finally:
            current_session_id.reset(second_context)

        first_context = current_session_id.set("first-session")
        try:
            with pytest.raises(ValueError, match="does not match"):
                await endpoint.result(token=second["token"])
            await endpoint.result(token=first["token"])
        finally:
            current_session_id.reset(first_context)

        second_context = current_session_id.set("second-session")
        try:
            await endpoint.result(token=second["token"])
        finally:
            current_session_id.reset(second_context)

    assert bridge.result_calls == 2


async def test_dialing_slot_expires_without_blocking_live_peer() -> None:
    bridge = _ProbeBridge(num_envs=2)

    async with _running_bridge(bridge) as endpoint:
        live_episode = await endpoint.reset(task="same")
        late_episode = await endpoint.reset(task="same")
        capability = (await endpoint.capabilities())[0]
        live = await RobotClient.connect(capability, token=live_episode["token"])
        try:
            assert not (await _observation(live))["terminated"]
            await live.send_action(np.array([3.0], dtype=np.float32))

            assert not (await _observation(live))["terminated"]
            assert len(bridge.step_calls) == 1
            np.testing.assert_allclose(bridge.step_calls[0], [[3.0], [0.0]])

            with pytest.raises(RuntimeError, match="initial connection deadline expired"):
                await RobotClient.connect(capability, token=late_episode["token"])
        finally:
            await _close_client(live)


async def test_connected_slow_policy_is_not_replaced_by_hold_actions() -> None:
    bridge = _ProbeBridge()

    async with _running_bridge(bridge) as endpoint:
        episode = await endpoint.reset()
        capability = (await endpoint.capabilities())[0]
        client = await RobotClient.connect(capability, token=episode["token"])
        try:
            await _observation(client)
            await asyncio.sleep(bridge.step_timeout * 3)
            assert bridge.step_calls == []

            await client.send_action(np.array([2.0], dtype=np.float32))
            await _observation(client)
            assert len(bridge.step_calls) == 1
            np.testing.assert_allclose(bridge.step_calls[0], [[2.0]])
        finally:
            await _close_client(client)


async def test_late_rollout_waits_for_a_fresh_reset_after_the_batch_has_stepped() -> None:
    bridge = _ProbeBridge(num_envs=2)

    async with _running_bridge(bridge) as endpoint:
        episode = await endpoint.reset(task="same")
        capability = (await endpoint.capabilities())[0]
        client = await RobotClient.connect(capability, token=episode["token"])
        waiting_reset: asyncio.Task[dict[str, Any]] | None = None
        try:
            await _observation(client)
            await client.send_action(np.array([1.0], dtype=np.float32))
            observation = await _observation(client)
            np.testing.assert_allclose(observation["data"]["state"], [1.0])

            waiting_reset = asyncio.create_task(endpoint.reset(task="next"))
            await asyncio.sleep(0.1)
            assert not waiting_reset.done()
            assert bridge.reset_calls == [{"task": "same"}]

            await client.send_action(np.array([2.0], dtype=np.float32))
            observation = await _observation(client)
            np.testing.assert_allclose(observation["data"]["state"], [2.0])
            assert (await endpoint.result(token=episode["token"]))["steps"] == 2

            next_episode = await asyncio.wait_for(waiting_reset, timeout=1)
            assert next_episode["prompt"] == "next"
            assert bridge.reset_calls == [{"task": "same"}, {"task": "next"}]
            await endpoint.result(token=next_episode["token"])
        finally:
            if waiting_reset is not None and not waiting_reset.done():
                waiting_reset.cancel()
                with suppress(asyncio.CancelledError):
                    await waiting_reset
            await _close_client(client)


async def test_reconnect_during_step_does_not_receive_a_duplicate_observation() -> None:
    bridge = _BlockedStepBridge()

    async with _running_bridge(bridge) as endpoint:
        first_episode = await endpoint.reset()
        second_episode = await endpoint.reset()
        capability = (await endpoint.capabilities())[0]
        first = await RobotClient.connect(capability, token=first_episode["token"])
        second = await RobotClient.connect(capability, token=second_episode["token"])
        reconnected: RobotClient | None = None
        try:
            await asyncio.gather(_observation(first), _observation(second))
            await _close_client(second)
            await asyncio.sleep(0.01)

            await first.send_action(np.array([1.0], dtype=np.float32))
            await bridge.step_finished.wait()

            reconnected = await RobotClient.connect(capability, token=second_episode["token"])
            reconnect_observation = await _observation(reconnected)
            np.testing.assert_allclose(reconnect_observation["data"]["state"], [1.0])

            bridge.release_step.set()
            first_observation = await _observation(first)
            np.testing.assert_allclose(first_observation["data"]["state"], [1.0])
            with pytest.raises(TimeoutError):
                await _observation(reconnected, within=bridge.step_timeout)

            await asyncio.gather(
                first.send_action(np.array([2.0], dtype=np.float32)),
                reconnected.send_action(np.array([3.0], dtype=np.float32)),
            )
            first_observation, second_observation = await asyncio.gather(
                _observation(first),
                _observation(reconnected),
            )
            np.testing.assert_allclose(first_observation["data"]["state"], [2.0])
            np.testing.assert_allclose(second_observation["data"]["state"], [2.0])
            assert len(bridge.step_calls) == 2
        finally:
            bridge.release_step.set()
            await _close_client(first)
            await _close_client(second)
            if reconnected is not None:
                await _close_client(reconnected)


async def test_all_terminal_slots_quiesce_until_result() -> None:
    bridge = _ProbeBridge(terminate_after={0: 1})

    async with _running_bridge(bridge) as endpoint:
        episode = await endpoint.reset()
        capability = (await endpoint.capabilities())[0]
        client = await RobotClient.connect(capability, token=episode["token"])
        try:
            assert not (await _observation(client))["terminated"]
            await client.send_action(np.array([1.0], dtype=np.float32))
            assert (await _observation(client))["terminated"]

            await asyncio.sleep(bridge.step_timeout * 3)
            assert len(bridge.step_calls) == 1
            result = await endpoint.result(token=episode["token"])
            assert result["steps"] == 1
        finally:
            await _close_client(client)


async def test_result_recycles_a_terminal_slot_for_a_fresh_episode() -> None:
    bridge = _ProbeBridge(terminate_after={0: 1})

    async with _running_bridge(bridge) as endpoint:
        capability = (await endpoint.capabilities())[0]
        first_episode = await endpoint.reset(task="first")
        first = await RobotClient.connect(capability, token=first_episode["token"])
        second: RobotClient | None = None
        try:
            await _observation(first)
            await first.send_action(np.array([1.0], dtype=np.float32))
            assert (await _observation(first))["terminated"]
            assert (await endpoint.result(token=first_episode["token"]))["steps"] == 1

            second_episode = await endpoint.reset(task="second")
            assert second_episode["token"] != first_episode["token"]
            second = await RobotClient.connect(capability, token=second_episode["token"])
            observation = await _observation(second)
            assert not observation["terminated"]
            np.testing.assert_allclose(observation["data"]["state"], [0.0])
        finally:
            await _close_client(first)
            if second is not None:
                await _close_client(second)

    assert bridge.reset_calls == [{"task": "first"}, {"task": "second"}]


async def test_terminal_slot_emits_once_while_live_peer_continues() -> None:
    bridge = _ProbeBridge(num_envs=2, terminate_after={0: 1})

    async with _running_bridge(bridge) as endpoint:
        first_episode = await endpoint.reset()
        second_episode = await endpoint.reset()
        capability = (await endpoint.capabilities())[0]
        first = await RobotClient.connect(capability, token=first_episode["token"])
        second = await RobotClient.connect(capability, token=second_episode["token"])
        try:
            await asyncio.gather(_observation(first), _observation(second))
            await asyncio.gather(
                first.send_action(np.array([1.0], dtype=np.float32)),
                second.send_action(np.array([2.0], dtype=np.float32)),
            )
            first_obs, second_obs = await asyncio.gather(_observation(first), _observation(second))
            assert first_obs["terminated"]
            assert not second_obs["terminated"]

            await second.send_action(np.array([4.0], dtype=np.float32))
            assert not (await _observation(second))["terminated"]
            assert len(bridge.step_calls) == 2
            np.testing.assert_allclose(bridge.step_calls[1], [[0.0], [4.0]])
            with pytest.raises(TimeoutError):
                await _observation(first, within=bridge.step_timeout)

            await _close_client(first)
            await asyncio.sleep(0.01)
            with pytest.raises(RuntimeError, match="terminated"):
                await RobotClient.connect(capability, token=first_episode["token"])
        finally:
            await _close_client(first)
            await _close_client(second)

"""Robot slot claims key on control session id — cancel/drop/shutdown all free."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from hud.environment.env import current_session_id
from hud.environment.robot.endpoint import RobotEndpoint


@pytest.fixture
def endpoint() -> RobotEndpoint:
    ep = RobotEndpoint.remote("127.0.0.1", 9)
    ep._call = AsyncMock()  # type: ignore[method-assign]
    return ep


@pytest.mark.asyncio
async def test_release_claim_frees_current_session_slot(endpoint: RobotEndpoint) -> None:
    endpoint._call.return_value = {"prompt": "p", "token": "slot-0-abcd"}  # type: ignore[attr-defined]
    token = current_session_id.set("sess-a")
    try:
        await endpoint.reset()
        assert endpoint._claims["sess-a"] == "slot-0-abcd"
        endpoint._call.return_value = {"score": 0.0}  # type: ignore[attr-defined]
        await endpoint.release_claim()
        assert "sess-a" not in endpoint._claims
        endpoint._call.assert_called_with("result", {"token": "slot-0-abcd"})  # type: ignore[attr-defined]
    finally:
        current_session_id.reset(token)


@pytest.mark.asyncio
async def test_release_claim_on_shutdown_frees_each_session(endpoint: RobotEndpoint) -> None:
    endpoint._claims["sess-a"] = "tok-a"
    endpoint._claims["sess-b"] = "tok-b"
    endpoint._call.return_value = {"score": 0.0}  # type: ignore[attr-defined]

    for sid in ("sess-a", "sess-b"):
        token = current_session_id.set(sid)
        try:
            await endpoint.release_claim()
        finally:
            current_session_id.reset(token)

    assert endpoint._claims == {}
    tokens = [c.args[1]["token"] for c in endpoint._call.call_args_list]  # type: ignore[attr-defined]
    assert sorted(tokens) == ["tok-a", "tok-b"]


@pytest.mark.asyncio
async def test_result_marks_claim_freed_so_disconnect_does_not_re_result(
    endpoint: RobotEndpoint,
) -> None:
    token = current_session_id.set("sess-a")
    try:
        endpoint._claims["sess-a"] = "tok-a"
        endpoint._call.return_value = {"score": 1.0, "success": True, "total_reward": 1.0}  # type: ignore[attr-defined]
        await endpoint.result(token="tok-a")
        assert endpoint._claims["sess-a"] == ""
        endpoint._call.reset_mock()  # type: ignore[attr-defined]
        await endpoint.release_claim()
        endpoint._call.assert_not_called()  # type: ignore[attr-defined]
    finally:
        current_session_id.reset(token)


@pytest.mark.asyncio
async def test_release_without_session_context_is_noop(endpoint: RobotEndpoint) -> None:
    endpoint._claims["sess-a"] = "tok-a"
    await endpoint.release_claim()
    assert endpoint._claims["sess-a"] == "tok-a"


@pytest.mark.asyncio
async def test_failed_release_rpc_retries_then_succeeds(endpoint: RobotEndpoint) -> None:
    token = current_session_id.set("sess-a")
    try:
        endpoint._claims["sess-a"] = "tok-a"
        endpoint._call.side_effect = [  # type: ignore[attr-defined]
            ConnectionError("sim down"),
            {"score": 0.0},
        ]
        await endpoint.release_claim()
        assert "sess-a" not in endpoint._claims
        assert endpoint._call.call_count == 2  # type: ignore[attr-defined]
    finally:
        current_session_id.reset(token)


@pytest.mark.asyncio
async def test_stop_drains_claims_that_cancel_failed_to_free(endpoint: RobotEndpoint) -> None:
    endpoint._claims["sess-a"] = "tok-a"
    endpoint._call.side_effect = [  # type: ignore[attr-defined]
        ConnectionError("sim down"),
        ConnectionError("sim down"),
        ConnectionError("sim down"),
    ]
    token = current_session_id.set("sess-a")
    try:
        await endpoint.release_claim()  # retries exhausted — claim kept
        assert endpoint._claims["sess-a"] == "tok-a"
    finally:
        current_session_id.reset(token)

    endpoint._call.side_effect = None  # type: ignore[attr-defined]
    endpoint._call.return_value = {"score": 0.0}  # type: ignore[attr-defined]
    await endpoint.stop()  # last chance while the control link is up
    assert endpoint._claims == {}

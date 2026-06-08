from __future__ import annotations

import pytest

from hud.client.client import HudProtocolError, _response_result


def test_response_result_requires_matching_jsonrpc_id() -> None:
    with pytest.raises(HudProtocolError, match="missing jsonrpc"):
        _response_result("hello", 1, {"id": 1, "result": {}})

    with pytest.raises(HudProtocolError, match="id mismatch"):
        _response_result("hello", 1, {"jsonrpc": "2.0", "id": 2, "result": {}})


def test_response_result_requires_one_result_or_error() -> None:
    with pytest.raises(HudProtocolError, match="result or error"):
        _response_result("hello", 1, {"jsonrpc": "2.0", "id": 1})

    with pytest.raises(HudProtocolError, match="result or error"):
        _response_result("hello", 1, {"jsonrpc": "2.0", "id": 1, "result": {}, "error": {}})


def test_response_result_parses_success_and_error() -> None:
    assert _response_result("hello", 1, {"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}) == {
        "ok": True,
    }

    with pytest.raises(HudProtocolError, match="hud rpc error -32001: unauthorized"):
        _response_result(
            "hello",
            1,
            {"jsonrpc": "2.0", "id": 1, "error": {"code": -32001, "message": "unauthorized"}},
        )

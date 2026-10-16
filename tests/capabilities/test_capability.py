"""Capability factories, manifest round trips, and ``Connection`` validation."""

from __future__ import annotations

from typing import Any

import pytest
from dirty_equals import IsStr

from hud.capabilities import Capability, Connection


@pytest.mark.parametrize(
    ("manifest", "params"),
    [
        (
            {"name": "t", "protocol": "mcp/2025-11-25", "url": "ws://t:80"},
            {"transport": "websocket"},
        ),
        (
            {"name": "t", "protocol": "mcp/2025-11-25", "url": "https://t:443", "params": None},
            {"transport": "streamable-http"},
        ),
        (
            {
                "name": "t",
                "protocol": "mcp/2025-11-25",
                "url": "http://t:80",
                "params": {"transport": "sse"},
            },
            {"transport": "sse"},
        ),
        ({"name": "s", "protocol": "ssh/2", "url": "ssh://s:22"}, {}),
    ],
    ids=["mcp over ws", "mcp over https", "mcp with its transport", "not mcp"],
)
def test_a_manifest_binding_without_params_gets_the_default_mcp_transport(
    manifest: dict[str, Any], params: dict[str, Any]
) -> None:
    assert Capability.from_manifest(manifest).params == params


def connection(**overrides: Any) -> Connection:
    fields: dict[str, Any] = {
        "name": "inference",
        "capability": "shell",
        "url": "https://inference.example/v1",
        "headers": {"Authorization": "Bearer secret"},
    }
    return Connection(**{**fields, **overrides})


def test_a_connection_carries_its_headers_only_on_the_wire() -> None:
    headers = {"Authorization": "Bearer secret"}
    bound = connection(headers=headers)
    headers["Authorization"] = "Bearer changed"

    assert bound.to_wire() == {
        "name": "inference",
        "capability": "shell",
        "url": "https://inference.example/v1",
        "headers": {"Authorization": "Bearer secret"},
    }
    assert Connection.from_wire(bound.to_wire()) == bound
    assert bound.client_url == IsStr(regex=r"http://inference-[0-9a-f]{12}\.hud\.invalid/v1")
    assert "secret" not in repr(bound) and "secret" not in bound.client_url


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"url": "https://user:pw@inference.example"}, "must not contain credentials"),
        ({"url": "https://inference.example/v1?key=1"}, "must not contain a query or fragment"),
        ({"headers": {"Bad Header": "x"}}, "invalid connection header name: 'Bad Header'"),
        (
            {"headers": {"X-Key": "a\r\nInjected: 1"}},
            "invalid connection header value for 'X-Key'",
        ),
    ],
)
def test_a_connection_refuses_names_urls_and_headers_that_could_inject(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        connection(**overrides)


@pytest.mark.parametrize(
    ("wire", "message"),
    [
        (["inference"], "connections must be objects"),
        (
            {"name": 1, "capability": "shell", "url": "https://x", "headers": {"A": "b"}},
            "connection name, capability, and url must be strings",
        ),
        (
            {
                "name": "inference",
                "capability": "shell",
                "url": "https://x",
                "headers": [["A", "b"]],
            },
            "connection headers must map strings to strings",
        ),
        (
            {"name": "inference", "capability": "shell", "url": "https://x", "headers": {"A": 1}},
            "connection headers must map strings to strings",
        ),
    ],
)
def test_a_connection_from_the_wire_must_be_well_typed(wire: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Connection.from_wire(wire)

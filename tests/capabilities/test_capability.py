"""Capability factories, manifest round trips, and ``Connection`` validation."""

from __future__ import annotations

import sys
from typing import Any

import pytest
from dirty_equals import IsStr

from hud.capabilities import Capability, Connection

DEFAULT_SHELL = "cmd" if sys.platform == "win32" else "bash"


@pytest.mark.parametrize(
    ("built", "url", "params"),
    [
        (
            lambda: Capability.ssh(url="workspace", host_pubkey="key"),
            "ssh://workspace:22",
            {"user": "agent", "host_pubkey": "key", "shell": DEFAULT_SHELL},
        ),
        (
            lambda: Capability.ssh(
                url="ssh://root@[::1]",
                user="root",
                host_pubkey="key",
                client_key="secret",
                shell="powershell",
                cwd="/work",
                isolation="bwrap",
                process_connections=True,
            ),
            "ssh://root@[::1]:22",
            {
                "user": "root",
                "host_pubkey": "key",
                "shell": "powershell",
                "client_key": "secret",
                "cwd": "/work",
                "isolation": "bwrap",
                "process_connections": True,
            },
        ),
        (
            lambda: Capability.ssh(url="host:2222", host_pubkey="key", client_key_path="/k"),
            "ssh://host:2222",
            {
                "user": "agent",
                "host_pubkey": "key",
                "shell": DEFAULT_SHELL,
                "client_key_path": "/k",
            },
        ),
        (lambda: Capability.cdp(url="localhost"), "ws://localhost:9222", {}),
        (
            lambda: Capability.cdp(url="ws://browser:9333/devtools/page/a", target_id="a"),
            "ws://browser:9333/devtools/page/a",
            {"target_id": "a"},
        ),
        (
            lambda: Capability.robot(url="sim", contract={"robot_type": "arm"}),
            "ws://sim:9091",
            {"contract": {"robot_type": "arm"}},
        ),
        (lambda: Capability.rfb(url="desktop"), "rfb://desktop:5900", {"display": 0}),
        (
            lambda: Capability.rfb(url="desktop", display=2, password="pw"),
            "rfb://desktop:5902",
            {"display": 2, "password": "pw"},
        ),
        (
            lambda: Capability.rfb(url="rfb://desktop:5999", display=2),
            "rfb://desktop:5999",
            {"display": 2},
        ),
        (
            lambda: Capability.mcp(url="http://tools/mcp"),
            "http://tools:80/mcp",
            {"transport": "streamable-http"},
        ),
        (
            lambda: Capability.mcp(url="https://tools/sse", transport="sse", auth_token="t"),
            "https://tools:443/sse",
            {"transport": "sse", "auth_token": "t"},
        ),
        (
            lambda: Capability.mcp(url="tools"),
            "ws://tools:80",
            {"transport": "websocket"},
        ),
        (lambda: Capability.mcp(url="wss://tools"), "wss://tools:443", {"transport": "websocket"}),
    ],
    ids=[
        "ssh defaults",
        "ssh everything over ipv6",
        "ssh explicit port and key path",
        "cdp defaults",
        "cdp devtools url",
        "robot defaults",
        "rfb display 0",
        "rfb display 2",
        "rfb explicit port wins",
        "mcp http",
        "mcp https sse",
        "mcp bare host",
        "mcp wss",
    ],
)
def test_factories_normalize_urls_and_params(built: Any, url: str, params: dict[str, Any]) -> None:
    capability = built()

    assert (capability.url, capability.params) == (url, params)
    assert Capability.from_manifest(capability.to_manifest()) == capability


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: Capability.mcp(url="stdio:server"),
            "only ws/wss/http/https URLs are supported, got 'stdio'",
        ),
        (
            lambda: Capability.mcp(url="ftp://tools"),
            "only ws/wss/http/https URLs are supported, got 'ftp'",
        ),
        (
            lambda: Capability.mcp(url="http://tools", transport="websocket"),
            "websocket transport requires a ws:// or wss:// URL",
        ),
        (
            lambda: Capability.mcp(url="ws://tools", transport="sse"),
            "sse transport requires an http:// or https:// URL",
        ),
        (lambda: Capability.ssh(url="ssh://", host_pubkey="key"), "invalid URL \\(no host\\)"),
    ],
    ids=["mcp stdio", "mcp ftp", "mcp websocket over http", "mcp sse over ws", "no host"],
)
def test_factories_refuse_urls_their_protocol_cannot_reach(build: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()


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
        ({"name": "Inference"}, "connection name must start with a lowercase letter"),
        ({"name": "1-inference"}, "connection name must start with a lowercase letter"),
        ({"name": "a" * 64}, "connection name must start with a lowercase letter"),
        ({"capability": ""}, "connection capability must not be empty or padded"),
        ({"capability": " shell"}, "connection capability must not be empty or padded"),
        ({"url": "ftp://inference.example"}, "connection url must be an HTTP\\(S\\) URL"),
        ({"url": "https://"}, "connection url must be an HTTP\\(S\\) URL"),
        ({"url": "https://user:pw@inference.example"}, "must not contain credentials"),
        ({"url": "https://inference.example/v1?key=1"}, "must not contain a query or fragment"),
        ({"url": "https://inference.example/v1#top"}, "must not contain a query or fragment"),
        ({"headers": {}}, "connection headers must not be empty"),
        ({"headers": {"Bad Header": "x"}}, "invalid connection header name: 'Bad Header'"),
        ({"headers": {"X-Key": ""}}, "invalid connection header value for 'X-Key'"),
        (
            {"headers": {"X-Key": "a\r\nInjected: 1"}},
            "invalid connection header value for 'X-Key'",
        ),
        ({"headers": {"X-Key": "a\nb"}}, "invalid connection header value for 'X-Key'"),
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

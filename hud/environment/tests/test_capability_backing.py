"""Env-run daemons publish capabilities at serve time, never at declaration.

``env.workspace(root)`` (and, generally, ``env.add_capability(...)`` from an
``@env.initialize`` hook) defers everything — keys, sockets, the directory —
until the env actually serves. The manifest carries the published address,
and ``env.stop()`` runs the matching shutdown hooks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from hud.capabilities import Capability
from hud.environment import Environment

from .conftest import served

if TYPE_CHECKING:
    from pathlib import Path


def test_attaching_a_workspace_writes_nothing(tmp_path: Path) -> None:
    env = Environment("pure")
    env.workspace(tmp_path / "root")

    assert env.capabilities == []  # published at serve time, not declaration
    assert not (tmp_path / "root").exists()


async def test_serving_publishes_the_workspace_capability(tmp_path: Path) -> None:
    env = Environment("ws-env")
    env.workspace(tmp_path / "root")

    async with served(env) as client:
        cap = client.binding("shell")
        assert cap.protocol == "ssh/2"
        assert cap.url.startswith("ssh://")
        assert cap.params["host_pubkey"].startswith("ssh-ed25519")
        assert (tmp_path / "root" / ".hud" / "ssh" / "host_ed25519").exists()


async def test_reconnecting_reuses_the_same_workspace(tmp_path: Path) -> None:
    from hud.clients import connect
    from hud.eval.runtime import _local

    env = Environment("ws-env")
    env.workspace(tmp_path / "root")

    # Client-side urls are per-connection (forwarded); the daemon's identity
    # is its host key, which only stays stable if the workspace is reused.
    async with _local(env) as runtime:
        async with connect(runtime) as client:
            first = client.binding("shell").params["host_pubkey"]
        async with connect(runtime) as client:
            assert client.binding("shell").params["host_pubkey"] == first


async def test_stop_tears_down_the_workspace(tmp_path: Path) -> None:
    import asyncio
    from urllib.parse import urlsplit

    env = Environment("ws-env")
    env.workspace(tmp_path / "root")

    async with served(env):
        # The substrate-local address (the manifest carries a forwarded one).
        backing_port = urlsplit(env.capability("shell").url).port
        assert backing_port is not None

    with pytest.raises(OSError):
        _, writer = await asyncio.open_connection("127.0.0.1", backing_port)
        writer.close()


async def test_restarting_replaces_the_published_address_without_duplicates(
    tmp_path: Path,
) -> None:
    env = Environment("ws-env")
    env.workspace(tmp_path / "root")

    async with served(env):
        pass
    async with served(env):
        assert [c.name for c in env.capabilities] == ["shell"]


async def test_any_initialize_hook_can_publish_a_capability() -> None:
    """Publication is protocol-agnostic: no SDK type per daemon kind."""
    import asyncio

    server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    torn_down = False

    env = Environment("browser-env")

    @env.initialize
    async def _up() -> None:
        env.add_capability(Capability.cdp(name="browser", url=f"ws://127.0.0.1:{port}"))

    @env.shutdown
    async def _down() -> None:
        nonlocal torn_down
        torn_down = True
        server.close()

    assert env.capabilities == []
    async with served(env) as client:
        cap = client.binding("browser")
        assert cap.protocol == "cdp/1.3"
        assert cap.url.startswith("ws://127.0.0.1:")
    assert torn_down  # env.stop() ran the shutdown hook


async def test_loopback_declarations_are_forwarded_and_remote_ones_pass_through() -> None:
    local = Capability.cdp(name="browser", url="ws://127.0.0.1:9222")
    remote = Capability.ssh(name="box", url="ssh://box.example.com:22", host_pubkey="ssh-ed25519 x")
    env = Environment("mixed-env", capabilities=[local, remote])

    async with served(env) as client:
        forwarded = client.binding("browser")
        # Loopback means substrate-local: the client substitutes a local
        # forwarder address, everything else about the capability intact.
        assert forwarded.url != local.url
        assert forwarded.url.startswith("ws://127.0.0.1:")
        assert (forwarded.name, forwarded.protocol) == (local.name, local.protocol)
        # Globally-reachable addresses are the client's to dial directly.
        assert client.binding("box") == remote


def test_a_capability_without_a_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="initialize hook"):
        Environment("bad", capabilities=[Capability(name="b", protocol="cdp/1.3", url="")])


def test_non_capability_entries_are_rejected() -> None:
    with pytest.raises(TypeError, match="expected Capability"):
        Environment("bad", capabilities=cast("list[Capability]", [object()]))

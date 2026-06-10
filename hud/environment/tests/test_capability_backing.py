"""Backed capabilities: declaration is pure data; daemons materialize at hello.

``Capability.shell(root)`` declares intent without an address. Importing or
constructing an env must not generate keys or bind sockets — the managed
workspace backing materializes when the env answers ``hello`` (the manifest
carries the resolved address), and ``env.stop()`` tears it down.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from hud.capabilities import Capability
from hud.environment import Environment

from .conftest import served

if TYPE_CHECKING:
    from pathlib import Path


def test_declaring_a_backed_shell_writes_nothing(tmp_path: Path) -> None:
    env = Environment("pure", capabilities=[Capability.shell(tmp_path / "root")])

    (entry,) = env.capabilities
    assert entry.protocol == "ssh/2"
    assert entry.url == ""  # backed: no address until the env serves
    assert not (tmp_path / "root").exists()


async def test_hello_materializes_a_managed_workspace(tmp_path: Path) -> None:
    env = Environment("ws-env", capabilities=[Capability.shell(tmp_path / "root")])

    async with served(env) as client:
        cap = client.binding("shell")
        assert cap.protocol == "ssh/2"
        assert cap.url.startswith("ssh://")
        assert cap.params["host_pubkey"].startswith("ssh-ed25519")
        assert (tmp_path / "root" / ".hud" / "ssh" / "host_ed25519").exists()


async def test_reconnecting_reuses_the_same_backing(tmp_path: Path) -> None:
    from hud.clients import connect
    from hud.environment.runtime import _local

    env = Environment("ws-env", capabilities=[Capability.shell(tmp_path / "root")])

    async with _local(env) as runtime:
        async with connect(runtime) as client:
            first = client.binding("shell").url
        async with connect(runtime) as client:
            assert client.binding("shell").url == first


async def test_stop_tears_down_the_materialized_backing(tmp_path: Path) -> None:
    import asyncio
    from urllib.parse import urlsplit

    env = Environment("ws-env", capabilities=[Capability.shell(tmp_path / "root")])

    async with served(env) as client:
        cap = client.binding("shell")
    port = urlsplit(cap.url).port
    assert port is not None

    with pytest.raises(OSError):
        _, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.close()


async def test_concrete_declarations_pass_through_unchanged() -> None:
    cap = Capability.cdp(name="browser", url="ws://127.0.0.1:9222")
    env = Environment("browser-env", capabilities=[cap])

    async with served(env) as client:
        assert client.binding("browser") == cap


async def test_backed_declaration_without_a_managed_backing_fails_loudly() -> None:
    from hud.clients import HudProtocolError

    env = Environment("bad", capabilities=[Capability(name="b", protocol="cdp/1.3", url="")])

    with pytest.raises(HudProtocolError, match="no managed backing"):
        async with served(env):
            pass


def test_non_capability_entries_are_rejected() -> None:
    with pytest.raises(TypeError, match="expected Capability"):
        Environment("bad", capabilities=cast("list[Capability]", [object()]))

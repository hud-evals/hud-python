"""The capability client contract, held by every client ``hud.capabilities`` exports.

Each client is a row, keyed by its exported name, with a backend: a served
workspace's SSH daemon, the fake DevTools endpoint, the fake VNC screen, or a
FastMCP server. An Environment publishes the backend, it is served on loopback,
and the client reaches it the way an agent does: at the binding the control
channel hands out, here through a relay (:func:`tests.fixtures.relay.relay`) that can
drop the connection. A client that cannot take part in a scenario is listed with
the reason beside that scenario's table, and a client the package exports with
neither fails :func:`test_every_exported_client_is_a_row_of_every_table`.
"""

from __future__ import annotations

import asyncio
import io
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

import httpx
import pytest
from fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from PIL import Image

import hud.capabilities
from hud.capabilities import Capability, CapabilityClient
from hud.clients import connect
from hud.environment import Environment
from hud.eval import LocalRuntime, Task
from tests.conftest import eventually
from tests.fixtures.asgi import mcp_server
from tests.fixtures.browser import fake_browser
from tests.fixtures.relay import relay
from tests.fixtures.screen import fake_screen

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine
    from contextlib import AbstractAsyncContextManager

    from tests.conftest import HudEnv
    from tests.fixtures.relay import FlakyRelay

COLOR = (10, 200, 30)


@dataclass(frozen=True)
class Backend:
    """What a row serves, and how a scenario drives and observes it."""

    env: Environment
    #: The capability's name in the manifest.
    name: str
    #: A call whose answer proves the client reached the backend, and that answer.
    probe: Callable[[Any], Awaitable[object]]
    answer: object
    #: A call the backend never finishes, under a timeout (``None``: no timeout).
    held: Callable[[Any, float | None], Coroutine[Any, Any, object]] | None = None
    #: Whether the held call reached the backend.
    reached: Callable[[], bool] = lambda: True
    #: Whether the backend ended the held call; ``None`` where it cannot tell.
    ended: Callable[[], bool] | None = None
    #: Whether the client has seen its connection drop.
    dropped: Callable[[Any], bool] = lambda client: True


def running(pid: int) -> bool:
    """Whether ``pid`` is a live process (an unreaped zombie counts as gone)."""
    stat = Path(f"/proc/{pid}/stat")
    try:
        return stat.read_text().split(") ", 1)[1][0] != "Z"
    except FileNotFoundError:
        return False


@asynccontextmanager
async def ssh(tmp_path: Path) -> AsyncIterator[Backend]:
    root = tmp_path / "workspace"
    root.mkdir()
    env = Environment("ssh")
    env.workspace(root, track_files=False)
    pid = root / "pid"

    async def probe(client: Any) -> object:
        return (await client.run("echo ok", timeout=10)).stdout

    yield Backend(
        env,
        "shell",
        probe,
        "ok\n",
        held=lambda client, timeout: client.run("echo $$ > pid; exec sleep 30", timeout=timeout),
        reached=lambda: pid.is_file() and bool(pid.read_text().strip()),
        ended=lambda: not running(int(pid.read_text())),
        dropped=lambda client: client.conn.is_closed(),
    )


@asynccontextmanager
async def cdp(tmp_path: Path) -> AsyncIterator[Backend]:
    replies = {"Page.navigate": {"result": {"frameId": "f-1"}}, "Runtime.evaluate": {"hold": True}}
    async with fake_browser(replies=replies) as browser:
        page = Capability.cdp(name="browser", url=f"{browser.url}/devtools/page/page-1")
        yield Backend(
            Environment("cdp", capabilities=[page]),
            "browser",
            lambda client: client.send("Page.navigate", {"url": "https://example.com"}),
            {"frameId": "f-1"},
            held=lambda client, timeout: asyncio.wait_for(
                client.send("Runtime.evaluate", {"expression": "1"}), timeout
            ),
            reached=lambda: any(c.method == "Runtime.evaluate" for c in browser.commands),
        )


@asynccontextmanager
async def rfb(tmp_path: Path) -> AsyncIterator[Backend]:
    async def probe(client: Any) -> object:
        data, _ = await client.screenshot_png()
        return Image.open(io.BytesIO(data)).convert("RGB").getpixel((0, 0))

    async with fake_screen(width=4, height=2, color=COLOR) as screen:
        yield Backend(
            Environment("rfb", capabilities=[Capability.rfb(name="screen", url=screen.url)]),
            "screen",
            probe,
            COLOR,
        )


@asynccontextmanager
async def mcp(tmp_path: Path) -> AsyncIterator[Backend]:
    started: list[str] = []
    tools = FastMCP("tools")

    @tools.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @tools.tool
    async def wait() -> str:
        """Never answer."""
        started.append("wait")
        await asyncio.Event().wait()
        return "unreachable"

    async def probe(client: Any) -> object:
        return (await client.call_tool("add", {"a": 2, "b": 3})).structuredContent

    async with mcp_server(tools) as capability:
        yield Backend(
            Environment("mcp", capabilities=[capability]),
            "tools",
            probe,
            {"result": 5},
            held=lambda client, timeout: asyncio.wait_for(client.call_tool("wait", {}), timeout),
            reached=lambda: started == ["wait"],
        )


BACKENDS: dict[str, Callable[[Path], AbstractAsyncContextManager[Backend]]] = {
    "SSHClient": ssh,
    "CDPClient": cdp,
    "RFBClient": rfb,
    "MCPClient": mcp,
}
#: Clients held to the contract by another area.
ELSEWHERE = {"RobotClient": "needs the robot extra; tests/robot/test_slots.py holds its contract"}


def client_class(name: str) -> type[CapabilityClient]:
    return getattr(hud.capabilities, name)


@asynccontextmanager
async def connected(name: str, tmp_path: Path) -> AsyncIterator[tuple[Any, Backend, FlakyRelay]]:
    """The row's client, connected to its served backend's binding through a relay."""
    async with (
        BACKENDS[name](tmp_path) as backend,
        LocalRuntime(backend.env)(Task(env=backend.env.name, id="scenario")) as runtime,
        connect(runtime) as control,
    ):
        binding = urlsplit(control.binding(backend.name).url)
        assert binding.hostname is not None and binding.port is not None
        async with relay(binding.hostname, binding.port) as flaky:
            user = f"{binding.username}@" if binding.username else ""
            netloc = f"{user}127.0.0.1:{flaky.port}"
            url = urlunsplit((binding.scheme, netloc, binding.path, binding.query, ""))
            capability = replace(control.binding(backend.name), url=url)
            client = await client_class(name).connect(capability)
            try:
                yield client, backend, flaky
            finally:
                await client.close()


@pytest.mark.parametrize("name", BACKENDS)
async def test_a_client_connects_at_its_binding_and_answers(name: str, tmp_path: Path) -> None:
    async with connected(name, tmp_path) as (client, backend, _):
        answered = await backend.probe(client)

    assert answered == backend.answer


@pytest.mark.parametrize("name", BACKENDS)
async def test_a_capability_published_at_serve_time_opens_through_the_control_port(
    name: str, tmp_path: Path
) -> None:
    async with (
        BACKENDS[name](tmp_path) as backend,
        LocalRuntime(backend.env)(Task(env=backend.env.name, id="scenario")) as runtime,
        connect(runtime) as control,
    ):
        opened = await control.open(backend.name)
        answered = await backend.probe(opened)
        again = await control.open(backend.name)

    assert isinstance(opened, client_class(name))
    assert again is opened
    assert answered == backend.answer


@asynccontextmanager
async def nothing_listening(
    scheme: str, protocol: str, path: str = ""
) -> AsyncIterator[Capability]:
    with socket.socket() as released:
        released.bind(("127.0.0.1", 0))
        port = released.getsockname()[1]
    url = f"{scheme}://127.0.0.1:{port}{path}"
    yield Capability.from_manifest({"name": "absent", "protocol": protocol, "url": url})


@asynccontextmanager
async def hangs_up(scheme: str, protocol: str) -> AsyncIterator[Capability]:
    """A server that closes every connection as soon as it accepts it."""
    server = await asyncio.start_server(lambda reader, writer: writer.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    async with server:
        url = f"{scheme}://127.0.0.1:{port}"
        yield Capability.from_manifest({"name": "absent", "protocol": protocol, "url": url})


@asynccontextmanager
async def browser_without_pages() -> AsyncIterator[Capability]:
    async with fake_browser(pages=[], can_create=False) as browser:
        yield Capability.cdp(url=browser.url)


NO_TARGET: dict[
    str, tuple[str, Callable[[], AbstractAsyncContextManager[Capability]], type, str | None]
] = {
    "ssh-nothing-listening": (
        "SSHClient",
        lambda: nothing_listening("ssh", "ssh/2"),
        ConnectionError,
        "SSH connection failed during handshake",
    ),
    "ssh-handshake-dropped": (
        "SSHClient",
        lambda: hangs_up("ssh", "ssh/2"),
        ConnectionError,
        "SSH connection failed during handshake",
    ),
    "cdp-nothing-listening": (
        "CDPClient",
        lambda: nothing_listening("ws", "cdp/1.3"),
        httpx.ConnectError,
        None,
    ),
    "cdp-no-page-target": (
        "CDPClient",
        browser_without_pages,
        ValueError,
        r"no CDP page target available at 127\.0\.0\.1:",
    ),
    "rfb-nothing-listening": (
        "RFBClient",
        lambda: nothing_listening("rfb", "rfb/3.8"),
        ConnectionRefusedError,
        None,
    ),
    "mcp-nothing-listening": (
        "MCPClient",
        lambda: nothing_listening("http", "mcp/2025-11-25", "/mcp"),
        McpError,
        "Transport error",
    ),
}


@pytest.mark.parametrize(
    ("name", "target", "error", "message"), NO_TARGET.values(), ids=NO_TARGET.keys()
)
async def test_a_client_with_nothing_to_connect_to_fails_clearly(
    hud_env: HudEnv,
    name: str,
    target: Callable[[], AbstractAsyncContextManager[Capability]],
    error: type[Exception],
    message: str | None,
) -> None:
    hud_env.set(HUD_CLIENT_TIMEOUT="1")

    async with target() as capability:
        with pytest.raises(error, match=message):
            await client_class(name).connect(capability)


RECONNECTING = ["SSHClient", "RFBClient", "MCPClient"]
NOT_RECONNECTING = {
    "CDPClient": "holds one page socket and does not reopen it; test_cdp.py holds what a drop "
    "does to its commands",
}


@pytest.mark.parametrize("name", RECONNECTING)
async def test_a_client_reconnects_after_its_connection_drops(name: str, tmp_path: Path) -> None:
    async with connected(name, tmp_path) as (client, backend, flaky):
        await backend.probe(client)
        accepted = flaky.accepted
        flaky.sever()
        await eventually(lambda: backend.dropped(client))
        answered = await backend.probe(client)

    assert answered == backend.answer
    assert flaky.accepted > accepted


async def cancel(backend: Backend, client: Any) -> None:
    assert backend.held is not None
    call = asyncio.create_task(backend.held(client, None))
    await eventually(backend.reached)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call


async def time_out(backend: Backend, client: Any) -> None:
    assert backend.held is not None
    with pytest.raises(TimeoutError):
        await backend.held(client, 0.5)
    await eventually(backend.reached)


ENDINGS = {"cancel": cancel, "timeout": time_out}
HOLDING = ["SSHClient", "CDPClient", "MCPClient"]
NOT_HOLDING = {"RFBClient": "a screenshot is one framebuffer read; there is no call to hold"}


@pytest.mark.parametrize("ending", ENDINGS)
@pytest.mark.parametrize("name", HOLDING)
async def test_a_cancelled_or_timed_out_call_ends_and_the_next_one_runs(
    name: str, ending: str, tmp_path: Path
) -> None:
    async with connected(name, tmp_path) as (client, backend, _):
        await ENDINGS[ending](backend, client)
        if backend.ended is not None:
            await eventually(backend.ended)
        answered = await backend.probe(client)

    assert answered == backend.answer


def test_every_exported_client_is_a_row_of_every_table() -> None:
    exported = {
        name
        for name in hud.capabilities.__all__
        if name.endswith("Client") and name != "CapabilityClient"
    }
    rows = BACKENDS.keys() | ELSEWHERE.keys()

    assert rows == exported
    assert {name for name, *_ in NO_TARGET.values()} | ELSEWHERE.keys() == exported
    assert set(RECONNECTING) | NOT_RECONNECTING.keys() == BACKENDS.keys()
    assert set(HOLDING) | NOT_HOLDING.keys() == BACKENDS.keys()

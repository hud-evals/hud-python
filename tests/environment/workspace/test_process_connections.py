"""Controller connections bound to one process: only that process reaches them.

A control session hands the environment a :class:`~hud.capabilities.Connection`;
the workspace serves it behind a credential-owning relay that only the process
it was requested for may connect to. The guard that enforces this has one
backend per kernel facility (seccomp user notification, ptrace), each held to
the same table; the served workspace is the sandbox-lane scenario.
"""

from __future__ import annotations

import asyncio
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import pytest

from hud.capabilities import Connection, SSHClient
from hud.clients import HudClient, HudProtocolError, connect
from hud.environment import Environment, Peer, WorkspaceRoute, process_guard
from hud.environment.egress import ANY_HOST, BRIDGE_PORT
from hud.environment.process_guard import ProcessConnectionGuard, process_connections_supported
from hud.eval import LocalRuntime, Task

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterator

pytestmark = pytest.mark.skipif(
    not process_connections_supported(), reason="process connection guards are unavailable"
)

ECHO = Task(env="sessions", id="echo")


class Recorded(BaseHTTPRequestHandler):
    """Answers ``ok`` and records the Authorization header it was sent."""

    authorizations: list[str | None]

    def do_GET(self) -> None:
        self.authorizations.append(self.headers.get("Authorization"))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: Any) -> None:
        pass


@contextmanager
def upstream() -> Iterator[tuple[int, list[str | None]]]:
    authorizations: list[str | None] = []
    handler = type("Upstream", (Recorded,), {"authorizations": authorizations})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], authorizations
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def fetch(url: str) -> str:
    return (
        "import urllib.request;"
        f"request=urllib.request.Request({url!r},headers={{'Authorization':'Bearer visible'}});"
        "print(urllib.request.urlopen(request,timeout=5).read().decode())"
    )


def threaded_fetch(url: str) -> str:
    return (
        "import threading,urllib.request;"
        "result=[];"
        "thread=threading.Thread(target=lambda:result.append("
        f"urllib.request.urlopen({url!r},timeout=5).read().decode()));"
        "thread.start();thread.join();print(result[0])"
    )


def direct_fetch(host: str, port: int, source_address: str = "None") -> str:
    """Fetch ``port`` on ``host`` directly, from ``source_address`` (Python source)."""
    return (
        "import http.client,socket;"
        f"c=http.client.HTTPConnection({host},{port},timeout=5,source_address={source_address});"
        "c.request('GET','/');print(c.getresponse().read().decode())"
    )


MSG_FASTOPEN = 0x20000000


def fast_open_fetch(send: str, host: str, port: int) -> str:
    """Fetch ``host:port`` over a socket that ``send`` opens with TCP Fast Open."""
    return (
        "import socket;"
        f"address=(socket.gethostbyname({host!r}),{port});"
        "request=b'GET / HTTP/1.1\\r\\nHost: relay\\r\\nConnection: close\\r\\n\\r\\n';"
        f"s=socket.socket();{send};"
        "print(s.recv(65536).decode())"
    )


def in_a_child(source: str) -> str:
    """Run ``source`` in a child process; print the child's exit status."""
    return (
        f"import subprocess,sys;print(subprocess.run([sys.executable,'-c',{source!r}]).returncode)"
    )


@contextmanager
def guard_directory() -> Iterator[Path]:
    """Outside the pytest tmp dir: the guard's unix socket path caps near 104 bytes."""
    directory = Path(tempfile.mkdtemp(dir="/tmp"))
    try:
        yield directory / "guard"
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def backend_works(backend: str) -> bool:
    return (
        sys.platform == "linux"
        and subprocess.run(
            [sys.executable, process_guard.__file__, "--probe", backend],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


@pytest.mark.e2e
@pytest.mark.sandbox
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            name,
            marks=pytest.mark.skipif(not backend_works(name), reason=f"no {name} guard backend"),
        )
        for name in ("notify", "ptrace")
    ],
)
async def test_only_the_guarded_process_reaches_its_protected_destination(
    backend: Literal["notify", "ptrace"],
) -> None:
    with upstream() as (protected, _), upstream() as (ordinary, _):
        protected_url = f"http://127.0.0.1:{protected}"
        child = (
            "import subprocess,sys,threading,urllib.request;"
            f"print(urllib.request.urlopen({protected_url!r},timeout=5).read().decode());"
            f"exec({threaded_fetch(protected_url)!r});"
            f"blocked=subprocess.run([sys.executable,'-c',{fetch(protected_url)!r}]);"
            "print(f'descendant={blocked.returncode}',flush=True);"
            f"permitted=subprocess.run([sys.executable,'-c',{fetch(f'http://127.0.0.1:{ordinary}')!r}]);"
            "print(f'ordinary={permitted.returncode}')"
        )
        target = ("127.0.0.1", protected)
        with guard_directory() as directory:
            guard = ProcessConnectionGuard(directory, {target}, {target}, backend=backend)
            guard.start()
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                str(guard.helper_path),
                "--backend",
                backend,
                str(guard.socket_path),
                "--",
                sys.executable,
                "-c",
                child,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                await guard.wait_ready()
                stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                guard.close()

    assert process.returncode == 0, stderr.decode(errors="replace")
    lines = stdout.decode().splitlines()
    assert lines[:2] == ["ok", "ok"]
    assert lines[2] != "descendant=0"
    assert lines[3:] == ["ok", "ordinary=0"]


def python(source: str) -> str:
    """A session command running ``source`` on the image's interpreter."""
    return f"exec python3 -c {shlex.quote(source)}"


IO_URING_DENIED = (
    "import ctypes,errno,sys;"
    "libc=ctypes.CDLL(None,use_errno=True);"
    "result=libc.syscall(425,0,None);"
    "sys.exit(0 if result == -1 and ctypes.get_errno() == errno.EPERM else 1)"
)


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_only_the_bound_session_process_reaches_a_controller_connection(
    tmp_path: Path,
) -> None:
    with (
        upstream() as (protected, protected_seen),
        upstream() as (neighbouring, neighbouring_seen),
        upstream() as (ordinary, ordinary_seen),
    ):
        connection = Connection(
            name="inference",
            capability="ssh",
            url=f"http://127.0.0.1:{protected}",
            headers={"Authorization": "Bearer scoped-runtime-token"},
        )
        # Two relays on one port listen on different loopback addresses.
        neighbour = Connection(
            name="neighbour",
            capability="ssh",
            url=f"http://127.0.0.1:{neighbouring}",
            headers={"Authorization": "Bearer neighbour-runtime-token"},
        )
        url = connection.client_url
        relay = f"socket.gethostbyname({connection.host!r})"
        env = Environment("process-connection")
        env.workspace(
            tmp_path / "root",
            peers=(Peer("ordinary.hud.invalid", 80, target=("127.0.0.1", ordinary)),),
            allowed_hosts={ANY_HOST},
            require_isolation=True,
            track_files=False,
        )
        cases = {
            "unbound": (fetch(url), False),
            "bound": (fetch(url), True),
            "bound-thread": (threaded_fetch(url), True),
            "bound-descendant": (in_a_child(fetch(url)), True),
            "bound-descendant-by-mapped-address": (
                in_a_child(direct_fetch(f"'::ffff:'+{relay}", connection.port)),
                True,
            ),
            "bound-descendant-through-the-proxy": (
                in_a_child(
                    "import http.client,sys;"
                    f"c=http.client.HTTPConnection('127.0.0.1',{BRIDGE_PORT},timeout=5);"
                    f"c.request('GET',{url!r});r=c.getresponse();"
                    "sys.exit(0 if r.status==200 and r.read()==b'ok' else 9)"
                ),
                True,
            ),
            "bound-io-uring": (IO_URING_DENIED, True),
            "bound-ordinary-peer": (fetch("http://ordinary.hud.invalid"), True),
            "unbound-ordinary-peer": (fetch("http://ordinary.hud.invalid"), False),
            "unbound-by-unspecified-address": (direct_fetch("'0.0.0.0'", connection.port), False),
            "unbound-by-unspecified-address-from-the-relay-address": (
                direct_fetch("'0.0.0.0'", connection.port, f"({relay},0)"),
                False,
            ),
            "unbound-by-unspecified-ipv6-address-from-the-mapped-relay-address": (
                direct_fetch("'::'", connection.port, f"('::ffff:'+{relay},0)"),
                False,
            ),
            "unbound-by-fast-open-sendto": (
                fast_open_fetch(
                    f"s.sendto(request,{MSG_FASTOPEN},address)", connection.host, connection.port
                ),
                False,
            ),
            "unbound-by-fast-open-sendmsg": (
                fast_open_fetch(
                    f"s.sendmsg([request],[],{MSG_FASTOPEN},address)",
                    connection.host,
                    connection.port,
                ),
                False,
            ),
        }
        results: dict[str, tuple[int | None, str]] = {}
        async with (
            LocalRuntime(env)(Task(env=env.name, id="test")) as runtime,
            connect(runtime, connections=(connection, neighbour)) as client,
        ):
            ssh = cast("SSHClient", await client.open("ssh"))
            advertised = ssh.capability.params["process_connections"]
            for case, (source, bound) in cases.items():
                process = await ssh.create_process(
                    python(source), connections=(connection,) if bound else ()
                )
                completed = await process.wait()
                results[case] = (completed.returncode, cast("bytes", completed.stdout).decode())

    assert advertised is True
    for case in (
        "unbound",
        "unbound-by-unspecified-address",
        "unbound-by-unspecified-address-from-the-relay-address",
        "unbound-by-unspecified-ipv6-address-from-the-mapped-relay-address",
        "unbound-by-fast-open-sendto",
        "unbound-by-fast-open-sendmsg",
    ):
        assert results[case][0] != 0, case
    assert results["bound"] == results["bound-thread"] == (0, "ok\n")
    for case in (
        "bound-descendant",
        "bound-descendant-by-mapped-address",
        "bound-descendant-through-the-proxy",
    ):
        assert results[case][0] == 0 and results[case][1].splitlines()[-1] != "0", case
    assert results["bound-io-uring"][0] == 0
    assert results["bound-ordinary-peer"] == results["unbound-ordinary-peer"] == (0, "ok\n")
    assert protected_seen == ["Bearer scoped-runtime-token"] * 2
    assert neighbouring_seen == []
    assert ordinary_seen == ["Bearer visible"] * 2


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_a_later_session_cannot_launch_against_a_released_connection(
    tmp_path: Path,
) -> None:
    with upstream() as (protected, _):
        connection = Connection(
            name="inference",
            capability="ssh",
            url=f"http://127.0.0.1:{protected}",
            headers={"Authorization": "Bearer scoped-runtime-token"},
        )
        env = Environment("process-connection")
        env.workspace(
            tmp_path / "root", allowed_hosts={ANY_HOST}, require_isolation=True, track_files=False
        )
        async with LocalRuntime(env)(Task(env=env.name, id="test")) as runtime:
            async with connect(runtime, connections=(connection,)) as first:
                ssh = cast("SSHClient", await first.open("ssh"))
                bound = await ssh.create_process(
                    python(fetch(connection.client_url)), connections=(connection,)
                )
                first_status = (await bound.wait()).returncode
            async with connect(runtime) as later:
                ssh = cast("SSHClient", await later.open("ssh"))
                stale = await ssh.create_process(
                    python(fetch(connection.client_url)), connections=(connection,)
                )
                result = await stale.wait()

    assert first_status == 0
    assert result.returncode == 1


def echo_env() -> Environment:
    env = Environment("sessions")

    @env.template()
    async def echo(tag: str):
        yield f"go {tag}"
        yield {"score": 1.0, "tag": tag}

    return env


def inference(token: str) -> Connection:
    return Connection(
        name="inference",
        capability="ssh",
        url="https://inference.hud.so",
        headers={"Authorization": f"Bearer {token}"},
    )


async def resolved_names(client: HudClient) -> list[str]:
    """The names a session of ``client`` resolves through its ``/etc/hosts``."""
    ssh = cast("SSHClient", await client.open("ssh"))
    hosts = str((await ssh.run("cat /etc/hosts", check=True)).stdout)
    return [name for line in hosts.splitlines() for name in line.split()[1:]]


async def a_finished_session_releases_its_connection(env: Environment, root: Path) -> None:
    env.workspace(root, network=False)
    async with LocalRuntime(env)(ECHO) as runtime:
        async with connect(runtime, connections=[inference("first")]) as first:
            await first.start_task("echo", {"tag": "a"})
            await first.grade({"answer": "x"})
        async with connect(runtime, connections=[inference("rotated")]) as second:
            await second.start_task("echo", {"tag": "b"})
            names = await resolved_names(second)
            assert names.count(inference("rotated").host) == 1
            assert (await second.grade({"answer": "x"}))["tag"] == "b"


async def a_live_session_keeps_its_connection_name(env: Environment, root: Path) -> None:
    env.workspace(root, network=False)
    async with LocalRuntime(env)(ECHO) as runtime, connect(runtime, connections=[inference("a")]):
        with pytest.raises(HudProtocolError, match="bound to another control session"):
            async with connect(runtime, connections=[inference("b")]):
                pass


async def a_finished_session_takes_its_route_with_it(env: Environment, root: Path) -> None:
    env.workspace(root, network=False)
    route = WorkspaceRoute("ssh", "inference.hud.so", 443)
    async with LocalRuntime(env)(ECHO) as runtime:
        async with connect(runtime, workspace_routes=[route, route]) as first:
            await first.start_task("echo", {"tag": "a"})
            assert (await resolved_names(first)).count("inference.hud.so") == 1
            await first.grade({"answer": "x"})
        async with connect(runtime) as later:
            await later.start_task("echo", {"tag": "later"})
            assert "inference.hud.so" not in await resolved_names(later)
            await later.grade({"answer": "x"})


@pytest.mark.e2e
@pytest.mark.sandbox
@pytest.mark.parametrize(
    "scenario",
    [
        a_finished_session_releases_its_connection,
        a_live_session_keeps_its_connection_name,
        a_finished_session_takes_its_route_with_it,
    ],
)
async def test_a_sessions_workspace_bindings_end_with_the_session(
    tmp_path: Path, scenario: Callable[[Environment, Path], Awaitable[None]]
) -> None:
    await scenario(echo_env(), tmp_path / "root")

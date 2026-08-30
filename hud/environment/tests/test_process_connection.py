"""Process-bound controller connection integration."""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, cast

import pytest

from hud.capabilities import Connection, SSHClient
from hud.clients import connect
from hud.environment import Environment, Peer
from hud.environment.egress import ANY_HOST, BRIDGE_PORT
from hud.environment.process_guard import ProcessConnectionGuard, process_connections_supported
from hud.eval import LocalRuntime, Task

pytestmark = pytest.mark.skipif(
    not process_connections_supported(),
    reason="process connection guards are unavailable",
)

_GUARD_PATH = Path(__file__).parents[1] / "process_guard.py"
_PTRACE_SUPPORTED = (
    sys.platform == "linux"
    and subprocess.run(
        [sys.executable, str(_GUARD_PATH), "--probe", "ptrace"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode
    == 0
)


class _ProtectedUpstream(BaseHTTPRequestHandler):
    authorization: str | None = None

    def do_GET(self) -> None:
        type(self).authorization = self.headers.get("Authorization")
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        pass


class _OrdinaryUpstream(_ProtectedUpstream):
    authorization: str | None = None


def _server(handler: type[BaseHTTPRequestHandler]) -> tuple[HTTPServer, threading.Thread]:
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _fetch_source(url: str) -> str:
    return (
        "import urllib.request;"
        f"request=urllib.request.Request({url!r},headers={{'Authorization':'Bearer visible'}});"
        "print(urllib.request.urlopen(request,timeout=5).read().decode())"
    )


def _fetch_script(url: str) -> str:
    return f"exec {shlex.quote(sys.executable)} -c {shlex.quote(_fetch_source(url))}"


def _threaded_fetch_source(url: str) -> str:
    return (
        "import threading,urllib.request;"
        "result=[];"
        f"request=urllib.request.Request({url!r});"
        "thread=threading.Thread(target=lambda:result.append("
        "urllib.request.urlopen(request,timeout=5).read().decode()));"
        "thread.start();thread.join();print(result[0])"
    )


def _proxy_fetch_source(url: str) -> str:
    return (
        "import http.client,sys;"
        f"connection=http.client.HTTPConnection('127.0.0.1',{BRIDGE_PORT},timeout=5);"
        f"connection.request('GET',{url!r});"
        "response=connection.getresponse();body=response.read();"
        "sys.exit(0 if response.status==200 and body==b'ok' else 9)"
    )


def _direct_fetch_source(host: str, port: int) -> str:
    return (
        "import http.client;"
        f"connection=http.client.HTTPConnection({host!r},{port},timeout=5);"
        "connection.request('GET','/');"
        "print(connection.getresponse().read().decode())"
    )


def _proxy_environment_source() -> str:
    return (
        "import os;"
        "print(os.environ.get('http_proxy',''));"
        "print(os.environ.get('https_proxy',''));"
        "print(os.environ.get('no_proxy',''))"
    )


def test_guard_projects_a_standalone_helper(tmp_path: Path) -> None:
    guard = ProcessConnectionGuard(tmp_path / "guard", set(), set(), backend="notify")
    try:
        guard.start()
        assert guard.helper_path.read_bytes() == _GUARD_PATH.read_bytes()
        assert guard.helper_path.stat().st_mode & 0o777 == 0o500
    finally:
        guard.close()


@pytest.mark.skipif(not _PTRACE_SUPPORTED, reason="ptrace guard backend is unavailable")
@pytest.mark.asyncio
async def test_ptrace_backend_emulates_connects_and_blocks_descendants(tmp_path: Path) -> None:
    protected, protected_thread = _server(_ProtectedUpstream)
    ordinary, ordinary_thread = _server(_OrdinaryUpstream)
    protected_url = f"http://127.0.0.1:{protected.server_address[1]}"
    ordinary_url = f"http://127.0.0.1:{ordinary.server_address[1]}"
    child = (
        "import subprocess,sys,threading,urllib.request;"
        f"protected={_fetch_source(protected_url)!r};"
        f"ordinary={_fetch_source(ordinary_url)!r};"
        f"print(urllib.request.urlopen({protected_url!r},timeout=5).read().decode());"
        "threaded=[];"
        f"thread=threading.Thread(target=lambda:threaded.append(urllib.request.urlopen({protected_url!r},timeout=5).read().decode()));"
        "thread.start();thread.join();print(threaded[0]);"
        "blocked=subprocess.run([sys.executable,'-c',protected]);"
        "print(f'blocked={blocked.returncode}');"
        "permitted=subprocess.run([sys.executable,'-c',ordinary]);"
        "print(f'ordinary={permitted.returncode}')"
    )
    target = ("127.0.0.1", protected.server_address[1])
    guard = ProcessConnectionGuard(tmp_path / "guard", {target}, {target}, backend="ptrace")
    process = None
    try:
        guard.start()
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(_GUARD_PATH),
            "--backend",
            "ptrace",
            str(guard.socket_path),
            "--",
            sys.executable,
            "-c",
            child,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await guard.wait_ready()
        stdout, stderr = await asyncio.wait_for(process.communicate(), 20)
        assert process.returncode == 0, stderr.decode(errors="replace")
        lines = stdout.decode().splitlines()
        assert lines[:2] == ["ok", "ok"]
        assert lines[2] != "blocked=0"
        assert lines[3:] == ["ok", "ordinary=0"]
    finally:
        if process is not None and process.returncode is None:
            process.kill()
            await process.wait()
        guard.close()
        for server, thread in (
            (protected, protected_thread),
            (ordinary, ordinary_thread),
        ):
            server.shutdown()
            server.server_close()
            thread.join()


async def test_only_bound_process_reaches_controller_connection(tmp_path: Path) -> None:
    protected, protected_thread = _server(_ProtectedUpstream)
    ordinary, ordinary_thread = _server(_OrdinaryUpstream)
    connection = Connection(
        name="inference",
        capability="ssh",
        url=f"http://127.0.0.1:{protected.server_address[1]}",
        headers={"Authorization": "Bearer scoped-runtime-token"},
    )
    env = Environment("process-connection")
    env.workspace(
        tmp_path / "root",
        peers=(
            Peer(
                "ordinary.hud.invalid",
                80,
                target=("127.0.0.1", ordinary.server_address[1]),
            ),
        ),
        allowed_hosts={ANY_HOST},
        require_isolation=True,
        track_files=False,
    )
    task = Task(env=env.name, id="test")
    try:
        async with (
            LocalRuntime(env)(task) as runtime,
            connect(runtime, connections=(connection,)) as client,
        ):
            ssh = cast("SSHClient", await client.open("ssh"))
            assert ssh.capability.params["process_connections"] is True

            unbound = await ssh.run(_fetch_script(connection.client_url), check=False)
            assert unbound.returncode != 0

            bound = await ssh.create_process(
                _fetch_script(connection.client_url),
                connections=(connection,),
            )
            completed = await bound.wait()
            assert completed.returncode == 0
            assert completed.stdout == b"ok\n"

            proxy_environment = await ssh.create_process(
                f"exec {shlex.quote(sys.executable)} -c {shlex.quote(_proxy_environment_source())}",
                connections=(connection,),
            )
            proxy_environment_result = await proxy_environment.wait()
            assert proxy_environment_result.returncode == 0
            assert isinstance(proxy_environment_result.stdout, bytes)
            proxy_values = proxy_environment_result.stdout.decode().splitlines()
            assert proxy_values[:2] == [f"http://127.0.0.1:{BRIDGE_PORT}"] * 2
            assert "ordinary.hud.invalid" in proxy_values[2].split(",")

            threaded = await ssh.create_process(
                f"exec {shlex.quote(sys.executable)} -c "
                f"{shlex.quote(_threaded_fetch_source(connection.client_url))}",
                connections=(connection,),
            )
            threaded_result = await threaded.wait()
            assert threaded_result.returncode == 0
            assert threaded_result.stdout == b"ok\n"

            child_source = (
                "import subprocess,sys;"
                f"result=subprocess.run([sys.executable,'-c',{_fetch_source(connection.client_url)!r}]);"
                "print(result.returncode)"
            )
            child = await ssh.create_process(
                f"exec {shlex.quote(sys.executable)} -c {shlex.quote(child_source)}",
                connections=(connection,),
            )
            child_result = await child.wait()
            assert child_result.returncode == 0
            assert isinstance(child_result.stdout, bytes)
            assert child_result.stdout.strip() != b"0"

            mapped_fetch = _direct_fetch_source("::ffff:127.0.0.1", connection.port)
            mapped_child_source = (
                "import subprocess,sys;"
                f"result=subprocess.run([sys.executable,'-c',{mapped_fetch!r}]);"
                "print(result.returncode)"
            )
            mapped_child = await ssh.create_process(
                f"exec {shlex.quote(sys.executable)} -c {shlex.quote(mapped_child_source)}",
                connections=(connection,),
            )
            mapped_child_result = await mapped_child.wait()
            assert mapped_child_result.returncode == 0
            assert isinstance(mapped_child_result.stdout, bytes)
            assert mapped_child_result.stdout.strip() != b"0"

            proxy_child_source = (
                "import subprocess,sys;"
                f"result=subprocess.run([sys.executable,'-c',{_proxy_fetch_source(connection.client_url)!r}]);"
                "print(result.returncode)"
            )
            proxy_child = await ssh.create_process(
                f"exec {shlex.quote(sys.executable)} -c {shlex.quote(proxy_child_source)}",
                connections=(connection,),
            )
            proxy_child_result = await proxy_child.wait()
            assert proxy_child_result.returncode == 0
            assert isinstance(proxy_child_result.stdout, bytes)
            assert proxy_child_result.stdout.strip() != b"0"

            io_uring_source = (
                "import ctypes,errno,sys;"
                "libc=ctypes.CDLL(None,use_errno=True);"
                "result=libc.syscall(425,0,None);"
                "sys.exit(0 if result == -1 and ctypes.get_errno() == errno.EPERM else 1)"
            )
            io_uring = await ssh.create_process(
                f"exec {shlex.quote(sys.executable)} -c {shlex.quote(io_uring_source)}",
                connections=(connection,),
            )
            io_uring_result = await io_uring.wait()
            assert io_uring_result.returncode == 0

            ordinary_result = await ssh.run(
                _fetch_script("http://ordinary.hud.invalid"),
                check=False,
            )
            assert ordinary_result.returncode == 0
            assert ordinary_result.stdout == "ok\n"
    finally:
        for server, thread in (
            (protected, protected_thread),
            (ordinary, ordinary_thread),
        ):
            server.shutdown()
            server.server_close()
            thread.join()

    assert _ProtectedUpstream.authorization == "Bearer scoped-runtime-token"
    assert _OrdinaryUpstream.authorization == "Bearer visible"

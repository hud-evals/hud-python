"""The sandbox a workspace's processes run in, observed from inside them.

Sessions run as the identity the workspace was given, in one sandbox they all
share, behind the network the task declared. Most rows here need Linux, root
and bubblewrap (the sandbox lane); the probe that decides whether bubblewrap
can sandbox at all runs anywhere, against stand-in ``bwrap`` executables.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from hud.environment import Peer, Workspace
from hud.environment.workspace import usable_bwrap

from .conftest import SANDBOX, shell

if TYPE_CHECKING:
    from tests.conftest import HudEnv

IDENTITY = "echo $(id -u) $(id -g) $(id -G)"


def namespace(kind: str) -> str:
    return os.readlink(f"/proc/self/ns/{kind}")


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        pytest.param(
            {"shell_uid": 1000},
            None,
            id="shell-uid-is-a-no-op-off-root",
            marks=pytest.mark.skipif(os.geteuid() == 0, reason="needs a non-root server"),
        ),
        pytest.param(
            {"require_isolation": True, "shell_uid": 1000},
            "1000 1000 1000",
            id="dropped-to-shell-uid",
            marks=SANDBOX,
        ),
        pytest.param(
            {"require_isolation": True, "shell_uid": 1000, "shell_gid": 2000},
            "1000 2000 2000",
            id="dropped-keeps-the-declared-group",
            marks=SANDBOX,
        ),
    ],
)
async def test_a_session_runs_as_the_identity_the_workspace_was_given(
    tmp_path: Path, options: dict[str, Any], expected: str | None
) -> None:
    async with shell(tmp_path / "root", **options) as (_, client):
        identity = (await client.run(IDENTITY, check=True)).stdout
        no_new_privs = (await client.run("grep NoNewPrivs /proc/self/status", check=False)).stdout

    if expected is None:
        assert str(identity).split()[0] == str(os.getuid())
    else:
        assert str(identity).strip() == expected
    if "shell_uid" in options and expected is not None:
        # A setuid binary must not let the dropped shell regain root.
        assert str(no_new_privs).split() == ["NoNewPrivs:", "1"]


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_a_dropped_session_has_a_world_writable_tmp(tmp_path: Path) -> None:
    async with shell(tmp_path / "root", require_isolation=True, shell_uid=1000) as (_, client):
        result = await client.run("stat -c %a /tmp && touch /tmp/scratch", check=False)

    assert (result.exit_status, str(result.stdout).strip()) == (0, "1777")


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_only_the_workspace_directory_is_handed_to_the_shell_uid(tmp_path: Path) -> None:
    """The handoff is O(1): recursing over baked content (node_modules, a venv)
    on the serving path would delay the control-port bind past the deploy
    readiness probe, so pre-staged content stays its author's."""
    root = tmp_path / "root"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("x = 1\n")

    async with shell(root, shell_uid=1000):
        owners = {
            path.name: path.stat().st_uid for path in (root, root / "pkg", root / "pkg" / "mod.py")
        }

    assert owners == {"root": 1000, "pkg": 0, "mod.py": 0}


def isolation_unavailable(_: pytest.MonkeyPatch, hud_env: HudEnv) -> dict[str, Any]:
    del hud_env
    if usable_bwrap() is not None:
        pytest.skip("this host can sandbox")
    return {"require_isolation": True}


def setpriv_missing(_: pytest.MonkeyPatch, hud_env: HudEnv) -> dict[str, Any]:
    hud_env.set(PATH="/nonexistent")
    return {"shell_uid": 1000}


def chown_denied(monkeypatch: pytest.MonkeyPatch, hud_env: HudEnv) -> dict[str, Any]:
    del hud_env

    def deny(*_: object) -> None:
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(os, "lchown", deny)
    return {"shell_uid": 1000}


@pytest.mark.parametrize(
    ("cause", "error", "message"),
    [
        pytest.param(
            isolation_unavailable,
            RuntimeError,
            "isolation was required",
            id="isolation-required-without-bwrap",
        ),
        pytest.param(
            setpriv_missing,
            RuntimeError,
            "privileges cannot be dropped",
            id="root-without-setpriv",
            marks=SANDBOX,
        ),
        pytest.param(
            chown_denied, PermissionError, "not permitted", id="handoff-chown-denied", marks=SANDBOX
        ),
    ],
)
async def test_a_workspace_that_cannot_wall_its_sessions_refuses_to_serve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hud_env: HudEnv,
    cause: Any,
    error: type[Exception],
    message: str,
) -> None:
    workspace = Workspace(tmp_path / "root", **cause(monkeypatch, hud_env))

    with pytest.raises(error, match=message):
        await workspace.start()


@pytest.mark.e2e
@pytest.mark.sandbox
@pytest.mark.parametrize("guest_path", ["/workspace", "/tmp/hud-coding/123/workspace"])
async def test_the_shell_user_reads_and_writes_the_guest_workspace(
    tmp_path: Path, guest_path: str
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "input.txt").write_text("hello\n")
    command = f"id -u; cat {guest_path}/input.txt; echo written > {guest_path}/output.txt"
    async with shell(
        root, guest_path=guest_path, network=False, shell_uid=1000, require_isolation=True
    ) as (workspace, client):
        session = await client.run(command, check=False)
        await workspace.terminate_sessions()
        captured = await workspace.run(["sh", "-c", command], scope="environment")

    assert (session.exit_status, session.stdout) == (0, "1000\nhello\n")
    assert (captured.returncode, captured.stdout) == (0, b"1000\nhello\n")
    assert (root / "output.txt").read_text() == "written\n"


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_concurrent_sessions_share_one_sandbox(tmp_path: Path) -> None:
    """An agent issues parallel tool calls; if each started its own sandbox,
    what one backgrounds would be invisible to the next."""
    async with shell(
        tmp_path / "root", guest_path="/app", network=True, require_isolation=True
    ) as (_, client):
        seen = await asyncio.gather(
            *(client.run("readlink /proc/self/ns/pid; pwd", check=True) for _ in range(4))
        )
        started = await client.run("sleep 300 >/dev/null 2>&1 & echo $!", check=True)
        visible = await client.run(f"kill -0 {str(started.stdout).strip()}", check=False)

    assert len({result.stdout for result in seen}) == 1
    assert str(seen[0].stdout).splitlines()[1] == "/app"
    assert visible.exit_status == 0


@pytest.mark.e2e
@pytest.mark.sandbox
@pytest.mark.parametrize(
    ("allowed_hosts", "network", "own_network"),
    [
        pytest.param({"example.com"}, True, True, id="a-policy-needs-a-network-to-apply-to"),
        pytest.param(set(), True, True, id="declared-unreachable"),
        pytest.param(None, False, True, id="no-network"),
        pytest.param(None, True, False, id="the-substrates-network"),
    ],
)
async def test_sessions_get_a_network_of_their_own_exactly_when_the_task_bounds_it(
    tmp_path: Path, allowed_hosts: set[str] | None, network: bool, own_network: bool
) -> None:
    async with shell(
        tmp_path / "root", allowed_hosts=allowed_hosts, network=network, require_isolation=True
    ) as (_, client):
        session = str((await client.run("readlink /proc/self/ns/net", check=True)).stdout).strip()

    assert (session != namespace("net")) is own_network


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_a_verifier_run_joins_the_sandbox_and_an_isolated_one_gets_its_own_network(
    tmp_path: Path,
) -> None:
    namespaces = "readlink /proc/self/ns/pid; readlink /proc/self/ns/net"
    probe = ["sh", "-c", namespaces]
    async with shell(tmp_path / "root", require_isolation=True) as (workspace, client):
        session = str((await client.run(namespaces, check=True)).stdout)
        joined = await workspace.run(probe, identity=None, max_wait=10)
        isolated = await workspace.run(probe, isolated=True, identity=None, max_wait=10)

    session_pid, session_net = session.split()
    assert joined.stdout.decode().split() == [session_pid, session_net]
    assert isolated.returncode == 0
    assert isolated.stdout.decode().split()[1] not in {session_net, namespace("net")}


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_a_verifier_gets_a_disposable_writable_hosts_file(tmp_path: Path) -> None:
    async with shell(tmp_path / "root", require_isolation=True) as (workspace, client):
        before = (await client.run("cat /etc/hosts", check=True)).stdout
        written = await workspace.run(
            ["sh", "-c", "echo '127.0.0.1 verifier-added' >> /etc/hosts && cat /etc/hosts"],
            identity=None,
            writable_hosts=True,
        )
        after = (await client.run("cat /etc/hosts", check=True)).stdout

    assert written.returncode == 0
    assert written.stdout.decode() == f"{before}127.0.0.1 verifier-added\n"
    assert after == before


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_ending_the_sessions_keeps_environment_processes_and_the_sandbox() -> None:
    # Somewhere another identity can traverse: the launch runs bwrap as uid 1000.
    parent = Path(tempfile.mkdtemp(dir="/tmp"))
    os.chmod(parent, 0o755)
    try:
        async with shell(parent / "root", network=True, require_isolation=True) as (
            workspace,
            client,
        ):
            infrastructure = await workspace.launch(
                ["sh", "-c", "id -u; id -g; exec sleep 300"],
                identity=(1000, 2000),
                no_new_privs=False,
                persistent=True,
                scope="environment",
            )
            ids = [await infrastructure.stdout.readline() for _ in range(2)]
            session = await client.conn.create_process("sleep 300")
            sandbox = (await client.run("readlink /proc/self/ns/pid", check=True)).stdout

            await workspace.terminate_sessions()

            await asyncio.wait_for(session.wait_closed(), 5)
            after = (await client.run("readlink /proc/self/ns/pid", check=True)).stdout
            still_running = infrastructure.returncode is None
            await infrastructure.terminate()
    finally:
        shutil.rmtree(parent, ignore_errors=True)

    assert ids == [b"1000\n", b"2000\n"]
    assert still_running
    assert after == sandbox


@pytest.mark.e2e
@pytest.mark.sandbox
async def test_the_sandbox_keeps_every_id_the_container_has(tmp_path: Path) -> None:
    uid_map = await asyncio.to_thread(Path("/proc/self/uid_map").read_text)
    ranges = [line.split() for line in uid_map.splitlines()]
    async with shell(tmp_path / "root", require_isolation=True) as (_, client):
        mapped = str((await client.run("cat /proc/self/uid_map", check=True)).stdout)

    assert [line.split() for line in mapped.splitlines()] == [
        [inside, inside, count] for inside, _, count in ranges
    ]


@pytest.mark.e2e
@pytest.mark.sandbox
@pytest.mark.parametrize(
    ("options", "present", "absent"),
    [
        pytest.param(
            {
                "peers": [Peer("db", 5432)],
                "local_aliases": ["main"],
                "ports": [5432],
                "allowed_hosts": {"pypi.org"},
                "shell_uid": 1000,
            },
            {"main": "127.0.0.1", "db": "127.0.0.2", socket.gethostname(): "127.0.0.1"},
            [],
            id="peers-and-aliases-resolve-for-any-identity",
        ),
        pytest.param(
            {"allowed_hosts": set()}, {"localhost": "127.0.0.1"}, [], id="localhost-without-peers"
        ),
        pytest.param(
            {"peers": [Peer("db", 5432)], "network": True},
            {},
            ["db"],
            id="the-substrates-hosts-where-the-network-is-shared",
        ),
    ],
)
async def test_a_declared_peer_is_a_name_sessions_resolve(
    tmp_path: Path,
    options: dict[str, Any],
    present: dict[str, str],
    absent: list[str],
) -> None:
    async with shell(tmp_path / "root", require_isolation=True, **options) as (_, client):
        hosts = str((await client.run("cat /etc/hosts", check=True)).stdout)

    resolves: dict[str, set[str]] = {}
    for line in hosts.splitlines():
        fields = line.split("#", 1)[0].split()
        for name in fields[1:]:
            resolves.setdefault(name, set()).add(fields[0])
    missing = {
        name: address
        for name, address in present.items()
        if address not in resolves.get(name, set())
    }
    assert missing == {}
    assert [name for name in absent if name in resolves] == []


FAKE_BWRAP = """#!/bin/sh
for arg in "$@"; do
  case "$arg" in {refused}) exit 1 ;; esac
done
{succeed}
"""


PUBLISHED_ISOLATION = """
import asyncio, sys
from hud.environment import Workspace

async def main():
    workspace = Workspace(sys.argv[1])
    print(workspace.capability().params["isolation"])
    await workspace.stop()

asyncio.run(main())
"""


@pytest.mark.parametrize(
    ("refused", "isolation"),
    [
        pytest.param("*", "none", id="unusable-install"),
        pytest.param("--never", "bwrap", id="usable-install"),
    ],
)
def test_a_workspace_publishes_bwrap_isolation_only_when_bubblewrap_works(
    tmp_path: Path, refused: str, isolation: str
) -> None:
    """An installed bwrap that cannot create namespaces must not be used."""
    tools = tmp_path / "bin"
    tools.mkdir()
    for name, script in {
        "bwrap": FAKE_BWRAP.format(refused=refused, succeed="exit 0"),
        "true": "#!/bin/sh\nexit 0\n",
    }.items():
        (tools / name).write_text(script)
        (tools / name).chmod(0o755)

    probe = subprocess.run(
        [sys.executable, "-c", PUBLISHED_ISOLATION, str(tmp_path / "root")],
        env={**os.environ, "PATH": str(tools)},
        capture_output=True,
        text=True,
        check=True,
    )

    assert probe.stdout.strip() == isolation

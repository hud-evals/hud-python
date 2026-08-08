from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from hud.environment.egress import BRIDGE_PORT
from hud.environment.seatbelt import (
    MACOS_PATH_TO_SEATBELT_EXECUTABLE,
    Seatbelt,
    SeatbeltPolicyInputs,
    generate_seatbelt_profile,
    policy_params,
    seatbelt_argv,
)
from hud.environment.workspace import Mount, Workspace


def test_sandbox_exec_path_is_hardcoded() -> None:
    assert MACOS_PATH_TO_SEATBELT_EXECUTABLE == "/usr/bin/sandbox-exec"


def test_profile_denies_default_and_allows_process() -> None:
    profile = generate_seatbelt_profile(
        SeatbeltPolicyInputs(writable_roots=(Path("/tmp/ws"),), readable_roots=())
    )
    assert "(version 1)" in profile
    assert "(deny default)" in profile
    assert "(allow process-exec)" in profile
    assert "(allow process-fork)" in profile
    assert "(allow file-read*)" not in profile  # noread-style: no global file-read*


def test_writable_roots_become_params() -> None:
    root = Path("/Users/me/proj").resolve()
    inputs = SeatbeltPolicyInputs(writable_roots=(root,), readable_roots=())
    params = policy_params(inputs)
    assert params["WRITABLE_ROOT_0"] == str(root)
    profile = generate_seatbelt_profile(inputs)
    assert '(subpath (param "WRITABLE_ROOT_0"))' in profile
    assert "file-write*" in profile


def test_readable_roots_parametrized() -> None:
    src = Path("/opt/data").resolve()
    inputs = SeatbeltPolicyInputs(
        writable_roots=(Path("/tmp/ws").resolve(),),
        readable_roots=(src,),
    )
    params = policy_params(inputs)
    assert params["READABLE_ROOT_0"] == str(src)
    profile = generate_seatbelt_profile(inputs)
    assert '(subpath (param "READABLE_ROOT_0"))' in profile


def test_isolated_network_allows_only_proxy_ports() -> None:
    profile = generate_seatbelt_profile(
        SeatbeltPolicyInputs(
            writable_roots=(Path("/tmp/ws"),),
            readable_roots=(),
            proxy_loopback_ports=(3128, 6379),
            allow_all_network=False,
        )
    )
    assert "3128" in profile
    assert "6379" in profile
    assert '(allow network-outbound (remote ip "127.0.0.1:3128"))' in profile
    assert '(allow network-outbound (remote ip "[::1]:3128"))' in profile
    assert '(allow network-outbound (remote ip "localhost:3128"))' in profile
    # Must not open every loopback port (egress boundary).
    assert '(allow network-outbound (remote ip "localhost:*"))' not in profile
    # Must not be unrestricted
    assert "(allow network*)" not in profile


def test_peer_loopback_endpoints_allow_127_0_0_n() -> None:
    profile = generate_seatbelt_profile(
        SeatbeltPolicyInputs(
            writable_roots=(Path("/tmp/ws"),),
            readable_roots=(),
            proxy_loopback_ports=(5432,),
            proxy_loopback_endpoints=(("127.0.0.2", 5432),),
            allow_all_network=False,
        )
    )
    assert '(allow network-outbound (remote ip "127.0.0.2:5432"))' in profile


def test_shared_network_allows_network_star() -> None:
    profile = generate_seatbelt_profile(
        SeatbeltPolicyInputs(
            writable_roots=(Path("/tmp/ws"),),
            readable_roots=(),
            allow_all_network=True,
        )
    )
    assert "(allow network*)" in profile


def test_seatbelt_argv_shape() -> None:
    argv = seatbelt_argv(
        ["/bin/echo", "hi"],
        profile="(version 1)(deny default)",
        params={"WRITABLE_ROOT_0": "/tmp/ws"},
    )
    assert argv[0] == "/usr/bin/sandbox-exec"
    assert argv[1:3] == ["-p", "(version 1)(deny default)"]
    assert "-DWRITABLE_ROOT_0=/tmp/ws" in argv
    assert argv[argv.index("--") + 1 :] == ["/bin/echo", "hi"]


def test_require_isolation_mentions_seatbelt_on_darwin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hud.environment.isolator.usable_bwrap", lambda: None)
    monkeypatch.setattr("hud.environment.isolator.usable_seatbelt", lambda: None)
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(RuntimeError, match=r"[Ss]eatbelt|[Ss]andbox"):
        Workspace(tmp_path / "root", require_isolation=True)


def test_darwin_selects_seatbelt_when_no_bwrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hud.environment.isolator.usable_bwrap", lambda: None)
    monkeypatch.setattr(
        "hud.environment.isolator.usable_seatbelt",
        lambda: Seatbelt("/usr/bin/sandbox-exec"),
    )
    monkeypatch.setattr(sys, "platform", "darwin")
    ws = Workspace(tmp_path / "root")
    assert ws._bwrap is None
    assert ws._seatbelt is not None
    assert ws.isolation_available


def test_seatbelt_policy_includes_extra_rw_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hud.environment.isolator.usable_bwrap", lambda: None)
    monkeypatch.setattr(
        "hud.environment.isolator.usable_seatbelt",
        lambda: Seatbelt("/usr/bin/sandbox-exec"),
    )
    extra = tmp_path / "extra"
    extra.mkdir()
    ws = Workspace(tmp_path / "root", mounts=(Mount("rw", src=str(extra), dst="/extra"),))
    inputs = ws._seatbelt_policy_inputs(proxy_ports=(3128,))
    assert any(p.resolve() == extra.resolve() for p in inputs.writable_roots)
    assert inputs.proxy_loopback_ports == (3128,)


# ─────────────── Task 3: sandbox spawn, session wrap, cleanup ────────────────


def _make_fake_proc(pid: int = 4242, ready_output: bytes = b"ready\n") -> object:
    """A minimal asyncio.subprocess.Process stand-in."""

    class FakeProc:
        def __init__(self) -> None:
            self.returncode = None
            self.pid = pid
            self.stdout = asyncio.StreamReader()
            self.stdout.feed_data(ready_output)
            self.stdout.feed_eof()
            self.stderr = asyncio.StreamReader()
            self.stderr.feed_eof()
            self.stdin = None

        async def wait(self) -> int:
            return 0

        def kill(self) -> None:
            self.returncode = -9

    return FakeProc()


@pytest.mark.asyncio
async def test_start_seatbelt_sandbox_wraps_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seatbelt holder is launched under sandbox-exec."""
    monkeypatch.setattr("hud.environment.isolator.usable_bwrap", lambda: None)
    monkeypatch.setattr(
        "hud.environment.isolator.usable_seatbelt",
        lambda: Seatbelt("/usr/bin/sandbox-exec"),
    )
    captured: list[list[str]] = []

    async def fake_exec(*argv: str, **kwargs: object) -> object:
        captured.append(list(argv))
        return _make_fake_proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    # network=True => owns_netns=False, so Egress / bridge skipped
    ws = Workspace(tmp_path / "root", network=True)
    pid = await ws._start_seatbelt_sandbox()
    assert pid == 4242
    assert captured, "no subprocess was launched"
    assert captured[0][0] == "/usr/bin/sandbox-exec"


@pytest.mark.asyncio
async def test_sandbox_pid_returns_holder_pid_for_seatbelt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """sandbox_pid() starts and returns the holder pid on seatbelt substrates."""
    monkeypatch.setattr("hud.environment.isolator.usable_bwrap", lambda: None)
    monkeypatch.setattr(
        "hud.environment.isolator.usable_seatbelt",
        lambda: Seatbelt("/usr/bin/sandbox-exec"),
    )

    async def fake_exec(*argv: str, **kwargs: object) -> object:
        return _make_fake_proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    ws = Workspace(tmp_path / "root", network=True)
    pid = await ws.sandbox_pid()
    assert pid == 4242


@pytest.mark.asyncio
async def test_discard_seatbelt_sandbox_kills_holder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """discard_sandbox() kills the holder and clears sandbox state."""
    monkeypatch.setattr("hud.environment.isolator.usable_bwrap", lambda: None)
    monkeypatch.setattr(
        "hud.environment.isolator.usable_seatbelt",
        lambda: Seatbelt("/usr/bin/sandbox-exec"),
    )

    async def fake_exec(*argv: str, **kwargs: object) -> object:
        return _make_fake_proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    ws = Workspace(tmp_path / "root", network=True)
    await ws._start_seatbelt_sandbox()
    assert ws._sandbox is not None
    assert ws._sandbox_init == 4242
    await ws.discard_sandbox()
    assert ws._sandbox is None
    assert ws._sandbox_init is None


@pytest.mark.asyncio
async def test_shell_argv_wraps_with_seatbelt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """shell_argv() wraps the session payload with sandbox-exec when seatbelt is active."""
    monkeypatch.setattr("hud.environment.isolator.usable_bwrap", lambda: None)
    monkeypatch.setattr(
        "hud.environment.isolator.usable_seatbelt",
        lambda: Seatbelt("/usr/bin/sandbox-exec"),
    )
    ws = Workspace(tmp_path / "root", network=True)
    argv = ws.shell_argv()
    assert argv[0] == "/usr/bin/sandbox-exec"
    assert "-p" in argv


def test_shell_argv_prefixes_setpriv_outside_seatbelt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Privilege drop must wrap sandbox-exec, not be skipped for Seatbelt."""
    monkeypatch.setattr("hud.environment.isolator.usable_bwrap", lambda: None)
    monkeypatch.setattr(
        "hud.environment.isolator.usable_seatbelt",
        lambda: Seatbelt("/usr/bin/sandbox-exec"),
    )
    ws = Workspace(tmp_path / "root", network=True, shell_uid=1000)
    monkeypatch.setattr(ws, "_drops_privileges", lambda: True)
    monkeypatch.setattr(ws, "_setpriv", lambda: "/usr/bin/setpriv")
    argv = ws.shell_argv("true")
    assert argv[:2] == ["/usr/bin/setpriv", "--reuid"]
    assert "/usr/bin/sandbox-exec" in argv
    assert argv.index("/usr/bin/setpriv") < argv.index("/usr/bin/sandbox-exec")


def test_capability_reports_seatbelt_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """capability() reports isolation='seatbelt' when seatbelt is the sandbox."""
    monkeypatch.setattr("hud.environment.isolator.usable_bwrap", lambda: None)
    monkeypatch.setattr(
        "hud.environment.isolator.usable_seatbelt",
        lambda: Seatbelt("/usr/bin/sandbox-exec"),
    )
    ws = Workspace(tmp_path / "root")
    cap = ws.capability()
    assert cap.params.get("isolation") == "seatbelt"


@pytest.mark.asyncio
async def test_start_seatbelt_sandbox_starts_host_bridge_when_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When owns_netns and allowed_hosts, a host-side bridge subprocess is started."""
    monkeypatch.setattr("hud.environment.isolator.usable_bwrap", lambda: None)
    monkeypatch.setattr(
        "hud.environment.isolator.usable_seatbelt",
        lambda: Seatbelt("/usr/bin/sandbox-exec"),
    )

    class _FakeWriter:
        def write(self, data: bytes) -> None:
            pass

        async def drain(self) -> None:
            pass

    def _ready_proc(pid: int) -> object:
        class P:
            returncode = None

            def __init__(self) -> None:
                self.pid = pid
                self.stdout = asyncio.StreamReader()
                self.stdout.feed_data(b"ready\n")
                self.stdout.feed_eof()
                self.stderr = asyncio.StreamReader()
                self.stderr.feed_eof()
                self.stdin = _FakeWriter()

            def kill(self) -> None:
                self.returncode = -9

            async def wait(self) -> int:
                return 0

        return P()

    call_count = 0
    launched_argv: list[list[str]] = []

    async def fake_exec(*argv: str, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        launched_argv.append(list(argv))
        return _ready_proc(9999 if call_count == 1 else 4242)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    import sys

    class _FakeEgress:
        def __init__(self, *a: object, **kw: object) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def environment(self, port: int = BRIDGE_PORT) -> dict[str, str]:
            return {}

        def bridge_command(
            self, port: int = BRIDGE_PORT, *, visitor_socket: object = None
        ) -> tuple[list[str], bytes]:
            return [sys.executable, "-c", "pass"], b"[]\n"

    monkeypatch.setattr("hud.environment.workspace.Egress", _FakeEgress)

    ws = Workspace(tmp_path / "root", allowed_hosts={"example.com"})
    pid = await ws._start_seatbelt_sandbox()
    assert pid == 4242
    # First call: bridge; second call: holder (sandbox-exec)
    assert len(launched_argv) == 2, f"expected 2 subprocesses, got {len(launched_argv)}"
    assert launched_argv[1][0] == "/usr/bin/sandbox-exec", "holder not wrapped by sandbox-exec"
    await ws.discard_sandbox()
    assert ws._host_bridge is None, "discard_sandbox must clear _host_bridge"


@pytest.mark.asyncio
async def test_seatbelt_proxy_ports_includes_bridge_port_when_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """BRIDGE_PORT appears in proxy ports when allowed_hosts is non-empty."""
    monkeypatch.setattr("hud.environment.isolator.usable_bwrap", lambda: None)
    monkeypatch.setattr(
        "hud.environment.isolator.usable_seatbelt",
        lambda: Seatbelt("/usr/bin/sandbox-exec"),
    )
    ws = Workspace(tmp_path / "root", allowed_hosts={"example.com"})
    ports = ws._seatbelt_proxy_ports()
    assert BRIDGE_PORT in ports


@pytest.mark.asyncio
async def test_seatbelt_proxy_ports_empty_when_shared_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No proxy ports are needed when the workspace shares the host network."""
    monkeypatch.setattr("hud.environment.isolator.usable_bwrap", lambda: None)
    monkeypatch.setattr(
        "hud.environment.isolator.usable_seatbelt",
        lambda: Seatbelt("/usr/bin/sandbox-exec"),
    )
    ws = Workspace(tmp_path / "root", network=True)
    ports = ws._seatbelt_proxy_ports()
    assert ports == ()

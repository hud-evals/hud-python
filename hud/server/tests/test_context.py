from __future__ import annotations

import os
import sys

try:
    import multiprocessing.connection as _mp_conn

    # Pull the exception dynamically; fall back to OSError if missing in stubs/runtime
    MPAuthenticationError: type[BaseException] = getattr(_mp_conn, "AuthenticationError", OSError)
except Exception:  # pragma: no cover
    MPAuthenticationError = OSError


from typing import TYPE_CHECKING

import pytest

from hud.server.context import attach_context, serve_context

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Context server uses UNIX domain sockets",
)


class CounterCtx:
    def __init__(self) -> None:
        self._n = 0

    def inc(self) -> int:
        self._n += 1
        return self._n

    def get(self) -> int:
        return self._n


def test_serve_and_attach_shared_state(tmp_path: Path) -> None:
    sock = str(tmp_path / "hud_ctx.sock")

    mgr = serve_context(CounterCtx(), sock_path=sock)
    try:
        c1 = attach_context(sock_path=sock)
        assert c1.get() == 0
        assert c1.inc() == 1

        # Second attachment sees the same underlying object
        c2 = attach_context(sock_path=sock)
        assert c2.get() == 1
        assert c2.inc() == 2
        assert c1.get() == 2  # shared state
    finally:
        mgr.shutdown()


def test_env_var_socket_path_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sock = str(tmp_path / "env_ctx.sock")
    monkeypatch.setenv("HUD_CTX_SOCK", sock)

    mgr = serve_context(CounterCtx(), sock_path=None)
    try:
        c = attach_context(sock_path=None)
        assert c.inc() == 1
        assert c.get() == 1
    finally:
        mgr.shutdown()
        monkeypatch.delenv("HUD_CTX_SOCK", raising=False)


def test_wrong_authkey_rejected(tmp_path: Path) -> None:
    sock = str(tmp_path / "auth_ctx.sock")
    mgr = serve_context(CounterCtx(), sock_path=sock, authkey=b"correct")
    try:
        with pytest.raises(
            (MPAuthenticationError, ConnectionRefusedError, BrokenPipeError, OSError)
        ):
            attach_context(sock_path=sock, authkey=b"wrong")
    finally:
        mgr.shutdown()


def test_attach_nonexistent_raises(tmp_path: Path) -> None:
    # ensure file truly doesn't exist
    sock = str(tmp_path / "missing.sock")
    if os.path.exists(sock):
        os.unlink(sock)

    with pytest.raises((FileNotFoundError, ConnectionRefusedError, OSError)):
        attach_context(sock_path=sock)


@pytest.mark.asyncio
async def test_run_context_server_handles_keyboardinterrupt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run_context_server should call manager.shutdown() when KeyboardInterrupt occurs."""
    # Capture serve_context() and the returned manager
    called = {"served": False, "shutdown": False, "addr": None}

    class _Mgr:
        def shutdown(self) -> None:
            called["shutdown"] = True

    def fake_serve(ctx, sock_path, authkey):
        called["served"] = True
        called["addr"] = sock_path
        return _Mgr()

    monkeypatch.setattr("hud.server.context.serve_context", fake_serve)

    # Make asyncio.Event().wait() raise KeyboardInterrupt immediately
    class _FakeEvent:
        async def wait(self) -> None:
            raise KeyboardInterrupt

    monkeypatch.setattr("hud.server.context.asyncio.Event", lambda: _FakeEvent())

    from hud.server.context import run_context_server

    await run_context_server(object(), sock_path=str(tmp_path / "ctx.sock"))

    assert called["served"] is True
    assert called["shutdown"] is True
    assert str(called["addr"]).endswith("ctx.sock")

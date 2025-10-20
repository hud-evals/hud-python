from __future__ import annotations

import sys
from unittest import mock

import pytest

from hud.cli.utils.local_runner import run_local_server, run_with_reload

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="Prefers Linux (signal/stdio nuances)"
)


@mock.patch("subprocess.run")
def test_run_local_server_no_reload_http(mock_run, monkeypatch):
    mock_run.return_value = mock.Mock(returncode=0)
    # Ensure sys.exit is raised with code 0
    with pytest.raises(SystemExit) as exc:
        run_local_server("server:app", transport="http", port=8765, verbose=True, reload=False)
    assert exc.value.code == 0
    # Verify the command contained port and no-banner
    args = mock_run.call_args[0][0]
    assert "--port" in args and "--no-banner" in args


@mock.patch("hud.cli.utils.local_runner.run_with_reload")
def test_run_local_server_reload_calls_reload(mock_reload):
    run_local_server("server:app", transport="stdio", port=None, verbose=False, reload=True)
    mock_reload.assert_called_once()


@pytest.mark.asyncio
async def test_run_with_reload_import_error(monkeypatch):
    # Force ImportError for watchfiles
    import builtins as _builtins

    real_import = _builtins.__import__

    def _imp(name, *args, **kwargs):
        if name == "watchfiles":
            raise ImportError("nope")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_builtins, "__import__", _imp)

    with pytest.raises(SystemExit) as exc:
        # run_with_reload is async in this module; await it
        await run_with_reload("server:app", transport="stdio", verbose=False)
    assert exc.value.code == 1

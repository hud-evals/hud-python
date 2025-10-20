from __future__ import annotations

import pytest

from hud.cli.utils.remote_runner import run_remote_server


def test_run_remote_server_requires_api_key(monkeypatch):
    # Ensure settings.api_key is None and no api_key provided
    from hud.cli.utils import remote_runner as mod

    monkeypatch.setattr(mod.settings, "api_key", None, raising=True)

    with pytest.raises(SystemExit) as exc:
        run_remote_server(
            image="img:latest",
            docker_args=[],
            transport="stdio",
            port=8765,
            url="https://api.example.com/mcp",
            api_key=None,
            run_id=None,
            verbose=False,
        )
    assert exc.value.code == 1

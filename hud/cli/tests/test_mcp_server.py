"""Tests for hud.cli.dev module."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from hud.cli.dev import (
    run_mcp_dev_server,
)


class TestRunMCPDevServer:
    """Test the main server runner."""

    def test_run_dev_server_image_not_found(self) -> None:
        """When using Docker mode without a lock file, exits with typer.Exit(1)."""
        import typer

        with (
            patch("hud.cli.dev.should_use_docker_mode", return_value=True),
            patch("hud.cli.dev.Path.cwd"),
            patch("hud.cli.dev.hud_console"),
            pytest.raises(typer.Exit),
        ):
            run_mcp_dev_server(
                module=None,
                stdio=False,
                port=8765,
                verbose=False,
                inspector=False,
                interactive=False,
                watch=[],
                docker=True,
                docker_args=[],
            )

    def test_run_dev_server_without_watch_uses_single_run(self, monkeypatch) -> None:
        """Without --watch, run once via _run_with_sigterm (no reloader)."""
        monkeypatch.delenv("_HUD_DEV_CHILD", raising=False)

        with (
            patch("hud.cli.dev.run_with_reload") as mock_reload,
            patch("hud.server.server._run_with_sigterm") as mock_sigterm,
        ):
            run_mcp_dev_server(
                module="server",
                stdio=True,
                port=8765,
                verbose=False,
                inspector=False,
                interactive=False,
                watch=None,
                docker=False,
                docker_args=[],
            )

        mock_sigterm.assert_called_once()
        mock_reload.assert_not_called()

    def test_run_dev_server_with_watch_uses_reloader(self, monkeypatch) -> None:
        """With --watch, use file-watcher reloader path."""
        monkeypatch.delenv("_HUD_DEV_CHILD", raising=False)

        with (
            patch("hud.cli.dev.run_with_reload") as mock_reload,
            patch("hud.server.server._run_with_sigterm") as mock_sigterm,
        ):
            run_mcp_dev_server(
                module="server",
                stdio=True,
                port=8765,
                verbose=False,
                inspector=False,
                interactive=False,
                watch=["tools"],
                docker=False,
                docker_args=[],
            )

        mock_reload.assert_called_once()
        mock_sigterm.assert_not_called()

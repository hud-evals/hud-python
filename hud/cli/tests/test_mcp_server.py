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

    def test_stdio_no_watch_runs_without_reload(self) -> None:
        """Stdio mode without --watch should run directly (no reload loop)."""
        with (
            patch.dict("os.environ", {}, clear=False),
            patch("hud.cli.dev.should_use_docker_mode", return_value=False),
            patch("hud.cli.dev.run_with_reload") as mock_reload,
            patch("hud.server.server._run_with_sigterm") as mock_run_sigterm,
        ):
            run_mcp_dev_server(
                module="pkg.main:env",
                stdio=True,
                port=8765,
                verbose=False,
                inspector=False,
                interactive=False,
                watch=None,
                docker=False,
                docker_args=[],
            )

        mock_reload.assert_not_called()
        mock_run_sigterm.assert_called_once()

    def test_http_no_watch_runs_without_reload(self) -> None:
        """HTTP mode without --watch should run directly (no reload loop)."""
        with (
            patch.dict("os.environ", {}, clear=False),
            patch("hud.cli.dev.should_use_docker_mode", return_value=False),
            patch("hud.cli.dev.run_with_reload") as mock_reload,
            patch("hud.server.server._run_with_sigterm") as mock_run_sigterm,
            patch("hud.cli.utils.logging.find_free_port", return_value=8765),
        ):
            run_mcp_dev_server(
                module="pkg.main:env",
                stdio=False,
                port=8765,
                verbose=False,
                inspector=False,
                interactive=False,
                watch=None,
                docker=False,
                docker_args=[],
            )

        mock_reload.assert_not_called()
        mock_run_sigterm.assert_called_once()

    def test_http_with_watch_uses_reload(self) -> None:
        """HTTP mode with --watch should use reload loop."""
        with (
            patch.dict("os.environ", {}, clear=False),
            patch("hud.cli.dev.should_use_docker_mode", return_value=False),
            patch("hud.cli.dev.run_with_reload") as mock_reload,
            patch("hud.server.server._run_with_sigterm") as mock_run_sigterm,
            patch("hud.cli.utils.logging.find_free_port", return_value=8765),
        ):
            run_mcp_dev_server(
                module="pkg.main:env",
                stdio=False,
                port=8765,
                verbose=False,
                inspector=False,
                interactive=False,
                watch=["."],
                docker=False,
                docker_args=[],
            )

        mock_reload.assert_called_once()
        mock_run_sigterm.assert_not_called()

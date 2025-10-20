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

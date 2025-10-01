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
        """Test handling when Docker image doesn't exist."""
        import click

        with (
            patch("hud.cli.dev.image_exists", return_value=False),
            patch("click.confirm", return_value=False),
            pytest.raises(click.Abort),
        ):
            run_mcp_dev_server(
                module=".",
                stdio=False,
                port=8765,
                verbose=False,
                inspector=False,
                interactive=False,
                watch=[],
                docker=False,
                docker_args=[],
            )

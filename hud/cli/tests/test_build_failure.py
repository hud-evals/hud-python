from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import typer

from hud.cli.build import build_environment

if TYPE_CHECKING:
    from pathlib import Path


@patch("hud.cli.build.compute_source_hash", return_value="deadbeef")
@patch(
    "hud.cli.build.analyze_mcp_environment",
    return_value={"initializeMs": 10, "toolCount": 0, "tools": []},
)
@patch("hud.cli.build.build_docker_image", return_value=True)
def test_build_label_rebuild_failure(_bd, _an, _hash, tmp_path: Path, monkeypatch):
    # Minimal environment dir
    env = tmp_path / "env"
    env.mkdir()
    (env / "Dockerfile").write_text("FROM python:3.11")

    # Ensure subprocess.run returns non-zero for the second build (label build)
    import types

    def run_side_effect(cmd, *a, **k):
        # Return 0 for first docker build, 1 for label build
        if isinstance(cmd, list) and cmd[:2] == ["docker", "build"] and "--label" in cmd:
            return types.SimpleNamespace(returncode=1, stderr="boom")
        return types.SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setenv("FASTMCP_DISABLE_BANNER", "1")
    with (
        patch("hud.cli.build.subprocess.run", side_effect=run_side_effect),
        pytest.raises(typer.Exit),
    ):
        build_environment(str(env), verbose=False)

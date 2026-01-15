from __future__ import annotations

from pathlib import Path


def test_hud_debug_directory_mode_accepts_dockerfile_hud_without_pyproject(
    tmp_path: Path, monkeypatch
) -> None:
    """
    Regression test for:
    - `hud debug .` treating '.' as an image name when the directory contains
      `Dockerfile.hud` but no `pyproject.toml`, resulting in:
        docker: invalid reference format
    """
    # Simulate a minimal environment directory (no pyproject.toml yet)
    (tmp_path / "Dockerfile.hud").write_text("FROM python:3.11\n", encoding="utf-8")

    # Run from inside the env dir, matching `hud debug .`
    monkeypatch.chdir(tmp_path)

    import hud.cli.__init__ as cli
    from hud.cli.utils import environment as env_utils

    # Avoid interactive prompts/builds during the test
    monkeypatch.setattr(env_utils, "image_exists", lambda _image: True)

    captured: dict[str, object] = {}

    async def _fake_debug_mcp_stdio(command, logger, max_phase: int = 5) -> int:  # type: ignore[no-untyped-def]
        captured["command"] = command
        return max_phase

    monkeypatch.setattr(cli, "debug_mcp_stdio", _fake_debug_mcp_stdio)

    # If directory detection fails, command would be: ["docker", "run", ..., "."]
    cli.debug(params=["."], config=None, cursor=None, build=False, max_phase=1)

    command = captured["command"]
    assert isinstance(command, list)
    assert command[-1] == f"{tmp_path.name}:dev"

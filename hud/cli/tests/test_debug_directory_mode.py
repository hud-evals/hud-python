from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def test_hud_debug_directory_mode_accepts_dockerfile_hud(
    tmp_path: Path, monkeypatch
) -> None:
    """Test that hud debug . works with Dockerfile.hud and pyproject.toml."""
    (tmp_path / "Dockerfile.hud").write_text("FROM python:3.11\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    import hud.cli.__init__ as cli
    from hud.cli.utils import environment as env_utils

    monkeypatch.setattr(env_utils, "image_exists", lambda _image: True)

    captured: dict[str, object] = {}

    async def _fake_debug_mcp_stdio(command, logger, max_phase: int = 5) -> int:  # type: ignore[no-untyped-def]
        captured["command"] = command
        return max_phase

    monkeypatch.setattr(cli, "debug_mcp_stdio", _fake_debug_mcp_stdio)
    cli.debug(params=["."], config=None, cursor=None, build=False, max_phase=1)

    command = captured["command"]
    assert isinstance(command, list)
    expected_name = tmp_path.name.replace("_", "-")
    assert command[-1] == f"{expected_name}:dev"

"""Configured action timing flows through task setup and the MCP tool."""

import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import Mock, call

import pytest
import yaml
from fastmcp import Client, FastMCP
from hud.environment.server import TaskRunner
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def game_env(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("libbet_action_frames_test", ROOT / "env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    emulator = Mock(spec=module.GameBoyEmulator)
    emulator.get_screen.return_value = Image.new("RGB", (160, 144))
    monkeypatch.setattr(module, "GameBoyEmulator", Mock(return_value=emulator))
    monkeypatch.setattr(module.games_loader, "GAMES_DIR", str(tmp_path))
    monkeypatch.delenv("VGBENCH_RECORD", raising=False)
    monkeypatch.delenv("VGBENCH_RECORD_DIR", raising=False)
    return module, emulator


def write_game(module, name, action_frames):
    directory = Path(module.games_loader.GAMES_DIR) / name
    directory.mkdir()
    config = {"emulator": "gba", "rom": "libbet.gb"}
    if action_frames is not None:
        config["action_frames"] = action_frames
    (directory / "config.yaml").write_text(yaml.safe_dump(config))


@pytest.mark.parametrize("configured,expected", [(None, 15), (9, 9), (0, 1), (1200, 600)])
def test_omitted_frames_use_the_game_default_in_prompt_and_tool(game_env, configured, expected):
    module, emulator = game_env
    write_game(module, "configured", configured)

    async def play():
        server = FastMCP("libbet-test")
        server.tool(module.press_buttons)
        runner = TaskRunner(module.play_game, {"game": "configured"})
        frame = await runner.start()
        try:
            assert f"press_buttons(buttons, frames={expected})" in frame["prompt"]
            async with Client(server) as client:
                await client.call_tool("press_buttons", {"buttons": ["A"]})
            emulator.step.assert_called_once_with({"A": True}, frames=expected)
        finally:
            await runner.grade({"answer": "done"})
        emulator.close.assert_called_once_with()

    asyncio.run(play())


@pytest.mark.parametrize("requested,expected", [(15, 15), (5, 5), (0, 1), (-2, 1), (1000, 600)])
def test_explicit_frames_override_the_game_default_and_keep_clamping(game_env, requested, expected):
    module, emulator = game_env
    write_game(module, "configured", 37)

    async def play():
        server = FastMCP("libbet-test")
        server.tool(module.press_buttons)
        runner = TaskRunner(module.play_game, {"game": "configured"})
        await runner.start()
        try:
            async with Client(server) as client:
                await client.call_tool("press_buttons", {"buttons": ["B"], "frames": requested})
            emulator.step.assert_called_once_with({"B": True}, frames=expected)
        finally:
            await runner.grade({"answer": "done"})

    asyncio.run(play())


def test_next_game_uses_its_own_action_default(game_env):
    module, emulator = game_env
    write_game(module, "first", 7)
    write_game(module, "second", 23)

    async def play():
        server = FastMCP("libbet-test")
        server.tool(module.press_buttons)
        async with Client(server) as client:
            for game in ("first", "second"):
                runner = TaskRunner(module.play_game, {"game": game})
                await runner.start()
                try:
                    await client.call_tool("press_buttons", {"buttons": ["RIGHT"]})
                finally:
                    await runner.grade({"answer": "done"})
        assert emulator.step.call_args_list == [
            call({"RIGHT": True}, frames=7),
            call({"RIGHT": True}, frames=23),
        ]
        assert emulator.close.call_count == 2

    asyncio.run(play())

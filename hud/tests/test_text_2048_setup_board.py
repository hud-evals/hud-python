"""Tests for the text-2048 setup board tool."""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

# Ensure the environment package is importable from the repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_SRC = REPO_ROOT / "environments" / "text_2048" / "src"
if str(ENV_SRC) not in sys.path:
    sys.path.append(str(ENV_SRC))

from hud_controller.setup import board as board_module  # noqa: E402


class DummyGame:
    """Minimal game stub for exercising setup_board."""

    def __init__(self) -> None:
        self.size = 0
        self.reset_calls: list[int] = []

    def reset(self, size: int = 4) -> None:
        self.size = size
        self.reset_calls.append(size)

    def get_board_ascii(self) -> str:
        return f"{self.size}x{self.size} board"

    def get_state(self) -> dict:
        return {
            "board": [[self.size]],
            "score": 0,
            "moves": 0,
            "game_over": False,
            "won": False,
            "highest_tile": self.size,
        }


@pytest.fixture(autouse=True)
def stub_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace telemetry trace context manager with a no-op."""

    @contextmanager
    def _noop_trace(*args, **kwargs):  # noqa: ANN001, ANN003
        yield

    monkeypatch.setattr(board_module, "trace", _noop_trace)


@pytest.mark.asyncio
async def test_setup_board_returns_ascii_and_json(monkeypatch: pytest.MonkeyPatch) -> None:
    game = DummyGame()
    monkeypatch.setattr(board_module.setup, "env", game, raising=False)

    result = await board_module.setup_board.fn(board_size=5)
    assert len(result) == 2

    ascii_block, json_block = result
    assert "5x5 game initialized" in ascii_block.text

    payload = json.loads(json_block.text)
    assert payload["board_size"] == 5
    assert payload["state"]["board"] == [[5]]


@pytest.mark.asyncio
async def test_setup_board_clamps_out_of_range_values(monkeypatch: pytest.MonkeyPatch) -> None:
    game = DummyGame()
    monkeypatch.setattr(board_module.setup, "env", game, raising=False)

    result = await board_module.setup_board.fn(board_size=99)
    ascii_block, json_block = result

    assert str(board_module.MAX_BOARD_SIZE) in ascii_block.text
    assert "using" in ascii_block.text.lower()

    payload = json.loads(json_block.text)
    assert payload["board_size"] == board_module.MAX_BOARD_SIZE
    assert payload["requested_board_size"] == 99

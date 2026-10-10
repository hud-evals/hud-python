"""``HUDConsole.select``, driven by keystrokes the way a user picks from a list."""

from __future__ import annotations

from typing import Any

import pytest
import typer
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput

from hud.utils import HUDConsole

CHOICES: list[str | dict[str, Any]] = [
    {"name": "A", "value": "a"},
    {"name": "B", "value": "b"},
    {"name": "C", "value": "c"},
]


def _select(keys: str, *, spaced: bool) -> str:
    with create_pipe_input() as keyboard:
        keyboard.send_text(keys)
        with create_app_session(input=keyboard, output=DummyOutput()):
            return HUDConsole().select("pick", CHOICES, spaced=spaced)


@pytest.mark.parametrize(
    ("keys", "spaced", "picked"),
    [
        pytest.param("\r", False, "a", id="enter-picks-the-first"),
        pytest.param("\x1b[B\r", False, "b", id="down-then-enter"),
        pytest.param("\x1b[B\r", True, "b", id="spaced-down-skips-the-separator"),
        pytest.param("\x1b[B\x1b[B\r", True, "c", id="spaced-down-twice"),
    ],
)
def test_arrow_keys_and_enter_pick_a_choice(keys: str, spaced: bool, picked: str) -> None:
    assert _select(keys, spaced=spaced) == picked


@pytest.mark.parametrize("spaced", [False, True])
def test_escape_cancels_the_selection(spaced: bool) -> None:
    with pytest.raises(typer.Exit):
        _select("\x1b", spaced=spaced)

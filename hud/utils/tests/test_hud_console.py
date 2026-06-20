"""``HUDConsole`` — smoke-exercise the output methods + check the pure formatters.

These mostly assert "doesn't raise" (output goes to a Rich console), which still
exercises the formatting branches; the ``format_*`` helpers return values
we can assert directly.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
import typer
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output import DummyOutput

from hud.utils.hud_console import HUDConsole


def _select_with_keystrokes(
    keys: str, choices: list[str | dict[str, Any]], *, spaced: bool = False
) -> str:
    """Drive ``HUDConsole.select`` with scripted keystrokes through a prompt_toolkit
    pipe input (no real TTY), returning the selected value."""
    with create_pipe_input() as inp:
        inp.send_text(keys)
        with create_app_session(input=inp, output=DummyOutput()):
            return HUDConsole().select("pick", choices, spaced=spaced)


def test_output_methods_do_not_raise() -> None:
    c = HUDConsole()
    c.header("Title")
    c.section_title("Section")
    c.success("ok")
    c.error("bad")
    c.warning("warn")
    c.info("info")
    c.print("plain")
    c.dim_info("key", "value")
    c.link("https://example.com")
    c.progress_message("working")
    c.hint("a hint")
    c.status_item("label", "value")
    c.command_example("hud eval tasks.json")
    c.key_value_table({"key": "value"})
    c.render_support_hint()


def test_debug_respects_logger_level() -> None:
    logger = logging.getLogger("test_hud_console_debug")
    c = HUDConsole(logger=logger)
    logger.setLevel(logging.DEBUG)
    c.debug("debug visible")
    logger.setLevel(logging.WARNING)
    c.debug("debug hidden")  # no-op when not verbose


def test_format_helpers_return_strings() -> None:
    c = HUDConsole()
    assert isinstance(c.format_tool_call("bash", {"command": "ls"}), str)
    assert isinstance(c.format_tool_result("output text"), str)
    assert isinstance(c.format_tool_result("error text", is_error=True), str)


def test_render_exception_does_not_raise() -> None:
    c = HUDConsole()
    try:
        raise ValueError("boom")
    except ValueError as exc:
        c.render_exception(exc)


def test_render_exception_request_error_details() -> None:
    from hud.utils.exceptions import HudRequestError

    c = HUDConsole()
    c.render_exception(HudRequestError("nope", status_code=403, response_text="forbidden"))


def _capture_select_choices(monkeypatch: pytest.MonkeyPatch, returns: str) -> dict[str, Any]:
    """Patch the ``questionary`` boundary so ``HUDConsole.select`` runs without a
    TTY, recording the choice list it builds."""
    import questionary

    captured: dict[str, Any] = {}

    class _App:
        def __init__(self) -> None:
            self.key_bindings = KeyBindings()

    class _Stub:
        def __init__(self) -> None:
            self.application = _App()

        def ask(self) -> str:
            return returns

    def _fake_select(message: str, **kwargs: Any) -> _Stub:
        captured["choices"] = kwargs["choices"]
        return _Stub()

    monkeypatch.setattr(questionary, "select", _fake_select)
    return captured


def test_select_spaced_interleaves_blank_separators(monkeypatch: pytest.MonkeyPatch) -> None:
    from questionary import Choice, Separator

    captured = _capture_select_choices(monkeypatch, returns="b")

    result = HUDConsole().select(
        "pick",
        [
            {"name": "A", "value": "a"},
            {"name": "B", "value": "b"},
            {"name": "C", "value": "c"},
        ],
        spaced=True,
    )

    assert result == "b"
    choices = captured["choices"]
    # A blank separator sits between consecutive choices, but never leads or trails.
    # (questionary's Separator subclasses Choice, so compare exact types here.)
    assert [type(x).__name__ for x in choices] == [
        "Choice",
        "Separator",
        "Choice",
        "Separator",
        "Choice",
    ]
    assert sum(type(x) is Choice for x in choices) == 3
    assert sum(isinstance(x, Separator) for x in choices) == 2


def test_select_without_spaced_has_no_separators(monkeypatch: pytest.MonkeyPatch) -> None:
    from questionary import Separator

    captured = _capture_select_choices(monkeypatch, returns="a")

    HUDConsole().select("pick", [{"name": "A", "value": "a"}, {"name": "B", "value": "b"}])

    assert not any(isinstance(x, Separator) for x in captured["choices"])


def test_select_escape_cancels() -> None:
    # questionary cancels on Ctrl+C only; we additionally bind Esc to cancel,
    # which surfaces as a typer.Exit through select's None handling.
    with pytest.raises(typer.Exit):
        _select_with_keystrokes("\x1b", [{"name": "A", "value": "a"}, {"name": "B", "value": "b"}])


def test_select_arrow_then_enter_selects_skipping_separators() -> None:
    # Down + Enter lands on the second real choice; the blank separator between
    # spaced choices is skipped, and the Esc binding doesn't break arrow keys.
    result = _select_with_keystrokes(
        "\x1b[B\r",
        [{"name": "A", "value": "a"}, {"name": "B", "value": "b"}, {"name": "C", "value": "c"}],
        spaced=True,
    )
    assert result == "b"

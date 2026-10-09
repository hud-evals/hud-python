# ruff: noqa: E501 -- snapshots quote product messages verbatim
"""Computer-use actions reach the environment's screen as RFB input events.

The environment publishes an RFB capability backed by a fake VNC screen that
records every key and pointer event it receives. Each row scripts one provider
to take an action and pins the events the screen received and the result the
model saw next.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from asyncvnc import key_codes
from inline_snapshot import snapshot

from hud.agents import ClaudeAgent, GeminiAgent, OpenAIAgent
from hud.agents.types import ClaudeConfig, GeminiConfig, OpenAIConfig
from hud.capabilities import Capability
from tests.agents.support import run_task, tool_results, wire, workspace_env
from tests.harness import (
    KeyEvent,
    Turn,
    call,
    computer_call,
    fake_screen,
    say,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from hud.agents.base import Agent
    from tests.harness import FakeScreen, HudEnv, Models

KEY_NAMES: dict[int, str] = {}
for _name, _code in key_codes.items():
    KEY_NAMES.setdefault(_code, _name)


def events(screen: FakeScreen) -> str:
    """The screen's input log in one line.

    ``+Key``/``-Key`` are presses and releases, a bare ``Key`` a press released
    straight away, and ``x,y bN`` a pointer event with button mask ``N``.
    """
    rendered: list[str] = []
    for event in screen.events:
        if isinstance(event, KeyEvent):
            name = KEY_NAMES[event.key]
            if not event.down and rendered and rendered[-1] == f"+{name}":
                rendered[-1] = name
            else:
                rendered.append(f"{'+' if event.down else '-'}{name}")
        else:
            rendered.append(f"{event.x},{event.y} b{event.buttons}")
    return " ".join(rendered)


def seen(entry: dict[str, Any]) -> str:
    """One tool result entry as the model sees it, in a line: images as ``<mime WxH>``."""
    entry = wire(entry)
    if entry.get("type") == "tool_result":
        parts = [block.get("text") or block["source"]["data"] for block in entry["content"]]
        return ("error: " if entry["is_error"] else "") + " ".join(parts)
    if entry.get("type") == "computer_call_output":
        acknowledged = [check["id"] for check in entry.get("acknowledged_safety_checks", [])]
        return " ".join([entry["output"]["image_url"], *(f"ack {id}" for id in acknowledged)])
    if "functionResponse" in entry:
        response = entry["functionResponse"]
        images = [part["inline_data"]["data"] for part in response.get("parts", [])]
        return " ".join([response["name"], json.dumps(response["response"]), *images])
    return "user: " + " ".join(block["text"] for block in entry["content"])


@dataclass(frozen=True)
class Row:
    agent: Callable[[], Agent]
    script: list[Turn]
    observed: Any


def claude() -> Agent:
    return ClaudeAgent(ClaudeConfig(model="claude-sonnet-4-6"))


def claude_does(*actions: dict[str, Any], observed: Any) -> Row:
    turns = [call("computer", **action) for action in actions]
    return Row(agent=claude, script=[*turns, say("done")], observed=observed)


CLAUDE = {
    "left-click-holding-ctrl": claude_does(
        {"action": "left_click", "coordinate": [10, 20], "text": "ctrl"},
        observed=snapshot(
            {
                "events": "10,20 b0 +Control_L 10,20 b1 10,20 b0 -Control_L",
                "results": ["<image/webp 200x100>"],
            }
        ),
    ),
    "right-click": claude_does(
        {"action": "right_click", "coordinate": [3, 4]},
        observed=snapshot({"events": "3,4 b0 3,4 b4 3,4 b0", "results": ["<image/webp 200x100>"]}),
    ),
    "middle-click": claude_does(
        {"action": "middle_click", "coordinate": [3, 4]},
        observed=snapshot({"events": "3,4 b0 3,4 b2 3,4 b0", "results": ["<image/webp 200x100>"]}),
    ),
    "double-click": claude_does(
        {"action": "double_click", "coordinate": [3, 4]},
        observed=snapshot(
            {"events": "3,4 b0 3,4 b1 3,4 b0 3,4 b1 3,4 b0", "results": ["<image/webp 200x100>"]}
        ),
    ),
    "triple-click": claude_does(
        {"action": "triple_click", "coordinate": [3, 4]},
        observed=snapshot(
            {
                "events": "3,4 b0 3,4 b1 3,4 b0 3,4 b1 3,4 b0 3,4 b1 3,4 b0",
                "results": ["<image/webp 200x100>"],
            }
        ),
    ),
    "mouse-move": claude_does(
        {"action": "mouse_move", "coordinate": [7, 8]},
        observed=snapshot({"events": "7,8 b0", "results": ["<image/webp 200x100>"]}),
    ),
    "mouse-down-then-up": claude_does(
        {"action": "mouse_move", "coordinate": [5, 6]},
        {"action": "left_mouse_down"},
        {"action": "mouse_move", "coordinate": [30, 20]},
        {"action": "left_mouse_up"},
        observed=snapshot(
            {
                "events": "5,6 b0 5,6 b1 30,20 b1 30,20 b0",
                "results": [
                    "<image/webp 200x100>",
                    "<image/webp 200x100>",
                    "<image/webp 200x100>",
                    "<image/webp 200x100>",
                ],
            }
        ),
    ),
    "type-with-newline-and-tab": claude_does(
        {"action": "type", "text": "a\nb\tC"},
        observed=snapshot({"events": "a Return b Tab C", "results": ["<image/webp 200x100>"]}),
    ),
    "key-sequence-repeated": claude_does(
        {"action": "key", "text": "ctrl+a Delete", "repeat": 2},
        observed=snapshot(
            {
                "events": "+Control_L a -Control_L Delete +Control_L a -Control_L Delete",
                "results": ["<image/webp 200x100>"],
            }
        ),
    ),
    "hold-key": claude_does(
        {"action": "hold_key", "text": "shift", "duration": 0.01},
        observed=snapshot({"events": "Shift_L", "results": ["<image/webp 200x100>"]}),
    ),
    "scroll-down": claude_does(
        {
            "action": "scroll",
            "coordinate": [5, 5],
            "scroll_direction": "down",
            "scroll_amount": 2,
        },
        observed=snapshot(
            {"events": "5,5 b0 5,5 b16 5,5 b0 5,5 b16 5,5 b0", "results": ["<image/webp 200x100>"]}
        ),
    ),
    "scroll-up-holding-shift": claude_does(
        {
            "action": "scroll",
            "coordinate": [5, 5],
            "scroll_direction": "up",
            "scroll_amount": 1,
            "text": "shift",
        },
        observed=snapshot(
            {
                "events": "5,5 b0 +Shift_L 5,5 b8 5,5 b0 -Shift_L",
                "results": ["<image/webp 200x100>"],
            }
        ),
    ),
    "scroll-left": claude_does(
        {
            "action": "scroll",
            "coordinate": [5, 5],
            "scroll_direction": "left",
            "scroll_amount": 3,
        },
        observed=snapshot({"events": "5,5 b0", "results": ["<image/webp 200x100>"]}),
    ),
    "drag": claude_does(
        {"action": "left_click_drag", "start_coordinate": [2, 2], "coordinate": [40, 2]},
        observed=snapshot(
            {"events": "2,2 b0 2,2 b1 21,2 b1 40,2 b1 40,2 b0", "results": ["<image/webp 200x100>"]}
        ),
    ),
    "wait": claude_does(
        {"action": "wait", "duration": 0.01},
        observed=snapshot({"events": "", "results": ["<image/webp 200x100>"]}),
    ),
    "cursor-position": claude_does(
        {"action": "mouse_move", "coordinate": [9, 9]},
        {"action": "cursor_position"},
        observed=snapshot({"events": "9,9 b0", "results": ["<image/webp 200x100>", "(9, 9)"]}),
    ),
    "zoom": claude_does(
        {"action": "zoom", "region": [0, 0, 32, 24]},
        observed=snapshot({"events": "", "results": ["<image/webp 32x24>"]}),
    ),
    "screenshot": claude_does(
        {"action": "screenshot"},
        observed=snapshot({"events": "", "results": ["<image/webp 200x100>"]}),
    ),
    "key-without-text": claude_does(
        {"action": "key"},
        observed=snapshot(
            {"events": "", "results": ["error: Error: `text` (key chord) is required for key"]}
        ),
    ),
    "move-without-coordinate": claude_does(
        {"action": "mouse_move"},
        observed=snapshot(
            {
                "events": "",
                "results": [
                    "error: Error: computer action 'mouse_move' failed: coordinate.x is required"
                ],
            }
        ),
    ),
    "unsupported-action": claude_does(
        {"action": "frobnicate"},
        observed=snapshot(
            {"events": "", "results": ["error: Error: unsupported computer action: 'frobnicate'"]}
        ),
    ),
}


def openai(**config: Any) -> Callable[[], Agent]:
    return lambda: OpenAIAgent(OpenAIConfig(model="gpt-5.6", **config))


def openai_does(*actions: dict[str, Any], observed: Any, **call: Any) -> Row:
    return Row(
        agent=openai(), script=[computer_call(*actions, **call), say("done")], observed=observed
    )


OPENAI = {
    "click-holding-ctrl": openai_does(
        {"type": "click", "x": 3, "y": 4, "button": "left", "keys": ["ctrl"]},
        observed=snapshot(
            {
                "events": "3,4 b0 +Control_L 3,4 b1 3,4 b0 -Control_L",
                "results": ["<image/png 200x100>"],
            }
        ),
    ),
    "right-and-wheel-click": openai_does(
        {"type": "click", "x": 3, "y": 4, "button": "right"},
        {"type": "click", "x": 3, "y": 4, "button": "wheel"},
        observed=snapshot(
            {
                "events": "3,4 b0 3,4 b4 3,4 b0 3,4 b0 3,4 b2 3,4 b0",
                "results": ["<image/png 200x100>"],
            }
        ),
    ),
    "double-click": openai_does(
        {"type": "double_click", "x": 3, "y": 4},
        observed=snapshot(
            {"events": "3,4 b0 3,4 b1 3,4 b0 3,4 b1 3,4 b0", "results": ["<image/png 200x100>"]}
        ),
    ),
    "scroll": openai_does(
        {"type": "scroll", "x": 5, "y": 5, "scroll_x": 4, "scroll_y": -2},
        observed=snapshot(
            {"events": "5,5 b0 5,5 b8 5,5 b0 5,5 b8 5,5 b0", "results": ["<image/png 200x100>"]}
        ),
    ),
    "type-and-keypress": openai_does(
        {"type": "type", "text": "hi\n"},
        {"type": "keypress", "keys": ["CTRL", "c"]},
        observed=snapshot(
            {"events": "h i Return +Control_L c -Control_L", "results": ["<image/png 200x100>"]}
        ),
    ),
    "move-drag-wait-screenshot": openai_does(
        {"type": "move", "x": 1, "y": 1},
        {"type": "drag", "path": [{"x": 2, "y": 2}, {"x": 20, "y": 2}]},
        {"type": "wait", "ms": 10},
        {"type": "screenshot"},
        observed=snapshot(
            {"events": "1,1 b0 2,2 b0 2,2 b1 20,2 b1 20,2 b0", "results": ["<image/png 200x100>"]}
        ),
    ),
    "response": openai_does(
        {"type": "response", "text": "all done"},
        observed=snapshot({"events": "", "results": ["<image/png 200x100>"]}),
    ),
    "invalid-action-stops-the-batch": openai_does(
        {"type": "move", "x": 3, "y": 4},
        {"type": "keypress", "keys": "ESC"},
        {"type": "type", "text": "later"},
        observed=snapshot(
            {
                "events": "3,4 b0",
                "results": [
                    "<image/png 200x100>",
                    """\
user: Computer call call_computer stopped with an error. Remaining actions in this call were not executed.
computer action 'keypress' failed: keys must be a list of non-empty strings\
""",
                ],
            }
        ),
    ),
    "unknown-action-type": openai_does(
        {"type": "frobnicate"},
        observed=snapshot(
            {
                "events": "",
                "results": [
                    "<image/png 200x100>",
                    """\
user: Computer call call_computer stopped with an error. Remaining actions in this call were not executed.
computer action 'frobnicate' failed: Invalid action type: frobnicate\
""",
                ],
            }
        ),
    ),
    "empty-actions": openai_does(
        observed=snapshot(
            {
                "events": "",
                "results": [
                    "<image/png 200x100>",
                    """\
user: Computer call call_computer stopped with an error. Remaining actions in this call were not executed.
actions list is empty\
""",
                ],
            }
        )
    ),
    "safety-checks-acknowledged": openai_does(
        {"type": "screenshot"},
        pending_safety_checks=[{"id": "sc_1", "code": "malicious_instructions", "message": "x"}],
        observed=snapshot({"events": "", "results": ["<image/png 200x100> ack sc_1"]}),
    ),
    "webp-screenshots": Row(
        agent=openai(screenshot_encoding={"mime_type": "image/webp", "quality": 42}),
        script=[computer_call({"type": "screenshot"}), say("done")],
        observed=snapshot({"events": "", "results": ["<image/webp 200x100>"]}),
    ),
}


def gemini(**config: Any) -> Callable[[], Agent]:
    return lambda: GeminiAgent(GeminiConfig(model="gemini-3.1-pro-preview", **config))


def gemini_does(*turns: Turn, observed: Any) -> Row:
    return Row(agent=gemini(), script=[*turns, say("done")], observed=observed)


GEMINI = {
    "open-web-browser": gemini_does(
        call("open_web_browser"),
        observed=snapshot(
            {"events": "", "results": ['open_web_browser {"success": true} <image/png 200x100>']}
        ),
    ),
    "scroll-document-down": gemini_does(
        call("scroll_document", direction="down"),
        observed=snapshot(
            {
                "events": "0,0 b16 0,0 b0 0,0 b16 0,0 b0 0,0 b16 0,0 b0",
                "results": ['scroll_document {"success": true} <image/png 200x100>'],
            }
        ),
    ),
    "go-back-and-forward": gemini_does(
        call("go_back"),
        call("go_forward"),
        observed=snapshot(
            {
                "events": "+Alt_L Left -Alt_L +Alt_L Right -Alt_L",
                "results": [
                    'go_back {"success": true} <image/png 200x100>',
                    'go_forward {"success": true} <image/png 200x100>',
                ],
            }
        ),
    ),
    "search": gemini_does(
        call("search"),
        observed=snapshot(
            {
                "events": "+Control_L l -Control_L h t t p s colon slash slash w w w period g o o g l e period c o m Return",
                "results": ['search {"success": true} <image/png 200x100>'],
            }
        ),
    ),
    "navigate": gemini_does(
        call("navigate", url="https://a.b"),
        observed=snapshot(
            {
                "events": "+Control_L l -Control_L h t t p s colon slash slash a period b Return",
                "results": ['navigate {"success": true} <image/png 200x100>'],
            }
        ),
    ),
    "key-combination": gemini_does(
        call("key_combination", keys="Control+a"),
        observed=snapshot(
            {
                "events": "+Control_L a -Control_L",
                "results": ['key_combination {"success": true} <image/png 200x100>'],
            }
        ),
    ),
    "key-combination-without-keys": gemini_does(
        call("key_combination", keys=3),
        observed=snapshot(
            {
                "events": "",
                "results": ['key_combination {"error": "keys must be a \'+\'-separated string"}'],
            }
        ),
    ),
}

ROWS = {
    **{f"claude-{name}": row for name, row in CLAUDE.items()},
    **{f"openai-{name}": row for name, row in OPENAI.items()},
    **{f"gemini-{name}": row for name, row in GEMINI.items()},
}


@pytest.mark.parametrize("row", ROWS.values(), ids=ROWS.keys())
async def test_a_computer_action_reaches_the_screen_as_rfb_input(
    row: Row, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script(row.script)

    async with fake_screen(width=200, height=100) as screen:
        screen_capability = Capability.rfb(url=screen.url)
        run = await run_task(
            workspace_env(tmp_path / "ws", capabilities=(screen_capability,)), row.agent()
        )

    assert run.trace.status == "completed"
    observed = {
        "events": events(screen),
        "results": [seen(entry) for entry in tool_results(models.requests())],
    }
    assert observed == row.observed


CLAUDE_MODELS = {
    "claude-sonnet-4-6": snapshot(
        {
            "beta": "computer-use-2025-11-24",
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
                {
                    "type": "computer_20251124",
                    "name": "computer",
                    "display_width_px": 200,
                    "display_height_px": 100,
                    "display_number": 1,
                    "enable_zoom": True,
                },
            ],
            "described_display": None,
        }
    ),
    "claude-sonnet-4-5": snapshot(
        {
            "beta": "computer-use-2025-01-24",
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
                {
                    "type": "computer_20250124",
                    "name": "computer",
                    "display_width_px": 200,
                    "display_height_px": 100,
                    "display_number": 1,
                },
            ],
            "described_display": None,
        }
    ),
    "claude-future-model": snapshot(
        {
            "beta": "computer-use-2025-11-24",
            "tools": [
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
                {
                    "type": "computer_20251124",
                    "name": "computer",
                    "display_width_px": 200,
                    "display_height_px": 100,
                    "display_number": 1,
                    "enable_zoom": True,
                },
            ],
            "described_display": None,
        }
    ),
    "qwen/qwen3.8-max": snapshot(
        {
            "beta": None,
            "tools": [
                {
                    "name": "bash",
                    "input_schema": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
                {
                    "name": "str_replace_based_edit_tool",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "enum": ["view", "create", "str_replace", "insert"],
                            },
                            "path": {"type": "string"},
                            "file_text": {"type": "string"},
                            "old_str": {"type": "string"},
                            "new_str": {"type": "string"},
                            "insert_line": {"type": "integer"},
                        },
                        "required": ["command", "path"],
                    },
                },
                {
                    "name": "computer",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": [
                                    "screenshot",
                                    "zoom",
                                    "left_click",
                                    "right_click",
                                    "middle_click",
                                    "double_click",
                                    "triple_click",
                                    "mouse_move",
                                    "left_mouse_down",
                                    "left_mouse_up",
                                    "type",
                                    "key",
                                    "hold_key",
                                    "scroll",
                                    "left_click_drag",
                                    "wait",
                                    "cursor_position",
                                ],
                            },
                            "coordinate": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "start_coordinate": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                            "text": {"type": "string"},
                            "duration": {"type": "number"},
                            "scroll_direction": {
                                "type": "string",
                                "enum": ["up", "down", "left", "right"],
                            },
                            "scroll_amount": {"type": "integer"},
                            "repeat": {"type": "integer"},
                            "region": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                        },
                        "required": ["action"],
                    },
                },
            ],
            "described_display": "200x100",
        }
    ),
}


@pytest.mark.parametrize(("model", "expected"), CLAUDE_MODELS.items(), ids=CLAUDE_MODELS.keys())
async def test_claude_advertises_the_computer_tool_its_model_supports(
    model: str, expected: Any, models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    """Claude models get the newest computer-use tool they support with its beta; other
    models served over the Messages API get plain schemas describing the display."""
    hud_env.set(HUD_API_KEY="k")
    models.script([say("done")])

    async with fake_screen(width=200, height=100) as screen:
        screen_capability = Capability.rfb(url=screen.url)
        await run_task(
            workspace_env(tmp_path / "ws", capabilities=(screen_capability,)),
            ClaudeAgent(ClaudeConfig(model=model)),
        )

    (request,) = models.requests()
    computer = next(tool for tool in request.body["tools"] if tool["name"] == "computer")
    display = re.search(r"display is (\d+x\d+) pixels", computer.get("description", ""))
    advertised = {
        "beta": request.headers.get("anthropic-beta"),
        "tools": wire(request.body["tools"]),
        "described_display": display and display[1],
    }
    assert advertised == expected


async def test_only_the_first_screen_is_driven(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([call("computer", action="left_click", coordinate=[1, 1]), say("done")])

    async with fake_screen() as first, fake_screen() as second:
        screens = (
            Capability.rfb(name="screen-0", url=first.url),
            Capability.rfb(name="screen-1", url=second.url),
        )
        await run_task(workspace_env(tmp_path / "ws", capabilities=screens), claude())

    assert (events(first), events(second), second.connections) == ("1,1 b0 1,1 b1 1,1 b0", "", 0)


async def test_gemini_keeps_screenshots_for_the_three_latest_computer_turns(
    models: Models, hud_env: HudEnv, tmp_path: Path
) -> None:
    hud_env.set(HUD_API_KEY="k")
    models.script([*(call("open_web_browser") for _ in range(5)), say("done")])

    async with fake_screen() as screen:
        screen_capability = Capability.rfb(url=screen.url)
        await run_task(
            workspace_env(tmp_path / "ws", capabilities=(screen_capability,)), gemini()()
        )

    responses = [part["functionResponse"] for part in tool_results(models.requests())]
    assert ["parts" in response for response in responses] == [False, False, True, True, True]

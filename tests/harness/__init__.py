"""The black-box test harness: fakes of everything outside the SDK.

Tests drive the SDK through its public boundaries (the ``hud`` CLI, exported
API, wire protocols, files) and fake only what lies outside it: the HUD
services, model providers, docker and VNC servers. The fixtures wiring these in
live in ``tests/conftest.py``.
"""

from .asgi import serve_asgi
from .cdp import Command, FakeBrowser, fake_browser
from .cli import Hud, Result, scrub
from .config import HudEnv
from .docker import FakeDocker
from .models import (
    ModelRequest,
    Models,
    ToolCall,
    Turn,
    call,
    calls,
    computer_call,
    fail,
    interrupted,
    say,
    shell_call,
    stream_error,
)
from .relay import FlakyRelay, relay
from .rfb import FakeScreen, KeyEvent, PointerEvent, fake_screen
from .scenario import RecordingProvider, ScriptedAgent, eventually, served, task_row
from .services import FakeServices, Reply, Request
from .spans import ROBOT_STEP_SCHEMA, spans, steps

__all__ = [
    "ROBOT_STEP_SCHEMA",
    "Command",
    "FakeBrowser",
    "FakeDocker",
    "FakeScreen",
    "FakeServices",
    "FlakyRelay",
    "Hud",
    "HudEnv",
    "KeyEvent",
    "ModelRequest",
    "Models",
    "PointerEvent",
    "RecordingProvider",
    "Reply",
    "Request",
    "Result",
    "ScriptedAgent",
    "ToolCall",
    "Turn",
    "call",
    "calls",
    "computer_call",
    "eventually",
    "fail",
    "fake_browser",
    "fake_screen",
    "interrupted",
    "relay",
    "say",
    "scrub",
    "serve_asgi",
    "served",
    "shell_call",
    "spans",
    "steps",
    "stream_error",
    "task_row",
]

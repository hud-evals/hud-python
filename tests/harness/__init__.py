"""The black-box test harness: fakes of everything outside the SDK.

Tests drive the SDK through its public boundaries (the ``hud`` CLI, exported
API, wire protocols, files) and fake only what lies outside it: the HUD
services, model providers, docker and VNC servers. The fixtures wiring these in
live in ``tests/conftest.py``.
"""

from .cli import Hud, Result, scrub
from .config import HudEnv
from .docker import FakeDocker
from .models import ModelRequest, Models, ToolCall, Turn, call, calls, fail, say
from .rfb import FakeScreen, KeyEvent, PointerEvent, fake_screen
from .scenario import RecordingProvider, ScriptedAgent, served, task_row
from .services import FakeServices, Reply, Request
from .spans import spans, steps

__all__ = [
    "FakeDocker",
    "FakeScreen",
    "FakeServices",
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
    "fail",
    "fake_screen",
    "say",
    "scrub",
    "served",
    "spans",
    "steps",
    "task_row",
]

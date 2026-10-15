"""The test harness: stand-ins for what tests cannot use for real.

Tests drive the SDK through its public boundaries (the ``hud`` CLI, exported
API, files) against the real outside world where they can: real models and
CLIs, Docker and Modal run in their own lanes. What stays here stands in for
what cannot run in a test, the HUD services and a desktop screen, and for
faults a real system will not produce on demand: scripted model replies,
dropped connections. The fixtures wiring these in live in ``tests/conftest.py``.
"""

from .asgi import serve_asgi
from .cdp import Command, FakeBrowser, fake_browser
from .cli import Hud, Result, scrub, words
from .config import HudEnv
from .control import ControlPeer, answer, control_peer, hang_up, hold
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
from .uv import Uv

__all__ = [
    "ROBOT_STEP_SCHEMA",
    "Command",
    "ControlPeer",
    "FakeBrowser",
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
    "Uv",
    "answer",
    "call",
    "calls",
    "computer_call",
    "control_peer",
    "eventually",
    "fail",
    "fake_browser",
    "fake_screen",
    "hang_up",
    "hold",
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
    "words",
]

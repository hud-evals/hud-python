"""Stand-ins for what lies outside the SDK: the user's machine, the HUD services, model
providers, a browser, a VNC screen, MCP servers, the network, and a raw peer on either
side of the control channel. The SDK code tests run is in ``tests/fixtures``."""

from __future__ import annotations

from .asgi import mcp_server, serve_asgi
from .browser import Command, FakeBrowser, fake_browser
from .internet import internet
from .machine import (
    ROBOT_STEP_SCHEMA,
    Hud,
    HudEnv,
    Result,
    Uv,
    scrub,
    spans,
    steps,
    words,
    write_files,
)
from .models import (
    ModelRequest,
    Models,
    ToolCall,
    Turn,
    call,
    calls,
    fail,
    say,
    shell_call,
    stream_error,
)
from .peers import (
    FRAME_LIMIT,
    SESSION_ID,
    ControlPeer,
    Wire,
    answer,
    control_peer,
    encode,
    hang_up,
    hold,
    wire,
)
from .platform import Platform
from .relay import FlakyRelay, relay
from .screen import FakeScreen, KeyEvent, PointerEvent, fake_screen
from .services import FakeServices, Reply, Request
from .waiting import eventually

__all__ = [
    "FRAME_LIMIT",
    "ROBOT_STEP_SCHEMA",
    "SESSION_ID",
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
    "Platform",
    "PointerEvent",
    "Reply",
    "Request",
    "Result",
    "ToolCall",
    "Turn",
    "Uv",
    "Wire",
    "answer",
    "call",
    "calls",
    "control_peer",
    "encode",
    "eventually",
    "fail",
    "fake_browser",
    "fake_screen",
    "hang_up",
    "hold",
    "internet",
    "mcp_server",
    "relay",
    "say",
    "scrub",
    "serve_asgi",
    "shell_call",
    "spans",
    "steps",
    "stream_error",
    "wire",
    "words",
    "write_files",
]

"""HUD environment: VideoGameBench Game Boy games as turn-based, scored evals.

A v6 HUD environment. An external HUD agent observes the game screen (returned as
an image from every action tool) and acts via button tools, served over an
in-process ``mcp`` capability. The emulator only advances inside tool calls, so
the game is naturally "paused" while the agent thinks — VideoGameBench's
``--lite`` semantics, which fit MCP cleanly.

Design notes:
  * VideoGameBench's agent / LLM client / real-time eval loop are intentionally
    NOT used — HUD's external agent replaces them.
  * Per-step scoring (checkpoint / RAM / exploration) runs inside each tool call,
    because a HUD template only yields twice (setup prompt, then final reward).
  * One emulator instance lives in module state. HUD runs one container per
    evaluation, so a single global instance is safe (no in-process parallelism).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import glob
import io
import os
import socket
import sys

# Make sibling modules importable regardless of how the server is launched.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections.abc import AsyncGenerator  # noqa: E402
from typing import Any  # noqa: E402

from hud import Environment  # noqa: E402
from hud.capabilities import Capability  # noqa: E402
from mcp.types import ContentBlock, ImageContent, TextContent  # noqa: E402
from PIL import Image  # noqa: E402

import games_loader  # noqa: E402
from emulator import GB_BUTTONS, GameBoyEmulator, sanitize_buttons  # noqa: E402

env = Environment(name="libbet")

# How far a single action may advance the game, to keep steps bounded.
MAX_FRAMES_PER_CALL = 600

# Upscale factor for the 160x144 Game Boy screen sent to the agent (and shown in
# the HUD trace UI). Nearest-neighbor keeps the pixel art crisp; bigger = easier
# for a vision model to read and clearer in the UI. Scoring uses the raw frame.
GB_SCALE = 3

# Shared control instructions prepended to every Game Boy game's prompt, so the
# agent gets consistent, tool-accurate command guidance (the per-game prompt.txt
# then adds game-specific lore + strategy, VideoGameBench-style).
GB_CONTROLS = """\
## Controls

You see the current Game Boy screen each turn and act with these tools:
- press_buttons(buttons, frames=15): press one or more buttons together this turn.
  Valid buttons: A, B, START, SELECT, UP, DOWN, LEFT, RIGHT.
  A = confirm / advance dialogue, B = cancel / back, START = open menu,
  SELECT = secondary, and the d-pad (UP/DOWN/LEFT/RIGHT) moves one tile per press.
  `frames` is how long the press is held (larger = the game advances further).
- wait(frames=30): let the game run with no input (animations, dialogue, scrolling).
- screenshot(): re-look at the current screen without acting.

Play one deliberate action per turn based on what is on screen. Press A to clear
text boxes; use the d-pad to walk; open the menu with START. If nothing changes,
try a different button or `wait`.
"""

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# In-process MCP server that exposes the agent-facing game tools as a capability.
_MCP_PORT: int = 0
_MCP_SERVER_TASK: asyncio.Task | None = None


def _resolve_record_dir(game: str) -> str | None:
    """Where to dump gameplay frames, if recording is enabled.

    Set ``VGBENCH_RECORD_DIR`` to record to a specific dir (use this so parallel
    runs of the SAME game don't clobber each other), or ``VGBENCH_RECORD=1`` to
    record to logs/rec/<game>/. Off by default / in production.
    """
    rd = os.environ.get("VGBENCH_RECORD_DIR")
    if not rd and os.environ.get("VGBENCH_RECORD"):
        rd = os.path.join("logs", "rec", game)
    if not rd:
        return None
    return rd if os.path.isabs(rd) else os.path.join(_REPO_DIR, rd)


# Module-global session (one game per container).
_session: dict[str, Any] = {
    "emulator": None,  # GameBoyEmulator
    "spec": None,  # GameSpec
    "board": None,  # ScoreBoard
    "steps": 0,
    "max_steps": 0,
    "record_dir": None,
    "frame_idx": 0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _teardown() -> None:
    emu = _session.get("emulator")
    if emu is not None:
        emu.close()
    _session.update(
        emulator=None,
        spec=None,
        board=None,
        steps=0,
        max_steps=0,
        record_dir=None,
        frame_idx=0,
    )


def _png_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _status(prefix: str) -> str:
    board = _session["board"]
    extra = f" | {board.summary()}" if board is not None else ""
    return f"{prefix} [step {_session['steps']}/{_session['max_steps']}{extra}]"


def _observe(text: str) -> list[ContentBlock]:
    """Return text + the current screen (upscaled) as MCP content blocks."""
    screen = _session["emulator"].get_screen()
    if GB_SCALE != 1:
        screen = screen.resize((screen.width * GB_SCALE, screen.height * GB_SCALE), Image.NEAREST)
    return [
        TextContent(type="text", text=text),
        ImageContent(type="image", data=_png_b64(screen), mimeType="image/png"),
    ]


def _no_game() -> list[ContentBlock]:
    return [TextContent(type="text", text="No game is loaded. The play-game task must be running first.")]


def _score_current() -> None:
    """Feed the current screen + RAM to the scorers (called after each action)."""
    emu, board = _session["emulator"], _session["board"]
    if board is not None and emu is not None:
        board.on_step(screen=emu.get_screen(), emulator=emu)


def _clamp_frames(frames: int) -> int:
    return max(1, min(int(frames), MAX_FRAMES_PER_CALL))


def _record_frame() -> None:
    """Save the current frame to the per-game recording dir, if recording."""
    rd = _session.get("record_dir")
    emu = _session.get("emulator")
    if not rd or emu is None:
        return
    try:
        emu.get_screen().save(os.path.join(rd, f"frame_{_session['frame_idx']:05d}.png"))
        _session["frame_idx"] += 1
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Agent-facing tools (registered on the in-process MCP server in @env.initialize)
# ---------------------------------------------------------------------------
async def press_buttons(buttons: list[str], frames: int = 15) -> list[ContentBlock]:
    """Press one or more Game Boy buttons, then return the new screen.

    Args:
        buttons: Buttons to press together this turn. Valid values:
            A, B, START, SELECT, UP, DOWN, LEFT, RIGHT.
        frames: How long the buttons are held / how far the game advances
            (1-600, default 15).
    """
    emu = _session["emulator"]
    if emu is None:
        return _no_game()
    valid, invalid = sanitize_buttons(buttons)
    if not valid:
        return [TextContent(type="text", text=f"No valid buttons in {buttons}. Valid buttons: {GB_BUTTONS}.")]
    emu.step({b: True for b in valid}, frames=_clamp_frames(frames))
    _session["steps"] += 1
    _score_current()
    _record_frame()
    note = f"Pressed {'+'.join(valid)}"
    if invalid:
        note += f" (ignored: {', '.join(map(str, invalid))})"
    return _observe(_status(note))


async def wait(frames: int = 30) -> list[ContentBlock]:
    """Advance the game without pressing anything (let animations/dialogue play).

    Args:
        frames: How many frames to advance (1-600, default 30).
    """
    emu = _session["emulator"]
    if emu is None:
        return _no_game()
    emu.no_op(_clamp_frames(frames))
    _session["steps"] += 1
    _score_current()
    _record_frame()
    return _observe(_status("Waited"))


async def screenshot() -> list[ContentBlock]:
    """Return the current game screen without advancing the game."""
    emu = _session["emulator"]
    if emu is None:
        return _no_game()
    return _observe(_status("Screenshot"))


# ---------------------------------------------------------------------------
# In-process MCP capability: serve the game tools to the agent
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


async def _listening(host: str, port: int, timeout: float = 15.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            socket.create_connection((host, port), timeout=0.5).close()
            return
        except OSError:
            await asyncio.sleep(0.1)
    raise RuntimeError(f"game MCP server never came up on {host}:{port}")


@env.initialize
async def _up() -> None:
    # Lazy import so `import tasks` (the task-collection path) stays free of
    # fastmcp/authlib import-time noise.
    from fastmcp import FastMCP

    global _MCP_PORT, _MCP_SERVER_TASK
    if _MCP_SERVER_TASK is None:
        server = FastMCP(name="gameboy")
        server.tool(press_buttons)
        server.tool(wait)
        server.tool(screenshot)
        _MCP_PORT = _free_port()
        _MCP_SERVER_TASK = asyncio.create_task(
            server.run_async(transport="http", host="127.0.0.1", port=_MCP_PORT, show_banner=False)
        )
        await _listening("127.0.0.1", _MCP_PORT)
    env.add_capability(Capability.mcp(name="gameboy", url=f"http://127.0.0.1:{_MCP_PORT}/mcp"))


@env.shutdown
async def _down() -> None:
    global _MCP_SERVER_TASK
    if _MCP_SERVER_TASK is not None:
        _MCP_SERVER_TASK.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await _MCP_SERVER_TASK
        _MCP_SERVER_TASK = None
    _teardown()


# ---------------------------------------------------------------------------
# Task: setup (boot game) -> agent plays via the mcp tools -> evaluate
# ---------------------------------------------------------------------------
@env.template(id="play-game", description="Explore Libbet through MCP tools, scored by visually distinct screens.")
async def play_game(
    game: str = "test", max_steps: int = 200, threshold: int | None = None
) -> AsyncGenerator[Any, str | None]:
    """Explore the bundled Game Boy game through screenshots and button presses.

    Args:
        game: Folder under games/ to load; the bundled game is "test" (Libbet).
        max_steps: Advisory step budget surfaced to the agent.
        threshold: Optional override of the checkpoint perceptual-hash threshold.
    """
    _teardown()
    spec = games_loader.load_game_spec(game, threshold_override=threshold)
    emu = GameBoyEmulator(render=False)
    emu.load_game(spec.rom_path, boot_frames=spec.boot_frames)
    if spec.preload_path:
        emu.pre_load_file(spec.preload_path)

    board = spec.new_board()
    _session.update(emulator=emu, spec=spec, board=board, steps=0, max_steps=max_steps)
    board.on_step(screen=emu.get_screen(), emulator=emu)  # initial snapshot

    rd = _resolve_record_dir(game)
    if rd:
        os.makedirs(rd, exist_ok=True)
        for old in glob.glob(os.path.join(rd, "frame_*.png")):
            os.remove(old)
        _session["record_dir"] = rd
        _session["frame_idx"] = 0
        _record_frame()  # capture the starting frame

    prompt = (
        f"{GB_CONTROLS}\n"
        f"## Game: {game}\n{spec.prompt}\n\n"
        f"Aim to explore within {max_steps} action steps. Scoring: {board.summary()}. "
        "Your score measures visual exploration; it does not check whether you completed a level or won the game."
    )
    answer = yield prompt

    result = board.result(done=True)
    result.info["game"] = game
    result.info["steps_used"] = _session["steps"]
    result.info["max_steps"] = max_steps
    if answer:
        result.info["agent_answer"] = str(answer)[:500]
    _teardown()
    yield result

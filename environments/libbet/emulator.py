"""Game Boy emulator wrapper (PyBoy) for the HUD VideoGameBench template.

Adapted from VideoGameBench's ``src/emulators/gba/interface.py`` (``GBAInterface``),
trimmed to a standalone, headless, *turn-based* backend. VideoGameBench bundles an
agent + LLM client + real-time eval loop around this emulator; in HUD that role is
played by the external agent, so only the emulator-driving parts are kept here.

The emulator only advances when a tool call drives it (``step`` / ``no_op``), which
is exactly VideoGameBench's ``--lite`` semantics and maps cleanly onto MCP's
request/response model.
"""

from __future__ import annotations

import os
from typing import Any

from PIL import Image
from pyboy import PyBoy
from pyboy.utils import WindowEvent

# Button name -> PyBoy press / release events. Mirrors VideoGameBench's BUTTON_MAP.
BUTTON_MAP = {
    "A": WindowEvent.PRESS_BUTTON_A,
    "B": WindowEvent.PRESS_BUTTON_B,
    "SELECT": WindowEvent.PRESS_BUTTON_SELECT,
    "START": WindowEvent.PRESS_BUTTON_START,
    "RIGHT": WindowEvent.PRESS_ARROW_RIGHT,
    "LEFT": WindowEvent.PRESS_ARROW_LEFT,
    "UP": WindowEvent.PRESS_ARROW_UP,
    "DOWN": WindowEvent.PRESS_ARROW_DOWN,
}
RELEASE_MAP = {
    "A": WindowEvent.RELEASE_BUTTON_A,
    "B": WindowEvent.RELEASE_BUTTON_B,
    "SELECT": WindowEvent.RELEASE_BUTTON_SELECT,
    "START": WindowEvent.RELEASE_BUTTON_START,
    "RIGHT": WindowEvent.RELEASE_ARROW_RIGHT,
    "LEFT": WindowEvent.RELEASE_ARROW_LEFT,
    "UP": WindowEvent.RELEASE_ARROW_UP,
    "DOWN": WindowEvent.RELEASE_ARROW_DOWN,
}

GB_BUTTONS = list(BUTTON_MAP.keys())

# Screen is 160x144 on every Game Boy / Color title.
SCREEN_SIZE = (160, 144)


def sanitize_buttons(buttons: list[str]) -> tuple[list[str], list[str]]:
    """Split a requested button list into (valid, invalid), de-duplicated.

    Also drops a simultaneous START+SELECT combo, which soft-resets many games
    (VideoGameBench's ``convert_to_dict`` guards against the same thing).
    """
    valid: list[str] = []
    invalid: list[str] = []
    for raw in buttons:
        b = str(raw).strip().upper()
        if b in BUTTON_MAP:
            if b not in valid:
                valid.append(b)
        else:
            invalid.append(raw)
    if "START" in valid and "SELECT" in valid:
        # Dropping SELECT is enough to avoid the reset combo.
        valid.remove("SELECT")
        invalid.append("SELECT (START+SELECT reset combo blocked)")
    return valid, invalid


class GameBoyEmulator:
    """Headless PyBoy wrapper exposing the minimal surface the HUD env needs."""

    def __init__(self, render: bool = False) -> None:
        self.pyboy: PyBoy | None = None
        self.render = render

    # -- lifecycle ----------------------------------------------------------
    def load_game(self, rom_path: str, boot_frames: int = 600) -> None:
        """Boot a ROM and run past the BIOS/boot logo.

        Args:
            rom_path: Path to a ``.gb`` / ``.gbc`` ROM.
            boot_frames: Number of frames to tick after boot to clear the
                Nintendo logo (and, for some titles, the attract loop).
        """
        if not os.path.exists(rom_path):
            raise FileNotFoundError(
                f"ROM not found: {rom_path}. ROMs are not bundled (copyright); "
                f"mount your legally-owned ROM at this path. See the README."
            )
        # PyBoy 2.x: "null" is the headless window; "SDL2" renders a window.
        self.pyboy = PyBoy(rom_path, window="SDL2" if self.render else "null")
        self.pyboy.set_emulation_speed(1 if self.render else 0)  # 0 = uncapped
        # render=True on the final tick so the screen buffer is populated; PyBoy
        # only updates `screen.image` for rendered frames.
        self.pyboy.tick(max(1, boot_frames), render=True, sound=False)

    def close(self) -> None:
        if self.pyboy is not None:
            try:
                self.pyboy.stop(save=False)
            except Exception:
                pass
            self.pyboy = None

    def _require(self) -> PyBoy:
        if self.pyboy is None:
            raise RuntimeError("No ROM loaded — call load_game() first.")
        return self.pyboy

    # -- stepping -----------------------------------------------------------
    def step(self, action: dict[str, bool], frames: int = 15) -> None:
        """Hold the pressed buttons for ``frames`` frames, then advance.

        ``frames`` is both how long the input is held (PyBoy auto-releases via
        the ``delay`` arg) and roughly how far the game advances.
        """
        pb = self._require()
        for button, pressed in action.items():
            if pressed and button in BUTTON_MAP:
                pb.send_input(BUTTON_MAP[button])
                pb.send_input(RELEASE_MAP[button], delay=frames)
        # Advance frames+1 ticks (release fires at `delay=frames`); render the
        # final frame so the screenshot reflects the post-action state.
        pb.tick(frames + 1, render=True, sound=False)

    def no_op(self, frames: int = 30) -> None:
        """Advance ``frames`` frames with no input."""
        pb = self._require()
        pb.tick(max(1, frames), render=True, sound=False)

    # -- observation --------------------------------------------------------
    def get_screen(self) -> Image.Image:
        """Current frame as an RGB PIL image (160x144)."""
        pb = self._require()
        return pb.screen.image.convert("RGB")

    def read_memory(self, addr: int, bank: int | None = None) -> int:
        """Read a single byte of Game Boy memory (for RAM-based scoring).

        ``bank`` selects a specific memory bank. For Game Boy Color games whose
        variables live in switchable WRAM (0xD000-0xDFFF), pass ``bank=1`` (etc.)
        to read that WRAM bank regardless of what's currently mapped. Omit it
        (DMG games / fixed WRAM 0xC000-0xCFFF) to read the currently-mapped bank.
        """
        pb = self._require()
        return int(pb.memory[addr] if bank is None else pb.memory[bank, addr])

    def read_memory_range(self, addr: int, length: int, bank: int | None = None) -> list[int]:
        pb = self._require()
        sl = slice(addr, addr + length)
        cells = pb.memory[sl] if bank is None else pb.memory[bank, sl]
        return [int(b) for b in cells]

    # -- save states (fast reset / skip-intro) ------------------------------
    def save_state(self, path: str) -> None:
        pb = self._require()
        with open(path, "wb") as f:
            pb.save_state(f)

    def load_state(self, path: str) -> None:
        pb = self._require()
        with open(path, "rb") as f:
            pb.load_state(f)
        pb.tick(30, render=True, sound=False)  # unstick after a state restore

    # -- preload (menu/difficulty setup) ------------------------------------
    def pre_load_file(self, preload_path: str) -> None:
        """Execute a VideoGameBench-style preload script.

        Supported commands (one per line; ``#`` comments allowed):
          * ``sleep N``       — advance N seconds (N*60 frames)
          * ``press_key BTN`` — tap a button (A/B/START/SELECT/UP/DOWN/LEFT/RIGHT)
        """
        pb = self._require()
        if not preload_path or not os.path.exists(preload_path):
            return
        with open(preload_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                cmd = parts[0].lower()
                if cmd == "sleep" and len(parts) >= 2:
                    pb.tick(max(1, int(float(parts[1]) * 60)), render=True, sound=False)
                elif cmd == "press_key" and len(parts) >= 2:
                    btn = parts[1].upper()
                    if btn in BUTTON_MAP:
                        self.step({btn: True}, frames=10)

    def observation(self) -> dict[str, Any]:
        return {"screen": self.get_screen(), "buttons": GB_BUTTONS}

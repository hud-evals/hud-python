"""Load a game definition from ``games/<game>/``.

Layout (mirrors VideoGameBench's ``configs/<game>/`` so its assets drop in):

    games/<game>/
      config.yaml      # rom, boot_frames, action_frames, scoring spec, prompt
      prompt.txt       # system prompt shown to the agent (optional)
      checkpoints/*.png # ordered checkpoint images (optional)
      preload.txt      # menu/difficulty setup script (optional)
      ram_map.py       # read(emulator)->dict + reward(values)->float (optional)

``config.yaml`` ``scoring`` is a list of scorer specs, e.g.::

    scoring:
      - type: exploration
        target: 25
        weight: 1.0
      - type: ram
        weight: 0.5
      - type: checkpoint
        threshold: 8
        weight: 0.5
"""

from __future__ import annotations

import glob
import importlib.util
import os
from dataclasses import dataclass, field
from typing import Any

import yaml
from PIL import Image

from scoring import (
    CheckpointScorer,
    ExplorationScorer,
    RamScorer,
    ScoreBoard,
    hash_image,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(ROOT, "games")
ROMS_DIR = os.path.join(ROOT, "roms")

DEFAULT_PROMPT = (
    "You are playing a Game Boy game. Each turn you see the current screen. "
    "Use the press_buttons tool (A, B, START, SELECT, UP, DOWN, LEFT, RIGHT) to play. "
    "Use wait to let the game advance and screenshot to look again without acting."
)


@dataclass
class GameSpec:
    """Everything needed to boot and score one game. ``new_board`` builds a
    fresh :class:`ScoreBoard` so re-running the scenario resets scoring state."""

    name: str
    rom_path: str
    prompt: str
    boot_frames: int
    action_frames: int
    preload_path: str | None
    _scorer_factories: list[Any] = field(default_factory=list)

    def new_board(self) -> ScoreBoard:
        return ScoreBoard([make() for make in self._scorer_factories])


def _load_ram_map(game_dir: str) -> Any:
    path = os.path.join(game_dir, "ram_map.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"scoring type 'ram' requires {path} (with read() and reward()).")
    spec = importlib.util.spec_from_file_location(f"ram_map_{os.path.basename(game_dir)}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_checkpoint_hashes(game_dir: str) -> list:
    cp_dir = os.path.join(game_dir, "checkpoints")
    files = sorted(
        glob.glob(os.path.join(cp_dir, "*.png")),
        key=lambda p: _natural_key(os.path.basename(p)),
    )
    return [hash_image(Image.open(f).convert("RGB")) for f in files]


def _natural_key(name: str):
    stem = os.path.splitext(name)[0]
    return (int(stem), name) if stem.isdigit() else (1 << 30, name)


def _build_scorer_factories(cfg: dict[str, Any], game_dir: str, threshold_override: int | None) -> list[Any]:
    specs = cfg.get("scoring")
    if not specs:
        # Sensible default: generic exploration reward (works for any ROM).
        specs = [{"type": "exploration"}]

    factories: list[Any] = []
    for spec in specs:
        kind = str(spec.get("type", "")).lower()
        weight = float(spec.get("weight", 1.0))

        if kind == "exploration":
            target = int(spec.get("target", 25))
            thr = int(spec.get("threshold", 4))
            factories.append(
                lambda target=target, thr=thr, weight=weight: ExplorationScorer(
                    target_unique=target, threshold=thr, weight=weight
                )
            )
        elif kind == "checkpoint":
            hashes = _load_checkpoint_hashes(game_dir)
            if not hashes:
                raise FileNotFoundError(f"scoring type 'checkpoint' but no PNGs in {game_dir}/checkpoints/")
            thr = threshold_override if threshold_override is not None else int(spec.get("threshold", 8))
            factories.append(
                lambda hashes=hashes, thr=thr, weight=weight: CheckpointScorer(hashes, threshold=thr, weight=weight)
            )
        elif kind == "ram":
            ram_map = _load_ram_map(game_dir)
            factories.append(lambda ram_map=ram_map, weight=weight: RamScorer(ram_map, weight=weight))
        else:
            raise ValueError(f"Unknown scoring type '{kind}' in {game_dir}/config.yaml")
    return factories


def list_games() -> list[str]:
    if not os.path.isdir(GAMES_DIR):
        return []
    return sorted(d for d in os.listdir(GAMES_DIR) if os.path.isfile(os.path.join(GAMES_DIR, d, "config.yaml")))


def load_game_spec(game: str, *, threshold_override: int | None = None) -> GameSpec:
    game_dir = os.path.join(GAMES_DIR, game)
    cfg_path = os.path.join(game_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"No game '{game}'. Available: {list_games()}. (Add games/{game}/config.yaml to define a new one.)"
        )
    cfg = yaml.safe_load(open(cfg_path)) or {}

    emulator = str(cfg.get("emulator", "gba")).lower()
    if emulator != "gba":
        raise ValueError(f"This template supports the Game Boy ('gba') backend; game '{game}' requests '{emulator}'.")

    rom_path = os.path.join(ROMS_DIR, cfg.get("rom") or f"{game}.gb")

    prompt_path = os.path.join(game_dir, "prompt.txt")
    if os.path.exists(prompt_path):
        prompt = open(prompt_path).read().strip()
    else:
        prompt = str(cfg.get("prompt", DEFAULT_PROMPT))

    preload_path = os.path.join(game_dir, "preload.txt")
    if not os.path.exists(preload_path):
        preload_path = None

    return GameSpec(
        name=game,
        rom_path=rom_path,
        prompt=prompt,
        boot_frames=int(cfg.get("boot_frames", 600)),
        action_frames=int(cfg.get("action_frames", 15)),
        preload_path=preload_path,
        _scorer_factories=_build_scorer_factories(cfg, game_dir, threshold_override),
    )

"""Pluggable scorers for the HUD VideoGameBench template.

A scenario builds a :class:`ScoreBoard` from a list of scorers. After every
agent action, ``ScoreBoard.on_step`` feeds the new screen + emulator to each
scorer; at the end ``ScoreBoard.result`` collapses them into a single HUD
``EvaluationResult`` with one ``SubScore`` per scorer.

Three scorers ship:

* :class:`CheckpointScorer` — VideoGameBench-style perceptual-hash matching
  against ordered checkpoint images. Sparse / staged progress.
* :class:`RamScorer` — *dense* reward read from per-game RAM (richer than
  upstream VideoGameBench, which is screenshot-only).
* :class:`ExplorationScorer` — generic, ROM-agnostic dense reward = number of
  distinct screens seen / a target. Used by the bundled test game.

Subscore weights are normalized so positive weights sum to 1.0 and
``reward == sum(value * weight)``, satisfying ``hud.graders.EvaluationResult``
validation.
"""

from __future__ import annotations

from typing import Any, Protocol

import imagehash
from hud.graders import EvaluationResult, SubScore
from PIL import Image

# Hash size used by VideoGameBench (src/utils.py). Bigger = finer-grained.
_HASH_SIZE = 16


def hash_image(img: Image.Image) -> imagehash.ImageHash:
    return imagehash.average_hash(img, hash_size=_HASH_SIZE)


def hamming(h1: imagehash.ImageHash, h2: imagehash.ImageHash) -> int:
    return h1 - h2


class Scorer(Protocol):
    name: str
    weight: float

    def on_step(self, *, screen: Image.Image | None, emulator: Any | None) -> None: ...
    def value(self) -> float: ...  # normalized progress in [0, 1]
    def info(self) -> dict[str, Any]: ...
    def summary(self) -> str: ...


class CheckpointScorer:
    """Advance through ordered checkpoints by perceptual-hash match.

    Ported from VideoGameBench's ``_check_checkpoint_progress`` /
    ``is_same_hash``. Progress = checkpoints reached / total; the final
    checkpoint corresponds to value 1.0.
    """

    def __init__(
        self,
        checkpoint_hashes: list[imagehash.ImageHash],
        threshold: int = 8,
        weight: float = 1.0,
    ) -> None:
        self.name = "checkpoint"
        self.hashes = checkpoint_hashes
        self.threshold = threshold
        self.weight = weight
        self.total = len(checkpoint_hashes)
        self.idx = 0

    def on_step(self, *, screen: Image.Image | None, emulator: Any | None = None) -> None:
        if screen is None or self.idx >= self.total:
            return
        h = hash_image(screen)
        # Advance ONLY when this frame matches the next expected checkpoint, one at
        # a time. (A forward scan would let a single frame that happens to be close
        # to a *later* checkpoint skip the ones in between and award unearned
        # credit — e.g. shipped Zelda checkpoints 4 and 9 are only a few bits apart.)
        if (h - self.hashes[self.idx]) <= self.threshold:
            self.idx += 1

    def value(self) -> float:
        return self.idx / self.total if self.total else 0.0

    def info(self) -> dict[str, Any]:
        return {"checkpoints_reached": self.idx, "checkpoints_total": self.total, "threshold": self.threshold}

    def summary(self) -> str:
        return f"checkpoints {self.idx}/{self.total}"


class ExplorationScorer:
    """Generic dense reward: distinct screens seen / target.

    ROM-agnostic — rewards the agent for making the screen change in novel ways.
    Used by the bundled test game so the template runs end-to-end without any
    game-specific RAM map or checkpoint images.
    """

    def __init__(self, target_unique: int = 25, threshold: int = 4, weight: float = 1.0) -> None:
        self.name = "exploration"
        self.target = max(1, target_unique)
        self.threshold = threshold
        self.weight = weight
        self.seen: list[imagehash.ImageHash] = []

    def on_step(self, *, screen: Image.Image | None, emulator: Any | None = None) -> None:
        if screen is None:
            return
        h = hash_image(screen)
        for prev in self.seen:
            if (h - prev) <= self.threshold:
                return
        self.seen.append(h)

    def value(self) -> float:
        return min(1.0, len(self.seen) / self.target)

    def info(self) -> dict[str, Any]:
        return {"unique_screens": len(self.seen), "target": self.target}

    def summary(self) -> str:
        return f"explored {len(self.seen)}/{self.target}"


class RamScorer:
    """Dense reward computed from per-game RAM values.

    ``ram_map`` is an object exposing:
      * ``read(emulator) -> dict``       — pull raw values from RAM
      * ``reward(values: dict) -> float``— collapse them to [0, 1]
    Reward is monotonic (max over the episode) so transient dips (e.g. menus)
    don't penalize the agent.
    """

    def __init__(self, ram_map: Any, weight: float = 1.0) -> None:
        self.name = "ram"
        self.ram_map = ram_map
        self.weight = weight
        self.best = 0.0
        self.last_values: dict[str, Any] = {}

    def on_step(self, *, screen: Image.Image | None = None, emulator: Any | None) -> None:
        if emulator is None:
            return
        try:
            self.last_values = self.ram_map.read(emulator)
            r = float(self.ram_map.reward(self.last_values))
        except Exception as exc:  # never let a flaky RAM read crash an eval
            self.last_values = {"error": str(exc)}
            return
        self.best = max(self.best, max(0.0, min(1.0, r)))

    def value(self) -> float:
        return self.best

    def info(self) -> dict[str, Any]:
        return {"ram": self.last_values, "best_reward": round(self.best, 4)}

    def summary(self) -> str:
        return f"ram {self.best:.2f}"


class ScoreBoard:
    """Holds the active scorers and produces the final EvaluationResult."""

    def __init__(self, scorers: list[Scorer]) -> None:
        if not scorers:
            raise ValueError("ScoreBoard needs at least one scorer")
        self.scorers = scorers

    def on_step(self, *, screen: Image.Image | None = None, emulator: Any | None = None) -> None:
        for s in self.scorers:
            s.on_step(screen=screen, emulator=emulator)

    def summary(self) -> str:
        return ", ".join(s.summary() for s in self.scorers)

    def result(self, *, done: bool = True) -> EvaluationResult:
        positive = sum(s.weight for s in self.scorers if s.weight > 0)
        norm = positive if positive > 0 else 1.0

        subscores: list[SubScore] = []
        info: dict[str, Any] = {}
        reward = 0.0
        for s in self.scorers:
            # Normalize by the positive-weight total so positive weights sum to 1.0;
            # negative weights are kept (the HUD SDK treats them as penalties)
            # rather than silently zeroed.
            w = s.weight / norm
            v = s.value()
            subscores.append(SubScore(name=s.name, weight=w, value=v))
            reward += w * v
            info[s.name] = s.info()

        # Clamp tiny float drift so it never trips ScenarioResult validation.
        reward = max(0.0, min(1.0, round(reward, 6)))
        return EvaluationResult(
            reward=reward,
            done=done,
            content=self.summary(),
            subscores=subscores,
            info=info,
        )

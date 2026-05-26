"""Scenario primitives.

A scenario is an async generator registered against an ``Env`` via
``@env.scenario(...)``. It yields twice:

  1. ``yield {"prompt": ..., "requires": [...]}`` — setup done, here is
     the task; runner returns this to the harness.
  2. ``yield {"score": ..., "reason": ...}`` — evaluation result, after
     the runner pushes ``asend(evaluate_payload)``.

Scenarios take arbitrary ``**kwargs``; the harness sends them as ``args``
on ``scenarios.start`` and the runner forwards them. Closures over the
env's sandbox + module-level state are fine — scenarios run inside the
env process.
"""

from __future__ import annotations

import contextlib
import inspect
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Any

ScenarioFn = Callable[..., AsyncGenerator[dict[str, Any], dict[str, Any]]]


@dataclass(slots=True)
class Scenario:
    id: str
    description: str
    func: ScenarioFn

    def manifest_entry(self) -> dict[str, Any]:
        return {"id": self.id, "description": self.description}


class ScenarioRunner:
    """Drives one scenario through its prompt -> evaluate lifecycle."""

    def __init__(self, scenario: Scenario, args: dict[str, Any] | None = None) -> None:
        self.scenario = scenario
        self._args = args or {}
        self._gen: AsyncGenerator[dict[str, Any], dict[str, Any]] | None = None

        # Fail fast on bad args (TypeError before any side-effects run).
        try:
            inspect.signature(scenario.func).bind(**self._args)
        except TypeError as exc:
            raise TypeError(
                f"scenario {scenario.id!r}: bad args {sorted(self._args)}: {exc}",
            ) from exc

    async def start(self) -> dict[str, Any]:
        self._gen = self.scenario.func(**self._args)
        prompt = await self._gen.__anext__()
        if not isinstance(prompt, dict) or "prompt" not in prompt:
            raise RuntimeError(
                f"scenario {self.scenario.id!r}: first yield must be a dict with 'prompt'",
            )
        return prompt

    async def evaluate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._gen is None:
            raise RuntimeError("scenario not started")
        try:
            evaluation = await self._gen.asend(payload)
        except StopAsyncIteration as exc:
            raise RuntimeError(
                f"scenario {self.scenario.id!r}: ended without yielding an evaluation",
            ) from exc
        if not isinstance(evaluation, dict) or "score" not in evaluation:
            raise RuntimeError(
                f"scenario {self.scenario.id!r}: second yield must be a dict with 'score'",
            )
        with contextlib.suppress(Exception):
            await self._gen.aclose()
        return evaluation

    async def cancel(self) -> None:
        if self._gen is not None:
            with contextlib.suppress(Exception):
                await self._gen.aclose()
            self._gen = None


__all__ = ["Scenario", "ScenarioFn", "ScenarioRunner"]

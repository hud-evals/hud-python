"""Ambient run configuration: placement and schedule for the rollout engine.

A :class:`RunConfig` carries *how/where* rollouts execute — never *what* runs
(tasks) or *who* runs it (the agent). :func:`configure` binds one for a scope;
the engine resolves explicit call-site arguments first, then the ambient
config, then defaults (``provision()`` placement by the row's env name,
``group=1``)::

    with hud.configure(on=spawn("envs/browser.py"), group=8):
        await taskset.run(agent)  # spawned placement, 8 per task
        await fix_bug(d=3).run(agent)  # spawned placement

Scopes nest by per-field merge: an inner ``configure(group=4)`` inherits the
enclosing placement. The binding is a contextvar, so it follows async tasks
spawned inside the scope (e.g. gathered rollouts).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from hud.environment.runtime import Provider


@dataclass(frozen=True)
class RunConfig:
    """How and where rollouts run: placement provider plus batch schedule."""

    on: Provider | None = None
    group: int | None = None
    max_concurrent: int | None = None

    def __post_init__(self) -> None:
        if self.group is not None and self.group < 1:
            raise ValueError("group must be >= 1")
        if self.max_concurrent is not None and self.max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")

    def override(
        self,
        *,
        on: Provider | None = None,
        group: int | None = None,
        max_concurrent: int | None = None,
    ) -> RunConfig:
        """A copy with the given fields replaced (``None`` keeps this config's value)."""
        cfg = self
        if on is not None:
            cfg = replace(cfg, on=on)
        if group is not None:
            cfg = replace(cfg, group=group)
        if max_concurrent is not None:
            cfg = replace(cfg, max_concurrent=max_concurrent)
        return cfg


_ACTIVE: ContextVar[RunConfig | None] = ContextVar("hud_run_config", default=None)


def active() -> RunConfig:
    """The ambient :class:`RunConfig` (all-default when no scope is open)."""
    return _ACTIVE.get() or RunConfig()


@contextmanager
def configure(
    *,
    on: Provider | None = None,
    group: int | None = None,
    max_concurrent: int | None = None,
) -> Iterator[RunConfig]:
    """Bind the ambient :class:`RunConfig` for a scope.

    Fields merge over the enclosing scope (``None`` inherits); explicit
    arguments at a run call site always win over the ambient config.
    """
    merged = active().override(on=on, group=group, max_concurrent=max_concurrent)
    token = _ACTIVE.set(merged)
    try:
        yield merged
    finally:
        _ACTIVE.reset(token)


__all__ = ["RunConfig", "configure"]

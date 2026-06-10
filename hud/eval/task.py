"""Task: one task row — an env reference, an id, bound args, and metadata.

``foo(x, y)`` (an ``@env.task`` factory call) returns one of these, carrying
the defining :class:`~hud.environment.Environment`. The env is declarative —
identity lives on it (``env.name``) and rows deserialized from data carry a
bare ``Environment(name)`` reference. Running a task never needs a live env:
the prompt and grading arrive over the wire from whatever substrate placement
brought up.

Placement is ``on: Provider | None`` (see :mod:`hud.environment.runtime`).
:meth:`Task.run` resolves explicit > ambient :func:`hud.eval.configure` scope >
HUD-hosted provisioning by env name; :meth:`Task.session` is plumbing — it
takes an explicit provider or provisions, never reading ambient state.
Platform sync lives in :mod:`hud.eval.sync`.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from hud.clients import connect
from hud.environment.runtime import provision

from .rollout import Run, rollout

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.agents.base import Agent
    from hud.environment import Environment
    from hud.environment.runtime import Provider


@dataclass
class Task:
    """One concrete task: an env reference plus data (id, args, metadata).

    Pure data — holds no execution state, so one ``Task`` can drive many
    concurrent rollouts. ``run`` it (or open a ``session``) for a live ``Run``;
    placement comes from ``on=`` (a provider) or defaults to HUD-hosted
    provisioning by ``env.name``.
    """

    env: Environment
    id: str
    args: dict[str, Any] = field(default_factory=dict)
    slug: str | None = None
    validation: list[dict[str, Any]] | None = None
    agent_config: dict[str, Any] | None = None
    columns: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Task needs a task id")

    def default_slug(self) -> str:
        """A stable slug from the task id, disambiguated by an args hash when present."""
        if not self.args:
            return self.id
        digest = hashlib.sha1(  # noqa: S324 - non-crypto, stable disambiguator
            json.dumps(self.args, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()[:8]
        return f"{self.id}-{digest}"

    # ─── the portable row shape ───────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the portable row: ``{"env": {"name": ...}, "task": id, "args": ...}``.

        Metadata fields (slug, validation, agent_config, columns) are included
        only when set.
        """
        data: dict[str, Any] = {
            "env": {"name": self.env.name},
            "task": self.id,
            "args": dict(self.args),
        }
        if self.slug is not None:
            data["slug"] = self.slug
        if self.validation is not None:
            data["validation"] = self.validation
        if self.agent_config is not None:
            data["agent_config"] = self.agent_config
        if self.columns is not None:
            data["columns"] = self.columns
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Build a task row from :meth:`to_dict` output (env as a bare name reference)."""
        from hud.environment import Environment

        env_data = data.get("env")
        env_name = env_data.get("name") if isinstance(env_data, dict) else None
        if not isinstance(env_name, str) or not env_name:
            raise ValueError(f"task entry needs env.name: {data!r}")
        task_id = data.get("task")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(f"task entry needs a task id: {data!r}")
        args = data.get("args", {})
        if not isinstance(args, dict):
            raise ValueError(f"task entry args must be an object: {data!r}")
        return cls(
            env=Environment(env_name),
            id=task_id,
            args=args,
            slug=data.get("slug"),
            validation=data.get("validation"),
            agent_config=data.get("agent_config"),
            columns=data.get("columns"),
        )

    # ─── execution ────────────────────────────────────────────────────

    async def run(self, agent: Agent, *, on: Provider | None = None) -> Run:
        """Execute this task with ``agent`` through the rollout engine.

        Method sugar for :func:`hud.eval.rollout` — full engine semantics:
        trace context, telemetry reporting, grading, and failure isolation.
        ``on`` is the placement provider for this execution; left unset it
        resolves from the ambient :func:`hud.eval.configure` scope.
        """
        return await rollout(self, agent, on=on)

    @asynccontextmanager
    async def session(self, on: Provider | None = None) -> AsyncIterator[Run]:
        """Bring up a substrate, start this task on it, and yield the live ``Run``.

        The one substrate-lifecycle path: acquire the placement, connect,
        start; grade and tear down on exit. ``on`` is a provider, called with
        this task row (each session acquires one fresh substrate for it);
        without one the task provisions a HUD-hosted substrate by its env
        name. Ambient :func:`hud.eval.configure` state is resolved by the
        engine (:func:`hud.eval.rollout`), never here.
        """
        provider = on or provision()
        async with provider(self) as runtime, connect(runtime) as client:
            run = Run(client, self.id, self.args)
            run._runtime = runtime.url  # the placement record for the receipt
            async with run:
                yield run


def task(
    env: Environment,
    id: str,
    *,
    slug: str | None = None,
    validation: list[dict[str, Any]] | None = None,
    agent_config: dict[str, Any] | None = None,
    columns: dict[str, Any] | None = None,
    **args: Any,
) -> Task:
    """Author a concrete :class:`Task` on an env: ``task(env, "id", arg=...)``."""
    return Task(
        env=env,
        id=id,
        args=args,
        slug=slug,
        validation=validation,
        agent_config=agent_config,
        columns=columns,
    )


__all__ = ["Task", "task"]

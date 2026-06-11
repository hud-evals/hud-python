"""Task: one task row — an env name, a task id, bound args, and metadata.

``foo(x, y)`` (an ``@env.task`` factory call) returns one of these. ``env``
is the environment's *name*: the join key between the data plane (rows) and
whatever placement can bring that environment up. Running a task never needs
a live env — the prompt and grading arrive over the wire from the substrate
the placement brought up — so the row holds the reference explicitly instead
of wrapping it in an :class:`~hud.environment.Environment` object.

The model *is* the row: field names are the wire keys, so plain pydantic
(``Task.model_validate(entry)`` / ``task.model_dump()``) is the whole codec —
there is no bespoke serialization layer.

Placement is ``runtime: Provider | None`` (see :mod:`.runtime`).
Execution lives entirely in :mod:`.rollout` and scheduling in
:mod:`.taskset` — :meth:`Task.run` is the single-task form of
``Taskset.run``, so the row is always an argument to the engine, never a
participant in it. Platform sync lives in :mod:`hud.eval.sync`.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from hud.agents.base import Agent

    from .job import Job
    from .runtime import Provider


class Task(BaseModel):
    """One concrete task: an env name plus data (id, args, metadata).

    Pure data — holds no execution state, so one ``Task`` can drive many
    concurrent rollouts. ``run`` it for a graded :class:`~hud.eval.job.Job`;
    placement comes from ``runtime=`` (a provider) or defaults to HUD-hosted
    provisioning by ``env``.
    """

    env: str = Field(min_length=1)
    id: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    slug: str | None = None
    validation: list[dict[str, Any]] | None = None
    agent_config: dict[str, Any] | None = None
    columns: dict[str, Any] | None = None

    def default_slug(self) -> str:
        """A stable slug from the task id, disambiguated by an args hash when present."""
        if not self.args:
            return self.id
        digest = hashlib.sha1(  # noqa: S324 - non-crypto, stable disambiguator
            json.dumps(self.args, sort_keys=True, default=str).encode("utf-8"),
        ).hexdigest()[:8]
        return f"{self.id}-{digest}"

    # ─── execution ────────────────────────────────────────────────────

    async def run(
        self,
        agent: Agent,
        *,
        runtime: Provider | None = None,
        group: int | None = None,
        max_concurrent: int | None = None,
        job: Job | None = None,
    ) -> Job:
        """Run this task with ``agent``: the single-task form of ``Taskset.run``.

        Identical scheduling semantics — one HUD job as the receipt (or an
        open ``job`` from :meth:`Job.start` to accumulate into), ``group``
        repeats sharing a group_id, ``max_concurrent`` capping parallelism —
        over a taskset of one. ``runtime`` is the placement provider; left
        unset it defaults to HUD-hosted provisioning by ``env`` name.
        """
        from .taskset import Taskset  # circular: taskset -> sync -> task

        return await Taskset(self.default_slug(), [self]).run(
            agent, runtime=runtime, group=group, max_concurrent=max_concurrent, job=job
        )


__all__ = ["Task"]

"""Task: one task row — an env name, a task id, bound args, and metadata.

``foo(x, y)`` (an ``@env.template`` factory call) returns one of these. ``env``
is the environment's *name*: the join key between the data plane (rows) and
whatever placement can bring that environment up. Running a task never needs
a live env — the prompt and grading arrive over the wire from the substrate
the placement brought up — so the row holds the reference explicitly instead
of wrapping it in an :class:`~hud.environment.Environment` object.

The model *is* the row: field names are the wire keys, so plain pydantic
(``Task.model_validate(entry)`` / ``task.model_dump()``) is the whole codec —
there is no bespoke serialization layer.

Placement is ``runtime: Provider | HostedRuntime | None`` (see :mod:`.runtime`).
Execution lives entirely in :mod:`.rollout` and scheduling in
:mod:`.taskset` — :meth:`Task.run` is the single-task form of
``Taskset.run``, so the row is always an argument to the engine, never a
participant in it. Platform sync lives in :mod:`hud.eval.sync`.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .runtime import RuntimeConfig

if TYPE_CHECKING:
    from hud.agents.base import Agent

    from .job import Job
    from .runtime import HostedRuntime, Provider


def _default_slug(data: dict[str, Any]) -> str:
    task_id = data.get("id")
    if not isinstance(task_id, str):
        return ""
    args = data.get("args")
    if not isinstance(args, dict) or not args:
        return task_id
    digest = hashlib.sha1(  # noqa: S324 - non-crypto, stable disambiguator
        json.dumps(args, sort_keys=True, default=str).encode("utf-8"),
    ).hexdigest()[:8]
    return f"{task_id}-{digest}"


class Task(BaseModel):
    """One concrete task: an env name plus data (id, args, metadata).

    Pure data — holds no execution state, so one ``Task`` can drive many
    concurrent rollouts. ``run`` it for a graded :class:`~hud.eval.job.Job`;
    placement comes from ``runtime=`` (a provider), else the HUD runtime
    tunnel by ``env`` name.
    """

    model_config = ConfigDict(validate_assignment=True)

    env: str = Field(min_length=1)
    id: str = Field(min_length=1)
    args: dict[str, Any] = Field(default_factory=dict)
    slug: str = Field(default_factory=_default_slug, min_length=1)
    validation: list[dict[str, Any]] | None = None
    agent_config: dict[str, Any] | None = None
    #: Arbitrary metadata fields surfaced as filterable columns / leaderboard
    #: facets on the platform (e.g. ``{"difficulty": "easy", "suite": "coding"}``).
    columns: dict[str, Any] | None = None
    #: Optional row-level runtime construction input. Runtime adapters apply the
    #: supported subset into their native launch shape or reject it.
    runtime_config: RuntimeConfig | None = None
    #: Optional agent-less task whose evaluation is the grade of record. The
    #: rollout completes this task first, then starts and grades the verifier
    #: with the same answer. Placement may reuse the live substrate when both
    #: tasks name the same environment.
    verifier: Task | None = None

    @model_validator(mode="after")
    def _reject_nested_verifier(self) -> Self:
        if self.verifier is not None and self.verifier.verifier is not None:
            raise ValueError("nested verifier tasks are not supported")
        return self

    # ─── execution ────────────────────────────────────────────────────

    async def run(
        self,
        agent: Agent,
        *,
        runtime: Provider | HostedRuntime | None = None,
        group: int | None = None,
        max_concurrent: int | None = None,
        job: Job | None = None,
        rollout_timeout: float | None = None,
    ) -> Job:
        """Run this task with ``agent``: the single-task form of ``Taskset.run``.

        Identical scheduling semantics — one HUD job as the receipt (or an
        open ``job`` from :meth:`Job.start` to accumulate into), ``group``
        repeats sharing a group_id, ``max_concurrent`` capping parallelism —
        over a taskset of one. ``runtime`` is the placement; left unset it
        falls back to the HUD runtime tunnel by ``env`` name. For a local
        run, pass one explicitly (``runtime=LocalRuntime("env.py")``).
        """
        from .taskset import Taskset  # circular: taskset -> sync -> task

        taskset = Taskset(self.slug, [self])
        return await taskset.run(
            agent,
            runtime=runtime,
            group=group,
            max_concurrent=max_concurrent,
            job=job,
            rollout_timeout=rollout_timeout,
        )


__all__ = ["RuntimeConfig", "Task"]

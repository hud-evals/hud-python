"""HUD eval: the v6 execution surface.

Define a :class:`Task` (a row pointing at its env), group many into a
:class:`Taskset`, and run agents against live :class:`~hud.eval.Run`s.
:func:`rollout` is the execution atom (one agent, one task, fully recorded);
``Task.run`` is its per-task sugar and ``Taskset.run`` the batch scheduler over
it. A :class:`Job` is the platform/batch receipt for a taskset run.

Placement is a provider passed at execution time (see
:mod:`hud.environment.runtime`): ``spawn`` a local source, ``provision`` a
HUD-hosted substrate, or attach to a ``Runtime(url)``. A :func:`configure`
scope binds ambient placement/schedule for every run inside it::

    from hud.eval import Taskset, configure
    from hud.environment import spawn

    run = await my_task(a=1).run(agent, on=spawn("env.py"))
    with configure(on=spawn("env.py"), group=8):
        job = await Taskset("demo", [task(d) for d in range(5)]).run(agent)
"""

from __future__ import annotations

from hud.types import Trace

from .chat import Chat
from .config import RunConfig, configure
from .job import Job
from .rollout import Grade, Run, rollout
from .sync import SyncPlan
from .task import Task, task
from .taskset import Taskset
from .training import HudTrainingClient, Rewarded, TrainingConfig, group_relative

__all__ = [
    "Chat",
    "Grade",
    "HudTrainingClient",
    "Job",
    "Rewarded",
    "Run",
    "RunConfig",
    "SyncPlan",
    "Task",
    "Taskset",
    "Trace",
    "TrainingConfig",
    "configure",
    "group_relative",
    "rollout",
    "task",
]

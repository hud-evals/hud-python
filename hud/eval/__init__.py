"""HUD eval: the v6 execution surface.

Define a :class:`Task` (a row pointing at its env), group many into a
:class:`Taskset`, and run agents against live :class:`~hud.eval.Run`s.
:func:`rollout` is the execution atom (one agent, one task, fully recorded);
``Task.run`` and ``Taskset.run`` are the scheduler over it, both returning a
:class:`Job` — the platform receipt. There are no standalone traces: every
run reports under a job.

This is the top layer: eval composes :mod:`hud.environment` and
:mod:`hud.agents`, which never import each other and never import eval back —
agents see eval only through the ``Run`` handle they are driven with. (Sole
exception: calling an ``@env.template`` declaration constructs the eval ``Task``
row.)

Placement is passed at execution time (see :mod:`.runtime`): ``LocalRuntime`` a
local source, ``DockerRuntime`` an image, ``Runtime(url)`` an env served
elsewhere (all :class:`Provider`s driven here), or ``HUDRuntime`` to run the
rollout on a HUD-leased box with the agent co-located with the env::

    from hud.eval import LocalRuntime, Taskset

    job = await my_task(a=1).run(agent, runtime=LocalRuntime("env.py"))
    job = await Taskset("demo", [my_task(d) for d in range(5)]).run(
        agent, runtime=LocalRuntime("env.py"), group=8
    )
"""

from __future__ import annotations

from hud.types import Trace

from .chat import Chat
from .job import Job
from .run import Grade, Run, rollout
from .runtime import (
    DaytonaRuntime,
    DockerRuntime,
    HUDRuntime,
    LocalRuntime,
    ModalRuntime,
    Provider,
    Runtime,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
)
from .sync import SyncPlan
from .task import Task
from .taskset import Taskset
from .training import HudTrainingClient, Rewarded, TrainingConfig, group_relative

__all__ = [
    "Chat",
    "DaytonaRuntime",
    "DockerRuntime",
    "Grade",
    "HUDRuntime",
    "HudTrainingClient",
    "Job",
    "LocalRuntime",
    "ModalRuntime",
    "Provider",
    "Rewarded",
    "Run",
    "Runtime",
    "RuntimeConfig",
    "RuntimeGPU",
    "RuntimeLimits",
    "RuntimeResources",
    "SyncPlan",
    "Task",
    "Taskset",
    "Trace",
    "TrainingConfig",
    "group_relative",
    "rollout",
]

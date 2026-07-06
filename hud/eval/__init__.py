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
local source served in-process, ``DockerRuntime`` an image,
``Runtime(url)`` an env served elsewhere, ``HUDRuntime`` a HUD runtime tunnel,
or ``HostedRuntime`` to run the whole rollout remotely on the platform::

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
from .run import Grade, Run, rollout, rollout_group
from .runtime import (
    DaytonaRuntime,
    DockerRuntime,
    HostedRuntime,
    HUDRuntime,
    LocalRuntime,
    ModalRuntime,
    Provider,
    Runtime,
    RuntimeConfig,
    RuntimeGPU,
    RuntimeLimits,
    RuntimeResources,
    SubprocessRuntime,
)
from .sync import SyncPlan
from .task import Task
from .taskset import Taskset

__all__ = [
    "Chat",
    "DaytonaRuntime",
    "DockerRuntime",
    "Grade",
    "HUDRuntime",
    "HostedRuntime",
    "Job",
    "LocalRuntime",
    "ModalRuntime",
    "Provider",
    "Run",
    "Runtime",
    "RuntimeConfig",
    "RuntimeGPU",
    "RuntimeLimits",
    "RuntimeResources",
    "SubprocessRuntime",
    "SyncPlan",
    "Task",
    "Taskset",
    "Trace",
    "rollout",
    "rollout_group",
]

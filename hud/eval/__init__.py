"""HUD eval: the v6 execution surface.

Define a :class:`Task` (a concrete task bound to an env/sandbox), group
many into a :class:`Taskset`, and run agents against live
:class:`~hud.client.Run`s. A :class:`Job` is the platform/batch receipt for a
taskset run; ``Run`` remains the execution atom agents drive.

    from hud.eval import Taskset, Task, launch

    job = await Taskset.from_tasks("demo", [task(d) for d in range(5)]).run(agent, group=8)
"""

from __future__ import annotations

from hud.client import Grade, Run
from hud.types import Trace

from .chat import Chat
from .job import Job
from .launch import launch
from .sandbox import (
    Channel,
    HudSandbox,
    LocalSandbox,
    RemoteSandbox,
    Sandbox,
    as_sandbox,
    load_environment,
    load_module,
    sandbox_from_ref,
)
from .task import Task, task
from .taskset import SyncPlan, Taskset
from .training import HudTrainingClient, Rewarded, TrainingConfig, group_relative

__all__ = [
    "Channel",
    "Chat",
    "Grade",
    "HudSandbox",
    "HudTrainingClient",
    "Job",
    "LocalSandbox",
    "RemoteSandbox",
    "Rewarded",
    "Run",
    "Sandbox",
    "SyncPlan",
    "Task",
    "Taskset",
    "Trace",
    "TrainingConfig",
    "as_sandbox",
    "group_relative",
    "launch",
    "load_environment",
    "load_module",
    "sandbox_from_ref",
    "task",
]

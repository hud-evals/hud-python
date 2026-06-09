"""HUD eval: the v6 execution surface.

Define a :class:`Task` (a concrete task bound to an env/sandbox), group
many into a :class:`Taskset`, ``launch`` a :class:`Sandbox`, and ship rewarded
:class:`~hud.client.Run`s to the :class:`HudTrainingClient`.

    from hud.eval import Taskset, Task, launch

    job = await Taskset.from_tasks("demo", [task(d) for d in range(5)]).run(agent, group=8)
"""

from __future__ import annotations

from .launch import launch
from .remote import submit_rollouts
from .sandbox import (
    Channel,
    HudSandbox,
    LocalSandbox,
    RemoteSandbox,
    Sandbox,
    as_sandbox,
    load_module,
    sandbox_from_ref,
)
from .task import Task, task
from .taskset import Job, SyncPlan, Taskset
from .training import HudTrainingClient, Rewarded, TrainingConfig, group_relative

__all__ = [
    "Channel",
    "HudSandbox",
    "HudTrainingClient",
    "Job",
    "LocalSandbox",
    "RemoteSandbox",
    "Rewarded",
    "Sandbox",
    "SyncPlan",
    "Task",
    "Taskset",
    "TrainingConfig",
    "as_sandbox",
    "group_relative",
    "launch",
    "load_module",
    "sandbox_from_ref",
    "submit_rollouts",
    "task",
]

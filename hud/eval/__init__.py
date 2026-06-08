"""HUD eval: the v6 execution surface.

Define a :class:`Variant` (a parameterized task bound to an env/sandbox), group
many into a :class:`Taskset`, ``launch`` a :class:`Sandbox`, and ship rewarded
:class:`~hud.client.Run`s to the :class:`HudTrainingClient`.

    from hud.eval import Taskset, Variant, launch

    runs = await Taskset(task(d) for d in range(5)).run(agent, group=8)
"""

from __future__ import annotations

from .launch import launch
from .sandbox import (
    LocalSandbox,
    RemoteSandbox,
    Runtime,
    Sandbox,
    as_sandbox,
    load_module,
    sandbox_from_ref,
)
from .source import collect_variants, load_variants, load_variants_json
from .taskset import Taskset
from .training import HudTrainingClient, Rewarded, TrainingConfig, group_relative
from .variant import Variant, variant

__all__ = [
    "HudTrainingClient",
    "LocalSandbox",
    "RemoteSandbox",
    "Rewarded",
    "Runtime",
    "Sandbox",
    "Taskset",
    "TrainingConfig",
    "Variant",
    "as_sandbox",
    "collect_variants",
    "group_relative",
    "launch",
    "load_module",
    "load_variants",
    "load_variants_json",
    "sandbox_from_ref",
    "variant",
]

"""Shared scaffolding for the RL-training cookbook scripts.

The training loop is agnostic to where rollouts come from — it only consumes
``job.runs`` (each carrying a trajectory + reward). So the real setup and the
local quickstart differ only in *which taskset* and *which runtime* you hand to
``Taskset.run``; the training code never changes.

``load_taskset_and_runtime()`` picks between them from the environment:

- ``HUD_TASKSET`` set — the real flow: load a taskset you already built and
  pushed (``hud deploy`` + ``hud sync``) from the platform with
  ``Taskset.from_api``, and run every rollout on a leased HUD box with
  ``HUDRuntime`` (the agent runs remotely, next to the env). Nothing local.
- unset — a self-contained quickstart: a tiny arithmetic taskset driven against
  the bundled ``env.py`` locally.
"""

from __future__ import annotations

import os
import random

from hud.eval import HUDRuntime, LocalRuntime, Provider, Taskset

from env import multiply


def load_taskset_and_runtime() -> tuple[Taskset, Provider | HUDRuntime]:
    """Resolve the rollout source from ``HUD_TASKSET`` (see module docstring)."""
    taskset_name = os.environ.get("HUD_TASKSET")
    if taskset_name:
        return Taskset.from_api(taskset_name), HUDRuntime()

    # Three-digit x two-digit multiplication *with* reasoning: hard enough that a
    # 4B reasoner is right only sometimes (a sub-1.0 baseline with within-group
    # variance — the GRPO signal). 2x2-with-CoT was ~100% and no-CoT was ~0%;
    # neither left a gradient, so we land in between.
    rng = random.Random(0)
    local = Taskset(
        "mult",
        [multiply(a=rng.randint(100, 999), b=rng.randint(11, 99)) for _ in range(4)],
    )
    return local, LocalRuntime("env.py")

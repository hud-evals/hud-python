"""Reproduce + verify: ``LocalRuntime`` reaps grandchild processes on teardown.

``LocalRuntime`` spawns ``python -m hud.environment.server <path>`` as a child
and terminates it on context exit. If the served env spawns its own subprocess —
a *grandchild* — a single-pid SIGTERM never reaches it. The fix spawns the child
in its own session and signals the whole process group (SIGTERM, then SIGKILL),
so grandchildren are reaped with the rollout.

This file is its own env source (self-spawning, ``LocalRuntime(__file__)``):

- imported in the child it only defines ``env``, whose ``@env.initialize``
  spawns a sleeper grandchild and records its pid to ``$GRANDCHILD_PID_FILE``;
- run directly it drives ``LocalRuntime`` against itself and reports whether
  that grandchild survived teardown.

    python hud/eval/tests/test_local_runtime_orphan.py   # manual repro
    pytest hud/eval/tests/test_local_runtime_orphan.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from hud.environment import Environment

env = Environment("orphan-env")


@env.initialize
async def _spawn_grandchild() -> None:
    # A long-lived grandchild the env "owns": not in the signal path of a
    # single-pid SIGTERM, so only a process-group kill reaps it.
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(100000)"])  # noqa: ASYNC220
    Path(os.environ["GRANDCHILD_PID_FILE"]).write_text(str(proc.pid))


# ─── repro (only runs in this process, never in the spawned child) ──────────


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just not ours to signal
    return True


async def _grandchild_survives_teardown() -> int:
    """Drive LocalRuntime against this file; return the grandchild pid if it
    outlived teardown, else 0."""
    import asyncio
    import tempfile

    from hud.eval.runtime import LocalRuntime
    from hud.eval.task import Task

    pid_file = Path(tempfile.mkstemp(suffix=".pid", prefix="orphan-")[1])
    os.environ["GRANDCHILD_PID_FILE"] = str(pid_file)
    try:
        async with LocalRuntime(__file__)(Task(env="orphan-env", id="noop")):
            pid = int(pid_file.read_text())  # initialize ran before the port line
            assert _alive(pid), "grandchild should be running while the server is up"
        await asyncio.sleep(0.5)  # let the cascade-kill land
        return pid if _alive(pid) else 0
    finally:
        pid_file.unlink(missing_ok=True)


async def test_local_runtime_kills_grandchildren() -> None:
    orphan = await _grandchild_survives_teardown()
    if orphan:
        os.kill(orphan, 9)  # don't leak it out of the test session
    assert not orphan, f"grandchild {orphan} was orphaned by LocalRuntime teardown"


if __name__ == "__main__":
    import asyncio

    orphan = asyncio.run(_grandchild_survives_teardown())
    if orphan:
        print(f"BUG: grandchild {orphan} still alive after teardown")  # noqa: T201
        os.kill(orphan, 9)
        sys.exit(1)
    print("OK: grandchild was reaped with the group")  # noqa: T201

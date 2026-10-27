"""What environment a process the workspace starts sees.

HUD's own variables in the serving process are a credential leak where they hold
a key, and a tell everywhere else: an agent that finds ``HUD_`` anything knows
what is running it. So no entry point hands them on. What the task declares is
the task's, HUD-shaped name or not, and the image's ``PATH`` (the interpreters
and tools the task installed) reaches every process.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

import pytest

from hud.capabilities import Connection, SSHClient
from hud.clients import connect
from hud.environment import Environment
from hud.environment.process_guard import process_connections_supported
from hud.eval import LocalRuntime, Task

from .conftest import SANDBOX, shell

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from tests.conftest import HudEnv

TASK_ENV = {"WORKSPACE_ENV": "present", "HUD_TASK_DECLARED": "mine"}

# What the workspace handed the session's shell, read before a login profile
# rewrites it where /proc shows it (Debian's /etc/profile resets PATH).
SHELL_ENV = "if [ -r /proc/$$/environ ]; then tr '\\0' '\\n' < /proc/$$/environ; else env; fi"


def parsed(output: str | bytes) -> dict[str, str]:
    text = output.decode() if isinstance(output, bytes) else output
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


async def ssh_session(root: Path, **options: Any) -> dict[str, str]:
    async with shell(root, env=TASK_ENV, **options) as (_, client):
        return parsed((await client.run(SHELL_ENV, check=True)).stdout or "")


async def unsandboxed_session(root: Path) -> dict[str, str]:
    return await ssh_session(root)


async def sandboxed_session(root: Path) -> dict[str, str]:
    return await ssh_session(root, require_isolation=True)


async def dropped_session(root: Path) -> dict[str, str]:
    return await ssh_session(root, require_isolation=True, shell_uid=1000)


async def bound_connection_session(root: Path) -> dict[str, str]:
    if not process_connections_supported():
        pytest.skip("process connection guards are unavailable")
    connection = Connection(
        name="inference",
        capability="ssh",
        url="http://127.0.0.1:9",
        headers={"Authorization": "Bearer scoped-runtime-token"},
    )
    env = Environment("bound")
    env.workspace(
        root, env=TASK_ENV, allowed_hosts={"pypi.org"}, require_isolation=True, track_files=False
    )
    async with (
        LocalRuntime(env)(Task(env=env.name, id="bound")) as runtime,
        connect(runtime, connections=(connection,)) as client,
    ):
        ssh = cast("SSHClient", await client.open("shell"))
        process = await ssh.create_process(SHELL_ENV, connections=(connection,))
        return parsed((await process.wait(check=True)).stdout or b"")


async def verifier_run(root: Path) -> dict[str, str]:
    async with shell(root, env=TASK_ENV, require_isolation=True) as (workspace, _):
        result = await workspace.run(
            ["env"],
            env={"VERIFIER_ONLY": "yes"},
            identity=None,
            inherit_workspace_env=False,
            allowed_hosts={"pypi.org"},
        )
    return parsed(result.stdout)


async def isolated_run(root: Path) -> dict[str, str]:
    async with shell(root, env=TASK_ENV, require_isolation=True) as (workspace, _):
        result = await workspace.run(["env"], isolated=True, identity=None)
    return parsed(result.stdout)


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param(
            unsandboxed_session, {**TASK_ENV, "ORDINARY": "kept"}, id="ssh-session-unsandboxed"
        ),
        pytest.param(
            sandboxed_session,
            {**TASK_ENV, "ORDINARY": "kept"},
            id="ssh-session-sandboxed",
            marks=SANDBOX,
        ),
        pytest.param(
            dropped_session,
            {**TASK_ENV, "ORDINARY": None, "HOME": "/workspace"},
            id="ssh-session-dropped-to-shell-uid",
            marks=SANDBOX,
        ),
        pytest.param(
            bound_connection_session,
            {**TASK_ENV, "ORDINARY": "kept", "http_proxy": "http://127.0.0.1:3128"},
            id="ssh-session-with-a-bound-connection",
            marks=SANDBOX,
        ),
        pytest.param(
            verifier_run,
            {
                **dict.fromkeys(TASK_ENV),
                "ORDINARY": "kept",
                "VERIFIER_ONLY": "yes",
            },
            id="verifier-run-without-workspace-env",
            marks=SANDBOX,
        ),
        pytest.param(
            isolated_run, {**TASK_ENV, "ORDINARY": "kept"}, id="isolated-run", marks=SANDBOX
        ),
    ],
)
async def test_hud_configuration_never_reaches_a_process_but_the_task_env_and_path_do(
    tmp_path: Path,
    hud_env: HudEnv,
    entry: Callable[[Path], Awaitable[dict[str, str]]],
    expected: dict[str, str | None],
) -> None:
    hud_env.set(
        HUD_API_KEY="super-secret",
        ORDINARY="kept",
        PATH=f"/task/bin{os.pathsep}{os.environ['PATH']}",
    )

    seen = await entry(tmp_path / "root")

    assert not [name for name in seen if name.startswith("HUD_") and name not in TASK_ENV]
    assert "super-secret" not in seen.values()
    assert "/task/bin" in seen["PATH"].split(os.pathsep)
    assert {name: seen.get(name) for name in expected} == expected

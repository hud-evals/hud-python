"""A workspace publishes its ``ssh/2`` binding when the environment serves, never before.

``env.workspace(root)`` is pure declaration: keys, sockets and the directory
materialize at serve time, the manifest carries the published address, and
stopping the environment takes the daemon down.
"""

from __future__ import annotations

import asyncio
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import asyncssh
import pytest

from hud.clients import connect
from hud.environment import Environment, Workspace
from hud.environment.workspace import usable_bwrap
from hud.eval import LocalRuntime, Task
from tests.fixtures.environments import served

from .conftest import SANDBOX, shell

if TYPE_CHECKING:
    from tests.conftest import HudEnv

TEMP = Path(tempfile.gettempdir()).resolve()


async def test_a_workspace_lives_exactly_while_its_environment_serves(tmp_path: Path) -> None:
    env = Environment("ws-env")
    env.workspace(tmp_path / "root")
    declared = (list(env.capabilities), (tmp_path / "root").exists())

    async with LocalRuntime(env)(Task(env=env.name, id="test")) as runtime:
        async with connect(runtime) as client:
            first = client.binding("shell").params["host_pubkey"]
        async with connect(runtime) as client:
            again = client.binding("shell").params["host_pubkey"]
        backing_port = urlsplit(env.capability("shell").url).port
    async with served(env):
        restarted = [capability.name for capability in env.capabilities]

    assert declared == ([], False)
    assert again == first
    assert restarted == ["shell", "filetracking"]
    with pytest.raises(OSError):
        _, writer = await asyncio.open_connection("127.0.0.1", backing_port)
        writer.close()


@pytest.mark.parametrize(
    ("options", "variables", "isolation", "published"),
    [
        pytest.param({}, {}, "none", ["shell", "filetracking"], id="tracked-by-default"),
        pytest.param(
            {}, {"HUD_FILE_TRACKING_ENABLED": "false"}, "none", ["shell"], id="tracking-off"
        ),
        pytest.param({"track_files": False}, {}, "none", ["shell"], id="the-workspace-opts-out"),
        pytest.param(
            {}, {}, "bwrap", ["shell", "filetracking"], id="sandboxed-host", marks=SANDBOX
        ),
    ],
)
async def test_serving_publishes_the_bindings_its_settings_ask_for(
    tmp_path: Path,
    hud_env: HudEnv,
    options: dict[str, Any],
    variables: dict[str, str],
    isolation: str,
    published: list[str],
) -> None:
    if isolation == "none" and usable_bwrap() is not None:
        pytest.skip("this host sandboxes every workspace")
    hud_env.set(**{"HUD_FILE_TRACKING_ENABLED": None, **variables})
    env = Environment("ws-env")
    env.workspace(tmp_path / "root", **options)

    async with served(env) as client:
        assert client.manifest is not None
        bindings = {binding.name: binding for binding in client.manifest.bindings}
        shell_binding = client.binding("shell")
        tracking = client.binding("filetracking") if "filetracking" in published else None

    assert list(bindings) == published
    assert bindings["shell"].params["isolation"] == isolation
    assert (shell_binding.protocol, urlsplit(shell_binding.url).scheme) == ("ssh/2", "ssh")
    assert shell_binding.params["host_pubkey"].startswith("ssh-ed25519")
    if tracking is not None:
        assert tracking.protocol == "filetracking/1"
        assert tracking.params == {
            "root": (tmp_path / "root").resolve().as_posix(),
            "setup_diff": True,
        }


async def test_session_credentials_live_privately_outside_the_served_root(tmp_path: Path) -> None:
    """The agent's shell root must not hold its own SSH key material."""
    async with shell(tmp_path / "root") as (workspace, client):
        credentials = Path(client.capability.params["client_key_path"]).parent
        mode = stat.S_IMODE(credentials.stat().st_mode)
        listing = await client.run("ls -A", check=True)
        other = Workspace(tmp_path / "other")
        other_credentials = Path(other.capability().params["client_key_path"]).parent
        await other.stop()

    assert credentials.resolve().is_relative_to(TEMP)
    assert not credentials.is_relative_to(workspace.root)
    assert mode == 0o700
    assert other_credentials != credentials
    assert listing.stdout == ""
    assert not credentials.exists()


async def test_the_ssh_daemon_serves_no_sftp_subsystem(tmp_path: Path) -> None:
    async with shell(tmp_path / "root") as (_, client):
        with pytest.raises(asyncssh.ChannelOpenError):
            await client.conn.start_sftp_client()
